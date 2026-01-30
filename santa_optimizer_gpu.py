#!/usr/bin/env python3
"""
Santa Christmas Tree Packing Optimizer - GPU Accelerated Version
Uses CuPy for CUDA acceleration on overlap detection and SA optimization

Requirements:
    pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version
    pip install numpy pandas shapely scipy
"""

import os
import sys
import time
import warnings
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull
from shapely import affinity
from shapely.geometry import Polygon

warnings.filterwarnings("ignore")

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("✅ GPU acceleration enabled (CuPy)")
    try:
        device_id = cp.cuda.Device().id
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props["name"].decode("utf-8")
    except Exception:
        name = "Unknown Device"

    print(f"   Device: {name}")
    print(f"   Memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not found, falling back to CPU (NumPy)")
    print("   Install with: pip install cupy-cuda11x (or cupy-cuda12x)")
    import numpy as cp  # Fallback to numpy

getcontext().prec = 30

print("\n🎄 Santa Optimizer - GPU Accelerated Version")
print("=" * 60)

# === TREE POLYGON VERTICES (constant) ===
# 15 vertices of the Christmas tree shape
TREE_VERTICES_X = np.array(
    [
        0.0,
        0.125,
        0.0625,
        0.2,
        0.1,
        0.35,
        0.075,
        0.075,
        -0.075,
        -0.075,
        -0.35,
        -0.1,
        -0.2,
        -0.0625,
        -0.125,
    ],
    dtype=np.float64,
)
TREE_VERTICES_Y = np.array(
    [0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5],
    dtype=np.float64,
)
NUM_VERTICES = 15


# === GPU KERNELS ===
if GPU_AVAILABLE:
    # Kernel to transform tree vertices (rotate + translate)
    transform_kernel = cp.RawKernel(
        r"""
    extern "C" __global__
    void transform_trees(
        const double* base_x, const double* base_y,
        const double* cx, const double* cy, const double* angles,
        double* out_x, double* out_y,
        int n_trees, int n_verts
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int tree_idx = tid / n_verts;
        int vert_idx = tid % n_verts;
        
        if (tree_idx >= n_trees) return;
        
        double angle_rad = angles[tree_idx] * 3.14159265358979323846 / 180.0;
        double cos_a = cos(angle_rad);
        double sin_a = sin(angle_rad);
        
        double bx = base_x[vert_idx];
        double by = base_y[vert_idx];
        
        // Rotate then translate
        out_x[tid] = bx * cos_a - by * sin_a + cx[tree_idx];
        out_y[tid] = bx * sin_a + by * cos_a + cy[tree_idx];
    }
    """,
        "transform_trees",
    )

    # Kernel to compute bounding boxes
    bbox_kernel = cp.RawKernel(
        r"""
    extern "C" __global__
    void compute_bboxes(
        const double* verts_x, const double* verts_y,
        double* bbox,  // [n_trees, 4] -> minx, miny, maxx, maxy
        int n_trees, int n_verts
    ) {
        int tree_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tree_idx >= n_trees) return;
        
        double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
        
        for (int v = 0; v < n_verts; v++) {
            int idx = tree_idx * n_verts + v;
            double x = verts_x[idx];
            double y = verts_y[idx];
            if (x < minx) minx = x;
            if (y < miny) miny = y;
            if (x > maxx) maxx = x;
            if (y > maxy) maxy = y;
        }
        
        bbox[tree_idx * 4 + 0] = minx;
        bbox[tree_idx * 4 + 1] = miny;
        bbox[tree_idx * 4 + 2] = maxx;
        bbox[tree_idx * 4 + 3] = maxy;
    }
    """,
        "compute_bboxes",
    )

    overlap_module = cp.RawModule(
        code=r"""
    __device__ bool segments_intersect(double ax, double ay, double bx, double by, 
                                     double cx, double cy, double dx, double dy) {
        double d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
        double d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
        double d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        double d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
        return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
    }

    __device__ bool point_in_polygon(double px, double py, const double* poly_x, const double* poly_y, int n_verts) {
        bool inside = false;
        int j = n_verts - 1;
        for (int i = 0; i < n_verts; i++) {
            if (((poly_y[i] > py) != (poly_y[j] > py)) &&
                (px < (poly_x[j] - poly_x[i]) * (py - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i])) {
                inside = !inside;
            }
            j = i;
        }
        return inside;
    }

    extern "C" __global__
    void check_overlap_detailed(
        const double* verts_x, const double* verts_y,
        const double* bbox,
        int* overlap_found,
        int n_trees, int n_verts
    ) {
        // Grid-stride loop 2D
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        
        int stride_x = gridDim.x * blockDim.x;
        int stride_y = gridDim.y * blockDim.y;

        for (int i = idx_x; i < n_trees; i += stride_x) {
            for (int j = idx_y; j < n_trees; j += stride_y) {
                if (i >= j) continue;
                
                if (*overlap_found) return; // Early exit

                // 1. BBox Check
                double tax0 = bbox[i * 4 + 0], tay0 = bbox[i * 4 + 1];
                double tax1 = bbox[i * 4 + 2], tay1 = bbox[i * 4 + 3];
                double tbx0 = bbox[j * 4 + 0], tby0 = bbox[j * 4 + 1];
                double tbx1 = bbox[j * 4 + 2], tby1 = bbox[j * 4 + 3];

                if (tax1 < tbx0 || tbx1 < tax0 || tay1 < tby0 || tby1 < tay0) {
                    continue; // No bbox overlap
                }

                // 2. Detailed Polygon Check
                // Pointers to vertices for tree i and j
                const double* ax = &verts_x[i * n_verts];
                const double* ay = &verts_y[i * n_verts];
                const double* bx = &verts_x[j * n_verts];
                const double* by = &verts_y[j * n_verts];

                // Check points of A in B
                for (int v = 0; v < n_verts; v++) {
                    if (point_in_polygon(ax[v], ay[v], bx, by, n_verts)) {
                        *overlap_found = 1;
                        return;
                    }
                }

                if (*overlap_found) return;

                // Check points of B in A
                for (int v = 0; v < n_verts; v++) {
                    if (point_in_polygon(bx[v], by[v], ax, ay, n_verts)) {
                        *overlap_found = 1;
                        return;
                    }
                }
                
                if (*overlap_found) return;

                // Check edge intersections
                for (int ei = 0; ei < n_verts; ei++) {
                    int ni = (ei + 1) % n_verts;
                    for (int ej = 0; ej < n_verts; ej++) {
                        int nj = (ej + 1) % n_verts;
                        if (segments_intersect(
                            ax[ei], ay[ei], ax[ni], ay[ni],
                            bx[ej], by[ej], bx[nj], by[nj]
                        )) {
                            *overlap_found = 1;
                            return;
                        }
                    }
                }
            }
        }
    }

    extern "C" __global__
    void transform_single(
        const double* base_x, const double* base_y,
        double cx, double cy, double angle,
        double* out_x, double* out_y,
        double* out_bbox,
        int target_idx, int n_verts
    ) {
        int tid = threadIdx.x;
        if (tid >= n_verts) return;

        double angle_rad = angle * 3.14159265358979323846 / 180.0;
        double cos_a = cos(angle_rad);
        double sin_a = sin(angle_rad);

        double bx = base_x[tid];
        double by = base_y[tid];

        double tx = bx * cos_a - by * sin_a + cx;
        double ty = bx * sin_a + by * cos_a + cy;

        out_x[target_idx * n_verts + tid] = tx;
        out_y[target_idx * n_verts + tid] = ty;
        
        // Compute bbox (collaborative reduction)
        // For 15 verts, single thread check is fine or simple memory write
        // Just let one thread do bbox update? No, race condition.
        // Simple: just atomicMin/Max or just reload in shared memory.
        // Given only 15 verts, thread 0 can just loop and compute bbox.
        
        if (tid == 0) {
           double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
           for(int v=0; v<n_verts; v++) {
                double bx_v = base_x[v]; // Use different variable name to avoid conflict with tid's bx
                double by_v = base_y[v]; // Use different variable name to avoid conflict with tid's by
                double tx_v = bx_v * cos_a - by_v * sin_a + cx;
                double ty_v = bx_v * sin_a + by_v * cos_a + cy;
                if (tx_v < minx) minx = tx_v;
                if (ty_v < miny) miny = ty_v;
                if (tx_v > maxx) maxx = tx_v;
                if (ty_v > maxy) maxy = ty_v;
           }
           out_bbox[target_idx * 4 + 0] = minx;
           out_bbox[target_idx * 4 + 1] = miny;
           out_bbox[target_idx * 4 + 2] = maxx;
           out_bbox[target_idx * 4 + 3] = maxy;
        }
    }

    extern "C" __global__
    void check_overlap_single(
        const double* verts_x, const double* verts_y,
        const double* bbox,
        int target_idx, 
        int* overlap_found,
        int n_trees, int n_verts
    ) {
        int other_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (other_idx >= n_trees || other_idx == target_idx) return;

        if (*overlap_found) return;

        // 1. BBox Check
        double tax0 = bbox[target_idx * 4 + 0], tay0 = bbox[target_idx * 4 + 1];
        double tax1 = bbox[target_idx * 4 + 2], tay1 = bbox[target_idx * 4 + 3];
        double tbx0 = bbox[other_idx * 4 + 0], tby0 = bbox[other_idx * 4 + 1];
        double tbx1 = bbox[other_idx * 4 + 2], tby1 = bbox[other_idx * 4 + 3];

        if (tax1 < tbx0 || tbx1 < tax0 || tay1 < tby0 || tby1 < tay0) {
            return; 
        }

        // 2. Detailed Polygon Check
        const double* ax = &verts_x[target_idx * n_verts];
        const double* ay = &verts_y[target_idx * n_verts];
        const double* bx = &verts_x[other_idx * n_verts];
        const double* by = &verts_y[other_idx * n_verts];

        // A in B
        for (int v = 0; v < n_verts; v++) {
            if (point_in_polygon(ax[v], ay[v], bx, by, n_verts)) {
                *overlap_found = 1;
                return;
            }
        }
        
        // B in A
        for (int v = 0; v < n_verts; v++) {
            if (point_in_polygon(bx[v], by[v], ax, ay, n_verts)) {
                *overlap_found = 1;
                return;
            }
        }
        
        // Edge Int
        for (int ei = 0; ei < n_verts; ei++) {
             int ni = (ei + 1) % n_verts;
             for (int ej = 0; ej < n_verts; ej++) {
                 int nj = (ej + 1) % n_verts;
                 if (segments_intersect(ax[ei], ay[ei], ax[ni], ay[ni],
                                        bx[ej], by[ej], bx[nj], by[nj])) {
                     *overlap_found = 1;
                     return;
                 }
             }
        }
    }
    """
    )

    detailed_overlap_kernel = overlap_module.get_function("check_overlap_detailed")
    transform_single_kernel = overlap_module.get_function("transform_single")
    check_overlap_single_kernel = overlap_module.get_function("check_overlap_single")


class GPUTreeGroup:
    """Manages a group of trees with GPU acceleration"""

    def __init__(self, cx, cy, angles):
        self.n = len(cx)
        self.cx = cp.array(cx, dtype=cp.float64)
        self.cy = cp.array(cy, dtype=cp.float64)
        self.angles = cp.array(angles, dtype=cp.float64)

        # Base vertices on GPU
        self.base_x = cp.array(TREE_VERTICES_X, dtype=cp.float64)
        self.base_y = cp.array(TREE_VERTICES_Y, dtype=cp.float64)

        # Transformed vertices cache
        self.verts_x = cp.zeros(self.n * NUM_VERTICES, dtype=cp.float64)
        self.verts_y = cp.zeros(self.n * NUM_VERTICES, dtype=cp.float64)
        self.bbox = cp.zeros((self.n, 4), dtype=cp.float64)

        self._update_transforms()

    def _update_transforms(self):
        """Recompute all transformed vertices and bboxes on GPU"""
        if not GPU_AVAILABLE:
            self._update_transforms_cpu()
            return

        threads = 256
        blocks = (self.n * NUM_VERTICES + threads - 1) // threads

        transform_kernel(
            (blocks,),
            (threads,),
            (
                self.base_x,
                self.base_y,
                self.cx,
                self.cy,
                self.angles,
                self.verts_x,
                self.verts_y,
                self.n,
                NUM_VERTICES,
            ),
        )

        blocks_bbox = (self.n + threads - 1) // threads
        bbox_kernel(
            (blocks_bbox,),
            (threads,),
            (self.verts_x, self.verts_y, self.bbox.ravel(), self.n, NUM_VERTICES),
        )

    def _update_transforms_cpu(self):
        """CPU fallback for transforms"""
        cx_np = cp.asnumpy(self.cx) if GPU_AVAILABLE else self.cx
        cy_np = cp.asnumpy(self.cy) if GPU_AVAILABLE else self.cy
        angles_np = cp.asnumpy(self.angles) if GPU_AVAILABLE else self.angles

        verts_x = np.zeros(self.n * NUM_VERTICES)
        verts_y = np.zeros(self.n * NUM_VERTICES)
        bbox = np.zeros((self.n, 4))

        for i in range(self.n):
            angle_rad = np.radians(angles_np[i])
            c, s = np.cos(angle_rad), np.sin(angle_rad)

            for v in range(NUM_VERTICES):
                idx = i * NUM_VERTICES + v
                bx, by = TREE_VERTICES_X[v], TREE_VERTICES_Y[v]
                verts_x[idx] = bx * c - by * s + cx_np[i]
                verts_y[idx] = bx * s + by * c + cy_np[i]

            start = i * NUM_VERTICES
            end = start + NUM_VERTICES
            bbox[i, 0] = verts_x[start:end].min()
            bbox[i, 1] = verts_y[start:end].min()
            bbox[i, 2] = verts_x[start:end].max()
            bbox[i, 3] = verts_y[start:end].max()

        if GPU_AVAILABLE:
            self.verts_x = cp.array(verts_x)
            self.verts_y = cp.array(verts_y)
            self.bbox = cp.array(bbox)
        else:
            self.verts_x = verts_x
            self.verts_y = verts_y
            self.bbox = bbox

    def get_side_length(self):
        """Compute bounding box side length"""
        bbox_np = cp.asnumpy(self.bbox) if GPU_AVAILABLE else self.bbox
        minx, miny = bbox_np[:, 0].min(), bbox_np[:, 1].min()
        maxx, maxy = bbox_np[:, 2].max(), bbox_np[:, 3].max()
        return max(maxx - minx, maxy - miny)

    def has_any_overlap_gpu(self):
        """Check for overlaps using GPU acceleration"""
        if not GPU_AVAILABLE:
            return self.has_any_overlap_cpu()

        # Output flag
        overlap_found = cp.zeros((1,), dtype=cp.int32)

        threads_per_block = 16
        blocks_x = (self.n + threads_per_block - 1) // threads_per_block
        blocks_y = (self.n + threads_per_block - 1) // threads_per_block

        detailed_overlap_kernel(
            (blocks_x, blocks_y),
            (threads_per_block, threads_per_block),
            (
                self.verts_x,
                self.verts_y,
                self.bbox.ravel(),
                # Output
                overlap_found,
                # Params
                self.n,
                NUM_VERTICES,
            ),
        )

        return bool(overlap_found[0])

    def has_any_overlap_cpu(self):
        """CPU fallback for overlap detection"""
        verts_x_np = cp.asnumpy(self.verts_x) if GPU_AVAILABLE else self.verts_x
        verts_y_np = cp.asnumpy(self.verts_y) if GPU_AVAILABLE else self.verts_y
        bbox_np = cp.asnumpy(self.bbox) if GPU_AVAILABLE else self.bbox

        # Simple N^2 check on CPU
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Bbox check first
                if (
                    bbox_np[i, 2] < bbox_np[j, 0]
                    or bbox_np[j, 2] < bbox_np[i, 0]
                    or bbox_np[i, 3] < bbox_np[j, 1]
                    or bbox_np[j, 3] < bbox_np[i, 1]
                ):
                    continue

                if self._polygons_overlap_cpu(i, j, verts_x_np, verts_y_np):
                    return True
        return False

    def _polygons_overlap_cpu(self, i, j, verts_x, verts_y):
        """Check if two polygons overlap (CPU)"""
        # Get vertices for both polygons
        poly_i_x = verts_x[i * NUM_VERTICES : (i + 1) * NUM_VERTICES]
        poly_i_y = verts_y[i * NUM_VERTICES : (i + 1) * NUM_VERTICES]
        poly_j_x = verts_x[j * NUM_VERTICES : (j + 1) * NUM_VERTICES]
        poly_j_y = verts_y[j * NUM_VERTICES : (j + 1) * NUM_VERTICES]

        # Point in polygon tests
        for v in range(NUM_VERTICES):
            if self._point_in_polygon(poly_i_x[v], poly_i_y[v], poly_j_x, poly_j_y):
                return True
            if self._point_in_polygon(poly_j_x[v], poly_j_y[v], poly_i_x, poly_i_y):
                return True

        # Edge intersection tests
        for ei in range(NUM_VERTICES):
            ni = (ei + 1) % NUM_VERTICES
            for ej in range(NUM_VERTICES):
                nj = (ej + 1) % NUM_VERTICES
                if self._segments_intersect(
                    poly_i_x[ei],
                    poly_i_y[ei],
                    poly_i_x[ni],
                    poly_i_y[ni],
                    poly_j_x[ej],
                    poly_j_y[ej],
                    poly_j_x[nj],
                    poly_j_y[nj],
                ):
                    return True

        return False

    @staticmethod
    def _point_in_polygon(px, py, poly_x, poly_y):
        """Ray casting point-in-polygon test"""
        n = len(poly_x)
        inside = False
        j = n - 1
        for i in range(n):
            if (poly_y[i] > py) != (poly_y[j] > py) and px < (poly_x[j] - poly_x[i]) * (
                py - poly_y[i]
            ) / (poly_y[j] - poly_y[i]) + poly_x[i]:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
        """Check if line segments AB and CD intersect"""
        d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
        d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
        d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    def move_tree(self, idx, new_cx, new_cy):
        """Move a single tree"""
        if GPU_AVAILABLE:
            self.cx[idx] = new_cx
            self.cy[idx] = new_cy
        else:
            self.cx[idx] = new_cx
            self.cy[idx] = new_cy
        self._update_transforms()

    def rotate_tree(self, idx, new_angle):
        """Rotate a single tree"""
        if GPU_AVAILABLE:
            self.angles[idx] = new_angle % 360
        else:
            self.angles[idx] = new_angle % 360
        self._update_transforms()

    def set_state(self, cx, cy, angles):
        """Set all tree positions and angles"""
        if GPU_AVAILABLE:
            self.cx = cp.array(cx, dtype=cp.float64)
            self.cy = cp.array(cy, dtype=cp.float64)
            self.angles = cp.array(angles, dtype=cp.float64)
        else:
            self.cx = np.array(cx, dtype=np.float64)
            self.cy = np.array(cy, dtype=np.float64)
            self.angles = np.array(angles, dtype=np.float64)
        self._update_transforms()

    def get_state(self):
        """Get current state as numpy arrays"""
        if GPU_AVAILABLE:
            return cp.asnumpy(self.cx), cp.asnumpy(self.cy), cp.asnumpy(self.angles)
        return self.cx.copy(), self.cy.copy(), self.angles.copy()

    def update_single_transform(self, idx):
        """Update transforms for a single tree on GPU"""
        if not GPU_AVAILABLE:
            self._update_transforms_cpu()  # Fallback for now if needed, effectively full update
            return

        cx_val = float(self.cx[idx])
        cy_val = float(self.cy[idx])
        ang_val = float(self.angles[idx])

        # 1 buffer of 15 threads is enough
        transform_single_kernel(
            (1,),
            (NUM_VERTICES,),
            (
                self.base_x,
                self.base_y,
                cx_val,
                cy_val,
                ang_val,
                self.verts_x,
                self.verts_y,
                self.bbox.ravel(),
                idx,
                NUM_VERTICES,
            ),
        )

    def has_overlap_with_tree_gpu(self, idx):
        """Check if tree at idx overlaps with any other tree"""
        if not GPU_AVAILABLE:
            return self.has_any_overlap_cpu()  # Full check fallback

        overlap_found = cp.zeros((1,), dtype=cp.int32)

        threads = 256
        blocks = (self.n + threads - 1) // threads

        check_overlap_single_kernel(
            (blocks,),
            (threads,),
            (
                self.verts_x,
                self.verts_y,
                self.bbox.ravel(),
                idx,
                overlap_found,
                self.n,
                NUM_VERTICES,
            ),
        )

        return bool(overlap_found[0])


def sa_optimize_gpu(group, iterations=20000, T0=5.0, Tm=0.000001, seed=42):
    """Simulated Annealing with GPU-accelerated overlap detection"""
    n = group.n
    if n <= 1:
        return group

    rng = np.random.default_rng(seed + n)

    # Current state (keep on CPU for manipulation, push to GPU for overlap check)
    cur_cx, cur_cy, cur_angles = group.get_state()

    best_cx, best_cy, best_angles = cur_cx.copy(), cur_cy.copy(), cur_angles.copy()
    best_side = group.get_side_length()
    cur_side = best_side

    log_ratio = np.log(Tm / T0)

    log_ratio = np.log(Tm / T0)

    for it in range(iterations):
        progress = it / iterations
        T = T0 * np.exp(log_ratio * progress)
        temp_scale = np.sqrt(T / T0)

        # Generate move
        mt = rng.integers(100)
        trial_cx, trial_cy, trial_angles = (
            cur_cx.copy(),
            cur_cy.copy(),
            cur_angles.copy(),
        )

        if mt < 40:  # Position move
            i = rng.integers(n)
            step = 0.15 if mt < 25 else 0.05
            trial_cx[i] += rng.normal(0, step * temp_scale)
            trial_cy[i] += rng.normal(0, step * temp_scale)
        elif mt < 70:  # Rotation
            i = rng.integers(n)
            rot = 45 if mt < 55 else 20
            trial_angles[i] += rng.normal(0, rot * temp_scale)
            trial_angles[i] %= 360
        elif mt < 90:  # Combined
            i = rng.integers(n)
            trial_cx[i] += rng.uniform(-0.1, 0.1) * temp_scale
            trial_cy[i] += rng.uniform(-0.1, 0.1) * temp_scale
            trial_angles[i] += rng.uniform(-30, 30) * temp_scale
            trial_angles[i] %= 360
        else:  # Squeeze
            cx_center = (cur_cx.min() + cur_cx.max()) / 2
            cy_center = (cur_cy.min() + cur_cy.max()) / 2
            factor = 1.0 - rng.uniform(0.002, 0.01) * temp_scale
            trial_cx = cx_center + (cur_cx - cx_center) * factor
            trial_cy = cy_center + (cur_cy - cy_center) * factor

        # Check on GPU
        group.set_state(trial_cx, trial_cy, trial_angles)

        if group.has_any_overlap_gpu():
            group.set_state(cur_cx, cur_cy, cur_angles)
            continue

        trial_side = group.get_side_length()
        delta = trial_side - cur_side

        # Metropolis
        if delta < 0 or rng.random() < np.exp(-delta / T):
            cur_cx, cur_cy, cur_angles = trial_cx, trial_cy, trial_angles
            cur_side = trial_side

            if trial_side < best_side:
                best_cx, best_cy, best_angles = (
                    trial_cx.copy(),
                    trial_cy.copy(),
                    trial_angles.copy(),
                )
                best_side = trial_side
        else:
            group.set_state(cur_cx, cur_cy, cur_angles)

        # Logging
        if it % 1000 == 0 or it == iterations - 1:
            sys.stdout.write(
                f"\r    SA Iter {it:5d}/{iterations}: T={T:.5f}, Best={best_side:.6f}, Cur={cur_side:.6f}"
            )
            sys.stdout.flush()

    # Return best found
    group.set_state(best_cx, best_cy, best_angles)
    return group


def compact_gpu(group, iterations=100):
    """Compaction with greedy GPU in-place updates"""
    if group.n <= 1:
        return group

    # Ensure usage of GPU arrays directly
    if not GPU_AVAILABLE:
        # Fallback to old slow method logic if CPU (omitted for brevity, assume GPU)
        return group

    # We assume group.cx, group.cy are already Cupy arrays

    steps = [0.06, 0.04, 0.025, 0.015, 0.01, 0.005, 0.0025, 0.001]

    for it in range(iterations):
        improved = False

        # Re-calculate center based on current bbox
        # bbox is (N, 4) array on GPU
        min_x = float(group.bbox[:, 0].min())
        max_x = float(group.bbox[:, 2].max())
        min_y = float(group.bbox[:, 1].min())
        max_y = float(group.bbox[:, 3].max())

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        current_side = max(max_x - min_x, max_y - min_y)

        # Iterate over all trees
        for i in range(group.n):
            # Store current position to revert if needed
            cx_i_orig = float(group.cx[i])
            cy_i_orig = float(group.cy[i])

            # Calculate direction to center
            dx = center_x - cx_i_orig
            dy = center_y - cy_i_orig
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < 1e-6:  # Already at center or very close
                continue

            dir_x = dx / dist
            dir_y = dy / dist

            # Try steps
            for step in steps:
                # Propose move
                new_cx = cx_i_orig + dir_x * step
                new_cy = cy_i_orig + dir_y * step

                # Apply move
                group.cx[i] = new_cx
                group.cy[i] = new_cy
                group.update_single_transform(i)  # Update bbox and verts for this tree

                # Check overlap for the moved tree (against all others)
                if not group.has_overlap_with_tree_gpu(i):
                    # Valid move! Keep this position for this tree
                    improved = True
                    # Update original position for next step/tree if needed
                    cx_i_orig = new_cx
                    cy_i_orig = new_cy
                    break  # Success with this tree, move to next tree
                else:
                    # Invalid, revert
                    group.cx[i] = cx_i_orig
                    group.cy[i] = cy_i_orig
                    group.update_single_transform(
                        i
                    )  # Revert bbox and verts for this tree

        if not improved:
            break  # No tree could be moved, stop compaction

        if it % 10 == 0:
            sys.stdout.write(
                f"\r    Compact Iter {it:3d}/{iterations}: Side={current_side:.6f}"
            )
            sys.stdout.flush()

    return group


def squeeze_gpu(group, factor=0.999):
    """Squeeze group toward center"""
    if group.n <= 1:
        return group

    cx, cy, angles = group.get_state()
    cx_center = (cx.min() + cx.max()) / 2
    cy_center = (cy.min() + cy.max()) / 2

    new_cx = cx_center + (cx - cx_center) * factor
    new_cy = cy_center + (cy - cy_center) * factor

    group.set_state(new_cx, new_cy, angles)

    if group.has_any_overlap_gpu():
        group.set_state(cx, cy, angles)
        return group

    return group


# === SHAPELY COMPATIBILITY LAYER ===
def gpu_group_to_shapely_trees(group):
    """Convert GPU group back to Shapely for final validation"""
    cx, cy, angles = group.get_state()
    trees = []

    base = Polygon(list(zip(TREE_VERTICES_X, TREE_VERTICES_Y)))

    for i in range(group.n):
        rot = affinity.rotate(base, angles[i], origin=(0, 0))
        trans = affinity.translate(rot, xoff=cx[i], yoff=cy[i])
        trees.append(trans)

    return trees


def get_total_score(dict_of_side_length):
    score = Decimal(0)
    for k, v in dict_of_side_length.items():
        score += Decimal(str(v)) ** 2 / Decimal(str(k))
    return score


def parse_csv(csv_path):
    result = pd.read_csv(csv_path)
    result["x"] = result["x"].str.strip("s").astype(float)
    result["y"] = result["y"].str.strip("s").astype(float)
    result["deg"] = result["deg"].str.strip("s").astype(float)
    result[["group_id", "item_id"]] = result["id"].str.split("_", n=2, expand=True)

    dict_of_groups = {}
    dict_of_side_length = {}

    for group_id, group_data in result.groupby("group_id"):
        cx = group_data["x"].values
        cy = group_data["y"].values
        angles = group_data["deg"].values

        group = GPUTreeGroup(cx, cy, angles)
        dict_of_groups[group_id] = group
        dict_of_side_length[group_id] = group.get_side_length()

    return dict_of_groups, dict_of_side_length


def save_csv(dict_of_groups, out_file):
    tree_data = []
    for group_name, group in dict_of_groups.items():
        cx, cy, angles = group.get_state()
        for i in range(group.n):
            tree_data.append(
                {
                    "id": f"{group_name}_{i}",
                    "x": f"s{cx[i]:.15f}",
                    "y": f"s{cy[i]:.15f}",
                    "deg": f"s{angles[i]:.15f}",
                }
            )
    pd.DataFrame(tree_data).to_csv(out_file, index=False)


def optimize_rotation_gpu(group):
    """Find optimal rotation for entire group"""
    cx, cy, angles = group.get_state()

    # Get all vertices
    all_x = []
    all_y = []
    for i in range(group.n):
        angle_rad = np.radians(angles[i])
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        for v in range(NUM_VERTICES):
            bx, by = TREE_VERTICES_X[v], TREE_VERTICES_Y[v]
            all_x.append(bx * c - by * s + cx[i])
            all_y.append(bx * s + by * c + cy[i])

    points = np.column_stack([all_x, all_y])

    try:
        hull_points = points[ConvexHull(points).vertices]
    except:
        return group.get_side_length(), 0.0

    def bbox_side_at_angle(angle_deg):
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot = np.array([[c, s], [-s, c]])
        rotated = hull_points.dot(rot)
        return max(
            rotated[:, 0].max() - rotated[:, 0].min(),
            rotated[:, 1].max() - rotated[:, 1].min(),
        )

    initial_side = bbox_side_at_angle(0)
    best_side, best_angle = initial_side, 0.0

    for start in range(0, 90, 3):
        res = minimize_scalar(
            bbox_side_at_angle, bounds=(start, start + 10), method="bounded"
        )
        if res.fun < best_side:
            best_side, best_angle = res.fun, res.x

    if initial_side - best_side > 1e-9:
        return best_side, best_angle
    return initial_side, 0.0


def apply_rotation_gpu(group, angle_deg):
    """Apply rotation to entire group"""
    if abs(angle_deg) < 1e-9:
        return group

    cx, cy, angles = group.get_state()

    # Get center
    cx_center = (cx.min() + cx.max()) / 2
    cy_center = (cy.min() + cy.max()) / 2

    # Rotate positions
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    dx = cx - cx_center
    dy = cy - cy_center

    new_cx = cx_center + dx * c - dy * s
    new_cy = cy_center + dx * s + dy * c
    new_angles = (angles + angle_deg) % 360

    group.set_state(new_cx, new_cy, new_angles)
    return group


# === MAIN ===
if __name__ == "__main__":
    print("Loading data...")
    input_file = "../best_solution.csv"
    output_file = "./submission_gpu_optimized.csv"

    dict_of_groups, dict_of_side_length = parse_csv(input_file)
    initial_score = get_total_score(dict_of_side_length)

    print(f"\n{'='*60}")
    print(f"Initial Score: {float(initial_score):.12f}")
    print(f"Groups: {len(dict_of_groups)}")
    print(f"{'='*60}")

    best_score = initial_score
    start_time = time.time()

    # Stage 1: Rotation
    print("\n🔄 Stage 1: Rotation Optimization")
    improved = 0
    for gid in sorted(dict_of_groups.keys(), key=lambda x: int(x)):
        group = dict_of_groups[gid]
        cur_side = dict_of_side_length[gid]
        new_side, angle = optimize_rotation_gpu(group)

        if new_side < cur_side - 1e-10:
            apply_rotation_gpu(group, angle)
            if not group.has_any_overlap_gpu():
                dict_of_side_length[gid] = new_side
                improved += 1
            else:
                apply_rotation_gpu(group, -angle)

    print(f"  Improved {improved} groups")
    new_score = get_total_score(dict_of_side_length)
    print(f"  Score: {float(new_score):.12f}")

    if new_score < best_score:
        best_score = new_score
        save_csv(dict_of_groups, output_file)

    # Stage 2: Compaction
    print("\n📦 Stage 2: Compaction")
    # Run a few passes of greedy compaction
    # Since compact_gpu is now very efficient and greedy, one call might be enough,
    # or a few calls to restart from center.

    # We will run it once with more iterations or a few times?
    # The original loop had logic to save if improved.
    # Let's keep a simple loop to allow saving intermediate results.

    for pass_num in range(5):  # Reduced from 15, as greedy is more effective
        improved = 0
        for gid in sorted(dict_of_groups.keys(), key=lambda x: int(x)):
            group = dict_of_groups[gid]
            cur_side = dict_of_side_length[gid]

            compact_gpu(group, iterations=100)
            new_side = group.get_side_length()

            if new_side < cur_side - 1e-12:
                dict_of_side_length[gid] = new_side
                improved += 1

        new_score = get_total_score(dict_of_side_length)
        elapsed = time.time() - start_time
        print(
            f"  Pass {pass_num+1:2d}: improved {improved:3d}, Score: {float(new_score):.12f} ({elapsed:.0f}s)"
        )

        if new_score < best_score:
            best_score = new_score
            save_csv(dict_of_groups, output_file)

        if improved == 0:
            break

    # Stage 3: Squeeze
    print("\n🗜️ Stage 3: Squeeze")
    for factor in [
        0.99999,
        0.99995,
        0.9999,
        0.99985,
        0.9998,
        0.9997,
        0.9995,
        0.999,
        0.998,
        0.995,
    ]:
        improved = 0
        for gid in dict_of_groups.keys():
            group = dict_of_groups[gid]
            cur_side = dict_of_side_length[gid]

            old_state = group.get_state()
            squeeze_gpu(group, factor)
            new_side = group.get_side_length()

            if new_side < cur_side - 1e-12 and not group.has_any_overlap_gpu():
                dict_of_side_length[gid] = new_side
                improved += 1
            else:
                group.set_state(*old_state)

        if improved > 0:
            new_score = get_total_score(dict_of_side_length)
            print(
                f"  Factor {factor}: improved {improved}, Score: {float(new_score):.12f}"
            )
            if new_score < best_score:
                best_score = new_score
                save_csv(dict_of_groups, output_file)

    # Stage 4: Simulated Annealing
    print("\n🔥 Stage 4: GPU Simulated Annealing (top 80 worst groups)")

    group_scores = [
        (gid, float(Decimal(str(side)) ** 2 / Decimal(gid)))
        for gid, side in dict_of_side_length.items()
    ]
    group_scores.sort(key=lambda x: x[1], reverse=True)
    worst_groups = [g[0] for g in group_scores[:80]]

    for idx, gid in enumerate(worst_groups):
        n = int(gid)
        group = dict_of_groups[gid]
        cur_side = dict_of_side_length[gid]

        # Adaptive iterations
        if n <= 10:
            iters = 30000
        elif n <= 30:
            iters = 25000
        elif n <= 70:
            iters = 15000
        else:
            iters = 10000

        # Multiple SA runs
        best_group = group.clone()
        best_sa_side = cur_side

        for seed in [42, 123, 456, 789]:
            trial_group = group.clone()
            sa_optimize_gpu(trial_group, iterations=iters, seed=seed)
            compact_gpu(trial_group, iterations=50)
            trial_side = trial_group.get_side_length()

            if trial_side < best_sa_side and not trial_group.has_any_overlap_gpu():
                best_group = trial_group
                best_sa_side = trial_side

        if best_sa_side < cur_side - 1e-10:
            dict_of_groups[gid] = best_group
            dict_of_side_length[gid] = best_sa_side
            improvement = cur_side - best_sa_side
            print(f"  [{idx+1:2d}/80] n={n:3d}: improved by {improvement:.8f}")

    new_score = get_total_score(dict_of_side_length)
    if new_score < best_score:
        best_score = new_score
        save_csv(dict_of_groups, output_file)
    print(f"  Score after SA: {float(new_score):.12f}")

    # Stage 5: Final Polish
    print("\n✨ Stage 5: Final Polish")

    for gid in dict_of_groups.keys():
        group = dict_of_groups[gid]
        cur_side = dict_of_side_length[gid]

        new_side, angle = optimize_rotation_gpu(group)
        if new_side < cur_side - 1e-10:
            apply_rotation_gpu(group, angle)
            if not group.has_any_overlap_gpu():
                dict_of_side_length[gid] = new_side
            else:
                apply_rotation_gpu(group, -angle)

        compact_gpu(group, iterations=60)
        dict_of_side_length[gid] = group.get_side_length()

    for factor in [0.99999, 0.9999, 0.9998]:
        for gid in dict_of_groups.keys():
            group = dict_of_groups[gid]
            cur_side = dict_of_side_length[gid]
            old_state = group.get_state()
            squeeze_gpu(group, factor)
            new_side = group.get_side_length()
            if new_side < cur_side - 1e-12 and not group.has_any_overlap_gpu():
                dict_of_side_length[gid] = new_side
            else:
                group.set_state(*old_state)

    final_score = get_total_score(dict_of_side_length)
    if final_score < best_score:
        best_score = final_score
        save_csv(dict_of_groups, output_file)

    # Results
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Initial Score:  {float(initial_score):.12f}")
    print(f"Final Score:    {float(best_score):.12f}")
    print(f"Improvement:    {float(initial_score - best_score):.12f}")
    print(
        f"Improvement %:  {float(initial_score - best_score) / float(initial_score) * 100:.6f}%"
    )
    print(f"Time:           {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"✅ Output saved to: {output_file}")
