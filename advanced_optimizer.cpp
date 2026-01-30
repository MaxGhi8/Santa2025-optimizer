// Advanced Optimizer - SA + Back Propagation + Global Rotation + Earthquake
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o advanced_optimizer advanced_optimizer.cpp
// Run: ./advanced_optimizer -iter 200000 --earthquake 5

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <string>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

constexpr int MAX_N = 250;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

struct FastRNG {
    uint64_t s[2];
    FastRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }
    inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1], r = s0 + s1;
        s1 ^= s0; s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); s[1] = rotl(s1, 37);
        return r;
    }
    inline double rf() { return (next() >> 11) * 0x1.0p-53; }
    inline double rf2() { return rf() * 2.0 - 1.0; }
    inline int ri(int n) { return next() % n; }
    inline double gaussian() {
        double u1 = rf() + 1e-10, u2 = rf();
        return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    }
};

alignas(64) const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
alignas(64) const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

struct Poly {
    double px[NV], py[NV];
    double x0, y0, x1, y1;
};

inline void getPoly(double cx, double cy, double deg, Poly& q) {
    double rad = deg * (PI / 180.0);
    double s = sin(rad), c = cos(rad);
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (int i = 0; i < NV; i++) {
        double x = TX[i] * c - TY[i] * s + cx;
        double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x; q.py[i] = y;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

inline bool pip(double px, double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.py[i] > py) != (q.py[j] > py) &&
            px < (q.px[j] - q.px[i]) * (py - q.py[i]) / (q.py[j] - q.py[i]) + q.px[i])
            in = !in;
        j = i;
    }
    return in;
}

inline bool segInt(double ax, double ay, double bx, double by,
                   double cx, double cy, double dx, double dy) {
    double d1 = (dx-cx)*(ay-cy) - (dy-cy)*(ax-cx);
    double d2 = (dx-cx)*(by-cy) - (dy-cy)*(bx-cx);
    double d3 = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
    double d4 = (bx-ax)*(dy-ay) - (by-ay)*(dx-ax);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}

inline bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        if (pip(a.px[i], a.py[i], b)) return true;
        if (pip(b.px[i], b.py[i], a)) return true;
    }
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segInt(a.px[i], a.py[i], a.px[ni], a.py[ni],
                      b.px[j], b.py[j], b.px[nj], b.py[nj])) return true;
        }
    }
    return false;
}

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    double gx0, gy0, gx1, gy1;

    void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    
    void updAll() {
        gx0 = gy0 = 1e9; gx1 = gy1 = -1e9;
        for (int i = 0; i < n; i++) {
            upd(i);
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }
    
    void updGlobal() {
        gx0 = gy0 = 1e9; gx1 = gy1 = -1e9;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    bool hasOvl(int i) const {
        for (int j = 0; j < n; j++)
            if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }

    bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return true;
        return false;
    }

    double side() const { return max(gx1 - gx0, gy1 - gy0); }
    double score() const { double s = side(); return s * s / n; }
    
    // Remove tree at index
    Cfg removeTree(int idx) const {
        Cfg c;
        c.n = n - 1;
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i != idx) {
                c.x[j] = x[i];
                c.y[j] = y[i];
                c.a[j] = a[i];
                j++;
            }
        }
        c.updAll();
        return c;
    }
    
    // Apply global rotation around center
    Cfg rotateGlobal(double angleDeg) const {
        Cfg c = *this;
        double cx = (gx0 + gx1) / 2;
        double cy = (gy0 + gy1) / 2;
        double rad = angleDeg * PI / 180.0;
        double co = cos(rad), si = sin(rad);
        for (int i = 0; i < n; i++) {
            double dx = x[i] - cx;
            double dy = y[i] - cy;
            c.x[i] = cx + dx * co - dy * si;
            c.y[i] = cy + dx * si + dy * co;
            c.a[i] = fmod(a[i] + angleDeg + 360.0, 360.0);
        }
        c.updAll();
        return c;
    }
    
    // Get indices of trees touching the bounding box
    vector<int> getBboxTouchingIndices() const {
        vector<int> touching;
        const double eps = 1e-9;
        for (int i = 0; i < n; i++) {
            // Check if any polygon vertex touches the bbox
            bool touches = false;
            for (int v = 0; v < NV; v++) {
                if (abs(pl[i].px[v] - gx0) < eps || abs(pl[i].px[v] - gx1) < eps ||
                    abs(pl[i].py[v] - gy0) < eps || abs(pl[i].py[v] - gy1) < eps) {
                    touches = true;
                    break;
                }
            }
            // Also check AABB
            if (abs(pl[i].x0 - gx0) < eps || abs(pl[i].x1 - gx1) < eps ||
                abs(pl[i].y0 - gy0) < eps || abs(pl[i].y1 - gy1) < eps) {
                touches = true;
            }
            if (touches) touching.push_back(i);
        }
        return touching;
    }
};


// SA optimizer
Cfg sa_optimize(Cfg c, int iterations, double T0, double Tmin, FastRNG& rng) {
    Cfg best = c, cur = c;
    double bestSide = best.side();
    double curSide = bestSide;
    double T = T0;
    double alpha = pow(Tmin / T0, 1.0 / iterations);
    
    for (int iter = 0; iter < iterations; iter++) {
        int moveType = rng.ri(10);
        double scale = T / T0;
        bool valid = true;
        
        if (moveType < 4) {
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            cur.x[i] += rng.gaussian() * 0.02 * scale;
            cur.y[i] += rng.gaussian() * 0.02 * scale;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i] = ox; cur.y[i] = oy; cur.upd(i); valid = false; }
        }
        else if (moveType < 6) {
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            double cx = (cur.gx0 + cur.gx1) / 2;
            double cy = (cur.gy0 + cur.gy1) / 2;
            double dx = cx - cur.x[i], dy = cy - cur.y[i];
            double d = sqrt(dx*dx + dy*dy);
            if (d > 1e-6) {
                cur.x[i] += (dx/d) * rng.rf() * 0.02 * scale;
                cur.y[i] += (dy/d) * rng.rf() * 0.02 * scale;
            }
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i] = ox; cur.y[i] = oy; cur.upd(i); valid = false; }
        }
        else if (moveType < 8) {
            int i = rng.ri(c.n);
            double oa = cur.a[i];
            cur.a[i] += rng.gaussian() * 10.0 * scale;
            while (cur.a[i] < 0) cur.a[i] += 360.0;
            while (cur.a[i] >= 360.0) cur.a[i] -= 360.0;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.a[i] = oa; cur.upd(i); valid = false; }
        }
        else if (moveType == 8) {
            double factor = 1.0 - rng.rf() * 0.002 * scale;
            double cx = (cur.gx0 + cur.gx1) / 2;
            double cy = (cur.gy0 + cur.gy1) / 2;
            Cfg trial = cur;
            for (int i = 0; i < c.n; i++) {
                trial.x[i] = cx + (cur.x[i] - cx) * factor;
                trial.y[i] = cy + (cur.y[i] - cy) * factor;
            }
            trial.updAll();
            if (!trial.anyOvl()) cur = trial; else valid = false;
        }
        else {
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i], oa = cur.a[i];
            cur.x[i] += rng.rf2() * 0.01 * scale;
            cur.y[i] += rng.rf2() * 0.01 * scale;
            cur.a[i] += rng.rf2() * 5.0 * scale;
            while (cur.a[i] < 0) cur.a[i] += 360.0;
            while (cur.a[i] >= 360.0) cur.a[i] -= 360.0;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i] = ox; cur.y[i] = oy; cur.a[i] = oa; cur.upd(i); valid = false; }
        }
        
        if (!valid) { T *= alpha; continue; }
        
        cur.updGlobal();
        double newSide = cur.side();
        double delta = newSide - curSide;
        
        bool accept = (delta < 0) || (T > 1e-10 && rng.rf() < exp(-delta / T));
        
        if (accept) {
            curSide = newSide;
            if (newSide < bestSide) { bestSide = newSide; best = cur; }
        } else {
            cur = best; curSide = bestSide;
        }
        
        T *= alpha;
        if (T < Tmin) T = Tmin;
    }
    return best;
}


// Local search
Cfg localSearch(Cfg c, int maxIter) {
    const double steps[] = {0.001, 0.0005, 0.0001};
    const double rots[] = {0.5, 0.1, 0.01};
    const int dx[] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dy[] = {0, 0, 1, -1, 1, -1, 1, -1};
    
    double bestSide = c.side();
    
    for (int iter = 0; iter < maxIter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            for (double step : steps) {
                double ox = c.x[i], oy = c.y[i];
                for (int d = 0; d < 8; d++) {
                    c.x[i] = ox + dx[d] * step; c.y[i] = oy + dy[d] * step; c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bestSide - 1e-14) { bestSide = c.side(); ox = c.x[i]; oy = c.y[i]; improved = true; }
                        else { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
                    } else { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
                }
                double cx = (c.gx0 + c.gx1) / 2, cy = (c.gy0 + c.gy1) / 2;
                double ddx = cx - c.x[i], ddy = cy - c.y[i];
                double dist = sqrt(ddx*ddx + ddy*ddy);
                if (dist > 1e-6) {
                    c.x[i] = ox + (ddx/dist) * step; c.y[i] = oy + (ddy/dist) * step; c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bestSide - 1e-14) { bestSide = c.side(); ox = c.x[i]; oy = c.y[i]; improved = true; }
                        else { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
                    } else { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
                }
            }
            for (double rot : rots) {
                double oa = c.a[i];
                for (double delta : {rot, -rot}) {
                    c.a[i] = fmod(oa + delta + 360.0, 360.0); c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bestSide - 1e-14) { bestSide = c.side(); oa = c.a[i]; improved = true; }
                        else { c.a[i] = oa; c.upd(i); }
                    } else { c.a[i] = oa; c.upd(i); }
                }
            }
        }
        if (!improved) break;
    }
    c.updGlobal();
    return c;
}

// Squeeze
Cfg squeeze(Cfg c) {
    double cx = (c.gx0 + c.gx1) / 2, cy = (c.gy0 + c.gy1) / 2;
    for (double f = 0.9999; f > 0.99; f -= 0.0001) {
        Cfg trial = c;
        for (int i = 0; i < c.n; i++) {
            trial.x[i] = cx + (c.x[i] - cx) * f;
            trial.y[i] = cy + (c.y[i] - cy) * f;
        }
        trial.updAll();
        if (!trial.anyOvl()) c = trial; else break;
    }
    return c;
}

// Earthquake: perturb solution to escape local minima
Cfg earthquake(Cfg c, double strength, FastRNG& rng) {
    Cfg original = c;
    
    // Perturb random subset of trees
    int numToPerturb = max(1, (int)(c.n * 0.3 + strength * 2));
    for (int k = 0; k < numToPerturb; k++) {
        int i = rng.ri(c.n);
        c.x[i] += rng.gaussian() * strength * 0.05;
        c.y[i] += rng.gaussian() * strength * 0.05;
        c.a[i] += rng.gaussian() * 15.0;
        while (c.a[i] < 0) c.a[i] += 360.0;
        while (c.a[i] >= 360.0) c.a[i] -= 360.0;
    }
    c.updAll();
    
    // Try to fix overlaps by pushing trees outward
    for (int iter = 0; iter < 1000; iter++) {
        bool fixed = true;
        for (int i = 0; i < c.n; i++) {
            if (c.hasOvl(i)) {
                fixed = false;
                double cx = (c.gx0 + c.gx1) / 2;
                double cy = (c.gy0 + c.gy1) / 2;
                double dx = c.x[i] - cx;
                double dy = c.y[i] - cy;
                double d = sqrt(dx*dx + dy*dy);
                if (d > 1e-6) {
                    c.x[i] += dx/d * 0.015;
                    c.y[i] += dy/d * 0.015;
                }
                c.a[i] += rng.rf2() * 10.0;
                while (c.a[i] < 0) c.a[i] += 360.0;
                while (c.a[i] >= 360.0) c.a[i] -= 360.0;
                c.upd(i);
            }
        }
        if (fixed) break;
    }
    c.updGlobal();
    
    // If still has overlaps, revert
    if (c.anyOvl()) return original;
    return c;
}

// Global rotation optimization
Cfg optimizeGlobalRotation(Cfg c) {
    double bestSide = c.side();
    Cfg bestCfg = c;
    
    for (double angle = 0.5; angle < 90.0; angle += 0.5) {
        Cfg rotated = c.rotateGlobal(angle);
        if (!rotated.anyOvl() && rotated.side() < bestSide) {
            bestSide = rotated.side();
            bestCfg = rotated;
        }
    }
    return bestCfg;
}

// Full optimization
Cfg optimizeFull(Cfg c, int iterations, double startTemp, double endTemp, uint64_t seed) {
    FastRNG rng(seed);
    
    // Try global rotation first
    c = optimizeGlobalRotation(c);
    
    // SA
    c = sa_optimize(c, iterations, startTemp, endTemp, rng);
    
    // Local search + squeeze
    c = localSearch(c, 30);
    c = squeeze(c);
    c = localSearch(c, 20);
    
    return c;
}

// CSV I/O
map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) return cfg;
    string ln; getline(f, ln);
    map<int, vector<tuple<int,double,double,double>>> data;
    while (getline(f, ln)) {
        size_t p1=ln.find(','), p2=ln.find(',',p1+1), p3=ln.find(',',p2+1);
        if (p1 == string::npos || p2 == string::npos || p3 == string::npos) continue;
        string id=ln.substr(0,p1), xs=ln.substr(p1+1,p2-p1-1), ys=ln.substr(p2+1,p3-p2-1), ds=ln.substr(p3+1);
        if(!xs.empty() && xs[0]=='s') xs=xs.substr(1);
        if(!ys.empty() && ys[0]=='s') ys=ys.substr(1);
        if(!ds.empty() && ds[0]=='s') ds=ds.substr(1);
        try {
            int n=stoi(id.substr(0,3)), idx=stoi(id.substr(4));
            data[n].push_back({idx, stod(xs), stod(ys), stod(ds)});
        } catch (...) { continue; }
    }
    for (auto& [n,v] : data) {
        Cfg c; c.n = n;
        for (auto& [i,x,y,d] : v) if (i < n) { c.x[i]=x; c.y[i]=y; c.a[i]=d; }
        c.updAll();
        cfg[n] = c;
    }
    return cfg;
}

void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(17) << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        if (cfg.count(n)) {
            const Cfg& c = cfg.at(n);
            for (int i = 0; i < n; i++)
                f << setfill('0') << setw(3) << n << "_" << i << ",s" << c.x[i] << ",s" << c.y[i] << ",s" << c.a[i] << "\n";
        }
    }
}

int main(int argc, char** argv) {
    string inputFile = "../best_solution.csv";
    string outputFile = "../submission.csv";
    int iterations = 1000000;
    bool doBackProp = true;
    int earthquakeRounds = 3;
    double startTemp = 10.0;
    double endTemp = 0.0001;
    double baseStrength = 1.5;
    int targetGroup = -1;
    int numRestarts = 0;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i+1 < argc) inputFile = argv[++i];
        else if (a == "-o" && i+1 < argc) outputFile = argv[++i];
        else if ((a == "-n" || a == "-iter") && i+1 < argc) iterations = stoi(argv[++i]);
        else if (a == "--no-backprop") doBackProp = false;
        else if ((a == "--earthquake" || a == "-earthquake") && i+1 < argc) earthquakeRounds = stoi(argv[++i]);
        else if ((a == "-t0" || a == "-t") && i+1 < argc) startTemp = stod(argv[++i]); // Added -t alias
        else if (a == "-tmin" && i+1 < argc) endTemp = stod(argv[++i]);
        else if ((a == "-s" || a == "-strength") && i+1 < argc) baseStrength = stod(argv[++i]);
        else if ((a == "-g" || a == "--group") && i+1 < argc) targetGroup = stoi(argv[++i]); // Added -g
        else if ((a == "-r" || a == "--restarts") && i+1 < argc) numRestarts = stoi(argv[++i]); // Added -r
    }
    
    // Adjust threading if running single group
    int numThreads = omp_get_max_threads();
    if (targetGroup != -1) {
       // If optimizing a single group, we might want to use threads for restarts if implemented, 
       // but here the loop structure is over groups. 
       // If simple, we keep the parallel loop but filter.
       // Or better, we just process that one group.
    }

    printf("Advanced Optimizer (SA + Global Rotation + Back Prop + Earthquake)\n");
    printf("OpenMP: %d threads\n", numThreads);
    printf("Input: %s, Output: %s\n", inputFile.c_str(), outputFile.c_str());
    if (targetGroup != -1) printf("Target Group: %d\n", targetGroup);
    if (numRestarts > 0) printf("Restarts: %d\n", numRestarts);
    printf("SA Iters: %d, BackProp: %s, Earthquake: %d rounds (Strength: %.2f)\n", iterations, doBackProp ? "ON" : "OFF", earthquakeRounds, baseStrength);
    printf("Temp: %.4f -> %.6f\n\n", startTemp, endTemp);
    
    auto t0 = chrono::high_resolution_clock::now();
    
    auto cfg = loadCSV(inputFile);
    if (cfg.empty()) { cerr << "Failed to load " << inputFile << endl; return 1; }
    printf("Loaded %zu configs\n", cfg.size());
    
    double initScore = 0;
    for (auto& [n, c] : cfg) initScore += c.score();
    printf("Initial Score: %.6f\n\n", initScore);
    
    vector<int> nvals;
    if (targetGroup != -1) {
        if (cfg.count(targetGroup)) {
            nvals.push_back(targetGroup);
        } else {
            cerr << "Group " << targetGroup << " not found in input!" << endl;
            return 1;
        }
    } else {
        for (auto& [n, c] : cfg) nvals.push_back(n);
    }
    
    map<int, Cfg> results;
    for (auto& [n, c] : cfg) results[n] = c;
    
    int improved = 0;
    
    printf("Phase 1: SA + Global Rotation...\n");
    
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < (int)nvals.size(); idx++) {
        int n = nvals[idx];
        Cfg c = cfg[n];
        double oldScore = c.score();
        
        Cfg bestOpt = c;
        double bestOptScore = oldScore;
        
        int runs = (numRestarts > 0) ? numRestarts : 1;
        
        for (int r = 0; r < runs; r++) {
            uint64_t seed = 42 + n * 1000 + idx + r * 12345;
            Cfg opt = optimizeFull(c, iterations, startTemp, endTemp, seed);
            
            if (!opt.anyOvl()) {
                double score = opt.score();
                if (score < bestOptScore) {
                    bestOptScore = score;
                    bestOpt = opt;
                }
            }
        }
        
        if (bestOptScore < oldScore - 1e-12) {
            #pragma omp critical
            {
                results[n] = bestOpt;
                improved++;
                // Save only if running full batch, or maybe just don't save periodically for single group mode to avoid races or IO overhead
                if (targetGroup == -1 && n % 20 == 0) saveCSV(outputFile, results); 
                printf("  N=%3d: %.6f -> %.6f (%.4f%%)\n", n, oldScore, bestOptScore, (oldScore - bestOptScore) / oldScore * 100);
                fflush(stdout);
            }
        }
    }
    
    // Only run phases 2 and 3 if operating on all groups, OR if specifically requested?
    // Earthquakes usually work on the result. If we are doing single group, we probably want earthquake on that group too.
    // Backprop requires neighbors so it only works if we optimize everything or at least have neighbors.
    // Assuming single group mode is mostly for independent SA optimization.
    
    if (targetGroup == -1 && doBackProp) {
        // ... (existing backprop code) ...
         printf("\nPhase 2: Back Propagation (removing trees)...\n");
        
        bool changed = true;
        int pass = 0;
        while (changed && pass < 5) {
            changed = false;
            pass++;
            
            for (int k = 200; k >= 2; k--) {
                if (!results.count(k) || !results.count(k-1)) continue;
                
                double sideK = results[k].side();
                double sideK1 = results[k-1].side();
                
                if (sideK < sideK1 - 1e-12) {
                    Cfg& cfgK = results[k];
                    double bestSide = sideK1;
                    Cfg bestCfg = results[k-1];
                    
                    // Only try removing trees that touch the bbox (smarter selection)
                    vector<int> touchingIndices = cfgK.getBboxTouchingIndices();
                    for (int removeIdx : touchingIndices) {
                        Cfg reduced = cfgK.removeTree(removeIdx);
                        if (!reduced.anyOvl()) {
                            reduced = squeeze(reduced);
                            reduced = localSearch(reduced, 50);
                            
                            if (!reduced.anyOvl() && reduced.side() < bestSide) {
                                bestSide = reduced.side();
                                bestCfg = reduced;
                            }
                        }
                    }
                    
                    if (bestSide < sideK1 - 1e-12 && !bestCfg.anyOvl()) {
                        double oldScore = results[k-1].score();
                        double newScore = bestCfg.score();
                        results[k-1] = bestCfg;
                        saveCSV(outputFile, results); // Save on backprop improve
                        printf("  N=%3d: %.6f -> %.6f (from N=%d, %.4f%%)\n", k-1, oldScore, newScore, k, (oldScore-newScore)/oldScore*100);
                        fflush(stdout);
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Phase 3: Earthquake 
    if (earthquakeRounds > 0) {
        printf("\nPhase 3: Earthquake (%d rounds)...\n", earthquakeRounds);
        
        for (int round = 1; round <= earthquakeRounds; round++) {
            printf("  Earthquake round %d...\n", round);
            int roundImproved = 0;
            double strength = baseStrength + (round - 1) * 0.5;  // Increase strength each round
            
            #pragma omp parallel for schedule(dynamic)
            for (int idx = 0; idx < (int)nvals.size(); idx++) {
                int n = nvals[idx];
                Cfg c = results[n];
                double oldSide = c.side();
                
                FastRNG rng(42 + n * 1000 + round * 7777 + idx);
                
                // Earthquake: shake and re-optimize
                Cfg shaken = earthquake(c, strength, rng);
                if (shaken.anyOvl()) continue;
                
                // Re-optimize
                shaken = sa_optimize(shaken, iterations / 2, 0.5, 0.0001, rng);
                shaken = localSearch(shaken, 30);
                shaken = squeeze(shaken);
                shaken = localSearch(shaken, 20);
                
                if (!shaken.anyOvl() && shaken.side() < oldSide - 1e-12) {
                    #pragma omp critical
                    {
                        if (shaken.side() < results[n].side()) {
                            results[n] = shaken;
                            roundImproved++;
                            // Only save if full run
                             if (targetGroup == -1) saveCSV(outputFile, results);
                            printf("    N=%3d: %.6f -> %.6f (%.4f%%)\n", n, c.score(), shaken.score(), (c.score() - shaken.score()) / c.score() * 100);
                            fflush(stdout);
                        }
                    }
                }
            }
            printf("  Round %d: %d configs improved\n", round, roundImproved);
        }
    }
    
    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration_cast<chrono::milliseconds>(t1-t0).count() / 1000.0;
    
    double finalScore = 0;
    for (auto& [n, c] : cfg) {
        // If we only updated one group, use the new result for that group and old results for others to calc score
        if (results.count(n)) finalScore += results[n].score();
        else finalScore += c.score(); 
    }
    
    printf("\n========================================\n");
    printf("Initial: %.6f\n", initScore);
    printf("Final:   %.6f\n", finalScore);
    printf("Improve: %.6f (%.4f%%)\n", initScore - finalScore, (initScore - finalScore) / initScore * 100);
    printf("Time: %.1fs\n", elapsed);
    printf("========================================\n");
    
    // For single group, we assume the caller wants to see just that group in the output or the full set?
    // The previous code always dumped everything. 'results' contains everything (copy of cfg) plus updates.
    // So writing 'results' is correct.
    saveCSV(outputFile, results);
    printf("Saved %s\n", outputFile.c_str());
    
    return 0;
}
