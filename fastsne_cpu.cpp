// fastsne_cpu: the fastsne pipeline on CPUs (OpenMP + FFTW), same inputs, outputs and model files.
//
//   g++ -O3 -march=native -fopenmp -std=c++17 -o fastsne_cpu fastsne_cpu.cpp -lfftw3f_omp -lfftw3f -lm
//
// Differences from the GPU version, chosen for CPU costs:
//   - PCA covariance from a row sample (--pca-sample, 1M rows)
//   - kNN index: two-level k-means (coarse clusters, then fine clusters inside each) so that assigning
//     points stays cheap; candidates are stored dimension-major per cluster and scanned in 4-query x
//     16-candidate register tiles
//   - repulsion: same single-splat FFT field (FFTW), bilinear splat (per-thread grids while the field is small)
//   - --sphere: same 3D FFT field as the GPU, with in-place transforms that skip the empty part of the padded grid
//     (~40% less FFT work) and real-valued kernel spectra (half the FFT memory); the cubic splat uses atomics on
//     points in kNN-cluster order
//   - --project: brute-force nearest reference cluster, then the same tiled scan over its probe list
//   - neighbour rows are re-sorted by index after the affinities, which speeds up both the attraction
//     loop (memory order) and the neighbour-overlap measure used by --dedup (sorted-list intersection)

#include "common.h"
#include <omp.h>
#include <fftw3.h>
#include <unordered_map>

struct F2 { float x, y; };
// native-width float vectors for the kNN scan tile (GCC lowers wider generic vectors through the stack);
// vfu is the same type without the alignment requirement, for direct unaligned loads
#ifdef __AVX512F__
constexpr int VW = 16;
#else
constexpr int VW = 8;
#endif
typedef float vf __attribute__((vector_size(VW * 4)));
typedef float vfu __attribute__((vector_size(VW * 4), aligned(4)));

// ---------------------------------------------------------------------------------------------
// Transform + PCA
// ---------------------------------------------------------------------------------------------
static void transform_row(float* x, int D, int mode) {
    if (mode == 0) {
        float mn = 1e30f, s = 0;
        for (int j = 0; j < D; j++) { float v = x[j]; if (v > 0 && v < mn) mn = v; }
        float eps = mn < 1e30f ? 0.5f * mn : 1.f;
        for (int j = 0; j < D; j++) { float v = logf(std::max(x[j], 0.f) + eps); x[j] = v; s += v; }
        s /= D;
        for (int j = 0; j < D; j++) x[j] -= s;
    } else if (mode == 1) {
        float s = 0;
        for (int j = 0; j < D; j++) s += std::max(x[j], 0.f);
        s = s > 0 ? 1.f / s : 1.f;
        for (int j = 0; j < D; j++) x[j] = sqrtf(std::max(x[j], 0.f) * s);
    }
}

static void transform_all(Data& d, int mode) {
#pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < d.chunks.size(); c++) {
        int64_t rows = d.cstart[c + 1] - d.cstart[c];
        for (int64_t r = 0; r < rows; r++) transform_row(d.chunks[c].data() + r * d.D, d.D, mode);
    }
}

// Y = (X - mu) V for every (transformed) row: N x DP row-major
static std::vector<float> pca_project(const Data& d, const std::vector<float>& muf, const std::vector<float>& Vp, int DP) {
    const int D = d.D;
    // V laid out [k][DP] for a vectorised inner loop
    std::vector<float> Vk((size_t)D * DP);
    for (int j = 0; j < DP; j++) for (int k = 0; k < D; k++) Vk[(size_t)k * DP + j] = Vp[(size_t)j * D + k];
    std::vector<float> Y((size_t)d.N * DP);
#pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < d.chunks.size(); c++) {
        int64_t rows = d.cstart[c + 1] - d.cstart[c];
        std::vector<float> acc(DP);
        for (int64_t r = 0; r < rows; r++) {
            const float* x = d.chunks[c].data() + r * D;
            std::fill(acc.begin(), acc.end(), 0.f);
            for (int k = 0; k < D; k++) { float v = x[k] - muf[k]; const float* vk = Vk.data() + (size_t)k * DP; for (int j = 0; j < DP; j++) acc[j] += v * vk[j]; }
            memcpy(Y.data() + (size_t)(d.cstart[c] + r) * DP, acc.data(), DP * 4);
        }
    }
    return Y;
}

// returns N x DP row-major (cluster order not applied yet); fills mu and Vp (D x DP, column j = component j)
static std::vector<float> transform_pca_cpu(Data& d, const Opts& o, int DP, int mode, std::vector<float>& muf, std::vector<float>& Vp) {
    const int D = d.D; const int64_t N = d.N; const size_t nch = d.chunks.size();
    transform_all(d, mode);
    // mean over all rows
    std::vector<double> mu(D, 0);
#pragma omp parallel
    {
        std::vector<double> loc(D, 0);
#pragma omp for schedule(dynamic) nowait
        for (size_t c = 0; c < nch; c++) {
            int64_t rows = d.cstart[c + 1] - d.cstart[c];
            const float* x = d.chunks[c].data();
            for (int64_t r = 0; r < rows; r++) for (int j = 0; j < D; j++) loc[j] += x[r * D + j];
        }
#pragma omp critical
        for (int j = 0; j < D; j++) mu[j] += loc[j];
    }
    muf.assign(D, 0.f);
    for (int j = 0; j < D; j++) { mu[j] /= N; muf[j] = (float)mu[j]; }
    // covariance on a row sample (every stride-th row)
    int64_t stride = std::max<int64_t>(1, N / std::max<int64_t>(1, o.pca_sample));
    std::vector<double> C((size_t)D * D, 0);
    int64_t ns = 0;
    // blocks of 64 sampled rows: each row of the (upper) covariance triangle stays in L1 while the block's rows
    // stream past it, instead of the whole D x D triangle streaming from memory once per row (memory bound: at
    // D = 1024 one row at a time took 300 s for 905k rows on 16 threads)
    const int RB = 64;
    const int64_t nsamp = (N + stride - 1) / stride, nblk = (nsamp + RB - 1) / RB;
#pragma omp parallel
    {
        std::vector<double> loc((size_t)D * D, 0);
        std::vector<double> xb((size_t)RB * D);
        int64_t nl = 0;
#pragma omp for schedule(dynamic, 4) nowait
        for (int64_t blk = 0; blk < nblk; blk++) {
            int nb = (int)std::min<int64_t>(RB, nsamp - blk * RB);
            for (int r = 0; r < nb; r++) {
                int64_t i = (blk * RB + r) * stride;
                size_t c = std::upper_bound(d.cstart.begin(), d.cstart.end(), i) - d.cstart.begin() - 1;
                const float* x = d.chunks[c].data() + (i - d.cstart[c]) * D;
                for (int j = 0; j < D; j++) xb[(size_t)r * D + j] = x[j] - mu[j];
            }
            for (int a = 0; a < D; a++) {
                double* row = loc.data() + (size_t)a * D;
                for (int r = 0; r < nb; r++) { const double* xr = xb.data() + (size_t)r * D; double va = xr[a]; for (int b = a; b < D; b++) row[b] += va * xr[b]; }
            }
            nl += nb;
        }
#pragma omp critical
        { for (size_t j = 0; j < C.size(); j++) C[j] += loc[j]; ns += nl; }
    }
    for (int a = 0; a < D; a++) for (int b = a; b < D; b++) { C[(size_t)a * D + b] /= ns; C[(size_t)b * D + a] = C[(size_t)a * D + b]; }
    double tr = 0; for (int j = 0; j < D; j++) tr += C[(size_t)j * D + j];
    std::vector<double> V, ev;
    sym_eigen(C, D, std::min(o.pca, D), V, ev);
    std::vector<int> ord(D); std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return ev[a] > ev[b]; });
    int npc = std::min(o.pca, D);
    double cum = 0; for (int j = 0; j < npc; j++) cum += ev[ord[j]];
    LOG("PCA (%lld sampled rows): %d comps explain %.1f%% of variance", (long long)ns, npc, 100 * cum / tr);
    Vp.assign((size_t)D * DP, 0.f);
    for (int j = 0; j < npc; j++) for (int i = 0; i < D; i++) Vp[(size_t)j * D + i] = (float)V[(size_t)i * D + ord[j]];
    return pca_project(d, muf, Vp, DP);
}

// ---------------------------------------------------------------------------------------------
// k-means helpers (centroids stored dimension-major for vectorised assignment)
// ---------------------------------------------------------------------------------------------
// nearest centroid of each point; Ct is DP x nc (dimension-major)
static void assign_points(const float* X, const int64_t* rows, int64_t n, int DP, const std::vector<float>& Ct, int nc, int* out, bool parallel) {
    auto body = [&](int64_t t, std::vector<float>& acc) {
        const float* x = X + (rows ? rows[t] : t) * DP;
        std::fill(acc.begin(), acc.end(), 0.f);
        for (int d = 0; d < DP; d++) { float xd = x[d]; const float* c = Ct.data() + (size_t)d * nc; for (int k = 0; k < nc; k++) { float e = xd - c[k]; acc[k] += e * e; } }
        int b = 0; float bv = acc[0];
        for (int k = 1; k < nc; k++) if (acc[k] < bv) { bv = acc[k]; b = k; }
        out[t] = b;
    };
    if (parallel) {
#pragma omp parallel
        {
            std::vector<float> acc(nc);
#pragma omp for schedule(static)
            for (int64_t t = 0; t < n; t++) body(t, acc);
        }
    } else {
        std::vector<float> acc(nc);
        for (int64_t t = 0; t < n; t++) body(t, acc);
    }
}

// Lloyd iterations on the rows listed in `rows` (or 0..n-1); returns centroids row-major nc x DP
static std::vector<float> kmeans(const float* X, const int64_t* rows, int64_t n, int DP, int nc, int iters, uint64_t seed, bool parallel) {
    std::mt19937_64 rng(seed);
    std::vector<float> C((size_t)nc * DP), Ct((size_t)DP * nc);
    for (int k = 0; k < nc; k++) { int64_t r = rng() % n; memcpy(&C[(size_t)k * DP], X + (rows ? rows[r] : r) * DP, DP * 4); }
    std::vector<int> a(n);
    for (int it = 0; it < iters; it++) {
        for (int k = 0; k < nc; k++) for (int d = 0; d < DP; d++) Ct[(size_t)d * nc + k] = C[(size_t)k * DP + d];
        assign_points(X, rows, n, DP, Ct, nc, a.data(), parallel);
        std::vector<double> S((size_t)nc * DP, 0); std::vector<int64_t> cnt(nc, 0);
        for (int64_t t = 0; t < n; t++) { const float* x = X + (rows ? rows[t] : t) * DP; double* s = &S[(size_t)a[t] * DP]; for (int d = 0; d < DP; d++) s[d] += x[d]; cnt[a[t]]++; }
        for (int k = 0; k < nc; k++) {
            if (cnt[k] == 0) { int64_t r = rng() % n; memcpy(&C[(size_t)k * DP], X + (rows ? rows[r] : r) * DP, DP * 4); continue; }
            for (int d = 0; d < DP; d++) C[(size_t)k * DP + d] = (float)(S[(size_t)k * DP + d] / cnt[k]);
        }
    }
    return C;
}

static std::vector<float> transpose_c(const std::vector<float>& C, int nc, int DP) {
    std::vector<float> Ct((size_t)DP * nc);
    for (int k = 0; k < nc; k++) for (int d = 0; d < DP; d++) Ct[(size_t)d * nc + k] = C[(size_t)k * DP + d];
    return Ct;
}

// ---------------------------------------------------------------------------------------------
// kNN: two-level k-means inverted file + register-tiled scan
// ---------------------------------------------------------------------------------------------
static inline void topk_insert(float* bd, int* bi, int K, float dd, int id) {
    int pos = K - 1;
    while (pos > 0 && bd[pos - 1] > dd) { bd[pos] = bd[pos - 1]; bi[pos] = bi[pos - 1]; pos--; }
    bd[pos] = dd; bi[pos] = id;
}

struct KnnOut { std::vector<int> perm, I, offs; std::vector<float> X, Dd, C; int nc = 0; };

// Inverted file over points already sorted by cluster: per cluster a dimension-major copy padded to a multiple of
// 16 with far-away points, and per cluster the np nearest clusters (by centroid, itself first) to scan
struct IVF {
    int nc = 0, DP = 0, np = 0;
    std::vector<int> offs, probes;
    std::vector<int64_t> toff;
    std::vector<float> Xt;
};

static void ivf_build(IVF& V, const float* X, const std::vector<int>& offs, const std::vector<float>& Cf, int nc, int DP, int nprobe) {
    V.nc = nc; V.DP = DP; V.offs = offs;
    V.toff.assign(nc + 1, 0);
    for (int k = 0; k < nc; k++) V.toff[k + 1] = V.toff[k] + (int64_t)((offs[k + 1] - offs[k] + 15) / 16 * 16) * DP;
    V.Xt.assign(V.toff[nc], 0.f);
#pragma omp parallel for schedule(dynamic, 16)
    for (int k = 0; k < nc; k++) {
        int s = offs[k], m = offs[k + 1] - s, mp = (m + 15) / 16 * 16;
        float* b = V.Xt.data() + V.toff[k];
        for (int d = 0; d < DP; d++) for (int t = 0; t < mp; t++) b[(size_t)d * mp + t] = t < m ? X[(size_t)(s + t) * DP + d] : 1e15f;
    }
    const int np = V.np = std::min(nprobe, nc);
    V.probes.assign((size_t)nc * np, 0);
    std::vector<float> Ct = transpose_c(Cf, nc, DP);
#pragma omp parallel
    {
        std::vector<float> acc(nc); std::vector<float> bd(np); std::vector<int> bi(np);
#pragma omp for schedule(dynamic, 16)
        for (int c = 0; c < nc; c++) {
            std::fill(acc.begin(), acc.end(), 0.f);
            for (int d = 0; d < DP; d++) { float x = Cf[(size_t)c * DP + d]; const float* ct = Ct.data() + (size_t)d * nc; for (int k = 0; k < nc; k++) { float e = x - ct[k]; acc[k] += e * e; } }
            acc[c] = -1;
            std::fill(bd.begin(), bd.end(), 1e30f); std::fill(bi.begin(), bi.end(), c);
            for (int k = 0; k < nc; k++) if (acc[k] < bd[np - 1]) topk_insert(bd.data(), bi.data(), np, acc[k], k);
            for (int j = 0; j < np; j++) V.probes[(size_t)c * np + j] = bi[j];
        }
    }
}

// K nearest indexed points of each query: rows qoffs[c]..qoffs[c+1] of Q are the queries of cluster c. With
// `self`, Q is the indexed set itself (qoffs == offs) and a query does not count as its own neighbour.
// Work items are runs of up to 256 queries of one cluster (so one big cluster still spreads over all threads).
// Loop order keeps candidates in cache: for each probed cluster, for each chunk of 1024 candidates (128 KB at 32
// dims), every 4-query tile of the run scans it in 4-query x 16-candidate register tiles.
static void ivf_scan(const IVF& V, const float* Q, const std::vector<int64_t>& qoffs, int K, bool self, int* I, float* Dd) {
    constexpr int QB = 4, CB = 16, QRUN = 256, TC = 1024;
    const int DP = V.DP, np = V.np;
    std::vector<std::pair<int, int64_t>> items;
    for (int c = 0; c < V.nc; c++) for (int64_t q = qoffs[c]; q < qoffs[c + 1]; q += QRUN) items.push_back({c, q});
#pragma omp parallel
    {
        std::vector<float> qv((size_t)QRUN * DP);
        std::vector<float> bd((size_t)QRUN * K); std::vector<int> bi((size_t)QRUN * K);
#pragma omp for schedule(dynamic, 1)
        for (size_t it = 0; it < items.size(); it++) {
            const int c = items[it].first;
            const int64_t s = items[it].second;
            const int m = (int)std::min<int64_t>(QRUN, qoffs[c + 1] - s), mt = (m + QB - 1) / QB * QB;
            for (int a = 0; a < mt; a++) memcpy(&qv[(size_t)a * DP], Q + (size_t)(s + std::min(a, m - 1)) * DP, DP * 4);   // pad with the last query
            std::fill(bd.begin(), bd.begin() + (size_t)m * K, 1e30f); std::fill(bi.begin(), bi.begin() + (size_t)m * K, -1);
            for (int j = 0; j < np; j++) {
                int p = V.probes[(size_t)c * np + j];
                int ps = V.offs[p], pm = V.offs[p + 1] - ps, pmp = (pm + 15) / 16 * 16;
                const float* blk = V.Xt.data() + V.toff[p];
                for (int tc = 0; tc < pmp; tc += TC) {
                    const int te = std::min(pmp, tc + TC);
                    for (int q0 = 0; q0 < mt; q0 += QB) {
                        const int nq = std::min(QB, m - q0);
                        const float* qt = &qv[(size_t)q0 * DP];
                        for (int t0 = tc; t0 < te; t0 += CB) {
                            // explicit native-width vectors with direct unaligned loads: the auto-vectorised loop
                            // ran at ~1 candidate-dim per cycle (stack round trips), this at ~6
                            constexpr int NV = CB / VW;
                            vf sa[QB][NV] = {};
                            for (int d = 0; d < DP; d++) {
                                vf col[NV];
                                for (int v = 0; v < NV; v++) col[v] = *(const vfu*)(blk + (size_t)d * pmp + t0 + v * VW);
                                for (int a = 0; a < QB; a++) { float qd = qt[(size_t)a * DP + d]; for (int v = 0; v < NV; v++) { vf e = qd - col[v]; sa[a][v] += e * e; } }
                            }
                            float acc[QB][CB];
                            memcpy(acc, sa, sizeof acc);
                            for (int a = 0; a < nq; a++) {
                                float* bda = &bd[(size_t)(q0 + a) * K]; int* bia = &bi[(size_t)(q0 + a) * K];
                                int64_t me = self ? s + q0 + a : -1;
                                for (int t = 0; t < CB; t++) {
                                    float dd = acc[a][t];
                                    if (dd < bda[K - 1] && ps + t0 + t != me && t0 + t < pm) topk_insert(bda, bia, K, dd, ps + t0 + t);
                                }
                            }
                        }
                    }
                }
            }
            for (int a = 0; a < m; a++) {
                int64_t q = s + a;
                for (int j = 0; j < K; j++) {
                    int b = bi[(size_t)a * K + j]; float dv = bd[(size_t)a * K + j];
                    if (b < 0) { b = self ? (int)q : 0; dv = 1e30f; }
                    I[q * K + j] = b; Dd[q * K + j] = dv;
                }
            }
        }
    }
}

static void knn_cpu(std::vector<float>& Y, int64_t N, int DP, const Opts& o, int K, KnnOut& R) {
    const int S = o.cluster_size;                 // target fine cluster size
    int nthreads = omp_get_max_threads();
    // 1. coarse clusters on a sample, assign everything
    int c1 = (int)std::max<int64_t>(1, std::min<int64_t>({(int64_t)std::max(4 * nthreads, 64), N / std::max(8 * S, 1), 4096}));
    int64_t ns1 = std::min<int64_t>(N, (int64_t)c1 * 256);
    std::vector<int64_t> srows(ns1);
    { std::mt19937_64 rng(o.seed); for (auto& r : srows) r = (int64_t)(rng() % N); }
    std::vector<float> C1 = kmeans(Y.data(), srows.data(), ns1, DP, c1, 6, o.seed, true);
    std::vector<int> a1(N);
    assign_points(Y.data(), nullptr, N, DP, transpose_c(C1, c1, DP), c1, a1.data(), true);
    std::vector<int64_t> o1(c1 + 1, 0);
    for (int64_t i = 0; i < N; i++) o1[a1[i] + 1]++;
    for (int k = 0; k < c1; k++) o1[k + 1] += o1[k];
    std::vector<int64_t> mem(N);
    { std::vector<int64_t> cur(o1.begin(), o1.end() - 1); for (int64_t i = 0; i < N; i++) mem[cur[a1[i]]++] = i; }
    LOG("kNN: %d coarse clusters", c1);
    // 2. fine clusters inside each coarse cluster (one coarse cluster per task)
    std::vector<int> nf(c1), foff(c1 + 1, 0);
    for (int g = 0; g < c1; g++) nf[g] = (int)std::max<int64_t>(1, ((o1[g + 1] - o1[g]) + S / 2) / S);
    for (int g = 0; g < c1; g++) foff[g + 1] = foff[g] + nf[g];
    const int nc = foff[c1];
    std::vector<float> Cf((size_t)nc * DP);
    std::vector<int> fa(N);   // fine cluster of each point
#pragma omp parallel for schedule(dynamic, 1)
    for (int g = 0; g < c1; g++) {
        int64_t m = o1[g + 1] - o1[g];
        if (m == 0) continue;
        const int64_t* rows = mem.data() + o1[g];
        int k = nf[g];
        std::vector<float> C;
        if (k == 1) {
            C.assign(DP, 0.f);
            for (int64_t t = 0; t < m; t++) for (int d = 0; d < DP; d++) C[d] += Y[rows[t] * DP + d] / m;
        } else {
            int64_t msamp = std::min<int64_t>(m, (int64_t)k * 64);
            std::vector<int64_t> sr(msamp);
            std::mt19937_64 rng(o.seed + g);
            for (auto& r : sr) r = rows[rng() % m];
            C = kmeans(Y.data(), sr.data(), msamp, DP, k, 5, o.seed + g, false);
        }
        std::vector<int> loc(m);
        if (k > 1) assign_points(Y.data(), rows, m, DP, transpose_c(C, k, DP), k, loc.data(), false);
        for (int64_t t = 0; t < m; t++) fa[rows[t]] = foff[g] + (k > 1 ? loc[t] : 0);
        memcpy(&Cf[(size_t)foff[g] * DP], C.data(), (size_t)k * DP * 4);
    }
    std::vector<int64_t>().swap(mem); std::vector<int>().swap(a1);
    // 3. sort points by fine cluster
    std::vector<int> offs(nc + 1, 0);
    for (int64_t i = 0; i < N; i++) offs[fa[i] + 1]++;
    for (int k = 0; k < nc; k++) offs[k + 1] += offs[k];
    R.perm.resize(N);
    { std::vector<int> cur(offs.begin(), offs.end() - 1); for (int64_t i = 0; i < N; i++) R.perm[cur[fa[i]]++] = (int)i; }
    std::vector<int>().swap(fa);
    R.X.resize((size_t)N * DP);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) memcpy(&R.X[(size_t)i * DP], &Y[(size_t)R.perm[i] * DP], DP * 4);
    std::vector<float>().swap(Y);
    int maxsz = 0; for (int k = 0; k < nc; k++) maxsz = std::max(maxsz, offs[k + 1] - offs[k]);
    IVF V;
    ivf_build(V, R.X.data(), offs, Cf, nc, DP, o.nprobe);
    double work = 0;
    for (int c = 0; c < nc; c++) { double cand = 0; for (int j = 0; j < V.np; j++) { int p = V.probes[(size_t)c * V.np + j]; cand += offs[p + 1] - offs[p]; } work += cand * (offs[c + 1] - offs[c]); }
    LOG("kNN: %d fine clusters (max size %d), %d probes, %.0f candidates/query", nc, maxsz, V.np, work / N);
    R.I.assign((size_t)N * K, 0); R.Dd.assign((size_t)N * K, 1e30f);
    std::vector<int64_t> qoffs(offs.begin(), offs.end());
    ivf_scan(V, R.X.data(), qoffs, K, true, R.I.data(), R.Dd.data());
    LOG("kNN scan done");
    // optional NN-descent rounds (in place)
    for (int r = 0; r < o.refine; r++) {
        int A = std::min(K, o.refine_a);
#pragma omp parallel
        {
            std::vector<float> bd(K); std::vector<int> bi(K);
#pragma omp for schedule(dynamic, 1024)
            for (int64_t q = 0; q < N; q++) {
                memcpy(bd.data(), &R.Dd[q * K], K * 4); memcpy(bi.data(), &R.I[q * K], K * 4);
                const float* x = &R.X[(size_t)q * DP];
                for (int a = 0; a < A; a++) {
                    int nb = bi[a]; if (nb == q) continue;
                    for (int b = 0; b < A; b++) {
                        int c = R.I[(int64_t)nb * K + b]; if (c == q) continue;
                        const float* y = &R.X[(size_t)c * DP];
                        float dd = 0; for (int dd_ = 0; dd_ < DP; dd_++) { float e = x[dd_] - y[dd_]; dd += e * e; }
                        if (dd >= bd[K - 1]) continue;
                        bool dup = false; for (int j = 0; j < K; j++) if (bi[j] == c) { dup = true; break; }
                        if (!dup) topk_insert(bd.data(), bi.data(), K, dd, c);
                    }
                }
                memcpy(&R.Dd[q * K], bd.data(), K * 4); memcpy(&R.I[q * K], bi.data(), K * 4);
            }
        }
        LOG("kNN refinement round %d done", r + 1);
    }
    if (o.check_knn > 0) {
        int nq = (int)std::min<int64_t>(o.check_knn, N);
        std::mt19937_64 rng(o.seed + 99);
        std::vector<int64_t> qs(nq); for (auto& q : qs) q = (int64_t)(rng() % N);
        int64_t hit = 0;
#pragma omp parallel for reduction(+:hit) schedule(dynamic)
        for (int t = 0; t < nq; t++) {
            int64_t q = qs[t];
            std::vector<float> bd(K, 1e30f); std::vector<int> bi(K, -1);
            const float* x = &R.X[(size_t)q * DP];
            for (int64_t j = 0; j < N; j++) {
                if (j == q) continue;
                const float* y = &R.X[(size_t)j * DP];
                float dd = 0; for (int d = 0; d < DP; d++) { float e = x[d] - y[d]; dd += e * e; }
                if (dd < bd[K - 1]) topk_insert(bd.data(), bi.data(), K, dd, (int)j);
            }
            std::vector<int> mine(R.I.begin() + q * K, R.I.begin() + (q + 1) * K);
            std::sort(mine.begin(), mine.end());
            for (int j = 0; j < K; j++) hit += std::binary_search(mine.begin(), mine.end(), bi[j]);
        }
        LOG("kNN recall@%d = %.3f (%d queries)", K, hit / (double)nq / K, nq);
    }
    R.offs = offs; R.C = Cf; R.nc = nc;
}

// ---------------------------------------------------------------------------------------------
// Affinities, neighbour overlap, masses
// ---------------------------------------------------------------------------------------------
static void perplexity_row(float* d, int K, float logperp, bool uniform, float scale) {
    int kv = 0; while (kv < K && d[kv] < 1e29f) kv++;
    if (uniform || kv == 0) { for (int j = 0; j < K; j++) d[j] = j < kv ? scale / kv : 0.f; return; }
    float d0 = d[0], lo = 0, hi = 1e30f, mean = 0;
    for (int j = 0; j < kv; j++) mean += d[j] - d0;
    mean /= kv;
    float beta = mean > 0 ? 1.f / mean : 1.f;
    for (int it = 0; it < 200; it++) {
        float s = 0, sd = 0;
        for (int j = 0; j < kv; j++) { float dj = d[j] - d0; float w = dj > 0 ? expf(-beta * dj) : 1.f; s += w; sd += dj > 0 ? w * dj : 0.f; }
        float diff = logf(s) + beta * sd / s - logperp;
        if (fabsf(diff) < 1e-5f) break;
        if (diff > 0) { lo = beta; beta = hi >= 1e29f ? beta * 2 : 0.5f * (beta + hi); if (beta > 1e30f) { beta = 1e30f; break; } }
        else { hi = beta; beta = 0.5f * (beta + lo); }
    }
    float s = 0;
    for (int j = 0; j < kv; j++) { float dj = d[j] - d0; s += dj > 0 ? expf(-beta * dj) : 1.f; }
    for (int j = 0; j < K; j++) { float dj = d[j] - d0; d[j] = j < kv ? scale * (dj > 0 ? expf(-beta * dj) : 1.f) / s : 0.f; }
}

// ---------------------------------------------------------------------------------------------
// Optimisation
// ---------------------------------------------------------------------------------------------
struct FieldCPU {
    int P = 0; float h = 0;
    float *grid = nullptr, *kreal = nullptr, *out = nullptr;
    fftwf_complex *spec = nullptr, *kspec = nullptr, *ospec = nullptr;
    fftwf_plan r2c = nullptr, c2r3 = nullptr;
    void release() {
        if (!P) return;
        fftwf_destroy_plan(r2c); fftwf_destroy_plan(c2r3);
        fftwf_free(grid); fftwf_free(kreal); fftwf_free(out); fftwf_free(spec); fftwf_free(kspec); fftwf_free(ospec);
        P = 0;
    }
    void setup(int G, float h_) {
        int P_ = 2 * G;
        if (P_ != P) {
            release();
            P = P_;
            size_t PP = (size_t)P * P, PC = (size_t)P * (P / 2 + 1);
            grid = fftwf_alloc_real(PP); memset(grid, 0, PP * 4); kreal = fftwf_alloc_real(PP); out = fftwf_alloc_real(3 * PP);
            spec = fftwf_alloc_complex(PC); kspec = fftwf_alloc_complex(3 * PC); ospec = fftwf_alloc_complex(3 * PC);
            // FFTW threads only pay off on big grids; small early-iteration grids run single-threaded
            fftwf_plan_with_nthreads(std::max(1, std::min(omp_get_max_threads(), (int)(PP / (128 * 128)))));
            r2c = fftwf_plan_dft_r2c_2d(P, P, grid, spec, FFTW_ESTIMATE);
            int dims[2] = {P, P};
            c2r3 = fftwf_plan_many_dft_c2r(2, dims, 3, ospec, nullptr, 1, (int)PC, out, nullptr, 1, (int)PP, FFTW_ESTIMATE);
            h = 0;
        }
        if (h_ != h) {
            h = h_;
            size_t PP = (size_t)P * P, PC = (size_t)P * (P / 2 + 1);
            for (int k = 0; k < 3; k++) {
#pragma omp parallel for schedule(static)
                for (int r = 0; r < P; r++) {
                    int dy = r < P / 2 ? r : r - P;
                    for (int c = 0; c < P; c++) {
                        int dx = c < P / 2 ? c : c - P;
                        float x = dx * h, y = dy * h, w = 1.f / (1.f + x * x + y * y);
                        float v = k == 0 ? w : k == 1 ? (c == P / 2 ? 0.f : -2.f * x * w * w) : (r == P / 2 ? 0.f : -2.f * y * w * w);
                        kreal[(size_t)r * P + c] = v;
                    }
                }
                fftwf_execute_dft_r2c(r2c, kreal, kspec + k * PC);
            }
            float sc = 1.f / (float)PP;
#pragma omp parallel for schedule(static)
            for (size_t t = 0; t < 3 * PC; t++) { kspec[t][0] *= sc; kspec[t][1] *= sc; }
        }
    }
};

static inline void atomic_addf(float* p, float v) {
#pragma omp atomic
    *p += v;
}

// reverse CSR of the kNN graph (in-edges per point, edges with p > 0) by counting sort
static void build_reverse(int64_t N, int K, const std::vector<int>& I, const std::vector<float>& Pw,
                          std::vector<int64_t>& roff, std::vector<int>& rsrc, std::vector<float>& rp) {
    roff.assign(N + 1, 0);
    std::vector<std::atomic<int>> cnt(N);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) cnt[i].store(0, std::memory_order_relaxed);
#pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < N * K; e++) if (Pw[e] > 0) cnt[I[e]].fetch_add(1, std::memory_order_relaxed);
    for (int64_t i = 0; i < N; i++) roff[i + 1] = roff[i] + cnt[i].load(std::memory_order_relaxed);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) cnt[i].store(0, std::memory_order_relaxed);
    rsrc.assign(roff[N], 0); rp.assign(roff[N], 0);
#pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < N * K; e++) {
        if (!(Pw[e] > 0)) continue;
        int dst = I[e];
        int64_t pos = roff[dst] + cnt[dst].fetch_add(1, std::memory_order_relaxed);
        rsrc[pos] = (int)(e / K); rp[pos] = Pw[e];
    }
}

static void run_tsne_cpu(std::vector<F2>& Y, int64_t N, int K, const std::vector<int>& I, const std::vector<float>& Pw,
                         const std::vector<float>& mass, const Opts& o) {
    std::vector<int64_t> roff; std::vector<int> rsrc; std::vector<float> rp;
    build_reverse(N, K, I, Pw, roff, rsrc, rp);
    const bool hasm = !mass.empty();
    std::vector<F2> U(N, {0, 0}), Gn(N, {1, 1}), A(N), Gp(N);
    FieldCPU F;
    const int nthr = omp_get_max_threads();
    std::vector<float> priv;   // per-thread splat grids for small fields
    float lr = o.lr > 0 ? o.lr : std::max(200.f, (float)N / o.ee);
    int total = o.ee_iters + o.iters, Gcur = 0;
    double t_rep = 0, t_att = 0, t_upd = 0, t_fft = 0;
    for (int it = 0; it < total; it++) {
        bool ee = it < o.ee_iters;
        float exag = ee ? o.ee : o.exag, mom = ee ? 0.5f : 0.8f;
        double t0 = now();
        float xmn = 1e30f, ymn = 1e30f, xmx = -1e30f, ymx = -1e30f;
#pragma omp parallel for reduction(min:xmn,ymn) reduction(max:xmx,ymx) schedule(static)
        for (int64_t i = 0; i < N; i++) { xmn = std::min(xmn, Y[i].x); xmx = std::max(xmx, Y[i].x); ymn = std::min(ymn, Y[i].y); ymx = std::max(ymx, Y[i].y); }
        float ext = std::max(xmx - xmn, ymx - ymn);
        float h = o.h;
        int need = (int)ceilf(ext / h) + 3, Gmax = o.max_grid / 2;
        if (need > Gmax) {
            float hh = ext / (Gmax - 3);
            h = o.h * powf(1.02f, ceilf(logf(hh / o.h) / logf(1.02f)));
            need = (int)ceilf(ext / h) + 3;
        }
        int G = std::max(16, fft_size(need));
        if (G < Gcur && G * 3 / 2 > Gcur) G = Gcur;
        Gcur = G;
        double tf0 = now();
        F.setup(G, h);
        const int P = F.P;
        const size_t PP = (size_t)P * P, PC = (size_t)P * (P / 2 + 1);
        float cx = 0.5f * (xmn + xmx), cy = 0.5f * (ymn + ymx);
        float x0 = cx - 0.5f * (G - 1) * h, y0 = cy - 0.5f * (G - 1) * h, ih = 1.f / h;
        // splat (only the G x G corner is non-zero)
        // only rows/cols < G + 1 are ever written (the grid is fully zeroed when allocated)
#pragma omp parallel for schedule(static)
        for (int r = 0; r < G + 1; r++) memset(F.grid + (size_t)r * P, 0, (size_t)(G + 1) * 4);
        const size_t GW = (size_t)(G + 1) * (G + 1);
        if (GW * nthr <= ((size_t)64 << 20)) {
            // small grid (early iterations, all points in few cells): atomics would serialise, so each
            // thread splats into its own (G+1)^2 copy and the copies are summed
            priv.resize(GW * nthr);
#pragma omp parallel
            {
                float* pg = priv.data() + GW * omp_get_thread_num();
                memset(pg, 0, GW * 4);
                const int W = G + 1;
#pragma omp for schedule(static)
                for (int64_t i = 0; i < N; i++) {
                    float u = (Y[i].x - x0) * ih, v = (Y[i].y - y0) * ih;
                    int ix = (int)floorf(u), iy = (int)floorf(v);
                    float fx = u - ix, fy = v - iy, m = hasm ? mass[i] : 1.f;
                    float* g = pg + (size_t)iy * W + ix;
                    g[0] += m * (1 - fx) * (1 - fy); g[1] += m * fx * (1 - fy);
                    g[W] += m * (1 - fx) * fy; g[W + 1] += m * fx * fy;
                }
#pragma omp for schedule(static)
                for (size_t c = 0; c < GW; c++) {
                    float sum = 0;
                    for (int t = 0; t < nthr; t++) sum += priv[GW * t + c];
                    F.grid[(c / W) * P + c % W] = sum;
                }
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < N; i++) {
                float u = (Y[i].x - x0) * ih, v = (Y[i].y - y0) * ih;
                int ix = (int)floorf(u), iy = (int)floorf(v);
                float fx = u - ix, fy = v - iy, m = hasm ? mass[i] : 1.f;
                float* g = F.grid + (size_t)iy * P + ix;
                atomic_addf(g, m * (1 - fx) * (1 - fy)); atomic_addf(g + 1, m * fx * (1 - fy));
                atomic_addf(g + P, m * (1 - fx) * fy); atomic_addf(g + P + 1, m * fx * fy);
            }
        }
        double tf1 = now();
        fftwf_execute(F.r2c);
#pragma omp parallel for schedule(static)
        for (size_t t = 0; t < PC; t++) {
            float sr = F.spec[t][0], si = F.spec[t][1];
            for (int k = 0; k < 3; k++) {
                const float* kk = F.kspec[k * PC + t];
                F.ospec[k * PC + t][0] = sr * kk[0] - si * kk[1];
                F.ospec[k * PC + t][1] = sr * kk[1] + si * kk[0];
            }
        }
        fftwf_execute(F.c2r3);
        double tf2 = now();
        // gather phi, dphi/dx, dphi/dy and Z
        double Z = 0;
        const float k10 = 1.f / (1.f + h * h), k11 = 1.f / (1.f + 2 * h * h);
#pragma omp parallel for reduction(+:Z) schedule(static)
        for (int64_t i = 0; i < N; i++) {
            float u = (Y[i].x - x0) * ih, v = (Y[i].y - y0) * ih;
            int ix = (int)floorf(u), iy = (int)floorf(v);
            float fx = u - ix, fy = v - iy;
            size_t b = (size_t)iy * P + ix;
            float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
            float r3[3];
            for (int k = 0; k < 3; k++) { const float* f = F.out + k * PP + b; r3[k] = w00 * f[0] + w10 * f[1] + w01 * f[P] + w11 * f[P + 1]; }
            float m = hasm ? mass[i] : 1.f;
            float sx0 = (1 - fx) * (1 - fx) + fx * fx, sx1 = 2 * fx * (1 - fx), sy0 = (1 - fy) * (1 - fy) + fy * fy, sy1 = 2 * fy * (1 - fy);
            float self = sx0 * sy0 + (sx1 * sy0 + sx0 * sy1) * k10 + sx1 * sy1 * k11;
            Z += (double)m * (r3[0] - m * self);
            Gp[i] = {r3[1], r3[2]};
        }
        double t1 = now();
        // attraction over out- and in-edges
#pragma omp parallel for schedule(dynamic, 4096)
        for (int64_t i = 0; i < N; i++) {
            float yx = Y[i].x, yy = Y[i].y, ax = 0, ay = 0;
            const int* nb = &I[i * K]; const float* pw = &Pw[i * K];
            for (int j = 0; j < K; j++) {
                float p = pw[j]; if (p == 0.f) continue;
                F2 q = Y[nb[j]]; float dx = yx - q.x, dy = yy - q.y, f = p / (1.f + dx * dx + dy * dy);
                ax += f * dx; ay += f * dy;
            }
            for (int64_t e = roff[i]; e < roff[i + 1]; e++) {
                F2 q = Y[rsrc[e]]; float dx = yx - q.x, dy = yy - q.y, f = rp[e] / (1.f + dx * dx + dy * dy);
                ax += f * dx; ay += f * dy;
            }
            A[i] = {ax, ay};
        }
        double t2 = now(), tchk = 0;   // the check is left out of the timings
        if (o.check_grad > 0 && (it == o.ee_iters - 1 || it == total - 1)) {
            // exact repulsion for a few points vs the grid field (same check as the GPU build)
            int nq = (int)std::min<int64_t>(o.check_grad, N);
            std::mt19937 r2(7); std::vector<int> qs(nq); for (auto& q : qs) q = (int)(r2() % N);
            double num = 0, den = 0, zsum = 0;
            for (int q : qs) {
                double z = 0, fx = 0, fy = 0;
#pragma omp parallel for reduction(+:z,fx,fy) schedule(static)
                for (int64_t j = 0; j < N; j++) {
                    if (j == q) continue;
                    float dx = Y[q].x - Y[j].x, dy = Y[q].y - Y[j].y, w = 1.f / (1.f + dx * dx + dy * dy), m = hasm ? mass[j] : 1.f;
                    z += m * w; fx += m * w * w * dx; fy += m * w * w * dy;
                }
                double ax = -0.5 * Gp[q].x, ay = -0.5 * Gp[q].y;
                num += (ax - fx) * (ax - fx) + (ay - fy) * (ay - fy); den += fx * fx + fy * fy;
                zsum += (hasm ? mass[q] : 1.f) * z;
            }
            LOG("grad check it %d: rel. error of repulsive force %.4f, Z grid %.4g vs sampled exact %.4g (h=%.3f, P=%d)",
                it, sqrt(num / den), Z, zsum / nq * N, h, P);
            tchk = now() - t2;
            t2 = now();
        }
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) {
            float m = hasm ? mass[i] : 1.f, rs = (float)(0.5 * m / Z);
            float gx = exag * A[i].x + rs * Gp[i].x, gy = exag * A[i].y + rs * Gp[i].y;
            F2 u = U[i], gn = Gn[i];
            gn.x = (std::signbit(gx) != std::signbit(u.x)) ? gn.x + 0.2f : std::max(gn.x * 0.8f, 0.01f);
            gn.y = (std::signbit(gy) != std::signbit(u.y)) ? gn.y + 0.2f : std::max(gn.y * 0.8f, 0.01f);
            u.x = mom * u.x - lr * gn.x * gx; u.y = mom * u.y - lr * gn.y * gy;
            Y[i].x += u.x; Y[i].y += u.y; U[i] = u; Gn[i] = gn;
        }
        if (it == o.ee_iters - 1) std::fill(U.begin(), U.end(), F2{0, 0});
        double t3 = now();
        t_rep += t1 - t0; t_fft += tf2 - tf1; t_att += t2 - t1 - tchk; t_upd += t3 - t2;
        if (it % 100 == 0 || it == total - 1)
            LOG("iter %4d  extent %8.2f  grid %5d  h %.3f  Z/N %.4g  (%.0f ms/iter: repulsion %.0f [fft %.0f], attraction %.0f)", it, ext, P, h, Z / N,
                1e3 * (t3 - t0 - tchk), 1e3 * (t1 - t0), 1e3 * (tf2 - tf1), 1e3 * (t2 - t1 - tchk));
    }
    LOG("optimisation time: repulsion %.1fs (FFT %.1fs), attraction %.1fs, update %.1fs", t_rep, t_fft, t_att, t_upd);
    F.release();
}

// ---------------------------------------------------------------------------------------------
// Spherical embedding (--sphere R), the GPU's run_tsne_sphere on the CPU: points on a sphere of radius R in 3D,
// Cauchy kernel on chord distance. Repulsion from one 3D FFT over the cube around the sphere (cubic Lagrange
// splat / gather; the grid is fixed for the whole run); gradients projected onto the tangent plane, each point put
// back on the sphere after its step. Optional springs toward per-rank group centroids (--anchor-file).
// CPU specifics: in-place FFTW transforms pruned to the occupied corner (PrunedFFT3), and the kernel spectra kept
// as one real array each (the potential kernel is even, so its spectrum is real; the gradient kernels are odd, so
// theirs are imaginary), which takes the FFT buffers to 28 bytes per padded cell (3.8 GB at the default 512^3)
// instead of the GPU's 56.
// ---------------------------------------------------------------------------------------------
struct F3 { float x, y, z; };

// cubic Lagrange weights for nodes -1, 0, 1, 2 around t in [0, 1)
static inline void lagrange4(float t, float* w) {
    w[0] = -t * (t - 1) * (t - 2) / 6; w[1] = (t + 1) * (t - 1) * (t - 2) / 2;
    w[2] = -(t + 1) * t * (t - 2) / 2; w[3] = (t + 1) * t * (t - 1) / 6;
}

// --anchor-file: per rank r, lab[r * N + i] = group of point i (cluster order, -1 = none), groups of rank r are
// offs[r] .. offs[r] + (number of groups) - 1 in the centroid array; w[r] = spring weight
struct AnchorsCPU { int nr = 0, total = 0; std::vector<int> lab, offs; std::vector<float> w; };

// The density is non-zero only in the [0, G)^3 corner of the padded P^3 grid (P = 2G), and only that corner of the
// convolution is read back, so each 3D transform runs as three 1D passes that skip the empty (forward) or unread
// (inverse) part: x over G^2 rows, y over G planes, z over everything, about 58% of the work of a full transform.
// In-place layout: P x P rows of Pc = P/2+1 complex values (2 Pc floats as real rows).
struct PrunedFFT3 {
    fftwf_plan fx = nullptr, fy = nullptr, fz = nullptr, iz = nullptr, iy = nullptr, ix = nullptr;
    // forward plans made on a, inverse plans on b (arrays of P * P * Pc complex values, FFTW-allocated)
    void plan(int P, int G, fftwf_complex* a, fftwf_complex* b, unsigned flags) {
        const int Pc = P / 2 + 1, Ps = 2 * Pc;
        fftwf_iodim d, h[2];
        d = {P, 1, 1}; h[0] = {G, P * Ps, P * Pc}; h[1] = {G, Ps, Pc};          // x: rows z < G, y < G
        fx = fftwf_plan_guru_dft_r2c(1, &d, 2, h, (float*)a, a, flags);
        d = {P, 1, 1}; h[0] = {G, P * Pc, P * Ps}; h[1] = {G, Pc, Ps};
        ix = fftwf_plan_guru_dft_c2r(1, &d, 2, h, b, (float*)b, flags);
        d = {P, Pc, Pc}; h[0] = {G, P * Pc, P * Pc}; h[1] = {Pc, 1, 1};         // y: planes z < G
        fy = fftwf_plan_guru_dft(1, &d, 2, h, a, a, FFTW_FORWARD, flags);
        iy = fftwf_plan_guru_dft(1, &d, 2, h, b, b, FFTW_BACKWARD, flags);
        d = {P, P * Pc, P * Pc}; h[0] = {P * Pc, 1, 1};                          // z: all P * Pc columns
        fz = fftwf_plan_guru_dft(1, &d, 1, h, a, a, FFTW_FORWARD, flags);
        iz = fftwf_plan_guru_dft(1, &d, 1, h, b, b, FFTW_BACKWARD, flags);
        if (!fx || !fy || !fz || !ix || !iy || !iz) { fprintf(stderr, "FFTW planning failed\n"); exit(1); }
    }
    void forward(fftwf_complex* a) const { fftwf_execute_dft_r2c(fx, (float*)a, a); fftwf_execute_dft(fy, a, a); fftwf_execute_dft(fz, a, a); }
    void inverse(fftwf_complex* b) const { fftwf_execute_dft(iz, b, b); fftwf_execute_dft(iy, b, b); fftwf_execute_dft_c2r(ix, b, (float*)b); }
    void release() { for (fftwf_plan q : {fx, fy, fz, ix, iy, iz}) if (q) fftwf_destroy_plan(q); }
};

// Spectra of the Cauchy kernel w = 1/(1+r^2) (Re, scaled by 1/P^3) and of its gradient -2 r w^2 (Im, per axis) on
// the padded P^3 grid of spacing h. grid is the in-place r2c work array (overwritten); ksp holds 4 * PC floats.
static void kernel_spectra3(int P, float h, fftwf_complex* grid, fftwf_plan r2cf, float* ksp) {
    const int Pc = P / 2 + 1, Ps = 2 * Pc;
    const size_t PC = (size_t)P * P * Pc;
    float* g = (float*)grid;
    double kept = 0, dropped = 0;
    const float sc = 1.f / ((float)P * P * P);
    for (int k = 0; k < 4; k++) {
#pragma omp parallel for schedule(static)
        for (int l = 0; l < P; l++)
            for (int r = 0; r < P; r++) {
                int dz = l < P / 2 ? l : l - P, dy = r < P / 2 ? r : r - P;
                float* row = g + ((size_t)l * P + r) * Ps;
                for (int c = 0; c < P; c++) {
                    int dx = c < P / 2 ? c : c - P;
                    float x = dx * h, y = dy * h, z = dz * h, w = 1.f / (1.f + x * x + y * y + z * z);
                    row[c] = k == 0 ? w : k == 1 ? (c == P / 2 ? 0.f : -2.f * x * w * w)
                           : k == 2 ? (r == P / 2 ? 0.f : -2.f * y * w * w) : (l == P / 2 ? 0.f : -2.f * z * w * w);
                }
            }
        fftwf_execute(r2cf);   // the kernels fill the whole grid: full transform
        double kk = 0, kd = 0;
#pragma omp parallel for reduction(max:kk,kd) schedule(static)
        for (size_t t = 0; t < PC; t++) {
            float re = grid[t][0], im = grid[t][1];
            ksp[k * PC + t] = (k == 0 ? re : im) * sc;
            kk = std::max(kk, (double)fabsf(k == 0 ? re : im)); kd = std::max(kd, (double)fabsf(k == 0 ? im : re));
        }
        kept = std::max(kept, kk); dropped = std::max(dropped, kd);
    }
    if (dropped > 1e-4 * kept) LOG("warning: kernel spectra not real/imaginary (%.2g of max)", dropped / kept);
}

static void run_tsne_sphere_cpu(std::vector<F3>& Y, int64_t N, int K, const std::vector<int>& I, const std::vector<float>& Pw,
                                const std::vector<float>& mass, const Opts& o, const AnchorsCPU& an) {
    const float R = o.sphere;
    std::vector<int64_t> roff; std::vector<int> rsrc; std::vector<float> rp;
    build_reverse(N, K, I, Pw, roff, rsrc, rp);
    const bool hasm = !mass.empty();
    const int nthr = omp_get_max_threads();
    // grid: the cube around the sphere with three cells of margin; h grows if the padded grid would exceed max-grid
    const int Gmax = o.max_grid / 2;
    const float h = std::max(o.h, (2 * R) / (Gmax - 6));
    int G = fft_size((int)ceilf(2 * R / h) + 6);
    if (G > Gmax) G = Gmax;
    const int P = 2 * G, Pc = P / 2 + 1, Ps = 2 * Pc;   // Ps: padded row length of the in-place real arrays
    const size_t PC = (size_t)P * P * Pc;                // complex values per spectrum
    const float x0 = -0.5f * G * h, ih = 1.f / h;
    fftwf_complex* grid = fftwf_alloc_complex(PC);       // density (padded real), then its spectrum
    fftwf_complex* out[4];                               // product spectra, then phi and grad phi (padded real)
    for (auto& a : out) a = fftwf_alloc_complex(PC);
    float* ksp = fftwf_alloc_real(4 * PC);               // kernel spectra: Re (phi), Im (d/dx, d/dy, d/dz), scaled
    if (!grid || !out[0] || !out[1] || !out[2] || !out[3] || !ksp) { fprintf(stderr, "cannot allocate the %d^3 FFT grid\n", P); exit(1); }
    fftwf_plan_with_nthreads(nthr);
    // the grid is fixed for the whole run, so measured plans pay off (planning overwrites the buffers: done first)
    double tp = now();
    PrunedFFT3 F;
    F.plan(P, G, grid, out[0], FFTW_MEASURE);
    fftwf_plan r2cf = fftwf_plan_dft_r2c_3d(P, P, P, (float*)grid, grid, FFTW_ESTIMATE);   // for the kernels only
    LOG("FFT plans (%d^3): %.1fs", P, now() - tp);
    auto zero = [&](fftwf_complex* a) {
#pragma omp parallel for schedule(static)
        for (int l = 0; l < P; l++) memset(a + (size_t)l * P * Pc, 0, (size_t)P * Pc * 8);
    };
    kernel_spectra3(P, h, grid, r2cf, ksp);
    LOG("sphere R %.1f: grid %d^3 (padded %d^3), h %.3f, %.1f GB", R, G, P, h, PC * 56 / 1e9);

    std::vector<F3> U(N, {0, 0, 0}), Gn(N, {1, 1, 1}), A(N), Gp(N);
    std::vector<double> cacc;
    std::vector<float> cent;
    if (an.nr) { cacc.resize((size_t)nthr * an.total * 3); cent.resize((size_t)an.total * 3); }
    const float lr = o.lr > 0 ? o.lr : std::max(200.f, (float)N / o.ee);
    const int total = o.ee_iters + o.iters;
    double t_spl = 0, t_fft = 0, t_gat = 0, t_att = 0, t_upd = 0;
    for (int it = 0; it < total; it++) {
        bool ee = it < o.ee_iters;
        float exag = ee ? o.ee : o.exag, mom = ee ? 0.5f : 0.8f;
        double t0 = now();
        zero(grid);
        float* g = (float*)grid;
        // splat (atomics: points are in kNN-cluster order, so a thread's points mostly share cells with each other)
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) {
            F3 y = Y[i];
            float u = (y.x - x0) * ih, v = (y.y - x0) * ih, s = (y.z - x0) * ih;
            int ix = (int)floorf(u), iy = (int)floorf(v), iz = (int)floorf(s);
            float wx[4], wy[4], wz[4];
            lagrange4(u - ix, wx); lagrange4(v - iy, wy); lagrange4(s - iz, wz);
            float m = hasm ? mass[i] : 1.f;
            size_t b = ((size_t)(iz - 1) * P + (iy - 1)) * Ps + (ix - 1);
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 4; r++) {
                    float wzy = m * wz[c] * wy[r];
                    float* row = g + b + ((size_t)c * P + r) * Ps;
                    for (int q = 0; q < 4; q++) atomic_addf(row + q, wzy * wx[q]);
                }
        }
        double t1 = now();
        F.forward(grid);
#pragma omp parallel for schedule(static)
        for (size_t t = 0; t < PC; t++) {
            float sr = grid[t][0], si = grid[t][1], k0 = ksp[t];
            out[0][t][0] = sr * k0; out[0][t][1] = si * k0;
            for (int k = 1; k < 4; k++) { float kk = ksp[k * PC + t]; out[k][t][0] = -si * kk; out[k][t][1] = sr * kk; }
        }
        for (int k = 0; k < 4; k++) F.inverse(out[k]);
        double t2 = now();
        // gather phi and grad phi; Z = sum_i m_i (phi_i - m_i), the self-interaction taken as the exact w(0) = 1
        const float* fo[4] = {(const float*)out[0], (const float*)out[1], (const float*)out[2], (const float*)out[3]};
        double Z = 0;
#pragma omp parallel for reduction(+:Z) schedule(static)
        for (int64_t i = 0; i < N; i++) {
            F3 y = Y[i];
            float u = (y.x - x0) * ih, v = (y.y - x0) * ih, s = (y.z - x0) * ih;
            int ix = (int)floorf(u), iy = (int)floorf(v), iz = (int)floorf(s);
            float wx[4], wy[4], wz[4];
            lagrange4(u - ix, wx); lagrange4(v - iy, wy); lagrange4(s - iz, wz);
            size_t b = ((size_t)(iz - 1) * P + (iy - 1)) * Ps + (ix - 1);
            float r4[4] = {0, 0, 0, 0};
            for (int c = 0; c < 4; c++)
                for (int rr = 0; rr < 4; rr++) {
                    float wzy = wz[c] * wy[rr];
                    const size_t o = b + ((size_t)c * P + rr) * Ps;
                    for (int q = 0; q < 4; q++) {
                        float w = wzy * wx[q];
                        for (int k = 0; k < 4; k++) r4[k] += w * fo[k][o + q];
                    }
                }
            float m = hasm ? mass[i] : 1.f;
            Z += (double)m * (r4[0] - m);
            Gp[i] = {r4[1], r4[2], r4[3]};
        }
        double t3 = now();
        // attraction over out- and in-edges
#pragma omp parallel for schedule(dynamic, 4096)
        for (int64_t i = 0; i < N; i++) {
            F3 yi = Y[i]; float ax = 0, ay = 0, az = 0;
            const int* nb = &I[i * K]; const float* pw = &Pw[i * K];
            for (int j = 0; j < K; j++) {
                float p = pw[j]; if (p == 0.f) continue;
                F3 q = Y[nb[j]]; float dx = yi.x - q.x, dy = yi.y - q.y, dz = yi.z - q.z, f = p / (1.f + dx * dx + dy * dy + dz * dz);
                ax += f * dx; ay += f * dy; az += f * dz;
            }
            for (int64_t e = roff[i]; e < roff[i + 1]; e++) {
                F3 q = Y[rsrc[e]]; float dx = yi.x - q.x, dy = yi.y - q.y, dz = yi.z - q.z, f = rp[e] / (1.f + dx * dx + dy * dy + dz * dz);
                ax += f * dx; ay += f * dy; az += f * dz;
            }
            A[i] = {ax, ay, az};
        }
        if (an.nr) {
            // group centroids (per-thread sums), then a spring toward each re-normalised centroid, added to the
            // attraction (so it is exaggerated with it, as on the GPU)
            const size_t T3 = (size_t)an.total * 3;
#pragma omp parallel
            {
                double* c = cacc.data() + T3 * omp_get_thread_num();
                std::fill(c, c + T3, 0.0);
#pragma omp for schedule(static)
                for (int64_t i = 0; i < N; i++)
                    for (int r = 0; r < an.nr; r++) {
                        int gr = an.lab[(size_t)r * N + i];
                        if (gr < 0) continue;
                        double* cc = c + 3 * (size_t)(an.offs[r] + gr);
                        cc[0] += Y[i].x; cc[1] += Y[i].y; cc[2] += Y[i].z;
                    }
#pragma omp for schedule(static)
                for (size_t t = 0; t < T3; t++) { double sum = 0; for (int th = 0; th < nthr; th++) sum += cacc[T3 * th + t]; cent[t] = (float)sum; }
#pragma omp for schedule(static)
                for (int64_t i = 0; i < N; i++) {
                    F3 y = Y[i], f = A[i];
                    for (int r = 0; r < an.nr; r++) {
                        int gr = an.lab[(size_t)r * N + i];
                        if (gr < 0) continue;
                        const float* cc = &cent[3 * (size_t)(an.offs[r] + gr)];
                        float l = sqrtf(cc[0] * cc[0] + cc[1] * cc[1] + cc[2] * cc[2]);
                        if (!(l > 0)) continue;
                        float s = R / l, wr = an.w[r] * (1.f / (float)N);
                        f.x += wr * (y.x - cc[0] * s); f.y += wr * (y.y - cc[1] * s); f.z += wr * (y.z - cc[2] * s);
                    }
                    A[i] = f;
                }
            }
        }
        double t4 = now();
        if (o.check_grad > 0 && (it == o.ee_iters - 1 || it == total - 1)) {
            int nq = (int)std::min<int64_t>(o.check_grad, N);
            std::mt19937 r2(7); std::vector<int> qs(nq); for (auto& q : qs) q = (int)(r2() % N);
            double num = 0, den = 0, zsum = 0;
            for (int q : qs) {
                double z = 0, fx = 0, fy = 0, fz = 0;
                const F3 yq = Y[q];
#pragma omp parallel for reduction(+:z,fx,fy,fz) schedule(static)
                for (int64_t j = 0; j < N; j++) {
                    if (j == q) continue;
                    float dx = yq.x - Y[j].x, dy = yq.y - Y[j].y, dz = yq.z - Y[j].z;
                    float w = 1.f / (1.f + dx * dx + dy * dy + dz * dz), m = hasm ? mass[j] : 1.f;
                    z += m * w; fx += m * w * w * dx; fy += m * w * w * dy; fz += m * w * w * dz;
                }
                double a[3] = {-0.5 * Gp[q].x, -0.5 * Gp[q].y, -0.5 * Gp[q].z}, ex[3] = {fx, fy, fz};
                for (int c = 0; c < 3; c++) { num += (a[c] - ex[c]) * (a[c] - ex[c]); den += ex[c] * ex[c]; }
                zsum += (hasm ? mass[q] : 1.f) * z;
            }
            LOG("grad check it %d: rel. error of repulsive force %.4f, Z grid %.4g vs sampled exact %.4g", it, sqrt(num / den), Z, zsum / nq * N);
        }
        double tu = now();   // the check is left out of the timings
        // tangent-plane step; the point goes back to radius R and the velocity is made tangent again
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) {
            float m = hasm ? mass[i] : 1.f, rs = (float)(0.5 * m / Z);
            F3 a = A[i], gp = Gp[i], y = Y[i], u = U[i], gn = Gn[i];
            float gv[3] = {exag * a.x + rs * gp.x, exag * a.y + rs * gp.y, exag * a.z + rs * gp.z};
            float n0[3] = {y.x / R, y.y / R, y.z / R};
            float gd = gv[0] * n0[0] + gv[1] * n0[1] + gv[2] * n0[2];
            float* uu = &u.x; float* gg = &gn.x; float* yy = &y.x;
            for (int c = 0; c < 3; c++) {
                float gc = gv[c] - gd * n0[c];
                gg[c] = (std::signbit(gc) != std::signbit(uu[c])) ? gg[c] + 0.2f : std::max(gg[c] * 0.8f, 0.01f);
                uu[c] = mom * uu[c] - lr * gg[c] * gc;
                yy[c] += uu[c];
            }
            float s = R / sqrtf(y.x * y.x + y.y * y.y + y.z * y.z);
            y.x *= s; y.y *= s; y.z *= s;
            float ud = (u.x * y.x + u.y * y.y + u.z * y.z) / (R * R);
            u.x -= ud * y.x; u.y -= ud * y.y; u.z -= ud * y.z;
            Y[i] = y; U[i] = u; Gn[i] = gn;
        }
        if (it == o.ee_iters - 1) std::fill(U.begin(), U.end(), F3{0, 0, 0});
        double t5 = now();
        t_spl += t1 - t0; t_fft += t2 - t1; t_gat += t3 - t2; t_att += t4 - t3; t_upd += t5 - tu;
        if (it % 50 == 0 || it == total - 1)
            LOG("iter %4d  Z/N %.4g  (%.0f ms/iter: splat %.0f, fft %.0f, gather %.0f, attraction%s %.0f)", it, Z / N,
                1e3 * (t5 - tu + t4 - t0), 1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2), an.nr ? " + anchors" : "", 1e3 * (t4 - t3));
    }
    LOG("optimisation time: splat %.1fs, FFT %.1fs, gather %.1fs, attraction%s %.1fs, update %.1fs", t_spl, t_fft, t_gat,
        an.nr ? " + anchors" : "", t_att, t_upd);
    F.release(); fftwf_destroy_plan(r2cf);
    fftwf_free(grid); for (auto a : out) fftwf_free(a); fftwf_free(ksp);
}

// ---------------------------------------------------------------------------------------------
// Free 3D embedding (--dims 3): the sphere's 3D FFT field (cubic Lagrange splat / gather, pruned in-place
// transforms, real / imaginary kernel spectra, 28 bytes per padded cell) on a plain 3D map. The padded grid P^3 is
// fixed for the run (planned once); each iteration centres it on the points' bounding box, and once the map
// outgrows it the spacing h grows with the extent in 1.02 steps (as in 2D), the kernel spectra being recomputed
// only when h changes. While the points still sit in a few cells (the first iterations), each thread splats into
// its own copy of the occupied box instead of contending on atomics.
// ---------------------------------------------------------------------------------------------
static void run_tsne_3d_cpu(std::vector<F3>& Y, int64_t N, int K, const std::vector<int>& I, const std::vector<float>& Pw,
                            const std::vector<float>& mass, const Opts& o) {
    std::vector<int64_t> roff; std::vector<int> rsrc; std::vector<float> rp;
    build_reverse(N, K, I, Pw, roff, rsrc, rp);
    const bool hasm = !mass.empty();
    const int nthr = omp_get_max_threads();
    const int G = fft_size_below(std::max(32, o.max_grid / 2));
    const int P = 2 * G, Pc = P / 2 + 1, Ps = 2 * Pc;
    const size_t PC = (size_t)P * P * Pc;
    fftwf_complex* grid = fftwf_alloc_complex(PC);
    fftwf_complex* out[4];
    for (auto& a : out) a = fftwf_alloc_complex(PC);
    float* ksp = fftwf_alloc_real(4 * PC);
    if (!grid || !out[0] || !out[1] || !out[2] || !out[3] || !ksp) { fprintf(stderr, "cannot allocate the %d^3 FFT grid\n", P); exit(1); }
    fftwf_plan_with_nthreads(nthr);
    double tp = now();
    PrunedFFT3 F;
    F.plan(P, G, grid, out[0], FFTW_MEASURE);
    fftwf_plan r2cf = fftwf_plan_dft_r2c_3d(P, P, P, (float*)grid, grid, FFTW_ESTIMATE);
    LOG("3D: grid %d^3 (padded %d^3), %.1f GB; FFT plans %.1fs", G, P, PC * 56 / 1e9, now() - tp);
    auto zero = [&](fftwf_complex* a) {
#pragma omp parallel for schedule(static)
        for (int l = 0; l < P; l++) memset(a + (size_t)l * P * Pc, 0, (size_t)P * Pc * 8);
    };

    std::vector<F3> U(N, {0, 0, 0}), Gn(N, {1, 1, 1}), A(N), Gp(N);
    std::vector<float> priv;
    const float lr = o.lr > 0 ? o.lr : std::max(200.f, (float)N / o.ee);
    const int total = o.ee_iters + o.iters;
    float hcur = 0;
    int nkern = 0;
    double t_spl = 0, t_fft = 0, t_gat = 0, t_att = 0, t_upd = 0, t_ker = 0;
    for (int it = 0; it < total; it++) {
        bool ee = it < o.ee_iters;
        float exag = ee ? o.ee : o.exag, mom = ee ? 0.5f : 0.8f;
        double t0 = now();
        float xmn = 1e30f, ymn = 1e30f, zmn = 1e30f, xmx = -1e30f, ymx = -1e30f, zmx = -1e30f;
#pragma omp parallel for reduction(min:xmn,ymn,zmn) reduction(max:xmx,ymx,zmx) schedule(static)
        for (int64_t i = 0; i < N; i++) {
            xmn = std::min(xmn, Y[i].x); xmx = std::max(xmx, Y[i].x); ymn = std::min(ymn, Y[i].y); ymx = std::max(ymx, Y[i].y);
            zmn = std::min(zmn, Y[i].z); zmx = std::max(zmx, Y[i].z);
        }
        float ext = std::max(xmx - xmn, std::max(ymx - ymn, zmx - zmn));
        // spacing: o.h while the map fits, else extent / (G - 6) rounded up to a power of 1.02 times o.h; kept when
        // the map shrinks by less than 10% of h, so the kernels are not recomputed back and forth
        float h = o.h;
        if (ext / o.h + 6 > G) h = o.h * powf(1.02f, ceilf(logf(ext / (G - 6) / o.h) / logf(1.02f)));
        if (hcur > 0 && h < hcur && h > hcur / 1.1f && ext / hcur + 6 <= G) h = hcur;
        if (h != hcur) {
            double tk = now();
            kernel_spectra3(P, h, grid, r2cf, ksp);
            hcur = h; nkern++;
            t_ker += now() - tk;
            t0 += now() - tk;   // kernel time is logged separately
        }
        const float ih = 1.f / h;
        const float x0 = 0.5f * (xmn + xmx) - 0.5f * G * h, y0 = 0.5f * (ymn + ymx) - 0.5f * G * h, z0 = 0.5f * (zmn + zmx) - 0.5f * G * h;
        zero(grid);
        float* g = (float*)grid;
        // occupied box of cells (stencil -1 .. +2 around each point's cell)
        const int bx0 = (int)floorf((xmn - x0) * ih) - 1, by0 = (int)floorf((ymn - y0) * ih) - 1, bz0 = (int)floorf((zmn - z0) * ih) - 1;
        const int BX = (int)floorf((xmx - x0) * ih) + 3 - bx0, BY = (int)floorf((ymx - y0) * ih) + 3 - by0, BZ = (int)floorf((zmx - z0) * ih) + 3 - bz0;
        const size_t BV = (size_t)BX * BY * BZ;
        if (BV * nthr <= ((size_t)64 << 20)) {
            priv.resize(BV * nthr);
#pragma omp parallel
            {
                float* pg = priv.data() + BV * omp_get_thread_num();
                memset(pg, 0, BV * 4);
#pragma omp for schedule(static)
                for (int64_t i = 0; i < N; i++) {
                    F3 y = Y[i];
                    float u = (y.x - x0) * ih, v = (y.y - y0) * ih, s = (y.z - z0) * ih;
                    int ix = (int)floorf(u), iy = (int)floorf(v), iz = (int)floorf(s);
                    float wx[4], wy[4], wz[4];
                    lagrange4(u - ix, wx); lagrange4(v - iy, wy); lagrange4(s - iz, wz);
                    float m = hasm ? mass[i] : 1.f;
                    size_t b = ((size_t)(iz - 1 - bz0) * BY + (iy - 1 - by0)) * BX + (ix - 1 - bx0);
                    for (int c = 0; c < 4; c++)
                        for (int r = 0; r < 4; r++) {
                            float wzy = m * wz[c] * wy[r];
                            float* row = pg + b + ((size_t)c * BY + r) * BX;
                            for (int q = 0; q < 4; q++) row[q] += wzy * wx[q];
                        }
                }
#pragma omp for schedule(static)
                for (size_t c = 0; c < BV; c++) {
                    float sum = 0;
                    for (int t = 0; t < nthr; t++) sum += priv[BV * t + c];
                    size_t qx = c % BX, qy = (c / BX) % BY, qz = c / ((size_t)BX * BY);
                    g[((qz + bz0) * P + (qy + by0)) * Ps + qx + bx0] = sum;
                }
            }
        } else {
            if (!priv.empty()) std::vector<float>().swap(priv);
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < N; i++) {
                F3 y = Y[i];
                float u = (y.x - x0) * ih, v = (y.y - y0) * ih, s = (y.z - z0) * ih;
                int ix = (int)floorf(u), iy = (int)floorf(v), iz = (int)floorf(s);
                float wx[4], wy[4], wz[4];
                lagrange4(u - ix, wx); lagrange4(v - iy, wy); lagrange4(s - iz, wz);
                float m = hasm ? mass[i] : 1.f;
                size_t b = ((size_t)(iz - 1) * P + (iy - 1)) * Ps + (ix - 1);
                for (int c = 0; c < 4; c++)
                    for (int r = 0; r < 4; r++) {
                        float wzy = m * wz[c] * wy[r];
                        float* row = g + b + ((size_t)c * P + r) * Ps;
                        for (int q = 0; q < 4; q++) atomic_addf(row + q, wzy * wx[q]);
                    }
            }
        }
        double t1 = now();
        F.forward(grid);
#pragma omp parallel for schedule(static)
        for (size_t t = 0; t < PC; t++) {
            float sr = grid[t][0], si = grid[t][1], k0 = ksp[t];
            out[0][t][0] = sr * k0; out[0][t][1] = si * k0;
            for (int k = 1; k < 4; k++) { float kk = ksp[k * PC + t]; out[k][t][0] = -si * kk; out[k][t][1] = sr * kk; }
        }
        for (int k = 0; k < 4; k++) F.inverse(out[k]);
        double t2 = now();
        // gather phi and grad phi; Z = sum_i m_i (phi_i - m_i), the self-interaction taken as the exact w(0) = 1
        const float* fo[4] = {(const float*)out[0], (const float*)out[1], (const float*)out[2], (const float*)out[3]};
        double Z = 0;
#pragma omp parallel for reduction(+:Z) schedule(static)
        for (int64_t i = 0; i < N; i++) {
            F3 y = Y[i];
            float u = (y.x - x0) * ih, v = (y.y - y0) * ih, s = (y.z - z0) * ih;
            int ix = (int)floorf(u), iy = (int)floorf(v), iz = (int)floorf(s);
            float wx[4], wy[4], wz[4];
            lagrange4(u - ix, wx); lagrange4(v - iy, wy); lagrange4(s - iz, wz);
            size_t b = ((size_t)(iz - 1) * P + (iy - 1)) * Ps + (ix - 1);
            float r4[4] = {0, 0, 0, 0};
            for (int c = 0; c < 4; c++)
                for (int rr = 0; rr < 4; rr++) {
                    float wzy = wz[c] * wy[rr];
                    const size_t o = b + ((size_t)c * P + rr) * Ps;
                    for (int q = 0; q < 4; q++) {
                        float w = wzy * wx[q];
                        for (int k = 0; k < 4; k++) r4[k] += w * fo[k][o + q];
                    }
                }
            float m = hasm ? mass[i] : 1.f;
            Z += (double)m * (r4[0] - m);
            Gp[i] = {r4[1], r4[2], r4[3]};
        }
        double t3 = now();
#pragma omp parallel for schedule(dynamic, 4096)
        for (int64_t i = 0; i < N; i++) {
            F3 yi = Y[i]; float ax = 0, ay = 0, az = 0;
            const int* nb = &I[i * K]; const float* pw = &Pw[i * K];
            for (int j = 0; j < K; j++) {
                float p = pw[j]; if (p == 0.f) continue;
                F3 q = Y[nb[j]]; float dx = yi.x - q.x, dy = yi.y - q.y, dz = yi.z - q.z, f = p / (1.f + dx * dx + dy * dy + dz * dz);
                ax += f * dx; ay += f * dy; az += f * dz;
            }
            for (int64_t e = roff[i]; e < roff[i + 1]; e++) {
                F3 q = Y[rsrc[e]]; float dx = yi.x - q.x, dy = yi.y - q.y, dz = yi.z - q.z, f = rp[e] / (1.f + dx * dx + dy * dy + dz * dz);
                ax += f * dx; ay += f * dy; az += f * dz;
            }
            A[i] = {ax, ay, az};
        }
        double t4 = now();
        if (o.check_grad > 0 && (it == o.ee_iters - 1 || it == total - 1)) {
            int nq = (int)std::min<int64_t>(o.check_grad, N);
            std::mt19937 r2(7); std::vector<int> qs(nq); for (auto& q : qs) q = (int)(r2() % N);
            double num = 0, den = 0, zsum = 0;
            for (int q : qs) {
                double z = 0, fx = 0, fy = 0, fz = 0;
                const F3 yq = Y[q];
#pragma omp parallel for reduction(+:z,fx,fy,fz) schedule(static)
                for (int64_t j = 0; j < N; j++) {
                    if (j == q) continue;
                    float dx = yq.x - Y[j].x, dy = yq.y - Y[j].y, dz = yq.z - Y[j].z;
                    float w = 1.f / (1.f + dx * dx + dy * dy + dz * dz), m = hasm ? mass[j] : 1.f;
                    z += m * w; fx += m * w * w * dx; fy += m * w * w * dy; fz += m * w * w * dz;
                }
                double a[3] = {-0.5 * Gp[q].x, -0.5 * Gp[q].y, -0.5 * Gp[q].z}, ex[3] = {fx, fy, fz};
                for (int c = 0; c < 3; c++) { num += (a[c] - ex[c]) * (a[c] - ex[c]); den += ex[c] * ex[c]; }
                zsum += (hasm ? mass[q] : 1.f) * z;
            }
            LOG("grad check it %d: rel. error of repulsive force %.4f, Z grid %.4g vs sampled exact %.4g (h=%.3f, P=%d)",
                it, sqrt(num / den), Z, zsum / nq * N, h, P);
        }
        double tu = now();   // the check is left out of the timings
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) {
            float m = hasm ? mass[i] : 1.f, rs = (float)(0.5 * m / Z);
            F3 a = A[i], gp = Gp[i], u = U[i], gn = Gn[i];
            float gv[3] = {exag * a.x + rs * gp.x, exag * a.y + rs * gp.y, exag * a.z + rs * gp.z};
            float* uu = &u.x; float* gg = &gn.x; float* yy = &Y[i].x;
            for (int c = 0; c < 3; c++) {
                gg[c] = (std::signbit(gv[c]) != std::signbit(uu[c])) ? gg[c] + 0.2f : std::max(gg[c] * 0.8f, 0.01f);
                uu[c] = mom * uu[c] - lr * gg[c] * gv[c];
                yy[c] += uu[c];
            }
            U[i] = u; Gn[i] = gn;
        }
        if (it == o.ee_iters - 1) std::fill(U.begin(), U.end(), F3{0, 0, 0});
        double t5 = now();
        t_spl += t1 - t0; t_fft += t2 - t1; t_gat += t3 - t2; t_att += t4 - t3; t_upd += t5 - tu;
        if (it % 100 == 0 || it == total - 1)
            LOG("iter %4d  extent %8.2f  h %.3f  Z/N %.4g  (%.0f ms/iter: splat %.0f, fft %.0f, gather %.0f, attraction %.0f)", it, ext, h, Z / N,
                1e3 * (t5 - tu + t4 - t0), 1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2), 1e3 * (t4 - t3));
    }
    LOG("optimisation time: splat %.1fs, FFT %.1fs, gather %.1fs, attraction %.1fs, update %.1fs; kernels %d times, %.1fs",
        t_spl, t_fft, t_gat, t_att, t_upd, nkern, t_ker);
    F.release(); fftwf_destroy_plan(r2cf);
    fftwf_free(grid); for (auto a : out) fftwf_free(a); fftwf_free(ksp);
}

// ---------------------------------------------------------------------------------------------
// Label propagation on the kNN graph (weights p_{j|i}), as the GPU's label_propagation: synchronous sweeps in
// which a random half of the nodes keep their label (avoids oscillation)
// ---------------------------------------------------------------------------------------------
static std::vector<int> label_propagation_cpu(const std::vector<int>& I, const std::vector<float>& Pw, int64_t N, int K, int sweeps, uint64_t seed) {
    std::vector<int> a(N), b(N);
    std::iota(a.begin(), a.end(), 0);
    for (int s = 0; s < sweeps; s++) {
        uint64_t sd = splitmix(seed + 0x1234 + s);
        int64_t ch = 0;
#pragma omp parallel reduction(+:ch)
        {
            std::vector<int> L(K); std::vector<float> W(K);
#pragma omp for schedule(dynamic, 4096)
            for (int64_t i = 0; i < N; i++) {
                int cur = a[i];
                if (splitmix(sd ^ (uint64_t)i) & 1) { b[i] = cur; continue; }
                int m = 0;
                for (int j = 0; j < K; j++) {
                    float w = Pw[i * K + j];
                    if (w == 0.f) continue;
                    int l = a[I[i * K + j]];
                    int t = 0; while (t < m && L[t] != l) t++;
                    if (t == m) { L[m] = l; W[m] = w; m++; } else W[t] += w;
                }
                int best = cur; float bw = -1;
                for (int t = 0; t < m; t++) if (W[t] > bw || (W[t] == bw && L[t] == cur)) { bw = W[t]; best = L[t]; }
                b[i] = best;
                ch += best != cur;
            }
        }
        std::swap(a, b);
        if (ch < N / 10000) break;
    }
    return a;
}

// ---------------------------------------------------------------------------------------------
// Saved model + projection of new points
// ---------------------------------------------------------------------------------------------
// Same placement as the GPU's k_place: the minimum of the t-SNE attraction sum_j w_j log(1 + |y - y_j|^2) over the
// reference neighbours (the weighted mean would put points whose neighbours sit in two clusters between them).
// Start from the best neighbour position, then fixed-point steps y = sum_j u_j y_j / sum_j u_j, u_j = w_j / (1 + d_j^2).
static F2 place_point(const int* nb, const float* w, int K, const float* Yr) {
    float best = INFINITY; F2 y = {NAN, NAN};
    for (int a = 0; a < K; a++) {
        if (!(w[a] > 0)) continue;
        float cx = Yr[2 * (size_t)nb[a]], cy = Yr[2 * (size_t)nb[a] + 1], obj = 0;
        for (int j = 0; j < K; j++) {
            if (!(w[j] > 0)) continue;
            float dx = cx - Yr[2 * (size_t)nb[j]], dy = cy - Yr[2 * (size_t)nb[j] + 1];
            obj += w[j] * log1pf(dx * dx + dy * dy);
        }
        if (obj < best) { best = obj; y = {cx, cy}; }
    }
    if (best == INFINITY) return y;
    for (int it = 0; it < 20; it++) {
        float sx = 0, sy = 0, su = 0;
        for (int j = 0; j < K; j++) {
            if (!(w[j] > 0)) continue;
            float rx = Yr[2 * (size_t)nb[j]], ry = Yr[2 * (size_t)nb[j] + 1], dx = y.x - rx, dy = y.y - ry, u = w[j] / (1.f + dx * dx + dy * dy);
            sx += u * rx; sy += u * ry; su += u;
        }
        y = {sx / su, sy / su};
    }
    return y;
}

// the same on a sphere of radius R (Y3: reference points as x, y, z): chord distances, each fixed-point step put
// back on the sphere; returns lon, lat in degrees
static F2 place_point_sphere(const int* nb, const float* w, int K, const float* Y3, float R) {
    float best = INFINITY, y[3] = {NAN, NAN, NAN};
    auto d2 = [&](const float* p, const float* q) { float a = p[0] - q[0], b = p[1] - q[1], c = p[2] - q[2]; return a * a + b * b + c * c; };
    for (int a = 0; a < K; a++) {
        if (!(w[a] > 0)) continue;
        const float* c = Y3 + 3 * (size_t)nb[a];
        float obj = 0;
        for (int j = 0; j < K; j++) if (w[j] > 0) obj += w[j] * log1pf(d2(c, Y3 + 3 * (size_t)nb[j]));
        if (obj < best) { best = obj; y[0] = c[0]; y[1] = c[1]; y[2] = c[2]; }
    }
    if (best == INFINITY) return {NAN, NAN};
    for (int it = 0; it < 20; it++) {
        float s[3] = {0, 0, 0};
        for (int j = 0; j < K; j++) {
            if (!(w[j] > 0)) continue;
            const float* r = Y3 + 3 * (size_t)nb[j];
            float u = w[j] / (1.f + d2(y, r));
            s[0] += u * r[0]; s[1] += u * r[1]; s[2] += u * r[2];
        }
        float l = R / sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        y[0] = s[0] * l; y[1] = s[1] * l; y[2] = s[2] * l;
    }
    return {atan2f(y[1], y[0]) * 57.29578f, asinf(fmaxf(-1.f, fminf(1.f, y[2] / R))) * 57.29578f};
}

// Streams the inputs in batches: same transform + PCA as the reference, nearest reference cluster, kNN against the
// reference's inverted file, then each point goes to the attraction minimum over its neighbours' positions.
// Reads models written by either build.
static void run_project_cpu(const Opts& o) {
    Model m;
    load_model(o.project, m);
    LOG("model: %lld reference points, %d -> %d dims, %d clusters", (long long)m.N, m.D, m.DP, m.nc);
    const int K = o.proj_k, DP = m.DP, nc = m.nc;
    IVF V;
    ivf_build(V, m.X.data(), m.offs, m.C, nc, DP, o.nprobe);
    std::vector<float>().swap(m.X);
    // the centroids as a one-cluster index, for exact nearest-centroid search with the same tiled scan
    IVF Vc;
    { std::vector<int> o1 = {0, nc}; std::vector<float> c0(DP, 0.f); ivf_build(Vc, m.C.data(), o1, c0, 1, DP, 1); }
    LOG("reference index built (%d probes)", V.np);
    std::vector<float> Y3;
    if (m.R > 0) {  // spherical model: lon, lat -> points on the sphere
        Y3.resize((size_t)m.N * 3);
        for (int64_t i = 0; i < m.N; i++) {
            float lo = m.Y[2 * i] / 57.29578f, la = m.Y[2 * i + 1] / 57.29578f;
            Y3[3 * i] = m.R * cosf(la) * cosf(lo); Y3[3 * i + 1] = m.R * cosf(la) * sinf(lo); Y3[3 * i + 2] = m.R * sinf(la);
        }
        LOG("spherical model, radius %.1f", m.R);
    }

    std::vector<std::string> files = expand_inputs(o);
    std::vector<std::vector<std::string>> batches(1);
    double acc = 0;
    for (auto& fn : files) {
        struct stat st; if (stat(fn.c_str(), &st) != 0) { fprintf(stderr, "cannot stat %s\n", fn.c_str()); exit(1); }
        int64_t mb = input_mem_bytes(fn, st.st_size);
        if (acc > 0 && acc + mb > o.batch_gb * 1e9) { batches.emplace_back(); acc = 0; }
        batches.back().push_back(fn); acc += mb;
    }
    FILE* f = fopen(o.out.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", o.out.c_str()); exit(1); }
    std::vector<char> buf(1 << 20);
    setvbuf(f, buf.data(), _IOFBF, buf.size());
    fputs(m.R > 0 ? "id\tlon\tlat\tnn_dist\n" : "id\tx\ty\tnn_dist\n", f);
    int64_t total = 0;
    double t0 = now();
    for (size_t b = 0; b < batches.size(); b++) {
        Data d;
        read_files(batches[b], o, d);
        if (d.D != m.D) { fprintf(stderr, "input has %d columns, model expects %d\n", d.D, m.D); exit(1); }
        const int64_t n = d.N;
        transform_all(d, m.mode);
        std::vector<float> Q = pca_project(d, m.mu, m.Vp, DP);
        std::vector<std::vector<float>>().swap(d.chunks);
        double tb = now();
        // group queries by nearest reference cluster: exact, as a K = 1 tiled scan over all centroids at once
        std::vector<int> a(n);
        {
            std::vector<float> dmin(n);
            std::vector<int64_t> all = {0, n};
            ivf_scan(Vc, Q.data(), all, 1, false, a.data(), dmin.data());
        }
        double tc = now();
        std::vector<int64_t> qoffs(nc + 1, 0);
        for (int64_t i = 0; i < n; i++) qoffs[a[i] + 1]++;
        for (int c = 0; c < nc; c++) qoffs[c + 1] += qoffs[c];
        std::vector<int64_t> pm(n);
        { std::vector<int64_t> cur(qoffs.begin(), qoffs.end() - 1); for (int64_t i = 0; i < n; i++) pm[cur[a[i]]++] = i; }
        std::vector<int>().swap(a);
        std::vector<float> Qs((size_t)n * DP);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; i++) memcpy(&Qs[(size_t)i * DP], &Q[(size_t)pm[i] * DP], DP * 4);
        std::vector<float>().swap(Q);
        std::vector<int> I((size_t)n * K); std::vector<float> Dd((size_t)n * K);
        ivf_scan(V, Qs.data(), qoffs, K, false, I.data(), Dd.data());
        std::vector<float>().swap(Qs);
        double td = now();
        std::vector<F2> Yo(n); std::vector<float> nno(n);
#pragma omp parallel for schedule(dynamic, 4096)
        for (int64_t i = 0; i < n; i++) {
            float* dd = &Dd[i * K];
            float nn = sqrtf(dd[0]);
            perplexity_row(dd, K, logf(o.proj_perp), false, 1.f);
            Yo[pm[i]] = m.R > 0 ? place_point_sphere(&I[i * K], dd, K, Y3.data(), m.R) : place_point(&I[i * K], dd, K, m.Y.data());
            nno[pm[i]] = nn;
        }
        for (int64_t i = 0; i < n; i++) {
            fwrite(d.ids.data() + d.idoff[i], 1, d.idoff[i + 1] - d.idoff[i], f);
            fprintf(f, "\t%.4f\t%.4f\t%.4g\n", Yo[i].x, Yo[i].y, nno[i]);
        }
        total += n;
        LOG("batch %zu/%zu: %lld points (%lld total, %.0f points/s; nearest cluster %.1fs, kNN %.1fs, place + write %.1fs)",
            b + 1, batches.size(), (long long)n, (long long)total, total / (now() - t0), tc - tb, td - tc, now() - td);
    }
    fclose(f);
    LOG("wrote %s", o.out.c_str());
}

int main(int argc, char** argv) {
    // OpenMP threads that spin at barriers collapse when other jobs share the CPU (a run that takes a minute
    // idle took >13 min with 6 of 16 hardware threads busy). libgomp reads OMP_WAIT_POLICY when it loads, so
    // when it is unset, set it to passive and re-execute (OMP_WAIT_POLICY=active restores spinning).
    if (!getenv("OMP_WAIT_POLICY")) {
        setenv("OMP_WAIT_POLICY", "passive", 1);
        execv("/proc/self/exe", argv);   // only returns on failure: carry on with the default policy
    }
    Opts o;
    parse_args(argc, argv, o);
    if (o.threads > 0) omp_set_num_threads(o.threads);
    if (o.nprobe < 0) o.nprobe = 32;
    if (o.refine < 0) o.refine = 1;
    if (o.cluster_size < 0) o.cluster_size = 1000;
    LOG("fastsne_cpu: %d threads", omp_get_max_threads());
    if (!o.project.empty()) { run_project_cpu(o); return 0; }
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());

    Data d;
    read_files(expand_inputs(o), o, d);
    if (!o.write_fsb.empty()) { write_fsb_file(d, o.write_fsb); return 0; }
    const int64_t N = d.N;
    int DP = std::max(8, (std::min(o.pca, d.D) + 7) / 8 * 8);
    int mode = o.transform == "clr" ? 0 : o.transform == "sqrt" ? 1 : 2;
    Model model;
    std::vector<float> Yp = transform_pca_cpu(d, o, DP, mode, model.mu, model.Vp);
    std::vector<std::vector<float>>().swap(d.chunks);
    LOG("transform + PCA done (%d dims)", DP);

    const int K = o.k;
    KnnOut R;
    knn_cpu(Yp, N, DP, o, K, R);
    model.C = R.C; model.offs = R.offs;

    // R.I / R.Dd: neighbours nearest first, squared distances
    if (!o.knn_out.empty()) {
        std::vector<int> Io((size_t)N * K);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) for (int j = 0; j < K; j++) Io[(size_t)R.perm[i] * K + j] = R.perm[R.I[(size_t)i * K + j]];
        FILE* g = fopen(o.knn_out.c_str(), "wb");
        if (!g) { fprintf(stderr, "cannot write %s\n", o.knn_out.c_str()); exit(1); }
        wr(g, Io.data(), Io.size() * 4); fclose(g);
        LOG("wrote kNN graph %s (%lld x %d)", o.knn_out.c_str(), (long long)N, K);
    }
    std::vector<float> rk, r1, idim;
    if (!o.diag.empty()) {  // kNN radii and MLE intrinsic dimension from the K distances
        rk.resize(N); r1.resize(N); idim.resize(N);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) {
            const float* d2 = &R.Dd[i * K];
            rk[i] = d2[K - 1]; r1[i] = d2[0];
            float lk = 0.5f * logf(std::max(d2[K - 1], 1e-20f)), s = 0;
            for (int j = 0; j < K - 1; j++) s += lk - 0.5f * logf(std::max(d2[j], 1e-20f));
            idim[i] = s > 0 ? (K - 2) / s : 100.f;
        }
    }

    // affinities, then rows re-sorted by neighbour index (memory order for the attraction loop, sorted lists
    // for the overlap measure)
    std::vector<float>& Pw = R.Dd;
    const bool need_share = o.dedup > 0 || !o.diag.empty();
    std::vector<float> share(need_share ? N : 0);
#pragma omp parallel
    {
        std::vector<std::pair<int, float>> row(K);
#pragma omp for schedule(dynamic, 4096)
        for (int64_t i = 0; i < N; i++) {
            perplexity_row(&Pw[i * K], K, logf(o.perplexity), o.uniform, 0.5f / (float)N);
            for (int j = 0; j < K; j++) row[j] = {R.I[i * K + j], Pw[i * K + j]};
            std::sort(row.begin(), row.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.first < b.first; });
            for (int j = 0; j < K; j++) { R.I[i * K + j] = row[j].first; Pw[i * K + j] = row[j].second; }
        }
    }
    LOG("affinities done (perplexity %.1f%s)", o.perplexity, o.uniform ? ", uniform" : "");

    // masses, in the GPU build's order: --shrink, then --dedup (multiplied in), then --mass-file (replaces both)
    std::vector<float> mass, mass_out;
    std::vector<int> groups;
    if (o.shrink > 0 || !o.groups_out.empty()) {
        groups = label_propagation_cpu(R.I, Pw, N, K, o.lp_sweeps, o.seed);
        std::vector<int> cnt(N, 0);
        for (auto g : groups) cnt[g]++;
        std::vector<int> sizes; for (auto c : cnt) if (c) sizes.push_back(c);
        std::sort(sizes.rbegin(), sizes.rend());
        LOG("label propagation: %zu groups, largest %d %d %d %d %d", sizes.size(), sizes[0], sizes.size() > 1 ? sizes[1] : 0,
            sizes.size() > 2 ? sizes[2] : 0, sizes.size() > 3 ? sizes[3] : 0, sizes.size() > 4 ? sizes[4] : 0);
        if (o.shrink > 0) {
            double ref = o.shrink_ref > 0 ? o.shrink_ref : std::max(100.0, N * 1e-3);
            mass.resize(N);
            double ms = 0;
            for (int64_t i = 0; i < N; i++) {
                double c = cnt[groups[i]];
                mass[i] = (float)std::max<double>(o.mass_min, c > ref ? pow(ref / c, o.shrink) : 1.0);
                ms += mass[i];
            }
            LOG("shrink: groups above %.0f points get mass (ref/n)^%.2f, mean mass %.3f", ref, o.shrink, ms / N);
        }
    }
    if (need_share) {
        // overlap of each point's neighbour set with those of its 8 strongest neighbours (sorted-list intersection)
#pragma omp parallel
        {
            std::vector<std::pair<float, int>> top(K);
#pragma omp for schedule(dynamic, 4096)
            for (int64_t i = 0; i < N; i++) {
                const int* a = &R.I[i * K];
                for (int j = 0; j < K; j++) top[j] = {-Pw[i * K + j], a[j]};
                int na = std::min(8, K);
                std::partial_sort(top.begin(), top.begin() + na, top.end());
                float sh = 0;
                for (int t = 0; t < na; t++) {
                    const int* b = &R.I[(int64_t)top[t].second * K];
                    int x = 0, y = 0, c = 0;
                    while (x < K && y < K) { if (a[x] < b[y]) x++; else if (a[x] > b[y]) y++; else { c++; x++; y++; } }
                    sh += c / (float)K;
                }
                share[i] = sh / na;
            }
        }
    }
    if (o.dedup > 0) {
        std::vector<float> t(share);
        std::nth_element(t.begin(), t.begin() + N / 2, t.end());
        float med = std::max(t[N / 2], 1e-3f);
        std::vector<float> lm(N), lm2(N);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) lm[i] = -o.dedup * std::max(0.f, logf(std::max(share[i], 1e-3f) / med) - o.dedup_margin);
        for (int r = 0; r < 3; r++) {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < N; i++) {
                float s = lm[i], w = 1;
                for (int j = 0; j < K; j++) if (Pw[i * K + j] > 0) { s += lm[R.I[i * K + j]]; w += 1; }
                lm2[i] = s / w;
            }
            std::swap(lm, lm2);
        }
        std::vector<float> m(N);
        double ms = 0; int64_t nsh = 0;
#pragma omp parallel for reduction(+:ms,nsh) schedule(static)
        for (int64_t i = 0; i < N; i++) { m[i] = std::max(o.mass_min, expf(lm[i])); ms += m[i]; nsh += m[i] < 0.5f; }
        if (!mass.empty()) for (int64_t i = 0; i < N; i++) m[i] = std::max(o.mass_min, m[i] * mass[i]);   // with --shrink
        mass = m; mass_out = m;
        LOG("dedup: mean mass %.3f, %.1f%% of points below 0.5", ms / N, 100.0 * nsh / N);
    }
    if (!o.mass_file.empty()) {
        std::vector<float> mi;
        FILE* f = fopen(o.mass_file.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", o.mass_file.c_str()); exit(1); }
        float v; while (fscanf(f, "%f", &v) == 1) mi.push_back(v); fclose(f);
        if ((int64_t)mi.size() != N) { fprintf(stderr, "mass file has %zu values, need %lld\n", mi.size(), (long long)N); exit(1); }
        mass.resize(N);
        for (int64_t i = 0; i < N; i++) mass[i] = mi[R.perm[i]];
    }

    // init: first two PCs scaled to std 1e-4 (plane), the first three so scaled (3D), or the direction of the
    // first three PCs (sphere)
    std::vector<F2> Y(N);
    double sd = 0;
    for (int64_t i = 0; i < N; i++) sd += (double)R.X[(size_t)i * DP] * R.X[(size_t)i * DP];
    sd = sqrt(sd / N);
    float sc = (float)(1e-4 / std::max(sd, 1e-12));
    for (int64_t i = 0; i < N; i++) Y[i] = {R.X[(size_t)i * DP] * sc, R.X[(size_t)i * DP + 1] * sc};
    std::vector<F3> Y3;
    if (o.dims == 3) {
        Y3.resize(N);
        for (int64_t i = 0; i < N; i++) { const float* p = &R.X[(size_t)i * DP]; Y3[i] = {p[0] * sc, p[1] * sc, p[2] * sc}; }
    } else if (o.sphere) {
        if (o.sphere < 0) o.sphere = sqrtf((float)N) / 12.f;
        Y3.resize(N);
        for (int64_t i = 0; i < N; i++) {
            const float* p = &R.X[(size_t)i * DP];
            float x = p[0], y = p[1], z = p[2], r = sqrtf(x * x + y * y + z * z);
            if (!(r > 0)) { x = 1; y = z = 0; r = 1; }
            Y3[i] = {o.sphere * x / r, o.sphere * y / r, o.sphere * z / r};
        }
    }
    if (!o.save_model.empty()) model.X = R.X;
    std::vector<float>().swap(R.X);

    double t_opt = now();
    if (!o.fixed_y.empty()) {
        // positions from an existing map, matched by id (x, y; or lon, lat with --sphere)
        std::unordered_map<std::string, F2> pos;
        FILE* f = fopen(o.fixed_y.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", o.fixed_y.c_str()); exit(1); }
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            char* t1 = strchr(line, '\t'); if (!t1) continue;
            *t1 = 0; float a, b;
            if (sscanf(t1 + 1, "%f\t%f", &a, &b) == 2) pos[line] = {a, b};
        }
        fclose(f);
        int64_t miss = 0;
        for (int64_t i = 0; i < N; i++) {
            int64_t r = R.perm[i];
            auto it = pos.find(std::string(d.ids.data() + d.idoff[r], d.idoff[r + 1] - d.idoff[r]));
            if (it == pos.end()) { miss++; Y[i] = {NAN, NAN}; } else Y[i] = it->second;
        }
        if (miss) { fprintf(stderr, "%lld of %lld input rows have no position in %s\n", (long long)miss, (long long)N, o.fixed_y.c_str()); exit(1); }
        LOG("positions from %s", o.fixed_y.c_str());
    } else if (o.dims == 3) {
        run_tsne_3d_cpu(Y3, N, K, R.I, Pw, mass, o);
    } else if (o.sphere) {
        AnchorsCPU an;
        if (!o.anchor_file.empty()) {
            std::vector<int> v;
            FILE* f = fopen(o.anchor_file.c_str(), "r");
            if (!f) { fprintf(stderr, "cannot open %s\n", o.anchor_file.c_str()); exit(1); }
            int x; while (fscanf(f, "%d", &x) == 1) v.push_back(x); fclose(f);
            if (v.size() % N) { fprintf(stderr, "anchor file has %zu values, not a multiple of %lld rows\n", v.size(), (long long)N); exit(1); }
            an.nr = (int)(v.size() / N);
            if ((int)o.anchor_w.size() != an.nr) { fprintf(stderr, "--anchor needs %d weights (one per rank column)\n", an.nr); exit(1); }
            an.lab.resize((size_t)an.nr * N); an.offs.resize(an.nr); an.w = o.anchor_w;
            for (int r = 0; r < an.nr; r++) {
                int mx = -1;
                for (int64_t i = 0; i < N; i++) { an.lab[(size_t)r * N + i] = v[(size_t)R.perm[i] * an.nr + r]; mx = std::max(mx, an.lab[(size_t)r * N + i]); }
                an.offs[r] = an.total; an.total += mx + 1;
            }
            LOG("anchors: %d ranks, %d groups", an.nr, an.total);
        }
        run_tsne_sphere_cpu(Y3, N, K, R.I, Pw, mass, o, an);
        for (int64_t i = 0; i < N; i++)  // lon, lat in degrees
            Y[i] = {atan2f(Y3[i].y, Y3[i].x) * 57.29578f, asinf(fmaxf(-1.f, fminf(1.f, Y3[i].z / o.sphere))) * 57.29578f};
    } else {
        run_tsne_cpu(Y, N, K, R.I, Pw, mass, o);
    }
    LOG("optimisation done in %.1fs", now() - t_opt);

    if (!o.save_model.empty()) {
        model.N = N; model.D = d.D; model.DP = DP; model.nc = R.nc; model.mode = mode;
        model.R = o.sphere;
        model.Y.resize((size_t)N * 2);
        for (int64_t i = 0; i < N; i++) { model.Y[2 * i] = Y[i].x; model.Y[2 * i + 1] = Y[i].y; }
        save_model(o.save_model, model);
        LOG("saved model %s", o.save_model.c_str());
    }
    std::vector<F2> Yo(o.dims == 3 ? 0 : N);
    std::vector<F3> Yo3(o.dims == 3 ? N : 0);
    for (int64_t i = 0; i < N; i++) if (o.dims == 3) Yo3[R.perm[i]] = Y3[i]; else Yo[R.perm[i]] = Y[i];
    FILE* f = fopen(o.out.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", o.out.c_str()); exit(1); }
    std::vector<char> buf(1 << 20);
    setvbuf(f, buf.data(), _IOFBF, buf.size());
    fputs(o.dims == 3 ? "id\tx\ty\tz\n" : o.sphere ? "id\tlon\tlat\n" : "id\tx\ty\n", f);
    for (int64_t i = 0; i < N; i++) {
        fwrite(d.ids.data() + d.idoff[i], 1, d.idoff[i + 1] - d.idoff[i], f);
        if (o.dims == 3) fprintf(f, "\t%.4f\t%.4f\t%.4f\n", Yo3[i].x, Yo3[i].y, Yo3[i].z);
        else fprintf(f, "\t%.4f\t%.4f\n", Yo[i].x, Yo[i].y);
    }
    fclose(f);
    LOG("wrote %s", o.out.c_str());
    if (!o.diag.empty()) {
        FILE* g = fopen(o.diag.c_str(), "w");
        if (!g) { fprintf(stderr, "cannot write %s\n", o.diag.c_str()); exit(1); }
        std::vector<float> a(N), b(N), mm(N, 1.f), id2(N), sh2(N);
        for (int64_t i = 0; i < N; i++) { int64_t r = R.perm[i]; a[r] = sqrtf(rk[i]); b[r] = sqrtf(r1[i]); id2[r] = idim[i]; sh2[r] = share[i]; }
        if (!mass_out.empty()) for (int64_t i = 0; i < N; i++) mm[R.perm[i]] = mass_out[i];
        fputs("id\trk\tr1\tlen_est\tmass\tidim\tshare\n", g);
        for (int64_t i = 0; i < N; i++) {
            fwrite(d.ids.data() + d.idoff[i], 1, d.idoff[i + 1] - d.idoff[i], g);
            fprintf(g, "\t%.4g\t%.4g\t%.0f\t%.3g\t%.3g\t%.3g\n", a[i], b[i], d.len_est[i], mm[i], id2[i], sh2[i]);
        }
        fclose(g);
    }
    if (!o.groups_out.empty()) {
        FILE* g = fopen(o.groups_out.c_str(), "w");
        if (!g) { fprintf(stderr, "cannot write %s\n", o.groups_out.c_str()); exit(1); }
        std::vector<int> go(N);
        for (int64_t i = 0; i < N; i++) go[R.perm[i]] = R.perm[groups[i]];
        fputs("id\tgroup\n", g);
        for (int64_t i = 0; i < N; i++) {
            fwrite(d.ids.data() + d.idoff[i], 1, d.idoff[i + 1] - d.idoff[i], g);
            fprintf(g, "\tg%d\n", go[i]);
        }
        fclose(g);
        LOG("wrote %s", o.groups_out.c_str());
    }
    return 0;
}
