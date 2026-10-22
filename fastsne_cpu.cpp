// fastsne_cpu: the fastsne pipeline on CPUs (OpenMP + FFTW), same inputs, outputs and model files.
//
//   g++ -O3 -march=native -fopenmp -std=c++17 -o fastsne_cpu fastsne_cpu.cpp -lfftw3f_omp -lfftw3f -lm
//
// Differences from the GPU version, chosen for CPU costs:
//   - PCA covariance from a row sample (--pca-sample, 1M rows)
//   - kNN index: two-level k-means (coarse clusters, then fine clusters inside each) so that assigning
//     points stays cheap; candidates are stored dimension-major per cluster and scanned in 4-query x
//     16-candidate register tiles
//   - repulsion: same single-splat FFT field (FFTW), bilinear splat with atomic adds
//   - neighbour rows are re-sorted by index after the affinities, which speeds up both the attraction
//     loop (memory order) and the neighbour-overlap measure used by --dedup (sorted-list intersection)

#include "common.h"
#include <omp.h>
#include <fftw3.h>

struct F2 { float x, y; };

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

// returns N x DP row-major (cluster order not applied yet); fills mu and Vp (D x DP, column j = component j)
static std::vector<float> transform_pca_cpu(Data& d, const Opts& o, int DP, int mode, std::vector<float>& muf, std::vector<float>& Vp) {
    const int D = d.D; const int64_t N = d.N; const size_t nch = d.chunks.size();
    // transform in place
#pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < nch; c++) {
        int64_t rows = d.cstart[c + 1] - d.cstart[c];
        for (int64_t r = 0; r < rows; r++) transform_row(d.chunks[c].data() + r * D, D, mode);
    }
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
#pragma omp parallel
    {
        std::vector<double> loc((size_t)D * D, 0);
        std::vector<double> xc(D);
        int64_t nl = 0;
#pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < N; i += stride) {
            size_t c = std::upper_bound(d.cstart.begin(), d.cstart.end(), i) - d.cstart.begin() - 1;
            const float* x = d.chunks[c].data() + (i - d.cstart[c]) * D;
            for (int j = 0; j < D; j++) xc[j] = x[j] - mu[j];
            for (int a = 0; a < D; a++) { double va = xc[a]; double* row = loc.data() + (size_t)a * D; for (int b = a; b < D; b++) row[b] += va * xc[b]; }
            nl++;
        }
#pragma omp critical
        { for (size_t j = 0; j < C.size(); j++) C[j] += loc[j]; ns += nl; }
    }
    for (int a = 0; a < D; a++) for (int b = a; b < D; b++) { C[(size_t)a * D + b] /= ns; C[(size_t)b * D + a] = C[(size_t)a * D + b]; }
    double tr = 0; for (int j = 0; j < D; j++) tr += C[(size_t)j * D + j];
    std::vector<double> V, ev;
    jacobi_eigen(C, D, V, ev);
    std::vector<int> ord(D); std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return ev[a] > ev[b]; });
    int npc = std::min(o.pca, D);
    double cum = 0; for (int j = 0; j < npc; j++) cum += ev[ord[j]];
    LOG("PCA (%lld sampled rows): %d comps explain %.1f%% of variance", (long long)ns, npc, 100 * cum / tr);
    Vp.assign((size_t)D * DP, 0.f);
    for (int j = 0; j < npc; j++) for (int i = 0; i < D; i++) Vp[(size_t)j * D + i] = (float)V[(size_t)i * D + ord[j]];
    // project: Y = (X - mu) V, V laid out [k][DP] for a vectorised inner loop
    std::vector<float> Vk((size_t)D * DP);
    for (int j = 0; j < DP; j++) for (int k = 0; k < D; k++) Vk[(size_t)k * DP + j] = Vp[(size_t)j * D + k];
    std::vector<float> Y((size_t)N * DP);
#pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < nch; c++) {
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
    // 4. dimension-major copy of each cluster, padded to a multiple of 16 with far-away points
    std::vector<int64_t> toff(nc + 1, 0);
    for (int k = 0; k < nc; k++) toff[k + 1] = toff[k] + (int64_t)((offs[k + 1] - offs[k] + 15) / 16 * 16) * DP;
    std::vector<float> Xt(toff[nc]);
#pragma omp parallel for schedule(dynamic, 16)
    for (int k = 0; k < nc; k++) {
        int s = offs[k], m = offs[k + 1] - s, mp = (m + 15) / 16 * 16;
        float* b = Xt.data() + toff[k];
        for (int d = 0; d < DP; d++) for (int t = 0; t < mp; t++) b[(size_t)d * mp + t] = t < m ? R.X[(size_t)(s + t) * DP + d] : 1e15f;
    }
    // 5. probe lists: nearest fine centroids of each fine centroid (including itself)
    int np = std::min(o.nprobe, nc);
    std::vector<int> probes((size_t)nc * np);
    {
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
                for (int j = 0; j < np; j++) probes[(size_t)c * np + j] = bi[j];
            }
        }
    }
    double work = 0;
    for (int c = 0; c < nc; c++) { double cand = 0; for (int j = 0; j < np; j++) { int p = probes[(size_t)c * np + j]; cand += offs[p + 1] - offs[p]; } work += cand * (offs[c + 1] - offs[c]); }
    LOG("kNN: %d fine clusters (max size %d), %d probes, %.0f candidates/query", nc, maxsz, np, work / N);
    // 6. scan: 4 queries x 16 candidates per register tile
    R.I.assign((size_t)N * K, 0); R.Dd.assign((size_t)N * K, 1e30f);
    constexpr int QB = 4, CB = 16;
#pragma omp parallel
    {
        std::vector<float> qv((size_t)QB * DP);
        std::vector<float> bd((size_t)QB * K); std::vector<int> bi((size_t)QB * K);
#pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < nc; c++) {
            int s = offs[c], m = offs[c + 1] - s;
            for (int q0 = 0; q0 < m; q0 += QB) {
                int nq = std::min(QB, m - q0);
                for (int a = 0; a < QB; a++) {
                    int qa = std::min(q0 + a, m - 1);   // duplicate the last query to fill the tile
                    memcpy(&qv[(size_t)a * DP], &R.X[(size_t)(s + qa) * DP], DP * 4);
                }
                std::fill(bd.begin(), bd.end(), 1e30f); std::fill(bi.begin(), bi.end(), -1);
                for (int j = 0; j < np; j++) {
                    int p = probes[(size_t)c * np + j];
                    int ps = offs[p], pm = offs[p + 1] - ps, pmp = (pm + 15) / 16 * 16;
                    const float* blk = Xt.data() + toff[p];
                    for (int t0 = 0; t0 < pmp; t0 += CB) {
                        float acc[QB][CB];
                        for (int a = 0; a < QB; a++) for (int t = 0; t < CB; t++) acc[a][t] = 0.f;
                        for (int d = 0; d < DP; d++) {
                            const float* col = blk + (size_t)d * pmp + t0;
                            for (int a = 0; a < QB; a++) {
                                float qd = qv[(size_t)a * DP + d];
                                for (int t = 0; t < CB; t++) { float e = qd - col[t]; acc[a][t] += e * e; }
                            }
                        }
                        for (int a = 0; a < nq; a++) {
                            float* bda = &bd[(size_t)a * K]; int* bia = &bi[(size_t)a * K];
                            int self = s + q0 + a;
                            for (int t = 0; t < CB; t++) {
                                float dd = acc[a][t];
                                if (dd < bda[K - 1] && ps + t0 + t != self && t0 + t < pm) topk_insert(bda, bia, K, dd, ps + t0 + t);
                            }
                        }
                    }
                }
                for (int a = 0; a < nq; a++) {
                    int64_t q = s + q0 + a;
                    for (int j = 0; j < K; j++) {
                        int b = bi[(size_t)a * K + j]; float dv = bd[(size_t)a * K + j];
                        if (b < 0) { b = (int)q; dv = 1e30f; }
                        R.I[q * K + j] = b; R.Dd[q * K + j] = dv;
                    }
                }
            }
        }
    }
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

static void run_tsne_cpu(std::vector<F2>& Y, int64_t N, int K, const std::vector<int>& I, const std::vector<float>& Pw,
                         const std::vector<float>& mass, const Opts& o) {
    // reverse edges (in-edges) by counting sort
    std::vector<int64_t> roff(N + 1, 0);
    std::vector<int> rsrc; std::vector<float> rp;
    {
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
        double t2 = now();
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
        t_rep += t1 - t0; t_fft += tf2 - tf1; t_att += t2 - t1; t_upd += t3 - t2;
        if (it % 100 == 0 || it == total - 1)
            LOG("iter %4d  extent %8.2f  grid %5d  h %.3f  Z/N %.4g  (%.0f ms/iter: repulsion %.0f [fft %.0f], attraction %.0f)", it, ext, P, h, Z / N,
                1e3 * (t3 - t0), 1e3 * (t1 - t0), 1e3 * (tf2 - tf1), 1e3 * (t2 - t1));
    }
    LOG("optimisation time: repulsion %.1fs (FFT %.1fs), attraction %.1fs, update %.1fs", t_rep, t_fft, t_att, t_upd);
    F.release();
}

int main(int argc, char** argv) {
    Opts o;
    parse_args(argc, argv, o);
    if (!o.project.empty()) { fprintf(stderr, "fastsne_cpu: --project is GPU-only for now; use fastsne\n"); return 1; }
    if (o.threads > 0) omp_set_num_threads(o.threads);
    if (o.nprobe < 0) o.nprobe = 8;
    if (o.refine < 0) o.refine = 0;
    if (o.cluster_size < 0) o.cluster_size = 1000;
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    LOG("fastsne_cpu: %d threads", omp_get_max_threads());

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

    // affinities, then rows re-sorted by neighbour index (memory order for the attraction loop, sorted lists
    // for the overlap measure)
    std::vector<float>& Pw = R.Dd;
    std::vector<float> share(o.dedup > 0 ? N : 0);
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

    std::vector<float> mass;
    if (o.dedup > 0) {
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
        mass.resize(N);
        double ms = 0; int64_t nsh = 0;
#pragma omp parallel for reduction(+:ms,nsh) schedule(static)
        for (int64_t i = 0; i < N; i++) { mass[i] = std::max(o.mass_min, expf(lm[i])); ms += mass[i]; nsh += mass[i] < 0.5f; }
        LOG("dedup: mean mass %.3f, %.1f%% of points below 0.5", ms / N, 100.0 * nsh / N);
    }

    // init: first two PCs scaled to std 1e-4
    std::vector<F2> Y(N);
    double sd = 0;
    for (int64_t i = 0; i < N; i++) sd += (double)R.X[(size_t)i * DP] * R.X[(size_t)i * DP];
    sd = sqrt(sd / N);
    float sc = (float)(1e-4 / std::max(sd, 1e-12));
    for (int64_t i = 0; i < N; i++) Y[i] = {R.X[(size_t)i * DP] * sc, R.X[(size_t)i * DP + 1] * sc};
    if (!o.save_model.empty()) model.X = R.X;
    std::vector<float>().swap(R.X);

    double t_opt = now();
    run_tsne_cpu(Y, N, K, R.I, Pw, mass, o);
    LOG("optimisation done in %.1fs", now() - t_opt);

    if (!o.save_model.empty()) {
        model.N = N; model.D = d.D; model.DP = DP; model.nc = R.nc; model.mode = mode;
        model.Y.resize((size_t)N * 2);
        for (int64_t i = 0; i < N; i++) { model.Y[2 * i] = Y[i].x; model.Y[2 * i + 1] = Y[i].y; }
        save_model(o.save_model, model);
        LOG("saved model %s", o.save_model.c_str());
    }
    std::vector<F2> Yo(N);
    for (int64_t i = 0; i < N; i++) Yo[R.perm[i]] = Y[i];
    FILE* f = fopen(o.out.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", o.out.c_str()); exit(1); }
    std::vector<char> buf(1 << 20);
    setvbuf(f, buf.data(), _IOFBF, buf.size());
    fputs("id\tx\ty\n", f);
    for (int64_t i = 0; i < N; i++) {
        fwrite(d.ids.data() + d.idoff[i], 1, d.idoff[i + 1] - d.idoff[i], f);
        fprintf(f, "\t%.4f\t%.4f\n", Yo[i].x, Yo[i].y);
    }
    fclose(f);
    LOG("wrote %s", o.out.c_str());
    return 0;
}
