// fastsne: GPU t-SNE for very large tetranucleotide-frequency (TNF) tables.
//
// Pipeline (everything after parsing runs on the GPU):
//   1. parse TSV rows "id \t f1 \t f2 ..." (multithreaded, many files ok)
//   2. per-row transform (CLR with a 0.5-count pseudocount by default), PCA to a few dims
//   3. approximate kNN: k-means inverted file, every query scans its cluster's nearest clusters
//   4. perplexity-calibrated (or uniform) affinities on the kNN graph
//   5. gradient descent; attraction over kNN edges, repulsion from a gridded density field:
//      phi = K * rho with K(r) = 1/(1+r^2), Z = sum phi, and sum_j w_ij^2 (y_i - y_j) = -grad(phi)/2,
//      so one forward FFT and three inverse FFTs per iteration give the full repulsive force.
//
// Optional per-point "mass" m_i scales the repulsion that point exerts and feels
// (q_ij ∝ m_i m_j w_ij); m_i < 1 acts as a local exaggeration that keeps huge
// homogeneous clusters compact instead of spreading into worm-like patterns.

#include "common.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CKFFT(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1); } } while (0)
#define CKBLAS(x) do { cublasStatus_t r_ = (x); if (r_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------------------------
// Transform + PCA
// ---------------------------------------------------------------------------------------------
__global__ void k_transform(float* X, int64_t m, int D, int mode) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= m) return;
    float* x = X + i * D;
    if (mode == 0) {           // clr, pseudocount = half the smallest nonzero frequency of the row
        float mn = 1e30f, s = 0;
        for (int j = 0; j < D; j++) { float v = x[j]; if (v > 0 && v < mn) mn = v; }
        float eps = mn < 1e30f ? 0.5f * mn : 1.f;
        for (int j = 0; j < D; j++) { float v = logf(fmaxf(x[j], 0.f) + eps); x[j] = v; s += v; }
        s /= D;
        for (int j = 0; j < D; j++) x[j] -= s;
    } else if (mode == 1) {    // sqrt (Hellinger)
        float s = 0;
        for (int j = 0; j < D; j++) s += fmaxf(x[j], 0.f);
        s = s > 0 ? 1.f / s : 1.f;
        for (int j = 0; j < D; j++) x[j] = sqrtf(fmaxf(x[j], 0.f) * s);
    }
}

__global__ void k_center(float* X, int64_t m, int D, const float* mu) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= m * D) return;
    X[i] -= mu[i % D];
}

// transform + center + project with a stored PCA (projection mode); returns device array n x DP
static float* apply_pca(Data& d, int mode, const std::vector<float>& muf, const std::vector<float>& Vp, int DP, cublasHandle_t blas) {
    const int D = d.D; const int64_t N = d.N;
    const int64_t CH = std::max<int64_t>(1, (256ll << 20) / (D * 4));
    float *dX, *dMu, *dV, *dY;
    CK(cudaMalloc(&dX, CH * D * 4)); CK(cudaMalloc(&dMu, D * 4)); CK(cudaMalloc(&dV, (size_t)D * DP * 4));
    CK(cudaMalloc(&dY, (size_t)N * DP * 4));
    CK(cudaMemcpy(dMu, muf.data(), D * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, Vp.data(), Vp.size() * 4, cudaMemcpyHostToDevice));
    float* stage; CK(cudaMallocHost(&stage, CH * D * 4));
    float one = 1, zero = 0;
    for (int64_t s = 0; s < N; s += CH) {
        int64_t m = std::min(CH, N - s);
        d.get_rows(s, m, stage);
        CK(cudaMemcpy(dX, stage, m * D * 4, cudaMemcpyHostToDevice));
        if (mode != 2) k_transform<<<(m + 255) / 256, 256>>>(dX, m, D, mode);
        k_center<<<(m * D + 255) / 256, 256>>>(dX, m, D, dMu);
        CKBLAS(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, DP, (int)m, D, &one, dV, D, dX, D, &zero, dY + s * DP, DP));
    }
    CK(cudaDeviceSynchronize());
    CK(cudaFreeHost(stage));
    cudaFree(dX); cudaFree(dMu); cudaFree(dV);
    return dY;
}

// returns device array N x DP (row-major), DP = pca dims padded to multiple of 4
static float* transform_pca(Data& d, const Opts& o, int DP, cublasHandle_t blas, std::vector<float>& muf, std::vector<float>& Vp) {
    const int D = d.D; const int64_t N = d.N;
    int mode = o.transform == "clr" ? 0 : o.transform == "sqrt" ? 1 : 2;
    const int64_t CH = std::max<int64_t>(1, (256ll << 20) / (D * 4));
    float *dX, *dOnes, *dSum, *dC;
    CK(cudaMalloc(&dX, CH * D * 4));
    CK(cudaMalloc(&dOnes, CH * 4));
    CK(cudaMalloc(&dSum, D * 4));
    CK(cudaMalloc(&dC, (size_t)D * D * 4));
    std::vector<float> ones(CH, 1.f);
    CK(cudaMemcpy(dOnes, ones.data(), CH * 4, cudaMemcpyHostToDevice));
    float* stage; CK(cudaMallocHost(&stage, CH * D * 4));
    float one = 1, zero = 0;
    // pass 1: transform (written back to host) + column sums
    std::vector<double> mu(D, 0);
    std::vector<float> tmp(D);
    for (int64_t s = 0; s < N; s += CH) {
        int64_t m = std::min(CH, N - s);
        d.get_rows(s, m, stage);
        CK(cudaMemcpy(dX, stage, m * D * 4, cudaMemcpyHostToDevice));
        if (mode != 2) k_transform<<<(m + 255) / 256, 256>>>(dX, m, D, mode);
        CKBLAS(cublasSgemv(blas, CUBLAS_OP_N, D, (int)m, &one, dX, D, dOnes, 1, &zero, dSum, 1));
        CK(cudaMemcpy(tmp.data(), dSum, D * 4, cudaMemcpyDeviceToHost));
        for (int j = 0; j < D; j++) mu[j] += tmp[j];
        CK(cudaMemcpy(stage, dX, m * D * 4, cudaMemcpyDeviceToHost));
        d.put_rows(s, m, stage);
    }
    muf.assign(D, 0.f);
    for (int j = 0; j < D; j++) { mu[j] /= N; muf[j] = (float)mu[j]; }
    float* dMu; CK(cudaMalloc(&dMu, D * 4));
    CK(cudaMemcpy(dMu, muf.data(), D * 4, cudaMemcpyHostToDevice));
    // pass 2: covariance
    std::vector<double> C((size_t)D * D, 0);
    std::vector<float> Cf((size_t)D * D);
    for (int64_t s = 0; s < N; s += CH) {
        int64_t m = std::min(CH, N - s);
        d.get_rows(s, m, stage);
        CK(cudaMemcpy(dX, stage, m * D * 4, cudaMemcpyHostToDevice));
        k_center<<<(m * D + 255) / 256, 256>>>(dX, m, D, dMu);
        CKBLAS(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, (int)m, &one, dX, D, dX, D, &zero, dC, D));
        CK(cudaMemcpy(Cf.data(), dC, (size_t)D * D * 4, cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < C.size(); j++) C[j] += Cf[j];
    }
    for (auto& c : C) c /= N;
    double tr = 0; for (int j = 0; j < D; j++) tr += C[(size_t)j * D + j];
    std::vector<double> V, ev;
    jacobi_eigen(C, D, V, ev);
    std::vector<int> ord(D); std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return ev[a] > ev[b]; });
    int npc = std::min(o.pca, D);
    double cum = 0; for (int j = 0; j < npc; j++) cum += ev[ord[j]];
    LOG("PCA: %d comps explain %.1f%% of variance (pc1 %.1f%%, pc2 %.1f%%)", npc, 100 * cum / tr,
        100 * ev[ord[0]] / tr, D > 1 ? 100 * ev[ord[1]] / tr : 0.0);
    // V padded: D x DP column-major (column j = eigenvector j)
    Vp.assign((size_t)D * DP, 0.f);
    for (int j = 0; j < npc; j++) for (int i = 0; i < D; i++) Vp[(size_t)j * D + i] = (float)V[(size_t)i * D + ord[j]];
    float* dV; CK(cudaMalloc(&dV, (size_t)D * DP * 4));
    CK(cudaMemcpy(dV, Vp.data(), Vp.size() * 4, cudaMemcpyHostToDevice));
    float* dY; CK(cudaMalloc(&dY, (size_t)N * DP * 4));
    for (int64_t s = 0; s < N; s += CH) {
        int64_t m = std::min(CH, N - s);
        d.get_rows(s, m, stage);
        CK(cudaMemcpy(dX, stage, m * D * 4, cudaMemcpyHostToDevice));
        k_center<<<(m * D + 255) / 256, 256>>>(dX, m, D, dMu);
        CKBLAS(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, DP, (int)m, D, &one, dV, D, dX, D, &zero, dY + s * DP, DP));
    }
    CK(cudaDeviceSynchronize());
    CK(cudaFreeHost(stage));
    cudaFree(dX); cudaFree(dOnes); cudaFree(dSum); cudaFree(dC); cudaFree(dMu); cudaFree(dV);
    return dY;
}

// ---------------------------------------------------------------------------------------------
// kNN: k-means inverted file
// ---------------------------------------------------------------------------------------------
template <int DP>
__global__ void k_assign(const float* __restrict__ X, int64_t n, const float* __restrict__ C, int nc, int* assign) {
    __shared__ float4 sc[128 * (DP / 4)];
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    float q[DP];
    if (i < n) {
#pragma unroll
        for (int j = 0; j < DP; j++) q[j] = X[i * DP + j];
    }
    float best = 1e30f; int bi = 0;
    for (int c0 = 0; c0 < nc; c0 += 128) {
        int cn = min(128, nc - c0);
        __syncthreads();
        for (int t = threadIdx.x; t < cn * (DP / 4); t += blockDim.x)
            sc[t] = reinterpret_cast<const float4*>(C + (int64_t)c0 * DP)[t];
        __syncthreads();
        if (i < n) {
            for (int c = 0; c < cn; c++) {
                float dd = 0;
#pragma unroll
                for (int j = 0; j < DP / 4; j++) {
                    float4 v = sc[c * (DP / 4) + j];
                    float a = q[4 * j] - v.x, b = q[4 * j + 1] - v.y, e = q[4 * j + 2] - v.z, f = q[4 * j + 3] - v.w;
                    dd += a * a + b * b + e * e + f * f;
                }
                if (dd < best) { best = dd; bi = c0 + c; }
            }
        }
    }
    if (i < n) assign[i] = bi;
}

__global__ void k_accum(const float* X, int64_t n, int DP, const int* assign, float* sums, int* counts) {
    int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (t >= n * DP) return;
    int64_t i = t / DP; int j = t % DP;
    int c = assign[i];
    atomicAdd(&sums[(int64_t)c * DP + j], X[t]);
    if (j == 0) atomicAdd(&counts[c], 1);
}

__global__ void k_gather_rows(const float* X, const int* idx, int64_t n, int DP, float* out) {
    int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (t >= n * DP) return;
    int64_t i = t / DP; int j = t % DP;
    out[t] = X[(int64_t)idx[i] * DP + j];
}

// nearest `np` centroids of each centroid (includes itself)
__global__ void k_probes(const float* C, int nc, int DP, int np, int* probes) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    float bd[64]; int bi[64];
    for (int j = 0; j < np; j++) { bd[j] = 1e30f; bi[j] = c; }
    for (int o = 0; o < nc; o++) {
        float dd = 0;
        for (int j = 0; j < DP; j++) { float a = C[(int64_t)c * DP + j] - C[(int64_t)o * DP + j]; dd += a * a; }
        if (o == c) dd = -1;
        if (dd < bd[np - 1]) {
            int p = np - 1;
            while (p > 0 && bd[p - 1] > dd) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
            bd[p] = dd; bi[p] = o;
        }
    }
    for (int j = 0; j < np; j++) probes[(int64_t)c * np + j] = bi[j];
}

#define QT 64
template <int DP, int K>
__global__ void __launch_bounds__(QT) k_knn(const float* __restrict__ X, const float* __restrict__ Q, const int* __restrict__ offs,
                                            const int* __restrict__ probes, int np, const int4* __restrict__ tiles,
                                            int* outI, float* outD) {
    __shared__ float4 sc[QT * (DP / 4)];
    int4 tl = tiles[blockIdx.x];
    int c = tl.x; int q = tl.y + threadIdx.x; bool active = threadIdx.x < tl.z;
    float qv[DP];
    if (active) {
#pragma unroll
        for (int j = 0; j < DP / 4; j++) {
            float4 v = reinterpret_cast<const float4*>(Q + (int64_t)q * DP)[j];
            qv[4 * j] = v.x; qv[4 * j + 1] = v.y; qv[4 * j + 2] = v.z; qv[4 * j + 3] = v.w;
        }
    }
    const int self = Q == X ? q : -1;   // exclude the query itself when querying the reference set
    float bd[K]; int bi[K];
#pragma unroll
    for (int j = 0; j < K; j++) { bd[j] = 1e30f; bi[j] = -1; }
    for (int p = 0; p < np; p++) {
        int cl = probes[(int64_t)c * np + p];
        int s = offs[cl], e = offs[cl + 1];
        for (int base = s; base < e; base += QT) {
            int cn = min(QT, e - base);
            __syncthreads();
            for (int t = threadIdx.x; t < cn * (DP / 4); t += QT)
                sc[t] = reinterpret_cast<const float4*>(X + (int64_t)base * DP)[t];
            __syncthreads();
            if (active) {
                for (int t = 0; t < cn; t++) {
                    float dd = 0;
#pragma unroll
                    for (int j = 0; j < DP / 4; j++) {
                        float4 v = sc[t * (DP / 4) + j];
                        float a = qv[4 * j] - v.x, b = qv[4 * j + 1] - v.y, f = qv[4 * j + 2] - v.z, g = qv[4 * j + 3] - v.w;
                        dd += a * a + b * b + f * f + g * g;
                    }
                    if (dd < bd[K - 1] && base + t != self) {
                        int pos = K - 1;
                        while (pos > 0 && bd[pos - 1] > dd) { bd[pos] = bd[pos - 1]; bi[pos] = bi[pos - 1]; pos--; }
                        bd[pos] = dd; bi[pos] = base + t;
                    }
                }
            }
        }
    }
    if (active) {
        for (int j = 0; j < K; j++) {
            int b = bi[j]; float dv = bd[j];
            if (b < 0) { b = self >= 0 ? q : 0; dv = 1e30f; }   // not enough candidates
            outI[(int64_t)q * K + j] = b; outD[(int64_t)q * K + j] = dv;
        }
    }
}

// NN-descent style refinement: try the first A neighbours of each of the first A neighbours
template <int DP, int K>
__global__ void k_refine(const float* __restrict__ X, int64_t n, int A, const int* I0, const float* D0,
                         int* I1, float* D1) {
    int64_t q = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (q >= n) return;
    float qv[DP];
#pragma unroll
    for (int j = 0; j < DP / 4; j++) {
        float4 v = reinterpret_cast<const float4*>(X + q * DP)[j];
        qv[4 * j] = v.x; qv[4 * j + 1] = v.y; qv[4 * j + 2] = v.z; qv[4 * j + 3] = v.w;
    }
    float bd[K]; int bi[K];
#pragma unroll
    for (int j = 0; j < K; j++) { bd[j] = D0[q * K + j]; bi[j] = I0[q * K + j]; }
    for (int a = 0; a < A; a++) {
        int nb = I0[q * K + a];
        if (nb == q) continue;
        for (int b = 0; b < A; b++) {
            int c = I0[(int64_t)nb * K + b];
            if (c == q) continue;
            const float4* cv = reinterpret_cast<const float4*>(X + (int64_t)c * DP);
            float dd = 0;
#pragma unroll
            for (int j = 0; j < DP / 4; j++) {
                float4 v = cv[j];
                float e0 = qv[4 * j] - v.x, e1 = qv[4 * j + 1] - v.y, e2 = qv[4 * j + 2] - v.z, e3 = qv[4 * j + 3] - v.w;
                dd += e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;
            }
            if (dd >= bd[K - 1]) continue;
            bool dup = false;
            for (int j = 0; j < K; j++) if (bi[j] == c) { dup = true; break; }
            if (dup) continue;
            int pos = K - 1;
            while (pos > 0 && bd[pos - 1] > dd) { bd[pos] = bd[pos - 1]; bi[pos] = bi[pos - 1]; pos--; }
            bd[pos] = dd; bi[pos] = c;
        }
    }
    for (int j = 0; j < K; j++) { I1[q * K + j] = bi[j]; D1[q * K + j] = bd[j]; }
}

// brute force for recall checks
__global__ void k_brute(const float* X, int64_t n, int DP, const int* qs, int nq, int K, int* outI) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nq) return;
    int q = qs[qi];
    float bd[128]; int bi[128];
    for (int j = 0; j < K; j++) { bd[j] = 1e30f; bi[j] = -1; }
    for (int64_t o = 0; o < n; o++) {
        if (o == q) continue;
        float dd = 0;
        for (int j = 0; j < DP; j++) { float a = X[(int64_t)q * DP + j] - X[o * DP + j]; dd += a * a; }
        if (dd < bd[K - 1]) {
            int p = K - 1;
            while (p > 0 && bd[p - 1] > dd) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
            bd[p] = dd; bi[p] = (int)o;
        }
    }
    for (int j = 0; j < K; j++) outI[(int64_t)qi * K + j] = bi[j];
}

template <int DP, int K>
static void launch_refine(const float* X, int64_t n, int A, const int* I0, const float* D0, int* I1, float* D1) {
    k_refine<DP, K><<<(n + 127) / 128, 128>>>(X, n, A, I0, D0, I1, D1);
}
template <int DP>
static void launch_refine_k(int K, const float* X, int64_t n, int A, const int* I0, const float* D0, int* I1, float* D1) {
    switch (K) {
        case 16: launch_refine<DP, 16>(X, n, A, I0, D0, I1, D1); break;
        case 32: launch_refine<DP, 32>(X, n, A, I0, D0, I1, D1); break;
        case 48: launch_refine<DP, 48>(X, n, A, I0, D0, I1, D1); break;
        case 64: launch_refine<DP, 64>(X, n, A, I0, D0, I1, D1); break;
        case 96: launch_refine<DP, 96>(X, n, A, I0, D0, I1, D1); break;
        default: fprintf(stderr, "unsupported k %d\n", K); exit(1);
    }
}
static void refine_dispatch(int DP, int K, const float* X, int64_t n, int A, const int* I0, const float* D0, int* I1, float* D1) {
    switch (DP) {
        case 8: launch_refine_k<8>(K, X, n, A, I0, D0, I1, D1); break;
        case 16: launch_refine_k<16>(K, X, n, A, I0, D0, I1, D1); break;
        case 24: launch_refine_k<24>(K, X, n, A, I0, D0, I1, D1); break;
        case 32: launch_refine_k<32>(K, X, n, A, I0, D0, I1, D1); break;
        case 48: launch_refine_k<48>(K, X, n, A, I0, D0, I1, D1); break;
        case 64: launch_refine_k<64>(K, X, n, A, I0, D0, I1, D1); break;
        default: fprintf(stderr, "unsupported pca dims %d\n", DP); exit(1);
    }
}
template <int DP, int K>
static void launch_knn(const float* X, const float* Q, const int* offs, const int* probes, int np, const int4* tiles, int ntiles, int* I, float* Dd) {
    k_knn<DP, K><<<ntiles, QT>>>(X, Q, offs, probes, np, tiles, I, Dd);
}
template <int DP>
static void launch_knn_k(int K, const float* X, const float* Q, const int* offs, const int* probes, int np, const int4* tiles, int ntiles, int* I, float* Dd) {
    switch (K) {
        case 16: launch_knn<DP, 16>(X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        case 32: launch_knn<DP, 32>(X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        case 48: launch_knn<DP, 48>(X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        case 64: launch_knn<DP, 64>(X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        case 96: launch_knn<DP, 96>(X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        default: fprintf(stderr, "unsupported k %d\n", K); exit(1);
    }
}
template <int DP>
static void launch_assign(const float* X, int64_t n, const float* C, int nc, int* a) {
    k_assign<DP><<<(n + 127) / 128, 128>>>(X, n, C, nc, a);
}
static void assign_dispatch(int DP, const float* X, int64_t n, const float* C, int nc, int* a) {
    switch (DP) {
        case 8: launch_assign<8>(X, n, C, nc, a); break;
        case 16: launch_assign<16>(X, n, C, nc, a); break;
        case 24: launch_assign<24>(X, n, C, nc, a); break;
        case 32: launch_assign<32>(X, n, C, nc, a); break;
        case 48: launch_assign<48>(X, n, C, nc, a); break;
        case 64: launch_assign<64>(X, n, C, nc, a); break;
        default: fprintf(stderr, "unsupported pca dims %d\n", DP); exit(1);
    }
}
static void knn_dispatch(int DP, int K, const float* X, const float* Q, const int* offs, const int* probes, int np, const int4* tiles, int ntiles, int* I, float* Dd) {
    switch (DP) {
        case 8: launch_knn_k<8>(K, X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        case 16: launch_knn_k<16>(K, X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        case 24: launch_knn_k<24>(K, X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        case 32: launch_knn_k<32>(K, X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        case 48: launch_knn_k<48>(K, X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        case 64: launch_knn_k<64>(K, X, Q, offs, probes, np, tiles, ntiles, I, Dd); break;
        default: fprintf(stderr, "unsupported pca dims %d\n", DP); exit(1);
    }
}

// Clusters the data, reorders it by cluster (perm[new] = old) and computes the kNN graph in the new order.
static void knn(float*& dX, int64_t N, int DP, const Opts& o, int K, std::vector<int>& perm, int*& dI, float*& dD,
                std::vector<float>& cent, std::vector<int>& offs_out) {
    std::mt19937_64 rng(o.seed);
    int nc = o.clusters > 0 ? o.clusters : (int)std::max<int64_t>(1, std::min<int64_t>(65536, N / 1000));
    int np = std::min(o.nprobe, nc);
    // k-means on a sample
    int64_t ns = std::min<int64_t>(N, (int64_t)nc * 64);
    std::vector<int> sidx(ns);
    if (ns == N) std::iota(sidx.begin(), sidx.end(), 0);
    else for (auto& s : sidx) s = (int)(rng() % N);
    int *dS, *dA, *dCnt;
    float *dXs, *dC, *dSum;
    CK(cudaMalloc(&dS, ns * 4)); CK(cudaMalloc(&dXs, ns * DP * 4));
    CK(cudaMalloc(&dC, (size_t)nc * DP * 4)); CK(cudaMalloc(&dSum, (size_t)nc * DP * 4));
    CK(cudaMalloc(&dCnt, nc * 4)); CK(cudaMalloc(&dA, N * 4));
    CK(cudaMemcpy(dS, sidx.data(), ns * 4, cudaMemcpyHostToDevice));
    k_gather_rows<<<(ns * DP + 255) / 256, 256>>>(dX, dS, ns, DP, dXs);
    std::vector<int> cinit(nc);
    for (auto& c : cinit) c = (int)(rng() % ns);
    int* dCi; CK(cudaMalloc(&dCi, nc * 4));
    CK(cudaMemcpy(dCi, cinit.data(), nc * 4, cudaMemcpyHostToDevice));
    k_gather_rows<<<((int64_t)nc * DP + 255) / 256, 256>>>(dXs, dCi, nc, DP, dC);
    std::vector<float> hsum((size_t)nc * DP), hc((size_t)nc * DP);
    std::vector<int> hcnt(nc);
    for (int it = 0; it < 8; it++) {
        assign_dispatch(DP, dXs, ns, dC, nc, dA);
        CK(cudaMemset(dSum, 0, (size_t)nc * DP * 4)); CK(cudaMemset(dCnt, 0, nc * 4));
        k_accum<<<(ns * DP + 255) / 256, 256>>>(dXs, ns, DP, dA, dSum, dCnt);
        CK(cudaMemcpy(hsum.data(), dSum, hsum.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hcnt.data(), dCnt, nc * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hc.data(), dC, hc.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<float> xs(DP);
        int empty = 0;
        for (int c = 0; c < nc; c++) {
            if (hcnt[c] == 0) {  // reseed from a random sample point
                empty++;
                int r = (int)(rng() % ns);
                CK(cudaMemcpy(xs.data(), dXs + (int64_t)r * DP, DP * 4, cudaMemcpyDeviceToHost));
                for (int j = 0; j < DP; j++) hc[(size_t)c * DP + j] = xs[j];
            } else for (int j = 0; j < DP; j++) hc[(size_t)c * DP + j] = hsum[(size_t)c * DP + j] / hcnt[c];
        }
        CK(cudaMemcpy(dC, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
    }
    assign_dispatch(DP, dX, N, dC, nc, dA);
    // sort points by cluster
    int* dP; CK(cudaMalloc(&dP, N * 4));
    thrust::sequence(thrust::device_ptr<int>(dP), thrust::device_ptr<int>(dP + N));
    thrust::stable_sort_by_key(thrust::device_ptr<int>(dA), thrust::device_ptr<int>(dA + N), thrust::device_ptr<int>(dP));
    std::vector<int> hA(N);
    CK(cudaMemcpy(hA.data(), dA, N * 4, cudaMemcpyDeviceToHost));
    perm.resize(N);
    CK(cudaMemcpy(perm.data(), dP, N * 4, cudaMemcpyDeviceToHost));
    float* dX2; CK(cudaMalloc(&dX2, (size_t)N * DP * 4));
    k_gather_rows<<<(N * DP + 255) / 256, 256>>>(dX, dP, N, DP, dX2);
    CK(cudaFree(dX)); dX = dX2;
    std::vector<int> offs(nc + 1, 0);
    for (int64_t i = 0; i < N; i++) offs[hA[i] + 1]++;
    for (int c = 0; c < nc; c++) offs[c + 1] += offs[c];
    cent = hc; offs_out = offs;
    int maxsz = 0; for (int c = 0; c < nc; c++) maxsz = std::max(maxsz, offs[c + 1] - offs[c]);
    // probes
    int* dProbes; CK(cudaMalloc(&dProbes, (size_t)nc * np * 4));
    k_probes<<<(nc + 63) / 64, 64>>>(dC, nc, DP, np, dProbes);
    std::vector<int> hprobes((size_t)nc * np);
    CK(cudaMemcpy(hprobes.data(), dProbes, hprobes.size() * 4, cudaMemcpyDeviceToHost));
    double work = 0;
    std::vector<int4> tiles;
    for (int c = 0; c < nc; c++) {
        int sz = offs[c + 1] - offs[c];
        double cand = 0; for (int p = 0; p < np; p++) { int cl = hprobes[(size_t)c * np + p]; cand += offs[cl + 1] - offs[cl]; }
        work += cand * sz;
        for (int t = 0; t < sz; t += QT) tiles.push_back(make_int4(c, offs[c] + t, std::min(QT, sz - t), 0));
    }
    // biggest work first for better load balance
    LOG("kNN: %d clusters (max size %d), %d probes, %.0f candidates/query", nc, maxsz, np, work / N);
    int4* dTiles; CK(cudaMalloc(&dTiles, tiles.size() * sizeof(int4)));
    CK(cudaMemcpy(dTiles, tiles.data(), tiles.size() * sizeof(int4), cudaMemcpyHostToDevice));
    int* dOffs; CK(cudaMalloc(&dOffs, (nc + 1) * 4));
    CK(cudaMemcpy(dOffs, offs.data(), (nc + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dI, (size_t)N * K * 4)); CK(cudaMalloc(&dD, (size_t)N * K * 4));
    knn_dispatch(DP, K, dX, dX, dOffs, dProbes, np, dTiles, (int)tiles.size(), dI, dD);
    CK(cudaDeviceSynchronize());
    LOG("kNN scan done");
    if (o.refine > 0) {
        // in place: a thread may see a neighbour's list mid-update, which is harmless for a heuristic search
        for (int r = 0; r < o.refine; r++) refine_dispatch(DP, K, dX, N, std::min(K, o.refine_a), dI, dD, dI, dD);
        CK(cudaDeviceSynchronize());
        LOG("kNN refined (%d rounds, %d x %d candidates)", o.refine, std::min(K, o.refine_a), std::min(K, o.refine_a));
    }
    if (o.check_knn > 0) {
        int nq = std::min<int64_t>(o.check_knn, N);
        std::vector<int> qs(nq); for (auto& q : qs) q = (int)(rng() % N);
        int *dQ, *dB; CK(cudaMalloc(&dQ, nq * 4)); CK(cudaMalloc(&dB, (size_t)nq * K * 4));
        CK(cudaMemcpy(dQ, qs.data(), nq * 4, cudaMemcpyHostToDevice));
        k_brute<<<(nq + 63) / 64, 64>>>(dX, N, DP, dQ, nq, K, dB);
        std::vector<int> hb((size_t)nq * K), ha(K);
        CK(cudaMemcpy(hb.data(), dB, hb.size() * 4, cudaMemcpyDeviceToHost));
        int64_t hit = 0, hit10 = 0;
        for (int qi = 0; qi < nq; qi++) {
            CK(cudaMemcpy(ha.data(), dI + (int64_t)qs[qi] * K, K * 4, cudaMemcpyDeviceToHost));
            std::sort(ha.begin(), ha.end());
            for (int j = 0; j < K; j++) {
                bool f = std::binary_search(ha.begin(), ha.end(), hb[(size_t)qi * K + j]);
                hit += f; if (j < 10) hit10 += f;
            }
        }
        LOG("kNN recall@%d = %.3f, recall of top-10 = %.3f (%d queries)", K, hit / (double)nq / K, hit10 / (double)nq / 10, nq);
        cudaFree(dQ); cudaFree(dB);
    }
    cudaFree(dS); cudaFree(dXs); cudaFree(dC); cudaFree(dSum); cudaFree(dCnt); cudaFree(dA);
    cudaFree(dCi); cudaFree(dP); cudaFree(dProbes); cudaFree(dTiles); cudaFree(dOffs);
}

// ---------------------------------------------------------------------------------------------
// Affinities
// ---------------------------------------------------------------------------------------------
// in: squared distances (sorted), out: p_{j|i} / (2N) in place; also writes beta (precision) per point
__global__ void k_perplexity(float* Dd, const int* I, int64_t n, int K, float logperp, bool uniform, float scale, float* sig) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* d = Dd + i * K;
    int kv = 0; while (kv < K && d[kv] < 1e29f) kv++;
    if (uniform || kv == 0) {
        for (int j = 0; j < K; j++) d[j] = j < kv ? scale / kv : 0.f;
        if (sig) sig[i] = kv ? sqrtf(d[kv - 1]) : 0.f;
        return;
    }
    float d0 = d[0];
    float lo = 0, hi = 1e30f, beta = 1.f;
    float mean = 0; for (int j = 0; j < kv; j++) mean += d[j] - d0; mean /= kv;
    beta = mean > 0 ? 1.f / mean : 1.f;
    for (int it = 0; it < 200; it++) {
        float s = 0, sd = 0;
        for (int j = 0; j < kv; j++) { float dj = d[j] - d0; float w = dj > 0 ? __expf(-beta * dj) : 1.f; s += w; sd += dj > 0 ? w * dj : 0.f; }
        float H = logf(s) + beta * sd / s;
        float diff = H - logperp;
        if (fabsf(diff) < 1e-5f) break;
        if (diff > 0) { lo = beta; beta = hi >= 1e29f ? beta * 2 : 0.5f * (beta + hi); if (beta > 1e30f) { beta = 1e30f; break; } }
        else { hi = beta; beta = 0.5f * (beta + lo); }
    }
    float s = 0;
    for (int j = 0; j < kv; j++) { float dj = d[j] - d0; s += dj > 0 ? __expf(-beta * dj) : 1.f; }
    for (int j = 0; j < K; j++) { float dj = d[j] - d0; d[j] = j < kv ? scale * (dj > 0 ? __expf(-beta * dj) : 1.f) / s : 0.f; }
    if (sig) sig[i] = rsqrtf(beta);
}

// ---------------------------------------------------------------------------------------------
// Gradient descent
// ---------------------------------------------------------------------------------------------
__global__ void k_bbox(const float2* Y, int64_t n, float4* out) {
    __shared__ float4 s[256];
    float4 b = make_float4(1e30f, 1e30f, -1e30f, -1e30f);
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float2 y = Y[i];
        b.x = fminf(b.x, y.x); b.y = fminf(b.y, y.y); b.z = fmaxf(b.z, y.x); b.w = fmaxf(b.w, y.y);
    }
    s[threadIdx.x] = b;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            float4 o = s[threadIdx.x + st];
            b.x = fminf(b.x, o.x); b.y = fminf(b.y, o.y); b.z = fmaxf(b.z, o.z); b.w = fmaxf(b.w, o.w);
            s[threadIdx.x] = b;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

struct GridP { float x0, y0, inv_h; int P; };

__global__ void k_splat(const float2* Y, const float* mass, int64_t n, GridP g, float* grid) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 y = Y[i];
    float u = (y.x - g.x0) * g.inv_h, v = (y.y - g.y0) * g.inv_h;
    int ix = (int)floorf(u), iy = (int)floorf(v);
    float fx = u - ix, fy = v - iy;
    float m = mass ? mass[i] : 1.f;
    int64_t b = (int64_t)iy * g.P + ix;
    atomicAdd(&grid[b], m * (1 - fx) * (1 - fy));
    atomicAdd(&grid[b + 1], m * fx * (1 - fy));
    atomicAdd(&grid[b + g.P], m * (1 - fx) * fy);
    atomicAdd(&grid[b + g.P + 1], m * fx * fy);
}

__global__ void k_kernel_grid(float* k0, float* k1, float* k2, int P, float h) {
    int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (t >= (int64_t)P * P) return;
    int r = t / P, c = t % P;
    int dy = r < P / 2 ? r : r - P, dx = c < P / 2 ? c : c - P;
    float x = dx * h, y = dy * h;
    float w = 1.f / (1.f + x * x + y * y);
    k0[t] = w;
    k1[t] = (c == P / 2) ? 0.f : -2.f * x * w * w;
    k2[t] = (r == P / 2) ? 0.f : -2.f * y * w * w;
}

__global__ void k_scale_c(cufftComplex* a, int64_t n, float s) {
    int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (t < n) { a[t].x *= s; a[t].y *= s; }
}

__global__ void k_specmul(const cufftComplex* S, const cufftComplex* Kf, int64_t n, cufftComplex* out) {
    int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (t >= n) return;
    cufftComplex s = S[t];
#pragma unroll
    for (int k = 0; k < 3; k++) {
        cufftComplex kk = Kf[k * n + t];
        out[k * n + t] = make_float2(s.x * kk.x - s.y * kk.y, s.x * kk.y + s.y * kk.x);
    }
}

// interpolates phi, dphi/dx, dphi/dy at each point; accumulates Z = sum_i m_i (phi_i - m_i * self_i)
__global__ void k_gather(const float2* Y, const float* mass, int64_t n, GridP g, float h, const float* field,
                         float2* gphi, double* Z) {
    __shared__ double sz[256];
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    double z = 0;
    if (i < n) {
        float2 y = Y[i];
        float u = (y.x - g.x0) * g.inv_h, v = (y.y - g.y0) * g.inv_h;
        int ix = (int)floorf(u), iy = (int)floorf(v);
        float fx = u - ix, fy = v - iy;
        int64_t b = (int64_t)iy * g.P + ix;
        int64_t PP = (int64_t)g.P * g.P;
        float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
        float r[3];
#pragma unroll
        for (int k = 0; k < 3; k++) {
            const float* f = field + k * PP + b;
            r[k] = w00 * f[0] + w10 * f[1] + w01 * f[g.P] + w11 * f[g.P + 1];
        }
        float m = mass ? mass[i] : 1.f;
        float sx0 = (1 - fx) * (1 - fx) + fx * fx, sx1 = 2 * fx * (1 - fx);
        float sy0 = (1 - fy) * (1 - fy) + fy * fy, sy1 = 2 * fy * (1 - fy);
        float k10 = 1.f / (1.f + h * h), k11 = 1.f / (1.f + 2 * h * h);
        float self = sx0 * sy0 + (sx1 * sy0 + sx0 * sy1) * k10 + sx1 * sy1 * k11;
        z = (double)m * (r[0] - m * self);
        gphi[i] = make_float2(r[1], r[2]);
    }
    sz[threadIdx.x] = z;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (threadIdx.x < st) sz[threadIdx.x] += sz[threadIdx.x + st]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(Z, sz[0]);
}

// Symmetric attraction without atomics: 8 lanes per point walk its K out-edges (row-major P, I)
// and its in-edges (reverse CSR roff/rsrc/rp), then reduce with shuffles.
__global__ void k_attract(const float2* __restrict__ Y, const int* __restrict__ I, const float* __restrict__ P,
                          const int64_t* __restrict__ roff, const int* __restrict__ rsrc, const float* __restrict__ rp,
                          int64_t n, int K, float2* attr) {
    int64_t gt = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t i = gt >> 3; int lane = gt & 7;
    bool valid = i < n;
    float ax = 0, ay = 0;
    if (valid) {
        float2 yi = Y[i];
        for (int j = lane; j < K; j += 8) {
            float p = P[i * K + j];
            if (p == 0.f) continue;
            float2 yj = Y[I[i * K + j]];
            float dx = yi.x - yj.x, dy = yi.y - yj.y;
            float f = p / (1.f + dx * dx + dy * dy);
            ax += f * dx; ay += f * dy;
        }
        for (int64_t e = roff[i] + lane; e < roff[i + 1]; e += 8) {
            float2 yj = Y[rsrc[e]];
            float dx = yi.x - yj.x, dy = yi.y - yj.y;
            float f = rp[e] / (1.f + dx * dx + dy * dy);
            ax += f * dx; ay += f * dy;
        }
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) { ax += __shfl_xor_sync(0xffffffffu, ax, o); ay += __shfl_xor_sync(0xffffffffu, ay, o); }
    if (valid && lane == 0) attr[i] = make_float2(ax, ay);
}

__global__ void k_count_dst(const int* I, const float* P, int64_t E, unsigned* cnt) {
    int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (e < E && P[e] > 0.f) atomicAdd(&cnt[I[e]], 1u);
}
__global__ void k_cnt_to_off(const unsigned* cnt, int64_t n, int64_t* off) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i <= n) off[i] = i < n ? cnt[i] : 0;
}
__global__ void k_scatter_rev(const int* I, const float* P, int64_t E, int K, const int64_t* off, unsigned* cur, int* rsrc, float* rp) {
    int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (e >= E || !(P[e] > 0.f)) return;
    int dst = I[e];
    int64_t pos = off[dst] + atomicAdd(&cur[dst], 1u);
    rsrc[pos] = (int)(e / K); rp[pos] = P[e];
}

// reverse CSR of the kNN graph (in-edges per point), by counting sort: needs no sort buffers
static void build_reverse(const int* dI, const float* dP, int64_t N, int K, int64_t*& roff, int*& rsrc, float*& rp) {
    int64_t E = N * K;
    unsigned* cnt;
    CK(cudaMalloc(&cnt, N * 4)); CK(cudaMemset(cnt, 0, N * 4));
    k_count_dst<<<(E + 255) / 256, 256>>>(dI, dP, E, cnt);
    CK(cudaMalloc(&roff, (N + 1) * 8));
    k_cnt_to_off<<<(N + 256) / 256, 256>>>(cnt, N, roff);
    thrust::exclusive_scan(thrust::device_ptr<int64_t>(roff), thrust::device_ptr<int64_t>(roff + N + 1), thrust::device_ptr<int64_t>(roff));
    int64_t tot; CK(cudaMemcpy(&tot, roff + N, 8, cudaMemcpyDeviceToHost));
    CK(cudaMalloc(&rsrc, std::max<int64_t>(tot, 1) * 4)); CK(cudaMalloc(&rp, std::max<int64_t>(tot, 1) * 4));
    CK(cudaMemset(cnt, 0, N * 4));
    k_scatter_rev<<<(E + 255) / 256, 256>>>(dI, dP, E, K, roff, cnt, rsrc, rp);
    CK(cudaFree(cnt));
}

__global__ void k_update(float2* Y, float2* upd, float2* gains, float2* attr, const float2* gphi, const float* mass,
                         int64_t n, float exag, const double* Z, float lr, float mom) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float m = mass ? mass[i] : 1.f;
    float rs = (float)(0.5 * m / *Z);
    float2 a = attr[i], gp = gphi[i];
    float gx = exag * a.x + rs * gp.x, gy = exag * a.y + rs * gp.y;
    float2 u = upd[i], gn = gains[i];
    gn.x = (signbit(gx) != signbit(u.x)) ? gn.x + 0.2f : fmaxf(gn.x * 0.8f, 0.01f);
    gn.y = (signbit(gy) != signbit(u.y)) ? gn.y + 0.2f : fmaxf(gn.y * 0.8f, 0.01f);
    u.x = mom * u.x - lr * gn.x * gx;
    u.y = mom * u.y - lr * gn.y * gy;
    float2 y = Y[i]; y.x += u.x; y.y += u.y;
    Y[i] = y; upd[i] = u; gains[i] = gn;
}

// exact repulsion for a few points (validation)
__global__ void k_exact_rep(const float2* Y, const float* mass, int64_t n, const int* qs, int nq, double* out) {
    __shared__ double s[3][256];
    int q = qs[blockIdx.x];
    float2 yq = Y[q];
    double z = 0, fx = 0, fy = 0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        if (j == q) continue;
        float2 yj = Y[j];
        float dx = yq.x - yj.x, dy = yq.y - yj.y;
        float w = 1.f / (1.f + dx * dx + dy * dy);
        float m = mass ? mass[j] : 1.f;
        z += m * w; fx += m * w * w * dx; fy += m * w * w * dy;
    }
    s[0][threadIdx.x] = z; s[1][threadIdx.x] = fx; s[2][threadIdx.x] = fy;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (threadIdx.x < st) for (int k = 0; k < 3; k++) s[k][threadIdx.x] += s[k][threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) for (int k = 0; k < 3; k++) out[blockIdx.x * 3 + k] = s[k][0];
}

struct Field {
    int P = 0, G = 0; float h = 0;
    float* grid = nullptr; float* kreal = nullptr; float* out = nullptr;
    cufftComplex *spec = nullptr, *kspec = nullptr, *ospec = nullptr;
    cufftHandle r2c = 0, r2c3 = 0, c2r3 = 0;
    void release() {
        if (!P) return;
        cudaFree(grid); cudaFree(kreal); cudaFree(out); cudaFree(spec); cudaFree(kspec); cudaFree(ospec);
        cufftDestroy(r2c); cufftDestroy(r2c3); cufftDestroy(c2r3); P = 0;
    }
    void setup(int G_, float h_) {
        int P_ = 2 * G_;
        if (P_ != P) {
            release();
            P = P_;
            int64_t PP = (int64_t)P * P, PC = (int64_t)P * (P / 2 + 1);
            CK(cudaMalloc(&grid, PP * 4)); CK(cudaMalloc(&kreal, 3 * PP * 4)); CK(cudaMalloc(&out, 3 * PP * 4));
            CK(cudaMalloc(&spec, PC * 8)); CK(cudaMalloc(&kspec, 3 * PC * 8)); CK(cudaMalloc(&ospec, 3 * PC * 8));
            CKFFT(cufftPlan2d(&r2c, P, P, CUFFT_R2C));
            int dims[2] = {P, P};
            CKFFT(cufftPlanMany(&r2c3, 2, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, 3));
            CKFFT(cufftPlanMany(&c2r3, 2, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, 3));
            h = 0;
        }
        G = G_;
        if (h_ != h) {
            h = h_;
            int64_t PP = (int64_t)P * P, PC = (int64_t)P * (P / 2 + 1);
            k_kernel_grid<<<(PP + 255) / 256, 256>>>(kreal, kreal + PP, kreal + 2 * PP, P, h);
            CKFFT(cufftExecR2C(r2c3, kreal, kspec));
            k_scale_c<<<(3 * PC + 255) / 256, 256>>>(kspec, 3 * PC, 1.f / (float)PP);
        }
    }
};

static void run_tsne(float2* dY, int64_t N, int K, const int* dI, const float* dP, const float* dMass, const Opts& o) {
    int64_t* roff; int* rsrc; float* rp;
    build_reverse(dI, dP, N, K, roff, rsrc, rp);
    int nb8 = (int)((N * 8 + 255) / 256);
    float2 *dU, *dG, *dA, *dGp; float4* dBB; double* dZ;
    CK(cudaMalloc(&dU, N * 8)); CK(cudaMalloc(&dG, N * 8)); CK(cudaMalloc(&dA, N * 8)); CK(cudaMalloc(&dGp, N * 8));
    CK(cudaMalloc(&dBB, 1024 * 16)); CK(cudaMalloc(&dZ, 8));
    CK(cudaMemset(dU, 0, N * 8)); CK(cudaMemset(dA, 0, N * 8));
    std::vector<float2> ones(N, make_float2(1, 1));
    CK(cudaMemcpy(dG, ones.data(), N * 8, cudaMemcpyHostToDevice));
    std::vector<float4> bb(1024);
    Field F;
    int nb = (int)((N + 255) / 256);
    float lr_all = o.lr > 0 ? o.lr : std::max(200.f, (float)N / o.ee);  // N/12 throughout, as in openTSNE
    int total = o.ee_iters + o.iters;
    double tf = 0, ta = 0, tsync = 0;
    for (int it = 0; it < total; it++) {
        bool ee = it < o.ee_iters;
        float exag = ee ? o.ee : o.exag;
        float mom = ee ? 0.5f : 0.8f;
        float lr = lr_all;
        double t1 = now();
        k_bbox<<<1024, 256>>>(dY, N, dBB);
        CK(cudaMemcpy(bb.data(), dBB, 1024 * 16, cudaMemcpyDeviceToHost));
        float4 b = bb[0];
        for (int j = 1; j < 1024; j++) { b.x = fminf(b.x, bb[j].x); b.y = fminf(b.y, bb[j].y); b.z = fmaxf(b.z, bb[j].z); b.w = fmaxf(b.w, bb[j].w); }
        float ext = fmaxf(b.z - b.x, b.w - b.y);
        tsync += now() - t1;
        // grid: G cells per side with spacing h, G >= ext/h + 3; padded FFT size 2G
        float h = o.h;
        int need = (int)ceilf(ext / h) + 3;
        int Gmax = o.max_grid / 2;
        if (need > Gmax) {
            // coarsen h in 2% steps so the kernel spectrum is rarely recomputed
            float hh = ext / (Gmax - 3);
            h = o.h * powf(1.02f, ceilf(logf(hh / o.h) / logf(1.02f)));
            need = (int)ceilf(ext / h) + 3;
        }
        int G = std::max(16, fft_size(need));
        if (G < F.G && G * 3 / 2 > F.G) G = F.G;  // hysteresis
        F.setup(G, h);
        GridP gp;
        float cx = 0.5f * (b.x + b.z), cy = 0.5f * (b.y + b.w);
        gp.x0 = cx - 0.5f * (G - 1) * h; gp.y0 = cy - 0.5f * (G - 1) * h; gp.inv_h = 1.f / h; gp.P = F.P;
        int64_t PP = (int64_t)F.P * F.P, PC = (int64_t)F.P * (F.P / 2 + 1);
        CK(cudaMemsetAsync(F.grid, 0, PP * 4));
        k_splat<<<nb, 256>>>(dY, dMass, N, gp, F.grid);
        CKFFT(cufftExecR2C(F.r2c, F.grid, F.spec));
        k_specmul<<<(PC + 255) / 256, 256>>>(F.spec, F.kspec, PC, F.ospec);
        CKFFT(cufftExecC2R(F.c2r3, F.ospec, F.out));
        CK(cudaMemsetAsync(dZ, 0, 8));
        k_gather<<<nb, 256>>>(dY, dMass, N, gp, h, F.out, dGp, dZ);
        k_attract<<<nb8, 256>>>(dY, dI, dP, roff, rsrc, rp, N, K, dA);
        if (o.check_grad > 0 && (it == o.ee_iters - 1 || it == total - 1)) {
            CK(cudaDeviceSynchronize());
            int nq = std::min<int64_t>(o.check_grad, N);
            std::mt19937 r2(7); std::vector<int> qs(nq); for (auto& q : qs) q = (int)(r2() % N);
            int* dQ; double* dO; CK(cudaMalloc(&dQ, nq * 4)); CK(cudaMalloc(&dO, nq * 3 * 8));
            CK(cudaMemcpy(dQ, qs.data(), nq * 4, cudaMemcpyHostToDevice));
            k_exact_rep<<<nq, 256>>>(dY, dMass, N, dQ, nq, dO);
            std::vector<double> ex(nq * 3); CK(cudaMemcpy(ex.data(), dO, nq * 24, cudaMemcpyDeviceToHost));
            std::vector<float2> gpv(N); CK(cudaMemcpy(gpv.data(), dGp, N * 8, cudaMemcpyDeviceToHost));
            std::vector<float> mv(N, 1.f); if (dMass) CK(cudaMemcpy(mv.data(), dMass, N * 4, cudaMemcpyDeviceToHost));
            double zh; CK(cudaMemcpy(&zh, dZ, 8, cudaMemcpyDeviceToHost));
            double num = 0, den = 0, zsum = 0;
            for (int q = 0; q < nq; q++) {
                double ax = -0.5 * gpv[qs[q]].x, ay = -0.5 * gpv[qs[q]].y;
                num += (ax - ex[q * 3 + 1]) * (ax - ex[q * 3 + 1]) + (ay - ex[q * 3 + 2]) * (ay - ex[q * 3 + 2]);
                den += ex[q * 3 + 1] * ex[q * 3 + 1] + ex[q * 3 + 2] * ex[q * 3 + 2];
                zsum += mv[qs[q]] * ex[q * 3];
            }
            LOG("grad check it %d: rel. error of repulsive force %.4f, Z grid %.4g vs sampled exact %.4g (h=%.3f, P=%d)",
                it, sqrt(num / den), zh, zsum / nq * N, h, F.P);
            cudaFree(dQ); cudaFree(dO);
        }
        k_update<<<nb, 256>>>(dY, dU, dG, dA, dGp, dMass, N, exag, dZ, lr, mom);
        if (it == 0) { size_t fr, tt; cudaMemGetInfo(&fr, &tt); LOG("GPU memory in use: %.1f GB of %.1f GB (incl. other processes)", (tt - fr) / 1e9, tt / 1e9); }
        if (it % 50 == 0 || it == total - 1) {
            CK(cudaDeviceSynchronize());
            double z; CK(cudaMemcpy(&z, dZ, 8, cudaMemcpyDeviceToHost));
            LOG("iter %4d  extent %8.2f  grid %5d  h %.3f  Z/N %.4g", it, ext, F.P, h, z / N);
        }
        (void)tf; (void)ta;
        if (it == o.ee_iters - 1) {  // reset momentum state like other implementations
            CK(cudaMemset(dU, 0, N * 8));
        }
    }
    CK(cudaDeviceSynchronize());
    F.release();
    cudaFree(dU); cudaFree(dG); cudaFree(dA); cudaFree(dGp); cudaFree(dBB); cudaFree(dZ);
    cudaFree(roff); cudaFree(rsrc); cudaFree(rp);
}

// ---------------------------------------------------------------------------------------------
// Label propagation on the kNN graph (weights p_{j|i}); half of the nodes update per sweep to avoid oscillation
// ---------------------------------------------------------------------------------------------
template <int K>
__global__ void k_lp(const int* __restrict__ I, const float* __restrict__ P, int64_t n, const int* lab, int* lab2,
                     uint64_t seed, int* changed) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cur = lab[i];
    if (splitmix(seed ^ (uint64_t)i) & 1) { lab2[i] = cur; return; }
    int L[K]; float W[K]; int m = 0;
    for (int j = 0; j < K; j++) {
        float w = P[i * K + j];
        if (w == 0.f) continue;
        int l = lab[I[i * K + j]];
        int t = 0; while (t < m && L[t] != l) t++;
        if (t == m) { L[m] = l; W[m] = w; m++; } else W[t] += w;
    }
    int best = cur; float bw = -1;
    for (int t = 0; t < m; t++) if (W[t] > bw || (W[t] == bw && L[t] == cur)) { bw = W[t]; best = L[t]; }
    lab2[i] = best;
    if (best != cur) atomicAdd(changed, 1);
}

static std::vector<int> label_propagation(const int* dI, const float* dP, int64_t N, int K, int sweeps, uint64_t seed) {
    int *a, *b, *dch;
    CK(cudaMalloc(&a, N * 4)); CK(cudaMalloc(&b, N * 4)); CK(cudaMalloc(&dch, 4));
    thrust::sequence(thrust::device_ptr<int>(a), thrust::device_ptr<int>(a + N));
    for (int s = 0; s < sweeps; s++) {
        CK(cudaMemset(dch, 0, 4));
        uint64_t sd = splitmix(seed + 0x1234 + s);
        int nb = (int)((N + 127) / 128);
        switch (K) {
            case 16: k_lp<16><<<nb, 128>>>(dI, dP, N, a, b, sd, dch); break;
            case 32: k_lp<32><<<nb, 128>>>(dI, dP, N, a, b, sd, dch); break;
            case 48: k_lp<48><<<nb, 128>>>(dI, dP, N, a, b, sd, dch); break;
            case 64: k_lp<64><<<nb, 128>>>(dI, dP, N, a, b, sd, dch); break;
            case 96: k_lp<96><<<nb, 128>>>(dI, dP, N, a, b, sd, dch); break;
        }
        std::swap(a, b);
        int ch; CK(cudaMemcpy(&ch, dch, 4, cudaMemcpyDeviceToHost));
        if (ch < N / 10000) break;
    }
    std::vector<int> lab(N);
    CK(cudaMemcpy(lab.data(), a, N * 4, cudaMemcpyDeviceToHost));
    cudaFree(a); cudaFree(b); cudaFree(dch);
    return lab;
}

// local structure diagnostics: MLE intrinsic dimension from the K distances, and mean overlap of the
// neighbour set with the neighbour sets of the first 8 neighbours
__global__ void k_localdim(const int* I, const float* D2, int64_t n, int K, float* idim, float* share) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float lk = 0.5f * logf(fmaxf(D2[i * K + K - 1], 1e-20f)), s = 0;
    for (int j = 0; j < K - 1; j++) s += lk - 0.5f * logf(fmaxf(D2[i * K + j], 1e-20f));
    idim[i] = s > 0 ? (K - 2) / s : 100.f;
    float sh = 0; int na = min(8, K);
    for (int a = 0; a < na; a++) {
        int nb = I[i * K + a]; int c = 0;
        for (int x = 0; x < K; x++) { int v = I[(int64_t)nb * K + x]; for (int y = 0; y < K; y++) if (I[i * K + y] == v) { c++; break; } }
        sh += c / (float)K;
    }
    share[i] = sh / na;
}

// average log-mass over the kNN neighbourhood so whole groups shrink together
__global__ void k_smooth_logm(const int* I, const float* P, int64_t n, int K, const float* in, float* out) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = in[i], w = 1.f;
    for (int j = 0; j < K; j++) if (P[i * K + j] > 0) { s += in[I[i * K + j]]; w += 1.f; }
    out[i] = s / w;
}

// ---------------------------------------------------------------------------------------------
// Density-adaptive masses: m_i = clamp((s_i / median s)^alpha, mass_min, 1), s_i = distance to the k-th neighbour
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// Saved model + projection of new points
// ---------------------------------------------------------------------------------------------
__global__ void k_place(const int* I, const float* W, int64_t n, int K, const float2* Yr, float2* out) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = 0, y = 0, ws = 0;
    for (int j = 0; j < K; j++) {
        float w = W[i * K + j];
        if (w <= 0) continue;
        float2 r = Yr[I[i * K + j]];
        x += w * r.x; y += w * r.y; ws += w;
    }
    out[i] = ws > 0 ? make_float2(x / ws, y / ws) : make_float2(NAN, NAN);
}

// Streams the inputs in batches: same transform + PCA as the reference, kNN against the reference with the
// reference's inverted file, then each point goes to the kernel-weighted mean of its neighbours' positions.
static void run_project(const Opts& o) {
    Model m;
    load_model(o.project, m);
    LOG("model: %lld reference points, %d -> %d dims, %d clusters", (long long)m.N, m.D, m.DP, m.nc);
    const int K = o.proj_k, DP = m.DP;
    int np = std::min(o.nprobe, m.nc);
    float *dX, *dC; float2* dYr; int *dOffs, *dProbes;
    CK(cudaMalloc(&dX, m.X.size() * 4)); CK(cudaMemcpy(dX, m.X.data(), m.X.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dYr, m.N * 8)); CK(cudaMemcpy(dYr, m.Y.data(), m.N * 8, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dC, m.C.size() * 4)); CK(cudaMemcpy(dC, m.C.data(), m.C.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dOffs, (m.nc + 1) * 4)); CK(cudaMemcpy(dOffs, m.offs.data(), (m.nc + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dProbes, (size_t)m.nc * np * 4));
    k_probes<<<(m.nc + 63) / 64, 64>>>(dC, m.nc, DP, np, dProbes);
    std::vector<float>().swap(m.X); std::vector<float>().swap(m.Y);
    cublasHandle_t blas; CKBLAS(cublasCreate(&blas));

    std::vector<std::string> files = expand_inputs(o);
    std::vector<std::vector<std::string>> batches(1);
    double acc = 0;
    for (auto& fn : files) {
        struct stat st; if (stat(fn.c_str(), &st) != 0) { fprintf(stderr, "cannot stat %s\n", fn.c_str()); exit(1); }
        if (acc > 0 && acc + st.st_size > o.batch_gb * 1e9) { batches.emplace_back(); acc = 0; }
        batches.back().push_back(fn); acc += st.st_size;
    }
    FILE* f = fopen(o.out.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", o.out.c_str()); exit(1); }
    std::vector<char> buf(1 << 20);
    setvbuf(f, buf.data(), _IOFBF, buf.size());
    fputs("id\tx\ty\tnn_dist\n", f);
    int64_t total = 0;
    double t0 = now();
    for (size_t b = 0; b < batches.size(); b++) {
        Data d;
        read_files(batches[b], o, d);
        if (d.D != m.D) { fprintf(stderr, "input has %d columns, model expects %d\n", d.D, m.D); exit(1); }
        const int64_t n = d.N;
        float* dQ = apply_pca(d, m.mode, m.mu, m.Vp, DP, blas);
        std::vector<std::vector<float>>().swap(d.chunks);
        // group queries by nearest reference centroid
        int *dA, *dPm;
        CK(cudaMalloc(&dA, n * 4)); CK(cudaMalloc(&dPm, n * 4));
        assign_dispatch(DP, dQ, n, dC, m.nc, dA);
        thrust::sequence(thrust::device_ptr<int>(dPm), thrust::device_ptr<int>(dPm + n));
        thrust::stable_sort_by_key(thrust::device_ptr<int>(dA), thrust::device_ptr<int>(dA + n), thrust::device_ptr<int>(dPm));
        float* dQs; CK(cudaMalloc(&dQs, (size_t)n * DP * 4));
        k_gather_rows<<<(n * DP + 255) / 256, 256>>>(dQ, dPm, n, DP, dQs);
        CK(cudaFree(dQ));
        std::vector<int> hA(n), pm(n);
        CK(cudaMemcpy(hA.data(), dA, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pm.data(), dPm, n * 4, cudaMemcpyDeviceToHost));
        std::vector<int4> tiles;
        for (int64_t s = 0; s < n;) {
            int64_t e = s; while (e < n && hA[e] == hA[s]) e++;
            for (int64_t t = s; t < e; t += QT) tiles.push_back(make_int4(hA[s], (int)t, (int)std::min<int64_t>(QT, e - t), 0));
            s = e;
        }
        int4* dTiles; CK(cudaMalloc(&dTiles, tiles.size() * sizeof(int4)));
        CK(cudaMemcpy(dTiles, tiles.data(), tiles.size() * sizeof(int4), cudaMemcpyHostToDevice));
        int* dI; float* dDd;
        CK(cudaMalloc(&dI, (size_t)n * K * 4)); CK(cudaMalloc(&dDd, (size_t)n * K * 4));
        knn_dispatch(DP, K, dX, dQs, dOffs, dProbes, np, dTiles, (int)tiles.size(), dI, dDd);
        std::vector<float> nn(n);
        CK(cudaMemcpy2D(nn.data(), 4, dDd, K * 4, 4, n, cudaMemcpyDeviceToHost));
        k_perplexity<<<(n + 255) / 256, 256>>>(dDd, dI, n, K, logf(o.proj_perp), false, 1.f, nullptr);
        float2* dOut; CK(cudaMalloc(&dOut, n * 8));
        k_place<<<(n + 255) / 256, 256>>>(dI, dDd, n, K, dYr, dOut);
        std::vector<float2> Yq(n), Yo(n);
        CK(cudaMemcpy(Yq.data(), dOut, n * 8, cudaMemcpyDeviceToHost));
        std::vector<float> nno(n);
        for (int64_t i = 0; i < n; i++) { Yo[pm[i]] = Yq[i]; nno[pm[i]] = sqrtf(nn[i]); }
        for (int64_t i = 0; i < n; i++) {
            fwrite(d.ids.data() + d.idoff[i], 1, d.idoff[i + 1] - d.idoff[i], f);
            fprintf(f, "\t%.4f\t%.4f\t%.4g\n", Yo[i].x, Yo[i].y, nno[i]);
        }
        cudaFree(dA); cudaFree(dPm); cudaFree(dQs); cudaFree(dTiles); cudaFree(dI); cudaFree(dDd); cudaFree(dOut);
        total += n;
        LOG("batch %zu/%zu: %lld points (%lld total, %.0f points/s)", b + 1, batches.size(), (long long)n, (long long)total, total / (now() - t0));
    }
    fclose(f);
    LOG("wrote %s", o.out.c_str());
}

int main(int argc, char** argv) {
    Opts o;
    parse_args(argc, argv, o);
    if (o.nprobe < 0) o.nprobe = 32;
    if (o.refine < 0) o.refine = 1;
    if (!o.project.empty()) { run_project(o); return 0; }

    Data d;
    read_files(expand_inputs(o), o, d);
    if (!o.write_fsb.empty()) { write_fsb_file(d, o.write_fsb); return 0; }
    const int64_t N = d.N;
    int DP = std::max(8, (std::min(o.pca, d.D) + 7) / 8 * 8);
    if (DP > 48) DP = 64; else if (DP > 32) DP = 48;
    cublasHandle_t blas; CKBLAS(cublasCreate(&blas));
    Model model;
    float* dX = transform_pca(d, o, DP, blas, model.mu, model.Vp);
    std::vector<std::vector<float>>().swap(d.chunks);
    LOG("transform + PCA done (%d dims)", DP);

    int K = o.k;
    std::vector<int> perm;
    int* dI; float* dD;
    knn(dX, N, DP, o, K, perm, dI, dD, model.C, model.offs);

    std::vector<float> rk(N), r1(N);
    CK(cudaMemcpy2D(rk.data(), 4, dD + (K - 1), K * 4, 4, N, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy2D(r1.data(), 4, dD, K * 4, 4, N, cudaMemcpyDeviceToHost));
    std::vector<float> idim, share;
    if (!o.diag.empty() || o.dedup > 0) {
        float *a, *b; CK(cudaMalloc(&a, N * 4)); CK(cudaMalloc(&b, N * 4));
        k_localdim<<<(N + 127) / 128, 128>>>(dI, dD, N, K, a, b);
        idim.resize(N); share.resize(N);
        CK(cudaMemcpy(idim.data(), a, N * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(share.data(), b, N * 4, cudaMemcpyDeviceToHost));
        cudaFree(a); cudaFree(b);
    }
    float* dSig; CK(cudaMalloc(&dSig, N * 4));
    k_perplexity<<<(N + 255) / 256, 256>>>(dD, dI, N, K, logf(o.perplexity), o.uniform, 0.5f / (float)N, dSig);
    CK(cudaDeviceSynchronize());
    LOG("affinities done (perplexity %.1f%s)", o.perplexity, o.uniform ? ", uniform" : "");

    // masses
    float* dMass = nullptr;
    std::vector<int> groups;
    std::vector<float> mass_out;
    if (o.shrink > 0 || !o.groups_out.empty()) {
        groups = label_propagation(dI, dD, N, K, o.lp_sweeps, o.seed);
        std::vector<int> cnt(N, 0);
        for (auto g : groups) cnt[g]++;
        std::vector<int> sizes; for (auto c : cnt) if (c) sizes.push_back(c);
        std::sort(sizes.rbegin(), sizes.rend());
        LOG("label propagation: %zu groups, largest %d %d %d %d %d", sizes.size(), sizes[0], sizes.size() > 1 ? sizes[1] : 0,
            sizes.size() > 2 ? sizes[2] : 0, sizes.size() > 3 ? sizes[3] : 0, sizes.size() > 4 ? sizes[4] : 0);
        if (o.shrink > 0) {
            double ref = o.shrink_ref > 0 ? o.shrink_ref : std::max(100.0, N * 1e-3);
            std::vector<float> m(N);
            double ms = 0;
            for (int64_t i = 0; i < N; i++) {
                double c = cnt[groups[i]];
                m[i] = (float)std::max<double>(o.mass_min, c > ref ? pow(ref / c, o.shrink) : 1.0);
                ms += m[i];
            }
            LOG("shrink: groups above %.0f points get mass (ref/n)^%.2f, mean mass %.3f", ref, o.shrink, ms / N);
            CK(cudaMalloc(&dMass, N * 4));
            CK(cudaMemcpy(dMass, m.data(), N * 4, cudaMemcpyHostToDevice));
        }
    }
    if (o.dedup > 0) {
        // redundancy: the kNN graph is chain-like (neighbours share most of their neighbours), as for overlapping
        // reads of very high coverage genomes. These form worms in t-SNE; shrinking their mass compacts them.
        if (share.empty()) {
            float *a, *b; CK(cudaMalloc(&a, N * 4)); CK(cudaMalloc(&b, N * 4));
            k_localdim<<<(N + 127) / 128, 128>>>(dI, dD, N, K, a, b);  // only share is used
            share.resize(N); CK(cudaMemcpy(share.data(), b, N * 4, cudaMemcpyDeviceToHost));
            cudaFree(a); cudaFree(b);
        }
        std::vector<float> t(share);
        std::nth_element(t.begin(), t.begin() + N / 2, t.end());
        float med = std::max(t[N / 2], 1e-3f);
        std::vector<float> lm(N);
        for (int64_t i = 0; i < N; i++) {
            float z = logf(std::max(share[i], 1e-3f) / med);
            lm[i] = -o.dedup * std::max(0.f, z - o.dedup_margin);
        }
        float *a, *b;
        CK(cudaMalloc(&a, N * 4)); CK(cudaMalloc(&b, N * 4));
        CK(cudaMemcpy(a, lm.data(), N * 4, cudaMemcpyHostToDevice));
        for (int r = 0; r < 3; r++) { k_smooth_logm<<<(N + 255) / 256, 256>>>(dI, dD, N, K, a, b); std::swap(a, b); }
        CK(cudaMemcpy(lm.data(), a, N * 4, cudaMemcpyDeviceToHost));
        cudaFree(a); cudaFree(b);
        std::vector<float> m(N);
        double ms = 0; int64_t nsh = 0;
        for (int64_t i = 0; i < N; i++) { m[i] = std::max(o.mass_min, expf(lm[i])); ms += m[i]; nsh += m[i] < 0.5f; }
        if (dMass) {  // combine with --shrink
            std::vector<float> m0(N); CK(cudaMemcpy(m0.data(), dMass, N * 4, cudaMemcpyDeviceToHost));
            for (int64_t i = 0; i < N; i++) m[i] = std::max(o.mass_min, m[i] * m0[i]);
        } else CK(cudaMalloc(&dMass, N * 4));
        CK(cudaMemcpy(dMass, m.data(), N * 4, cudaMemcpyHostToDevice));
        LOG("dedup: mean mass %.3f, %.1f%% of points below 0.5", ms / N, 100.0 * nsh / N);
        mass_out = m;
    }
    if (!o.mass_file.empty()) {
        std::vector<float> m(N, 1.f), mi;
        FILE* f = fopen(o.mass_file.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", o.mass_file.c_str()); exit(1); }
        float v; while (fscanf(f, "%f", &v) == 1) mi.push_back(v); fclose(f);
        if ((int64_t)mi.size() != N) { fprintf(stderr, "mass file has %zu values, need %lld\n", mi.size(), (long long)N); exit(1); }
        for (int64_t i = 0; i < N; i++) m[i] = mi[perm[i]];
        if (!dMass) CK(cudaMalloc(&dMass, N * 4));
        CK(cudaMemcpy(dMass, m.data(), N * 4, cudaMemcpyHostToDevice));
    }

    // init: first two PCs, scaled to std 1e-4 (dX is sorted N x DP)
    std::vector<float> pc(N * 2);
    CK(cudaMemcpy2D(pc.data(), 8, dX, DP * 4, 8, N, cudaMemcpyDeviceToHost));
    double sd = 0; for (int64_t i = 0; i < N; i++) sd += (double)pc[2 * i] * pc[2 * i];
    sd = sqrt(sd / N);
    float sc = (float)(1e-4 / std::max(sd, 1e-12));
    for (auto& v : pc) v *= sc;
    if (!o.save_model.empty()) {
        model.X.resize((size_t)N * DP);
        CK(cudaMemcpy(model.X.data(), dX, model.X.size() * 4, cudaMemcpyDeviceToHost));
    }
    CK(cudaFree(dX));
    float2* dY; CK(cudaMalloc(&dY, N * 8));
    CK(cudaMemcpy(dY, pc.data(), N * 8, cudaMemcpyHostToDevice));

    double t_opt = now();
    run_tsne(dY, N, K, dI, dD, dMass, o);
    LOG("optimisation done in %.1fs", now() - t_opt);
    { size_t fr, tt; cudaMemGetInfo(&fr, &tt); LOG("GPU memory in use at end: %.1f GB of %.1f GB (incl. other processes)", (tt - fr) / 1e9, tt / 1e9); }

    std::vector<float2> Y(N);
    CK(cudaMemcpy(Y.data(), dY, N * 8, cudaMemcpyDeviceToHost));
    if (!o.save_model.empty()) {
        model.N = N; model.D = d.D; model.DP = DP; model.nc = (int)model.C.size() / DP;
        model.mode = o.transform == "clr" ? 0 : o.transform == "sqrt" ? 1 : 2;
        model.Y.resize((size_t)N * 2);
        memcpy(model.Y.data(), Y.data(), N * 8);
        save_model(o.save_model, model);
        std::vector<float>().swap(model.X);
        LOG("saved model %s", o.save_model.c_str());
    }
    std::vector<float2> Yo(N);
    for (int64_t i = 0; i < N; i++) Yo[perm[i]] = Y[i];
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
    if (!o.diag.empty()) {
        FILE* g = fopen(o.diag.c_str(), "w");
        std::vector<float> a(N), b(N);
        for (int64_t i = 0; i < N; i++) { a[perm[i]] = sqrtf(rk[i]); b[perm[i]] = sqrtf(r1[i]); }
        std::vector<float> mm(N, 1.f), id2(N), sh2(N);
        if (!mass_out.empty()) for (int64_t i = 0; i < N; i++) mm[perm[i]] = mass_out[i];
        for (int64_t i = 0; i < N; i++) { id2[perm[i]] = idim[i]; sh2[perm[i]] = share[i]; }
        fputs("id\trk\tr1\tlen_est\tmass\tidim\tshare\n", g);
        for (int64_t i = 0; i < N; i++) {
            fwrite(d.ids.data() + d.idoff[i], 1, d.idoff[i + 1] - d.idoff[i], g);
            fprintf(g, "\t%.4g\t%.4g\t%.0f\t%.3g\t%.3g\t%.3g\n", a[i], b[i], d.len_est[i], mm[i], id2[i], sh2[i]);
        }
        fclose(g);
    }
    if (!o.groups_out.empty()) {
        FILE* g = fopen(o.groups_out.c_str(), "w");
        std::vector<int> go(N);
        for (int64_t i = 0; i < N; i++) go[perm[i]] = perm[groups[i]];
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
