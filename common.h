// Shared host code for fastsne (CUDA) and fastsne_cpu (OpenMP): options, input parsing (TSV / .fsb),
// PCA eigen-decomposition, FFT sizes and the model file format.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double T0 = now();
#define LOG(...) do { fprintf(stderr, "[%7.2fs] ", now() - T0); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } while (0)

// ---------------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------------
struct Opts {
    std::vector<std::string> inputs;
    std::string out = "tsne.tsv";
    std::string transform = "clr";
    int pca = 32;
    int k = 48;
    float perplexity = -1;      // default k/3
    bool uniform = false;
    int clusters = 0;           // 0 = auto
    int nprobe = -1;            // -1: 32
    int refine = -1;            // NN-descent rounds (-1: 1)
    int refine_a = 16;          // neighbours (and their neighbours) tried per round
    int iters = 750;
    int ee_iters = 250;
    float ee = 12.f;
    float exag = 1.f;           // exaggeration after the early phase
    float lr = -1;              // default N/ee
    float h = 0.25f;            // grid spacing in embedding units
    int max_grid = 8192;        // max padded FFT size
    double sample = 1.0;        // keep each row with this probability
    uint64_t seed = 1;
    int threads = 0;
    int check_knn = 0;
    int check_grad = 0;
    float shrink = 0.f;         // group-size mass exponent (0 = off)
    float dedup = 0.f;          // redundancy mass exponent (0 = off)
    float dedup_margin = 0.3f;  // only reads with log(share/median) above margin are shrunk
    double shrink_ref = 0;      // groups larger than this are shrunk (default max(100, N/1000))
    int lp_sweeps = 30;
    std::string groups_out;     // write label-propagation groups
    std::string diag;           // write per-point kNN radius diagnostics
    float mass_min = 0.02f;
    std::string mass_file;      // optional per-row masses (one float per line, input order)
    bool write_pca = false;
    int64_t pca_sample = 1000000;  // CPU: rows used for the PCA covariance
    int cluster_size = -1;      // CPU: target kNN fine-cluster size
    std::string write_fsb;      // convert the inputs to one .fsb file and exit
    std::string save_model;     // write the reference (PCA, index, coordinates, embedding) for --project
    std::string project;        // projection mode: place inputs onto a saved model
    int proj_k = 16;
    float proj_perp = 5.f;
    double batch_gb = 8.0;      // text per projection batch
};

// ---------------------------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------------------------
static double POW10[700];
static void init_pow10() { for (int i = 0; i < 700; i++) POW10[i] = pow(10.0, i - 350); }

static inline bool isdig(char c) { return c >= '0' && c <= '9'; }

// returns pointer after the number, or nullptr if no number
static inline const char* parse_float(const char* p, const char* end, float* out) {
    while (p < end && (*p == ' ' || *p == '\t')) p++;
    if (p >= end) return nullptr;
    bool neg = false;
    if (*p == '-') { neg = true; p++; } else if (*p == '+') p++;
    uint64_t mant = 0; int nd = 0, e10 = 0; bool any = false;
    while (p < end && isdig(*p)) { any = true; if (nd < 18) { mant = mant * 10 + (*p - '0'); if (mant) nd++; } else e10++; p++; }
    if (p < end && *p == '.') {
        p++;
        while (p < end && isdig(*p)) { any = true; if (nd < 18) { mant = mant * 10 + (*p - '0'); if (mant) nd++; e10--; } p++; }
    }
    if (!any) return nullptr;  // header, nan, garbage
    if (p < end && (*p == 'e' || *p == 'E')) {
        p++; bool en = false;
        if (p < end && *p == '-') { en = true; p++; } else if (p < end && *p == '+') p++;
        int e = 0; while (p < end && isdig(*p)) { e = e * 10 + (*p - '0'); p++; }
        e10 += en ? -e : e;
    }
    double v = (double)mant;
    if (e10 < -340) v = 0; else if (e10 > 340) v = INFINITY; else v *= POW10[e10 + 350];
    *out = (float)(neg ? -v : v);
    return p;
}

struct Unit { int file; int64_t start, end; };
struct UnitResult { std::vector<char> ids; std::vector<uint32_t> idlen; std::vector<float> vals; int64_t rows = 0, bad = 0; };

HD static inline uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}

static void parse_unit(const std::string& path, const Unit& u, int* Dp, std::atomic<int>* Dset,
                       double sample, uint64_t seed, int unit_id, UnitResult& R) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path.c_str()); return; }
    int64_t s = u.start > 0 ? u.start - 1 : 0;
    int64_t want = u.end - s;
    std::vector<char> buf;
    int64_t got = 0;
    // read [s, end) plus overrun until a newline after end
    int64_t cap = want + (1 << 16);
    buf.resize(cap);
    for (;;) {
        while (got < (int64_t)buf.size()) {
            ssize_t r = pread(fd, buf.data() + got, buf.size() - got, s + got);
            if (r <= 0) break;
            got += r;
        }
        if (got < (int64_t)buf.size()) break;  // EOF
        // check for newline after 'want'
        bool found = false;
        for (int64_t i = want; i < got; i++) if (buf[i] == '\n') { found = true; break; }
        if (found) break;
        buf.resize(buf.size() * 2);
    }
    close(fd);
    const char* b = buf.data();
    const char* e = b + got;
    const char* p = b;
    if (u.start > 0) {
        // line starts in this unit only if preceded by '\n'
        if (*p != '\n') { while (p < e && *p != '\n') p++; }
        p++;
    }
    const char* limit = b + want;  // lines must start before this
    std::vector<float> row;
    int64_t lineno = 0;
    while (p < e && p < limit) {
        const char* ls = p;
        const char* le = (const char*)memchr(p, '\n', e - p);
        if (!le) le = e;
        p = le + 1;
        const char* lend = le;
        if (lend > ls && lend[-1] == '\r') lend--;
        if (lend == ls) continue;
        lineno++;
        if (sample < 1.0) {
            uint64_t hsh = splitmix(seed ^ ((uint64_t)unit_id << 32) ^ (uint64_t)lineno);
            if ((hsh >> 11) * (1.0 / 9007199254740992.0) >= sample) continue;
        }
        const char* t = (const char*)memchr(ls, '\t', lend - ls);
        if (!t) { R.bad++; continue; }
        row.clear();
        const char* q = t + 1;
        float v;
        bool ok = true;
        while (q < lend) {
            const char* r = parse_float(q, lend, &v);
            if (!r) break;
            if (r < lend && *r != '\t' && *r != ' ') { ok = false; break; }
            row.push_back(v);
            q = r;
        }
        if (!ok || row.empty()) { R.bad++; continue; }  // header or garbage
        int D = *Dp;
        if (D == 0) {
            int expected = 0;
            if (Dset->compare_exchange_strong(expected, (int)row.size())) *Dp = (int)row.size();
            D = Dset->load();
            *Dp = D;
        }
        if ((int)row.size() != D) { R.bad++; continue; }
        R.ids.insert(R.ids.end(), ls, t);
        R.idlen.push_back((uint32_t)(t - ls));
        R.vals.insert(R.vals.end(), row.begin(), row.end());
        R.rows++;
    }
}

static bool is_fsb(const std::string& p) { return p.size() > 4 && p.compare(p.size() - 4, 4, ".fsb") == 0; }

// binary input written by tools/duckdb_export.py: "FSB1", int32 D, int64 N, N*D float32, ids joined by '\n'
static void parse_fsb(const std::string& path, std::atomic<int>* Dset, double sample, uint64_t seed, int unit_id, UnitResult& R) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return; }
    char mg[4]; int32_t D; int64_t N;
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "FSB1", 4) || fread(&D, 4, 1, f) != 1 || fread(&N, 8, 1, f) != 1) {
        fprintf(stderr, "%s: not an FSB1 file\n", path.c_str()); fclose(f); return;
    }
    int expected = 0;
    if (!Dset->compare_exchange_strong(expected, D) && expected != D) {
        fprintf(stderr, "%s has %d columns, expected %d; skipped\n", path.c_str(), D, expected); fclose(f); return;
    }
    std::vector<float> X((size_t)N * D);
    if (fread(X.data(), 4, X.size(), f) != X.size()) { fprintf(stderr, "%s truncated\n", path.c_str()); fclose(f); return; }
    std::vector<char> ids;
    char buf[1 << 16]; size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) ids.insert(ids.end(), buf, buf + r);
    fclose(f);
    size_t p = 0;
    for (int64_t i = 0; i < N; i++) {
        size_t e = p; while (e < ids.size() && ids[e] != '\n') e++;
        bool keep = true;
        if (sample < 1.0) {
            uint64_t hsh = splitmix(seed ^ ((uint64_t)unit_id << 32) ^ (uint64_t)(i + 1));
            keep = (hsh >> 11) * (1.0 / 9007199254740992.0) < sample;
        }
        if (keep) {
            R.ids.insert(R.ids.end(), ids.begin() + p, ids.begin() + e);
            R.idlen.push_back((uint32_t)(e - p));
            R.vals.insert(R.vals.end(), X.begin() + (size_t)i * D, X.begin() + (size_t)(i + 1) * D);
            R.rows++;
        }
        p = e + 1;
    }
}

struct Data {
    int64_t N = 0; int D = 0;
    // rows stay in the per-unit parse buffers (no second full copy); cstart[c] = first row of chunk c
    std::vector<std::vector<float>> chunks;
    std::vector<int64_t> cstart;
    // copies rows [s, s+m) to/from a contiguous buffer
    void get_rows(int64_t s, int64_t m, float* dst) const { rows_io(s, m, dst, false); }
    void put_rows(int64_t s, int64_t m, const float* src) { rows_io(s, m, const_cast<float*>(src), true); }
    void rows_io(int64_t s, int64_t m, float* buf, bool put) const {
        size_t c = std::upper_bound(cstart.begin(), cstart.end(), s) - cstart.begin() - 1;
        while (m > 0) {
            int64_t off = s - cstart[c], n = std::min<int64_t>(m, cstart[c + 1] - s);
            float* p = const_cast<float*>(chunks[c].data()) + off * D;
            if (put) memcpy(p, buf, n * D * 4); else memcpy(buf, p, n * D * 4);
            buf += n * D; s += n; m -= n; c++;
        }
    }
    std::vector<char> ids;
    std::vector<uint64_t> idoff;   // N+1
    std::vector<float> len_est;    // 1 / smallest nonzero frequency (~ read length for raw TNF of reads)
};

static std::vector<std::string> expand_inputs(const Opts& o) {
    std::vector<std::string> files;
    for (auto& in : o.inputs) {
        if (!in.empty() && in[0] == '@') {
            FILE* f = fopen(in.c_str() + 1, "r");
            if (!f) { fprintf(stderr, "cannot open list %s\n", in.c_str() + 1); exit(1); }
            char line[8192];
            while (fgets(line, sizeof line, f)) {
                size_t L = strlen(line);
                while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
                if (L) files.push_back(line);
            }
            fclose(f);
        } else files.push_back(in);
    }
    return files;
}

static void read_files(const std::vector<std::string>& files, const Opts& o, Data& d) {
    init_pow10();
    const int64_t UNIT = 32ll << 20;
    std::vector<Unit> units;
    int64_t total = 0;
    for (int i = 0; i < (int)files.size(); i++) {
        struct stat st;
        if (stat(files[i].c_str(), &st) != 0) { fprintf(stderr, "cannot stat %s\n", files[i].c_str()); exit(1); }
        int64_t sz = st.st_size; total += sz;
        if (is_fsb(files[i])) { units.push_back({i, -1, -1}); continue; }  // one unit per binary file
        for (int64_t s = 0; s < sz; s += UNIT) units.push_back({i, s, std::min(sz, s + UNIT)});
    }
    LOG("reading %zu files, %.2f GB, %zu units", files.size(), total / 1e9, units.size());
    std::vector<UnitResult> res(units.size());
    std::atomic<size_t> next(0);
    std::atomic<int> Dset(0);
    int nth = o.threads > 0 ? o.threads : (int)std::thread::hardware_concurrency();
    std::vector<std::thread> th;
    std::vector<int> Dloc(nth, 0);
    for (int t = 0; t < nth; t++) th.emplace_back([&, t]() {
        for (;;) {
            size_t u = next++;
            if (u >= units.size()) break;
            if (units[u].start < 0) parse_fsb(files[units[u].file], &Dset, o.sample, o.seed, (int)u, res[u]);
            else parse_unit(files[units[u].file], units[u], &Dloc[t], &Dset, o.sample, o.seed, (int)u, res[u]);
        }
    });
    for (auto& t : th) t.join();
    d.D = Dset.load();
    int64_t N = 0, bad = 0, idbytes = 0;
    for (auto& r : res) { N += r.rows; bad += r.bad; idbytes += r.ids.size(); }
    d.N = N;
    LOG("parsed %lld rows x %d cols (%lld lines skipped)", (long long)N, d.D, (long long)bad);
    if (N == 0) { fprintf(stderr, "no data\n"); exit(1); }
    d.ids.resize(idbytes);
    d.idoff.resize(N + 1);
    int64_t r0 = 0, b0 = 0;
    for (auto& r : res) {
        memcpy(d.ids.data() + b0, r.ids.data(), r.ids.size());
        for (int64_t i = 0; i < r.rows; i++) { d.idoff[r0 + i] = b0; b0 += r.idlen[i]; }
        if (r.rows) {
            d.cstart.push_back(r0);
            r.vals.shrink_to_fit();
            d.chunks.push_back(std::move(r.vals));
        }
        r0 += r.rows;
        std::vector<char>().swap(r.ids); std::vector<uint32_t>().swap(r.idlen);
    }
    d.cstart.push_back(N);
    d.idoff[N] = b0;
    d.len_est.resize(N);
    {
        std::vector<std::thread> th;
        int nth = std::max(1u, std::thread::hardware_concurrency());
        for (int t = 0; t < nth; t++) th.emplace_back([&, t]() {
            for (size_t c = t; c < d.chunks.size(); c += nth) {
                for (int64_t i = d.cstart[c]; i < d.cstart[c + 1]; i++) {
                    const float* x = d.chunks[c].data() + (i - d.cstart[c]) * d.D;
                    float mn = 1e30f;
                    for (int j = 0; j < d.D; j++) { float v = x[j]; if (v > 0 && v < mn) mn = v; }
                    d.len_est[i] = mn < 1e30f ? 1.f / mn : 0.f;
                }
            }
        });
        for (auto& x : th) x.join();
    }
}


// symmetric Jacobi eigen-decomposition, A (n x n, row-major) destroyed; V columns = eigenvectors
static void jacobi_eigen(std::vector<double>& A, int n, std::vector<double>& V, std::vector<double>& ev) {
    V.assign((size_t)n * n, 0); for (int i = 0; i < n; i++) V[(size_t)i * n + i] = 1;
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0;
        for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) off += A[(size_t)p * n + q] * A[(size_t)p * n + q];
        if (off < 1e-22) break;
        for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) {
            double apq = A[(size_t)p * n + q];
            if (fabs(apq) < 1e-300) continue;
            double app = A[(size_t)p * n + p], aqq = A[(size_t)q * n + q];
            double theta = (aqq - app) / (2 * apq);
            double t = (theta >= 0 ? 1 : -1) / (fabs(theta) + sqrt(theta * theta + 1));
            double c = 1 / sqrt(t * t + 1), s = t * c;
            for (int k = 0; k < n; k++) {
                double akp = A[(size_t)k * n + p], akq = A[(size_t)k * n + q];
                A[(size_t)k * n + p] = c * akp - s * akq; A[(size_t)k * n + q] = s * akp + c * akq;
            }
            for (int k = 0; k < n; k++) {
                double apk = A[(size_t)p * n + k], aqk = A[(size_t)q * n + k];
                A[(size_t)p * n + k] = c * apk - s * aqk; A[(size_t)q * n + k] = s * apk + c * aqk;
            }
            for (int k = 0; k < n; k++) {
                double vkp = V[(size_t)k * n + p], vkq = V[(size_t)k * n + q];
                V[(size_t)k * n + p] = c * vkp - s * vkq; V[(size_t)k * n + q] = s * vkp + c * vkq;
            }
        }
    }
    ev.resize(n); for (int i = 0; i < n; i++) ev[i] = A[(size_t)i * n + i];
}


static int fft_size(int need) {
    // smallest 2^a 3^b >= need
    int best = 1 << 30;
    for (int64_t p2 = 1; p2 < (1 << 30); p2 *= 2)
        for (int64_t v = p2; v < (1 << 30); v *= 3) if (v >= need && v < best) best = (int)v;
    return best;
}


struct Model {
    int64_t N = 0; int D = 0, DP = 0, nc = 0, mode = 0;
    std::vector<float> mu, Vp, C, X, Y;   // X: N x DP (cluster order), Y: N x 2 (same order)
    std::vector<int> offs;
};
static const char MODEL_MAGIC[8] = {'F', 'S', 'N', 'E', 'M', 'D', 'L', '1'};

static void wr(FILE* f, const void* p, size_t n) { if (n && fwrite(p, 1, n, f) != n) { fprintf(stderr, "write error\n"); exit(1); } }
static void rd(FILE* f, void* p, size_t n) { if (n && fread(p, 1, n, f) != n) { fprintf(stderr, "model file truncated\n"); exit(1); } }

static void save_model(const std::string& path, const Model& m) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
    wr(f, MODEL_MAGIC, 8);
    int64_t hdr[5] = {m.N, m.D, m.DP, m.nc, m.mode};
    wr(f, hdr, sizeof hdr);
    wr(f, m.mu.data(), m.mu.size() * 4); wr(f, m.Vp.data(), m.Vp.size() * 4); wr(f, m.C.data(), m.C.size() * 4);
    wr(f, m.offs.data(), m.offs.size() * 4); wr(f, m.X.data(), m.X.size() * 4); wr(f, m.Y.data(), m.Y.size() * 4);
    fclose(f);
}

static void load_model(const std::string& path, Model& m) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    char mg[8]; rd(f, mg, 8);
    if (memcmp(mg, MODEL_MAGIC, 8)) { fprintf(stderr, "%s is not a fastsne model\n", path.c_str()); exit(1); }
    int64_t hdr[5]; rd(f, hdr, sizeof hdr);
    m.N = hdr[0]; m.D = (int)hdr[1]; m.DP = (int)hdr[2]; m.nc = (int)hdr[3]; m.mode = (int)hdr[4];
    m.mu.resize(m.D); m.Vp.resize((size_t)m.D * m.DP); m.C.resize((size_t)m.nc * m.DP); m.offs.resize(m.nc + 1);
    m.X.resize((size_t)m.N * m.DP); m.Y.resize((size_t)m.N * 2);
    rd(f, m.mu.data(), m.mu.size() * 4); rd(f, m.Vp.data(), m.Vp.size() * 4); rd(f, m.C.data(), m.C.size() * 4);
    rd(f, m.offs.data(), m.offs.size() * 4); rd(f, m.X.data(), m.X.size() * 4); rd(f, m.Y.data(), m.Y.size() * 4);
    fclose(f);
}


static void usage() {
    fprintf(stderr,
        "fastsne -i input.tsv [-i more.tsv | -i @filelist] -o out.tsv [options]\n"
        "  input rows: id<TAB>v1<TAB>v2... (header lines are skipped)\n"
        "  --transform clr|sqrt|none   per-row transform (clr)\n"
        "  --pca N            PCA dimensions (32; 8..64)\n"
        "  --k N              neighbours (48; 16,32,48,64,96)\n"
        "  --perplexity P     (k/3)    --uniform   uniform kNN affinities\n"
        "  --clusters N --nprobe N     kNN inverted file (N/1000, 32)\n"
        "  --refine R --refine-a A     NN-descent rounds over A x A neighbours of neighbours (1, 16)\n"
        "  --iters N --ee-iters N --ee X  (750, 250, 12)\n"
        "  --exag X           exaggeration after the early phase (1)\n"
        "  --lr X             learning rate (N/12)\n"
        "  --h X              repulsion grid spacing (0.25)   --max-grid N (8192)\n"
        "  --shrink G         shrink big homogeneous groups: kNN label-propagation groups with n > ref points\n"
        "                     get repulsion mass (ref/n)^G, so their area grows like n^(1-G) (0 = off; try 0.5)\n"
        "  --dedup A          shrink chain-like redundant groups (neighbours share most of their neighbours, e.g.\n"
        "                     overlapping reads of very high coverage genomes, which t-SNE draws as worms):\n"
        "                     mass exp(-A * max(0, log(share/median) - T)), smoothed over neighbours (0 = off; try 2)\n"
        "  --dedup-margin T   (0.3)\n"
        "  --shrink-ref N     (max(100, N/1000))   --mass-min M (0.02)   --lp-sweeps N (30)\n"
        "  --groups FILE      write label-propagation group per id\n"
        "  --mass-file F      per-row masses, one per line in input order\n"
        "  --sample F         keep each row with probability F\n"
        "  --seed N --threads N --check-knn NQ --check-grad NQ\n"
        "  --save-model FILE  save the reference for later projection\n"
        "  fastsne_cpu only: --pca-sample N (1000000) --cluster-size N (kNN fine clusters)\n"
        "  --write-fsb FILE   convert all inputs (after --sample) to one binary .fsb file and exit\n"
        "projection: fastsne --project model.bin -i @files.txt -o placed.tsv [--proj-k 16] [--proj-perp 5]\n"
        "            [--nprobe 32] [--batch-gb 8] [--sample F]; output id x y nn_dist\n");
    exit(1);
}


static void parse_args(int argc, char** argv, Opts& o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "-i") o.inputs.push_back(nx());
        else if (a == "-o") o.out = nx();
        else if (a == "--transform") o.transform = nx();
        else if (a == "--pca") o.pca = atoi(nx().c_str());
        else if (a == "--k") o.k = atoi(nx().c_str());
        else if (a == "--perplexity") o.perplexity = atof(nx().c_str());
        else if (a == "--uniform") o.uniform = true;
        else if (a == "--clusters") o.clusters = atoi(nx().c_str());
        else if (a == "--nprobe") o.nprobe = atoi(nx().c_str());
        else if (a == "--refine") o.refine = atoi(nx().c_str());
        else if (a == "--refine-a") o.refine_a = atoi(nx().c_str());
        else if (a == "--iters") o.iters = atoi(nx().c_str());
        else if (a == "--ee-iters") o.ee_iters = atoi(nx().c_str());
        else if (a == "--ee") o.ee = atof(nx().c_str());
        else if (a == "--exag") o.exag = atof(nx().c_str());
        else if (a == "--lr") o.lr = atof(nx().c_str());
        else if (a == "--h") o.h = atof(nx().c_str());
        else if (a == "--max-grid") o.max_grid = atoi(nx().c_str());
        else if (a == "--sample") o.sample = atof(nx().c_str());
        else if (a == "--seed") o.seed = strtoull(nx().c_str(), 0, 10);
        else if (a == "--threads") o.threads = atoi(nx().c_str());
        else if (a == "--check-knn") o.check_knn = atoi(nx().c_str());
        else if (a == "--check-grad") o.check_grad = atoi(nx().c_str());
        else if (a == "--shrink") o.shrink = atof(nx().c_str());
        else if (a == "--dedup") o.dedup = atof(nx().c_str());
        else if (a == "--dedup-margin") o.dedup_margin = atof(nx().c_str());
        else if (a == "--shrink-ref") o.shrink_ref = atof(nx().c_str());
        else if (a == "--lp-sweeps") o.lp_sweeps = atoi(nx().c_str());
        else if (a == "--groups") o.groups_out = nx();
        else if (a == "--diag") o.diag = nx();
        else if (a == "--mass-min") o.mass_min = atof(nx().c_str());
        else if (a == "--mass-file") o.mass_file = nx();
        else if (a == "--write-pca") o.write_pca = true;
        else if (a == "--save-model") o.save_model = nx();
        else if (a == "--project") o.project = nx();
        else if (a == "--proj-k") o.proj_k = atoi(nx().c_str());
        else if (a == "--proj-perp") o.proj_perp = atof(nx().c_str());
        else if (a == "--batch-gb") o.batch_gb = atof(nx().c_str());
        else if (a == "--pca-sample") o.pca_sample = atoll(nx().c_str());
        else if (a == "--cluster-size") o.cluster_size = atoi(nx().c_str());
        else if (a == "--write-fsb") o.write_fsb = nx();
        else usage();
    }
    if (o.inputs.empty()) usage();
    if (o.perplexity <= 0) o.perplexity = o.k / 3.f;
    if (o.nprobe > 0) o.nprobe = std::max(1, std::min(o.nprobe, 64));
}

// all parsed rows -> one .fsb file (same layout tools/duckdb_export.py writes)
static void write_fsb_file(const Data& d, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
    int32_t D = d.D; int64_t N = d.N;
    wr(f, "FSB1", 4); wr(f, &D, 4); wr(f, &N, 8);
    for (const auto& c : d.chunks) wr(f, c.data(), c.size() * 4);
    for (int64_t i = 0; i < N; i++) {
        wr(f, d.ids.data() + d.idoff[i], d.idoff[i + 1] - d.idoff[i]);
        if (i + 1 < N) wr(f, "\n", 1);
    }
    fclose(f);
    LOG("wrote %s (%lld rows)", path.c_str(), (long long)N);
}
