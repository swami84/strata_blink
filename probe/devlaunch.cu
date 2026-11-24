// devlaunch.cu - Blink (arXiv 2604.07609) section 4.2 / Fig. 3 mechanism probe, for this machine (RTX 3090, sm_86).
//
// A "step graph" stands in for one decode step: K chained kernels (K=48 ~ one kernel per layer), each spinning for
// `work_ns` of GPU time; the last one advances a token (tok = lcg(tok)) and bumps a completion counter.
// The same N steps are driven five ways and the per-step overhead is reported:
//
//   host-sync  CPU-resident loop (Blink Fig. 3 baseline): H2D token -> host graph launch -> D2H token -> stream sync
//   host-async host launches back to back, no sync (pure host launch throughput; no per-step decision on the host)
//   dev-ff     Blink: a persistent scheduler graph launches the step graph fire-and-forget, polls the completion
//              counter, publishes the token to a ring; after 120 launches it tail-launches itself (window recovery)
//   dev-tail   tail launch only: the scheduler tail-launches the step graph and then itself, every step
//   doorbell   Strata's per-layer pattern: a GPU thread rings a doorbell in mapped host memory and spins until a
//              host thread acks it (one GPU<->CPU round trip per layer; 48 per verify window)
//
// --stress N runs N CPU threads streaming over private 64 MiB buffers (LLC + DRAM pressure, like a parallel build)
// for the duration of every mode.  All modes must end on the same token (checked).
//
// build: nvcc -O3 -std=c++17 -arch=sm_86 -rdc=true devlaunch.cu -lcudadevrt -o devlaunch

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                                     \
    do {                                                                                                          \
        cudaError_t e_ = (x);                                                                                     \
        if (e_ != cudaSuccess) {                                                                                  \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));             \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

static constexpr int kFF = 120;   // CUDA's limit of outstanding fire-and-forget launches per parent graph execution

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__host__ __device__ __forceinline__ int lcg(int x) { return (int) (((unsigned) x * 1103515245u + 12345u) & 0x7fffffffu); }

// one "layer": spin work_ns; the last layer advances the token and signals completion
__global__ void layer_kernel(int* tok, unsigned* done, int64_t work_ns, int last) {
    if (work_ns > 0) {
        const uint64_t t0 = gtimer();
        while ((int64_t) (gtimer() - t0) < work_ns) {}
    }
    if (last && threadIdx.x == 0 && blockIdx.x == 0) {
        *tok = lcg(*tok);
        __threadfence();
        atomicAdd(done, 1u);
    }
}

struct Sched {
    int total;          // steps to run
    int done_steps;     // steps published
    int pending;        // dev-tail: a step was launched by the previous invocation
    int relaunches;     // dev-ff: tail relaunches of the scheduler (window recoveries)
};

// Blink's scheduler: fire-and-forget the step graph, poll for completion, publish, repeat; tail-relaunch every kFF
__global__ void sched_ff(const cudaGraphExec_t* steps, int ncopy, int64_t gap_ns, Sched* s, volatile unsigned* done,
                         volatile int* tok, int* ring, uint64_t* stamp) {
    if (threadIdx.x != 0) return;
    int launched = 0;
    while (s->done_steps < s->total) {
        if (launched == kFF) {
            s->relaunches++;
            cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
            return;
        }
        const unsigned target = (unsigned) s->done_steps + 1u;
        const cudaError_t e = cudaGraphLaunch(steps[s->done_steps % ncopy], cudaStreamGraphFireAndForget);
        if (e != cudaSuccess) {
            s->pending = (int) e;   // report the failure to the host
            s->total = -1;
            return;
        }
        ++launched;
        while (*done < target) {}
        if (gap_ns > 0) {
            const uint64_t g0 = gtimer();
            while ((int64_t) (gtimer() - g0) < gap_ns) {}
        }
        __threadfence();
        ring[s->done_steps] = *tok;
        stamp[s->done_steps] = gtimer();
        s->done_steps = (int) target;
    }
}

// tail launch only: run the step after this invocation, then this scheduler again
__global__ void sched_tail(cudaGraphExec_t step, Sched* s, volatile int* tok, int* ring, uint64_t* stamp) {
    if (threadIdx.x != 0) return;
    if (s->pending) {
        ring[s->done_steps] = *tok;
        stamp[s->done_steps] = gtimer();
        s->done_steps++;
        s->pending = 0;
    }
    if (s->done_steps < s->total) {
        if (cudaGraphLaunch(step, cudaStreamGraphTailLaunch) != cudaSuccess ||
            cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch) != cudaSuccess) {
            s->total = -1;
            return;
        }
        s->pending = 1;
    }
}

// Strata's doorbell: per "layer" the GPU publishes, the host acks, the GPU continues
__global__ void doorbell_kernel(volatile int* bell, volatile int* ack, int n, uint64_t* stamp) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 1; i <= n; ++i) {
        *bell = i;
        __threadfence_system();
        while (*ack < i) {}
        stamp[i - 1] = gtimer();
    }
}

// ---------------------------------------------------------------------------------------------------------------

struct Stress {
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    void start(int n) {
        for (int i = 0; i < n; ++i)
            th.emplace_back([this, i] {
                const size_t sz = (64u << 20) / sizeof(uint64_t);
                std::vector<uint64_t> buf(sz, (uint64_t) i);
                uint64_t acc = 0, idx = (uint64_t) i * 7919u;
                while (!stop.load(std::memory_order_relaxed)) {
                    for (size_t k = 0; k < sz; k += 8) {              // stream (bandwidth)
                        buf[k] += acc;
                        acc ^= buf[k];
                    }
                    for (int k = 0; k < 200000; ++k) {               // random reads (LLC / TLB misses)
                        idx = idx * 6364136223846793005ull + 1442695040888963407ull;
                        acc += buf[(idx >> 17) % sz];
                    }
                }
                if (acc == 42) std::printf(" ");
            });
    }
    void end() {
        stop = true;
        for (auto& t : th) t.join();
        th.clear();
        stop = false;
    }
};

struct Stats {
    double mean_us, p50_us, p99_us, p999_us, max_us;
};
static Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[std::min(v.size() - 1, (size_t) (p * (double) (v.size() - 1) + 0.5))]; };
    double s = 0;
    for (double x : v) s += x;
    return {s / (double) v.size(), q(0.50), q(0.99), q(0.999), v.back()};
}
static std::vector<double> diffs_us(const std::vector<uint64_t>& ns, uint64_t t0) {
    std::vector<double> d(ns.size());
    uint64_t prev = t0;
    for (size_t i = 0; i < ns.size(); ++i) {
        d[i] = (double) (ns[i] - prev) / 1e3;
        prev = ns[i];
    }
    return d;
}

static cudaGraphExec_t make_step(cudaStream_t s, int K, int* tok, unsigned* done, int64_t work_ns, bool device) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int k = 0; k < K; ++k) layer_kernel<<<1, 32, 0, s>>>(tok, done, work_ns, k == K - 1);
    CK(cudaStreamEndCapture(s, &g));
    cudaGraphExec_t ge;
    CK(cudaGraphInstantiateWithFlags(&ge, g, device ? cudaGraphInstantiateFlagDeviceLaunch : 0));
    if (device) CK(cudaGraphUpload(ge, s));
    CK(cudaGraphDestroy(g));
    return ge;
}

int main(int argc, char** argv) {
    int N = 20000, K = 48, stress = 0, dev = 0, ncopy = 1;
    int64_t work_ns = 0, gap_ns = 0;
    std::string modes = "host-sync,host-async,dev-ff,dev-tail,doorbell";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { return std::string(argv[++i]); };
        if (a == "--steps") N = std::stoi(nx());
        else if (a == "--kernels") K = std::stoi(nx());
        else if (a == "--work-ns") work_ns = std::stoll(nx());
        else if (a == "--stress") stress = std::stoi(nx());
        else if (a == "--device") dev = std::stoi(nx());
        else if (a == "--copies") ncopy = std::stoi(nx());
        else if (a == "--gap-ns") gap_ns = std::stoll(nx());
        else if (a == "--modes") modes = nx();
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    CK(cudaSetDevice(dev));
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, dev));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    int *tok, *ring;
    unsigned* done;
    Sched* sch;
    uint64_t* stamp;
    CK(cudaMalloc(&tok, sizeof(int)));
    CK(cudaMalloc(&done, sizeof(unsigned)));
    CK(cudaMalloc(&ring, sizeof(int) * (size_t) N));
    CK(cudaMalloc(&sch, sizeof(Sched)));
    CK(cudaMalloc(&stamp, sizeof(uint64_t) * (size_t) N));
    int* h_tok;
    CK(cudaHostAlloc(&h_tok, sizeof(int), cudaHostAllocDefault));
    int* h_map;   // [0] bell, [1] ack
    CK(cudaHostAlloc(&h_map, 64, cudaHostAllocMapped));
    int* d_map;
    CK(cudaHostGetDevicePointer((void**) &d_map, h_map, 0));

    cudaGraphExec_t step_host = make_step(s, K, tok, done, work_ns, false);
    cudaGraphExec_t step_dev = make_step(s, K, tok, done, work_ns, true);
    std::vector<cudaGraphExec_t> h_steps(ncopy);
    h_steps[0] = step_dev;
    for (int c = 1; c < ncopy; ++c) h_steps[c] = make_step(s, K, tok, done, work_ns, true);
    cudaGraphExec_t* d_steps;
    CK(cudaMalloc(&d_steps, sizeof(cudaGraphExec_t) * (size_t) ncopy));
    CK(cudaMemcpy(d_steps, h_steps.data(), sizeof(cudaGraphExec_t) * (size_t) ncopy, cudaMemcpyHostToDevice));

    int expect = 1;
    for (int i = 0; i < N; ++i) expect = lcg(expect);

    std::printf("# %s sm_%d%d, %d SMs | steps %d, kernels/step %d, work/kernel %lld ns, stress threads %d\n", p.name,
                p.major, p.minor, p.multiProcessorCount, N, K, (long long) work_ns, stress);
    std::printf("%-11s %9s %9s %9s %9s %9s %10s  %s\n", "mode", "mean_us", "p50_us", "p99_us", "p99.9_us", "max_us",
                "total_ms", "check");

    auto reset = [&](int t0) {
        CK(cudaMemcpy(tok, &t0, sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemset(done, 0, sizeof(unsigned)));
        Sched z{N, 0, 0, 0};
        CK(cudaMemcpy(sch, &z, sizeof z, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
    };
    auto report = [&](const char* name, const std::vector<double>& d, double total_ms, bool ok, const char* note) {
        Stats st = stats_of(d);
        std::printf("%-11s %9.2f %9.2f %9.2f %9.2f %9.1f %10.1f  %s%s\n", name, st.mean_us, st.p50_us, st.p99_us,
                    st.p999_us, st.max_us, total_ms, ok ? "ok" : "MISMATCH", note);
        std::fflush(stdout);
    };
    auto has = [&](const char* m) { return modes.find(m) != std::string::npos; };

    // warm up both instantiations
    reset(1);
    for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(step_host, s));
    for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(step_dev, s));
    CK(cudaStreamSynchronize(s));

    Stress st;
    if (stress > 0) {
        st.start(stress);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    using clk = std::chrono::steady_clock;

    if (has("host-sync")) {   // CPU-resident loop: the host sees every token and launches the next step
        reset(1);
        std::vector<double> d(N);
        *h_tok = 1;
        const auto t0 = clk::now();
        auto prev = t0;
        for (int i = 0; i < N; ++i) {
            CK(cudaMemcpyAsync(tok, h_tok, sizeof(int), cudaMemcpyHostToDevice, s));
            CK(cudaGraphLaunch(step_host, s));
            CK(cudaMemcpyAsync(h_tok, tok, sizeof(int), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            const auto now = clk::now();
            d[i] = std::chrono::duration<double, std::micro>(now - prev).count();
            prev = now;
        }
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        report("host-sync", d, ms, *h_tok == expect, "");
    }
    if (has("host-async")) {   // launch throughput only (no host decision per step)
        reset(1);
        std::vector<double> d(N);
        const auto t0 = clk::now();
        auto prev = t0;
        for (int i = 0; i < N; ++i) {
            CK(cudaGraphLaunch(step_host, s));
            const auto now = clk::now();
            d[i] = std::chrono::duration<double, std::micro>(now - prev).count();
            prev = now;
        }
        CK(cudaStreamSynchronize(s));
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        int out = 0;
        CK(cudaMemcpy(&out, tok, sizeof(int), cudaMemcpyDeviceToHost));
        report("host-async", d, ms, out == expect, "  (per-step = host enqueue time; total includes the drain)");
    }
    auto run_dev = [&](const char* name, bool ff) {
        reset(1);
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        if (ff) sched_ff<<<1, 32, 0, s>>>(d_steps, ncopy, gap_ns, sch, done, tok, ring, stamp);
        else sched_tail<<<1, 32, 0, s>>>(step_dev, sch, tok, ring, stamp);
        CK(cudaStreamEndCapture(s, &g));
        cudaGraphExec_t ge;
        CK(cudaGraphInstantiateWithFlags(&ge, g, cudaGraphInstantiateFlagDeviceLaunch));
        CK(cudaGraphUpload(ge, s));
        CK(cudaStreamSynchronize(s));
        const auto t0 = clk::now();
        CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        Sched hs{};
        CK(cudaMemcpy(&hs, sch, sizeof hs, cudaMemcpyDeviceToHost));
        if (hs.total < 0 || hs.done_steps != N) {
            std::printf("%-11s device launch FAILED (done %d of %d): %s\n", name, hs.done_steps, N,
                        hs.total < 0 ? cudaGetErrorString((cudaError_t) hs.pending) : "-");
            CK(cudaGraphExecDestroy(ge));
            CK(cudaGraphDestroy(g));
            return;
        }
        std::vector<uint64_t> ts(N);
        CK(cudaMemcpy(ts.data(), stamp, sizeof(uint64_t) * (size_t) N, cudaMemcpyDeviceToHost));
        std::vector<int> r(N);
        CK(cudaMemcpy(r.data(), ring, sizeof(int) * (size_t) N, cudaMemcpyDeviceToHost));
        std::vector<uint64_t> tail(ts.begin() + 1, ts.end());   // step-to-step intervals on the GPU clock
        char note[96];
        std::snprintf(note, sizeof note, ff ? "  (%d window recoveries via tail launch)" : "", hs.relaunches);
        report(name, diffs_us(tail, ts[0]), ms, r[N - 1] == expect, note);
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
    };
    if (has("dev-ff")) run_dev("dev-ff", true);
    if (has("dev-tail")) run_dev("dev-tail", false);

    if (has("doorbell")) {   // GPU<->CPU round trip through mapped pinned memory, as Strata's per-layer doorbell
        volatile int* bell = h_map;
        volatile int* ack = h_map + 1;
        *bell = 0;
        *ack = 0;
        std::atomic<bool> fin{false};
        std::thread host([&] {
            for (int i = 1; i <= N; ++i) {
                while (*bell < i) {}
                *ack = i;
            }
            fin = true;
        });
        const auto t0 = clk::now();
        doorbell_kernel<<<1, 32, 0, s>>>(d_map, d_map + 1, N, stamp);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(s));
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        host.join();
        std::vector<uint64_t> ts(N);
        CK(cudaMemcpy(ts.data(), stamp, sizeof(uint64_t) * (size_t) N, cudaMemcpyDeviceToHost));
        std::vector<uint64_t> tail(ts.begin() + 1, ts.end());
        report("doorbell", diffs_us(tail, ts[0]), ms, fin.load(), "  (one GPU->CPU->GPU round trip; host spins)");
    }
    if (stress > 0) st.end();
    return 0;
}
