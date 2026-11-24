// ffsync.cu - does a host stream sync on a launcher graph also wait for the graph it fire-and-forget launched from
// the device?  The child spins ~2 ms and then writes a value; the launcher kernel launches it and exits at once.
// build: nvcc -O3 -std=c++17 -arch=sm_86 -rdc=true ffsync.cu -lcudadevrt -o ffsync
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                                                     \
    do {                                                                                                          \
        cudaError_t e_ = (x);                                                                                     \
        if (e_ != cudaSuccess) {                                                                                  \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));             \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

__global__ void child(int* out, int v) {
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 > 2000000ull) break;
    }
    *out = v;
}
__global__ void launcher(cudaGraphExec_t g, int* err) {
    if (threadIdx.x == 0) *err = (int) cudaGraphLaunch(g, cudaStreamGraphFireAndForget);
}

int main() {
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    int *out, *err;
    CK(cudaMallocManaged(&out, sizeof(int)));
    CK(cudaMallocManaged(&err, sizeof(int)));
    for (int round = 1; round <= 3; ++round) {
        *out = 0;
        *err = -1;
        cudaGraph_t gc, gl;
        cudaGraphExec_t ec, el;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        child<<<1, 1, 0, s>>>(out, 1000 + round);
        CK(cudaStreamEndCapture(s, &gc));
        CK(cudaGraphInstantiateWithFlags(&ec, gc, cudaGraphInstantiateFlagDeviceLaunch));
        CK(cudaGraphUpload(ec, s));
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        launcher<<<1, 32, 0, s>>>(ec, err);
        CK(cudaStreamEndCapture(s, &gl));
        CK(cudaGraphInstantiateWithFlags(&el, gl, round == 3 ? 0 : cudaGraphInstantiateFlagDeviceLaunch));
        CK(cudaStreamSynchronize(s));
        CK(cudaGraphLaunch(el, s));
        CK(cudaStreamSynchronize(s));
        const int seen = *out;   // read right after the sync, before anything else
        std::printf("round %d (launcher instantiated %s): launch err %d, value after stream sync = %d -> %s\n", round,
                    round == 3 ? "without DeviceLaunch" : "with DeviceLaunch", *err, seen,
                    seen == 1000 + round ? "sync COVERS the child" : "sync does NOT cover the child");
        CK(cudaDeviceSynchronize());
    }
    return 0;
}
