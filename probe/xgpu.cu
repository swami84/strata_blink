// xgpu.cu - cross-GPU signalling for a two-stage GPU-resident scheduler (Blink extended to a layer split).
// One persistent thread per GPU ping-pongs a counter N times; reports the round-trip time per hop pair.
//   path "host": both flags in mapped pinned host memory (works without peer access)
//   path "peer": each flag in the *reader's* VRAM, written by the other GPU through peer access (if the driver allows it)
// build: nvcc -O3 -std=c++17 -arch=sm_86 xgpu.cu -o xgpu
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
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

// GPU0 ("ping") writes `out = i`, waits for `in == i`; GPU1 ("pong") waits for `in == i`, writes `out = i`.
__global__ void pinger(volatile int* out, volatile int* in, int n, int first) {
    if (threadIdx.x | blockIdx.x) return;
    for (int i = 1; i <= n; ++i) {
        if (first) {
            *out = i;
            __threadfence_system();
            while (*in < i) {}
        } else {
            while (*in < i) {}
            *out = i;
            __threadfence_system();
        }
    }
}

static double run(int* a_for_1, int* b_for_0, int n, const char* name) {
    // a_for_1: written by GPU0, read by GPU1;  b_for_0: written by GPU1, read by GPU0
    cudaStream_t s0, s1;
    CK(cudaSetDevice(0)); CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaSetDevice(1)); CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    const auto t0 = std::chrono::steady_clock::now();
    CK(cudaSetDevice(1)); pinger<<<1, 32, 0, s1>>>(b_for_0, a_for_1, n, 0); CK(cudaGetLastError());
    CK(cudaSetDevice(0)); pinger<<<1, 32, 0, s0>>>(a_for_1, b_for_0, n, 1); CK(cudaGetLastError());
    CK(cudaStreamSynchronize(s0));
    CK(cudaSetDevice(1)); CK(cudaStreamSynchronize(s1));
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
    std::printf("%-5s GPU0->GPU1->GPU0 round trip: %.2f us (%d round trips)\n", name, us, n);
    return us;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 100000;
    int a01 = 0, a10 = 0;
    CK(cudaDeviceCanAccessPeer(&a01, 0, 1));
    CK(cudaDeviceCanAccessPeer(&a10, 1, 0));
    std::printf("peer access: GPU0->GPU1 %s, GPU1->GPU0 %s\n", a01 ? "yes" : "no", a10 ? "yes" : "no");

    // host path
    int* h;
    CK(cudaSetDevice(0));
    CK(cudaHostAlloc(&h, 256, cudaHostAllocMapped | cudaHostAllocPortable));
    h[0] = h[32] = 0;
    int* d;   // mapped pointers are identical across devices with UVA
    CK(cudaHostGetDevicePointer((void**) &d, h, 0));
    run(d, d + 32, n, "host");

    if (a01 && a10) {
        CK(cudaSetDevice(0)); cudaError_t e0 = cudaDeviceEnablePeerAccess(1, 0);
        CK(cudaSetDevice(1)); cudaError_t e1 = cudaDeviceEnablePeerAccess(0, 0);
        if ((e0 != cudaSuccess && e0 != cudaErrorPeerAccessAlreadyEnabled) ||
            (e1 != cudaSuccess && e1 != cudaErrorPeerAccessAlreadyEnabled)) {
            std::printf("peer enable failed: %s / %s\n", cudaGetErrorString(e0), cudaGetErrorString(e1));
            return 0;
        }
        int *f_on_1, *f_on_0;   // each flag lives in its reader's VRAM
        CK(cudaSetDevice(1)); CK(cudaMalloc(&f_on_1, 256)); CK(cudaMemset(f_on_1, 0, 256));
        CK(cudaSetDevice(0)); CK(cudaMalloc(&f_on_0, 256)); CK(cudaMemset(f_on_0, 0, 256));
        CK(cudaDeviceSynchronize());
        run(f_on_1, f_on_0, n, "peer");
    }
    return 0;
}
