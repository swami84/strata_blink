// stress.cpp - CPU interferer for the Blink replication: N threads, each streaming over a private 64 MiB buffer and
// doing random reads in it (LLC + DRAM + TLB pressure, roughly what a parallel build or pbzip2 does to a host).
// Runs until killed (or for --seconds S).   build: g++ -O2 -pthread stress.cpp -o stress
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    int n = 16;
    double seconds = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
    }
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    for (int t = 0; t < n; ++t)
        th.emplace_back([&stop, t] {
            const size_t sz = (64u << 20) / sizeof(uint64_t);
            std::vector<uint64_t> buf(sz, (uint64_t) t);
            uint64_t acc = 0, idx = (uint64_t) t * 7919u;
            while (!stop.load(std::memory_order_relaxed)) {
                for (size_t k = 0; k < sz; k += 8) {
                    buf[k] += acc;
                    acc ^= buf[k];
                }
                for (int k = 0; k < 200000; ++k) {
                    idx = idx * 6364136223846793005ull + 1442695040888963407ull;
                    acc += buf[(idx >> 17) % sz];
                }
            }
            if (acc == 42) std::printf(" ");
        });
    std::fprintf(stderr, "stress: %d threads running\n", n);
    if (seconds > 0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        stop = true;
    }
    for (auto& x : th) x.join();
    return 0;
}
