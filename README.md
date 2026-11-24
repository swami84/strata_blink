# Blink on Strata: a GPU-resident decode loop for Qwen3.8-Flash-Next

Testing the idea of [Blink](https://arxiv.org/abs/2604.07609) (Siavashi et al., 2026), CPU-free LLM inference, on a
local MoE setup: **Strata v0.2.15** serving **Qwen3.8-Flash-Next** (QFN) on **2x RTX 3090** (300 W, PCIe 4.0 x8),
Ryzen 7 7800X3D (8C/16T), 60 GB RAM, CUDA 13.4.

## Why

Blink argues that LLM serving stacks keep the CPU on the per-token path (scheduling, launches, syncs), so a busy CPU
slows inference badly. It moves the decode loop onto the GPU with a persistent scheduler kernel and device-side CUDA
graph launch, and moves request handling onto a SmartNIC.

Questions for this machine:
1. Does the problem exist for Strata, which already runs each verify window as one CUDA graph?
2. Can Blink's GPU-side loop be built for a two-GPU MoE engine, and what does it buy?

The SmartNIC half does not apply: there is no DPU here, and GeForce cards have no GPUDirect RDMA.

## What was done

1. **Measured the problem.** Strata's own timing switches, idle vs. a CPU interferer (`probe/stress`: 16 threads
   streaming through RAM, like a parallel build).
2. **Probed the mechanism** on the 3090 (`probe/`):
   - device-side graph launch with tail relaunch works;
   - a GPU-to-GPU flag through mapped host memory takes 1.8 µs (there is no P2P).
3. **Made QFN fit in VRAM.** IQ2_XS with `"layer_split": "23"` holds all 24,576 experts on the two cards, so neither
   GPU calls the CPU per layer.
4. **Built Blink for Strata** (`patches/`, about 1,400 lines, enabled with `STRATA_BLINK=1`):
   - one persistent scheduler kernel per GPU, coordinating through a control block in mapped host memory;
   - window choice, both stages, accept and commit, MTP drafting, the penalized pick, and the PLE (n-gram) row
     lookup all run on the GPUs;
   - the host only prints tokens and forwards "stop";
   - it is used only where the output is exactly the stock loop's, otherwise the request falls back to stock.
5. **Evaluated** stock vs. Blink, idle and busy, with three ways of placing the other work (below), and checked
   that the output is identical.

## Results

QFN IQ2_XS, same flags for both engines (`--suffix-draft 0 --mmap-experts --ple-io ram`), 1024-token decodes.

| Decode tok/s | Idle | Busy | Kept |
|---|---:|---:|---:|
| Stock, other work on any core | 183 | 119 | 0.65x |
| **Blink, other work on any core** | 181 | **183** | **1.01x** |
| Stock, Strata on 4 cores, other work on the rest | 185 | 183 | 0.99x |
| Blink, Strata on 4 cores, other work on the rest | 186 | 187 | 1.00x |
| Stock, Strata on 1 core, other work on the rest | 186 | 161 | 0.87x |
| **Blink, Strata on 1 core, other work on the rest** | 187 | **186** | **0.99x** |

- **Exact:** Blink's output is token-for-token the stock output: on fixed prompts idle and under load, across a
  three-turn conversation (same cache reuse), and after a mid-stream cancel.
- **Idle speed:** no gain. Strata's host overhead is already small (about 1-1.5 ms of a 15-18 ms window).
- **Time to first token under load:** not improved. Prompt reading is still host-driven: 0.44 s idle, 2.2 s
  busy with Blink, 1.7 s busy with stock.
- **IQ3_S** (82% of experts in VRAM; the CPU computes the rest) keeps only 0.29x of its speed when the CPU is busy.
  Blink does not cover it.

## Conclusions

- **The problem is real for Strata:** a busy CPU costs 35-70% of decode speed.
- **Blink removes it for decode** on a two-GPU MoE setup, with exact output, as long as the model fits in VRAM.
- **Core pinning gets the same throughput without Blink** (0.99x), but costs 4 of 8 cores. Blink needs 1.
- **On an idle machine, Blink adds nothing.** It only pays when other work shares the CPU and pinning is unwanted.
- **Open:** GPU-side prefill (time to first token), the lookup drafter, partially-resident models (IQ3_S), and an
  IQ2_XS vs. IQ3_S quality check.

## Repo layout

| Path | What |
|---|---|
| `patches/` | Strata v0.2.15 changes, each cumulative: `m1-devlaunch`, `m2-gpu-loop`, `m3-gpu-ple`, `blink-final` (use this one) |
| `probe/` | `devlaunch.cu` (launch modes), `xgpu.cu` (GPU-to-GPU flags), `ffsync.cu`, `stress.cpp` (interferer) |
| `configs/` | engine configs used for every run |
| `run_strata_timing.sh`, `run_blink_matrix.sh` | the benchmark runs |
| `exactness.py`, `robustness.py`, `summarize_timing.py` | output comparison and summaries |
| `download_iq2_xs.sh` | pinned model download and pack |
| `results/` | raw results (`strata/*.jsonl`, `exact/`, `robust/`, `devlaunch_*.txt`) |

**Reproduce:**
1. Get the engine: `git clone -b v0.2.15 https://github.com/Niko1221/Strata`, then
   `git apply patches/blink-final.patch`, then build with CUDA for sm_86.
2. Set up the model and the benchmark harness as in the configs (paths assume `../basalt_vs_strata`).
3. Run `./run_blink_matrix.sh`.
