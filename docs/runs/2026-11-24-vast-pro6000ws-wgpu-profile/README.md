# 2026-11-24: where a WebGPU H3 step goes (Vast RTX PRO 6000 Workstation Edition)

Vast contract 62385143: RTX PRO 6000 Blackwell Workstation Edition, 600 W,
3090 MHz max, driver 615.71.09, Vulkan 1.4.351 (Texas); ~40 min. It
destroyed itself from its last command. Binary eaea24a.

`h3 bench` used synthetic weights at the real geometry (50 blocks, the
h3-plug4 draft config, 124 frames, dense attention), two forwards each.
Commands are `cmd2`–`cmd4.sh`, outputs `out2`–`out4.txt`, with per-run logs
and memory samples. `bench --forwards` replaces its old `--steps`, which
clashed with the shared `--steps` and made every `h3 bench` call fail.

## Checks

| backend | result |
|---|---|
| WebGPU (Vulkan, NVIDIA, not llvmpipe) | 152 / 152 ops |
| CUDA | 152 / 152 ops |

## Per forward

| run | per forward | peak GPU |
|---|---|---|
| WebGPU 480p int4 | 56.1 / 57.8 s (1.12–1.16 s per block) | 21.7 GB |
| WebGPU 480p native (f16 weights) | 58.1 / 60.1 s | 53.6 GB |
| WebGPU 480p int4, `H3_WGPU_F16=1` | 56.1 / 57.8 s (the int4 GEMMs have no f16 path) | 21.7 GB |
| WebGPU 288p int4 | 16.1 / 16.4 s | 18.3 GB |
| CUDA 480p int4 | 3.82 / 3.83 s (76 ms per block) | 18.5 GB |

The first forward of each WebGPU pair is consistently ~1.7 s faster than the
second; the reason is not known. The WebGPU forwards here match the ~55 s
`generate` steps of `2026-11-24-runpod-roundC` (Server Edition, 595.112.07).

## GPU time by kernel (timestamp queries, both forwards)

**WebGPU, 480p int4** (114.2 s):

| kernel | time | share | per call |
|---|---|---|---|
| `flash` (attention) | 42.8 s | 37.5 % | 428 ms |
| `linear3_q4` 28672×5376 (FFN up) | 28.2 s | 24.7 % | 282 ms |
| `linear3_q4` 21504×5376 (QKV) | 21.2 s | 18.5 % | 212 ms |
| `linear3_q4` 5376×14336 (FFN down) | 14.3 s | 12.6 % | 143 ms |
| `linear3_q4` 5376×7168 (out) | 7.1 s | 6.2 % | 71 ms |
| elementwise + head_split + norm | 0.52 s | **0.5 %** | 0.6–1.1 ms |

**CUDA, same shapes** (7.65 s, synced per op):

| kernel | time | share | per call |
|---|---|---|---|
| attention (mma2) | 2.72 s | 35.5 % | 27 ms |
| FFN up | 1.77 s | | 17.7 ms |
| QKV | 1.33 s | | 13.3 ms |
| FFN down | 0.90 s | | 9.0 ms |
| out | 0.45 s | | 4.5 ms |
| dequant int4 | 0.15 s | | |
| elementwise (head_split, swiglu, norms, gate) | 0.31 s | 4 % | |

**WebGPU 288p**: GEMMs 75.6 %, flash 19.4 %, elementwise 0.6 %.

**WebGPU 480p native**: GEMMs (`linear3_f16`) 63 %, flash 36 %, elementwise
0.5 %, so the per-call times are close to int4's.

## What this says

**16-bit activation storage would not make WebGPU faster.**
- The memory-bound kernels it would speed up are 0.5 % of the step.
- What it would buy is memory: activations are f32 on WebGPU.

**The gap to CUDA is the GEMMs and attention, ~16× each.**
- Attention: 428 against 27 ms.
- FFN up: 282 against 17.7 ms.
- CUDA uses tensor cores; WGSL cannot reach them.
- h3-rs still runs the older `linear3` / `flash` kernels. fastwan-rs has
  since moved to `linear4` (128×128 tiles, 8×8 per thread) and `flash3`
  (4 threads per softmax row). Its native WebGPU step went from 10.7 to
  6.67 s on an RTX 4090 (`fastwan-rs: docs/runs/2026-11-14-vast-4090-webgpu-kernels`),
  with Chrome at 5.75 s. Porting those two kernels is the WebGPU speed lever
  here: they cover 99 % of the step.

**The memory notes.**
- The synthetic bench peaks at 21.7 GB at 480p int4, where `generate` peaked
  at 47.3 GB. The extra 25 GB comes from something `generate` does outside
  the forward (the load-time transient in `2026-11-24-runpod-roundC`).
- The native-weight run needs 53.6 GB: f16 weights.
