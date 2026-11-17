# 2026-11-07: RTX PRO 6000 Max-Q, decode stage after the host-side rewrite

Vast instance 59598940 (offer 46501555, the same machine as
`2026-11-07-vast-pro6000-clips`): NVIDIA RTX PRO 6000 Blackwell **Max-Q
Workstation Edition** (sm_120, 96 GB, **300 W** power limit, boost 3090 MHz),
driver 610.57.04, 16 CPU cores, 124 GB RAM, Switzerland, $1.56/h; 22 min,
self-destroyed. Image `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, release
build of commit `Decode stage: fold the audio upsampler crop…`
(`--features cuda`). Weights: the FastH3-V2 tree (138 GiB) in 362 s. Raw
output: `output.txt`; the profiled run in full: `v2-profiled.log`;
`clocks.log` is `nvidia-smi` every 2 s during the unprofiled run.

Conformance first: `h3 check --backend cuda` **97 / 97** at 3e-2.

## The clip (V2 bf16, seed 7, the fox prompt, 1344×768×124)

Same clip as the previous record (`fox-v2-bf16-frames.png`: frames 0 / 40 /
80 / 120); the profiled and unprofiled runs produced byte-identical `.y4m`.

| stage | before (`…-pro6000-clips`) | now | GPU time in ops (profiled) |
|---|---|---|---|
| denoise, 8 forwards | 79.5 s | 79.2 s (9.57 → 10.03 s / forward) | 79.0 s |
| **video decode**, 7 chunks | **20.5 s** | **11.9 s** (12.3 s profiled) | 10.7 s (1.53 s / chunk) |
| **audio decode** | **0.46 s** | **0.24 s** | 0.24 s |
| total after load | 101.1 s | 91.9 s | |
| peak device memory | 56.2 GB | 56.2 GB | |

- **Video decode** is now GPU-bound: 11.9 s wall against 10.7 s of
  kernels. The host side went from ~10 s to ~1.2 s for the 7 chunks: tiles
  are decoded one ahead of their download (the GPU runs tile `i + 1` while
  the host converts tile `i`), the stitch is one row-parallel pass, the
  chunk crop / temporal blend / denormalise is one fused pass, the 8-bit
  conversion is one task per frame, and the CUDA download converts
  bf16 → f32 in parallel. The decoder kernels are unchanged: the two big
  GEMMs are 58 % of the chunk at 208 TFLOP/s (tile batches of ~1.3 k
  rows), attention 12 %.
- **Audio decode**: the 36 634 device-to-device copies are gone. The
  anti-aliasing upsampler's crop was symmetric, so it is the transposed
  conv's own padding, and the ×ratio moved into the filter (exact in
  bf16). The stage is 0.24 s = 189 ms `conv1d` (510 calls) + 45 ms
  `conv_transpose1d` (268 calls); `copy (dtod)` is 42 calls, 0.3 ms.
- **Denoise is 100 % GPU**: 79.0 s in ops for 79.0 s of wall. Per block
  197 ms: VSA fine 53.8 ms (27 %), the six block GEMMs 130 ms (66 %, 246
  TFLOP/s), head_split 4 ms, coarse + top-k 2.7 ms, pool 1.2 ms.

## The clocks: this card runs the denoise at 1.4 GHz

`clocks.log` during the denoise (46 samples with power > 200 W): SM clock
**1380 – 1987 MHz, median 1413 MHz** against the 3090 MHz boost, power
278 – 304 W (median 300.0 W), 77 – 91 °C, `clocks_throttle_reasons.active
= 0x4` (SW power cap) in 45 of the 46 samples. The step time grows 9.57 →
10.03 s across the 8 steps as the card warms into the cap. The bench's
185 ms / block (`…-pro6000-clips`) was a short, cold run at higher clocks;
the clip's 197 ms / block is the sustained number on this 300 W card. A
600 W Server / Workstation Edition of the same chip (fastvideo-rs's gold
GPU) does not sit in this cap: expect the denoise gap to it to be mostly
clocks, not kernels. Record the power limit with every number.

## Against fastvideo-rs (RTX PRO 6000 Blackwell **Server Edition, 600 W**)

`lab/gold/fasth3-*.toml`, `docs/perf/h3-nvfp4-tier-options.md`,
`docs/techniques.md` in that repo; 1344×768×124, 8 forwards:

| | fastvideo-rs, 600 W | h3-rs, 300 W Max-Q |
|---|---|---|
| FastH3-V2 denoise, VSA 0.8 | 45.8 s (MXFP8 linears) | 79.2 s (bf16) |
| V2-NVFP4 + FP4 attention denoise | 31.9 s | — |
| Trim NVFP4 denoise | 32.7 s (24.6 s with FP4 VSA) | 67.8 s |
| FastH3 4-step V1 denoise, VSA 0.9 | 19.8 s | 35.0 s (int8) |
| 36-block video decode | 6.3 s bf16 / 4.7 s MXFP8 (GEMMs at 397 TFLOP/s) | 11.9 s (GEMMs at 208 TFLOP/s) |
| peak device memory, V2 | 61.0 GiB | 56.2 GB |

The remaining decode gap is GPU work: small tile batches run the GEMMs at
half of fastvideo-rs's efficiency, and it decodes in MXFP8. The denoise
gap is the 300 W clocks first, then MXFP8 / NVFP4 tensor-core GEMMs
(ours are bf16 at 246 TFLOP/s; int8 and NVFP4 weights expand to bf16 per
GEMM) and the FP4 VSA fine stage.
