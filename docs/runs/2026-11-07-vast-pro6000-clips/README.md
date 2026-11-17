# 2026-11-07: RTX PRO 6000 (Blackwell), four clips with per-stage profiles

Vast instance 59578829 (offer 46501555): NVIDIA RTX PRO 6000 Blackwell
**Max-Q Workstation Edition** (sm_120, 96 GB, **300 W** power limit, boost
3090 MHz), driver 610.57.04, 16 CPU cores, 124 GB RAM, 17 Gbit/s down at
$0.0011/GB, Switzerland, $1.52/h; 1 h 10 min, self-destroyed. Image
`nvidia/cuda:12.8.1-runtime-ubuntu24.04`, binary = release build of
fb-series commit `CUDA VSA: shifts for the tile index…` (`--features cuda`).
Weights: 222 GB from the R2 mirror in 634 s (FastH3-V2 tree with text
encoder, Trim NVFP4 ComfyUI file + Trim VAE, MiniMax-H3 base transformer,
FastH3 4-step V1 VSA LoRA, TAEH3). Raw output: `output.txt`.

The Max-Q is the 300 W card; the Server / Workstation Editions run 600 W
and will be faster. The kernels are compiled for `compute_120` by NVRTC at
start-up: this is the first run of any of them on Blackwell.

## Conformance

`h3 check --backend cuda`: **97 / 97** for the default kernels (fine 3d,
coarse4, vector pool), the v1 fine body, the paired-tile body, the v1 dense
body and the scalar attention / conv kernels.

## Bench (synthetic weights, 1344×768×124, 37 966 tokens, 4 blocks)

| | per block | of block |
|---|---|---|
| VSA 0.8 tile 64 (default kernels) | **184.7 ms** | |
| fine stage 3d / 3b / v1 | 50.8 / 58.7 / 76.5 ms | 27.5 % |
| GEMMs (cuBLASLt bf16) | 121 ms at 266 TFLOP/s | 65 % |
| head_split | 0.91 ms × 4 | 2.0 % |
| coarse + top-k (4 query tiles) | 2.4 ms | 1.3 % |
| pool (vector) | 1.08 ms | 0.6 % |
| rms_norm | 0.96 ms × 2.25 | 1.2 % |
| dense attention | 305.5 ms per block, attention 184 ms | |
| Omni, tile 128 | 168.8 ms per block, fine 35.5 ms | |
| 832×480×124 | 65.2 ms per block, fine 11.1 ms | |

Against the L40S (350 W, sm_89) with the same binary: 230 → 185 ms per
block, fine 66.5 → 50.8 ms, GEMMs 230 → 266 TFLOP/s.

## Clips (seed 7, the fox prompt, 1344×768×124, 24 fps, 5.2 s, stereo)

| | FastH3-V2 bf16 | FastH3-V2 int8 | FastH3-Trim, ComfyUI NVFP4 | FastH3 4-step V1 (base + VSA LoRA), int8 |
|---|---|---|---|---|
| forwards | 8 | 8 | 8 (42 blocks) | 4 |
| per forward | 9.6–10.1 s | 9.6–10.1 s | 8.2–8.6 s | 8.6–8.9 s |
| denoise | **79.5 s** | 79.9 s | **67.8 s** | **35.0 s** |
| video decode (7 chunks) | 20.5 s | 20.6 s | 17.4 s | 20.4 s |
| audio decode | 0.46 s | 0.46 s | 0.46 s | 0.46 s |
| AdaLN tables (once per ladder) | 201 s | (shared) | 30 s | 232 s |
| peak device memory | 56.2 GB | 36.9 GB | 24.4 GB | 37.0 GB |
| wall, incl. load + prompt | 5 m 36 s | 6 m 03 s | 3 m 48 s | 5 m 48 s |

`fox-*-frames.png` hold frames 0 / 40 / 80 / 120. V2 bf16 and V2 int8 are
the same scene; `ffmpeg psnr` between them is 18.5 dB (Y 14.0): the DMD
sampler amplifies int8's weight rounding into different snow and fur
texture while the composition and motion stay. The Trim NVFP4 clip is the
Trim scene of the earlier runs; the 4-step V1 (MiniMax-H3 base weights
with the VSA LoRA's rank-64 pairs, `.diff`s and the `.set_weight` gate) is
a different composition, as a different model should be.

## Per-stage kernel profiles (`H3_PROFILE=1` now prints one table per stage)

- **Denoise, V2 bf16**: 9.6–10.1 s per forward against the bench's 9.2 s
  for 50 blocks of the synthetic model: the real forward adds the refiner,
  the heads and the 33 text rows.
- **Video decode**: each of the 7 chunks is 1.47 s of GPU work (ViT
  decoder: the two 16384×2048 / 2048×8192 GEMMs are 58 %, attention 12 %)
  = 10.3 s, against 20.5 s wall: **half of the decode is host-side**
  (f32 frame download, conversion to RGB, stitching). The next decode step
  is on-device RGB conversion and overlapping the download with the next
  chunk.
- **Audio decode**: 0.44 s of GPU time, 46 % of it in 36 634 tiny
  device-to-device copies (0.006 ms each) around the BigVGAN activations:
  fusing them is a small, certain win.
- The prompt-encoding and prepare tables were cut by the run script's
  `tail`; the stage timings are in `output.txt`.
