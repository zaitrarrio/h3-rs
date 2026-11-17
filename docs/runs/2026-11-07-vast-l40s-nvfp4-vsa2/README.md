# 2026-11-07: L40S — NVFP4 scale layout, ComfyUI single-file Trim, attention v2 A/B

Vast instance 59505873 (offer 58838435): NVIDIA L40S 48 GB (sm_89), driver
595.91.07, power limit 350 W, 2.9 Gbit/s down at $0.0017/GB, New York,
$0.47/h all-in; image `nvidia/cuda:12.8.1-runtime-ubuntu24.04`. Binary:
release build of eba864b (`--features cuda`). Raw output: `output.txt`.
Weights: the text encoder, tokenizer, VAE and audio VAE of the FastH3-V2
mirror, the FastH3-Trim bf16 tree's VAE and inference contract, and the
ComfyUI single-file `fastvideo_fasth3_trim_8step_nvfp4.safetensors`
(11 GiB): 90 GB in 254 s.

(A Tesla V100 was rented first by mistake, sm_70, and destroyed after a
minute: the search now filters `compute_cap ≥ 800`.)

## NVFP4 scale layout (found offline, before the run)

`Nvfp4::unswizzle_scales` had reduced to the identity permutation. Against
the bf16 twin of the same matrix (`transformer_blocks.0.attn.to_q.weight`
of the FastH3-Trim HF tree vs rows 0–5375 of the ComfyUI file's
`blocks.0.attn.qkv_proj`):

| hypothesis | mean |err| / mean |ref| |
|---|---|
| first code low nibble, any scale layout | 1.50 |
| first code high nibble, scales as stored (= old "unswizzle") | 0.46 |
| first code high nibble, torchao `to_blocked` unswizzle | **0.090**, correlation 0.9955, per-group scale ratio median 1.0007 (IQR 0.979–1.023) |

0.09 is NVFP4's own quantization noise. `final_layer.video_out.weight`
equals `proj_out.weight` exactly (the negation in ComfyUI's forward is its
sampler convention, not the weights). `adaln_t_table` rows 0 / 512 / 1024
start 0.195 / 0.680 / 1.039: a smooth curve in the clean-time t.

## Conformance

`h3 check --backend cuda`: 95 / 95 with the v2 kernels (default in this
binary) and 95 / 95 with `H3_CUDA_ATTN_V1=1`. The five new cases are the
NVFP4 linears in ComfyUI's nibble order.

## Attention kernel A/B (1344×768×124, 37 966 tokens, 4 blocks, synthetic weights)

| | v1 (16 rows / warp, 64-key chunks) | v2 (32 rows / warp, 32-key chunks) |
|---|---|---|
| VSA 0.8 tile 64, per block | **253.8 ms** | 273.7 ms |
| VSA fine stage per call | **81.6 ms** | 105.5 ms (2 warps / block) |
| dense attention, per block | 432–446 ms | **385–394 ms** |
| Omni VSA 0.9 tile 128, per block | 224.8 ms | 224.0 ms (4 warps / block) |
| GEMMs | 221 TFLOP/s | 227 TFLOP/s |

Reading: the VSA fine stage is **not** shared-memory-bound. Halving the
smem reads per mma (v2) changed nothing on 128-token tiles and the 2-warp
blocks for 64-token tiles lost 29 % (half the threads issuing the gathered
`cp.async` loads, twice the barriers per key). Dense attention gained 11 %,
and that is the configuration where v2 also halves the K/V traffic per
query row (128-row blocks). So the fine stage is bound by the gathered K/V
loads: every selected tile is fetched again by every query tile that
selects it (≈ 2.3 GB per head per block at this geometry). The next
experiments are a deeper `cp.async` pipeline (two chunks in flight) and
processing two adjacent query tiles per block where their selections
overlap, not more register blocking. Dispatch after this run: v2 for dense
attention, v1 for VSA (`H3_CUDA_VSA_V2=1` / `H3_CUDA_ATTN_V1=1` to flip).

The L40S itself: 254 ms per block at VSA 0.8 vs 470 ms on the RTX A6000
(1.85×; the GEMMs run at 221 vs 121 TFLOP/s).

## The ComfyUI single-file NVFP4 Trim, end to end

Two rentals (the first L40S, 59505873, then 59507253 on the same host class
after the first destroyed itself on schedule; $0.47/h each).

First attempt: `ComfySource` loaded the file (AdaLN tables from
`adaln_t_table` in 30 s instead of 300 s from the bf16 tree, since the
rank-16 projections are 16-wide; DiT load 49 s) and the generation died
with `CUDA_ERROR_OUT_OF_MEMORY` after the prompt: the CUDA backend expanded
NVFP4 to bf16 at load, 36 GB for the 42 blocks.

Second attempt, with NVFP4 kept 4-bit on the device and expanded into the
shared bf16 scratch before each GEMM (`fw_dequant_nvfp4`, the int8 / int4
path): `h3 check` 95 / 95, `h3 plan` 20.1 GB predicted, and the clip:

| | FastH3-Trim, ComfyUI NVFP4 file, L40S |
|---|---|
| load (mmap, 4-bit upload) | 48.8 s |
| prompt (Qwen3-VL, 33 rows) | 127.5 s |
| denoise, 8 forwards | **94.9 s** (11.4–12.1 s each; v1 VSA kernels) |
| decode (Trim fast decoder) | 21.7 s |
| peak device memory (`nvidia-smi`, 5 s samples) | **24.3 GB** |

`fox-trim-nvfp4-frames.png` (frames 0 / 40 / 80 / 120) is the same scene
as the bf16-tree Trim clip of the A6000 run at the same seed: the loader's
nibble order, scale layout, `[gate | value]` swap and table convention are
all right, since any one of them wrong would not produce a fox. Per step
the L40S is 1.7× the A6000 (20.3 s, int8 from the bf16 tree).

Not yet: the ComfyUI file on WebGPU (the default `weight_nvfp4` expands to
f16 pairs, 2 B per weight), and a PSNR between the NVFP4 and bf16 Trim
clips (both exist in R2 under `h3-runs/`, not yet compared).
