# 2026-11-08/09: RTX PRO 6000 Server Edition (600 W): the official variants, and INT8 Q·Kᵀ

Vast instance 59800772 (offer 59376656): NVIDIA RTX PRO 6000 Blackwell
**Server Edition** (sm_120, 96 GB, **600 W**), driver 615.71.09, 256 CPU
cores, Michigan, $1.54/h. Weights: the MiniMax-H3 base tree + its
`transformer_ref` (265 GB from R2 in 487 s), then from Hugging Face in the
background: Pika-H3-Flex adapters (5.2 GB), Open-H3-Regenerate-2K (20 GB),
FastH3-Omni `transformer_ref` (66 GB), and for the INT8 Q·Kᵀ clip pair the
FastH3-V2 transformer. Raw outputs `output-N-*.txt` (trace lines and
presigned URLs stripped), per-clip logs `*.log`, frame grids `*-frames.png`.

Binaries: `h3-bin` = `498adb4^` (the validation round), `h3-bin2` = 498adb4
(INT8 Q·Kᵀ behind `--vsa-qk8`), `h3-bin3` = dc4e0c8 (two-stage K mean,
single-read quantizer).

Conformance: `h3 check --backend cuda` **109 / 109** (h3-bin) and
**115 / 115** with the six `vsa … qk8` cases (h3-bin2), tolerance 3e-2.

## Base MiniMax-H3 (`--variant h3`, bf16, dense attention, 49 Euler forwards)

AdaLN tables (`h3 adaln-tables --variant h3`, 99 timesteps): 960 MB in
197 s.

| | load | prompt (Qwen3-VL on the host) | denoise | total |
|---|---|---|---|---|
| T2VA 1344×768×124, seed 7, the fox prompt | 83.7 s | 96.1 s | **524.7 s** (49 × 10.70–10.72 s) | 12 m 00 s wall |

`base-t2va-frames.png` (frames 0 / 20 / 40 / 60): a coherent fox at dawn
in a snowy pine forest, so the base sampler (49 Euler forwards on the `h3` preset)
runs end to end on real weights. The host-side text encoder
is now the second-largest cost of a base clip after the denoise.

The first FL2VA and Ref2VA attempts asked for 61 frames and were refused by
the pipeline's own geometry guard (`H3 generates 5–15 s at 24 fps; aligned
num_frames 73 is outside 124–362`): the floor is 124 frames. They were
re-run at 124 frames (below).

## INT8 Q·Kᵀ in the VSA fine stage (`--vsa-qk8`, 498adb4 → dc4e0c8)

SageAttention's recipe: per head K is shifted by its mean over the
sequence (softmax does not see the shift), Q and K − mean are rounded to
int8 with one scale per (head, 64-row tile), the fine stage's Q·Kᵀ runs
`mma.sync m16n8k32 s8`, scores are rescaled in f32, P·V stays bf16. The
CPU oracle (`h3_core::vsa::qk8_round`) and the WebGPU emulation match the
CUDA kernels in the six new `vsa … qk8` check cases (rel 7e-3–1.7e-2 at
tolerance 3e-2; `output-2-int8-qk-bench.txt`).

Bench (`bench --variant fasth3-v2 --weights mxfp8`, synthetic weights,
37 966 tokens, `H3_PROFILE=1`, per fine-stage call over 100 calls):

| | fine stage | K mean | Q/K quant | forward (unprofiled, steps 1–2) |
|---|---|---|---|---|
| bf16 Q·Kᵀ (fine3d) | 35.43 ms | | | 4.79–4.80 s |
| `--vsa-qk8`, fine3e (3 blocks/SM), h3-bin2 | 29.16 ms | 2.00 ms (one block per head) | 1.27 ms | 4.61 s (profiled) |
| `--vsa-qk8`, fine3f (4 blocks/SM) | 34.54 ms | | | |
| `--vsa-qk8`, fine3e, h3-bin3 (two-stage K mean, single-read quant) | 29.15 ms | **0.42 + 0.01 ms** | **1.08 ms** | **4.54–4.55 s** |

fine3f (4 blocks per SM, fewer registers) is slower than fine3e, as the
d → c step was for the bf16 body: the fine stage wants the registers. The
int8 body itself is 1.21× faster than bf16; the helpers cost 1.5 ms per
call, so the stage is 30.6 vs 35.4 ms per call and the forward **5.2 %**
faster.

Clip pair (V2, seed 7, the fox prompt, 1344×768×124, `--weights mxfp8
--decoder-weights mxfp8` quantized at load from the bf16 tree, h3-bin3):

| | (a) bf16 attention | (b) `--vsa-qk8` |
|---|---|---|
| denoise, 8 forwards | 39.16 s (5.07, then 4.84–4.89 s) | **37.18 s** (4.82, then 4.60–4.64 s) |
| video decode | 5.71 s | 5.83 s |
| total after load | 45.41 s | **43.56 s** |
| load (quantizing at load) | 85.8 s | 83.6 s |

PSNR (b) vs (a): 17.5 dB (Y 13.0, U 28.0, V 29.9), the same band as MXFP8
vs bf16 weights (16.7 dB): the 8-step DMD sampler turns any perturbation
into a different sample of the same scene. The grids
`v2-mxfp8-bf16-attention-frames.png` / `v2-mxfp8-int8-qk-frames.png`
(frames 0 / 40 / 80 / 120) show the same fox, trees and light with small
pose differences and no artifacts. `--vsa-qk8` stays opt-in like
`--weights mxfp8`, and is part of the fastest measured recipe.
