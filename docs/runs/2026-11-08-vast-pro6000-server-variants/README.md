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

## The other official variants (h3-bin4 = 6fc41e2 unless noted)

All at 1344×768×124, seed 7, the fox prompt; keyframes / the reference
image are frames of the previous run's V2 fox clip (`fox.y4m`, re-encoded
from `fox-b-best.mp4`). Timings on the 600 W Server Edition:

| variant, task | load | prompt rows (Qwen3-VL, host) | denoise | decode | status |
|---|---|---|---|---|---|
| `h3` T2VA (bf16, dense) | 83.7 s | 33 in 96 s | 524.7 s = 49 × 10.71 s | | clip (`base-t2va-frames.png`) |
| `h3` FL2VA, first + last keyframe | 68.3 s | 2065 in 70 s | 621.4 s = 49 × 12.69 s | | clip (`base-fl2va-frames.png`) |
| `h3` Ref2VA (`transformer_ref`), one image | 100.8 s | 7209 in 126 s | 896.0 s = 49 × 18.29 s | | clip (`base-ref2va-frames.png`) |
| `pika-flex` FL2VA / Ref2VA, 768p bucket | 83 s | | | | AdaLN tables (13 renoise knots, 126 MB) in 145 s; both clips **panicked** after load: `renoise needs noise` (below) |
| `fasth3-omni` Ref2VA (PDD, VSA 0.9 tile 128) | 90.7 s | 7209 in 122 s | 112.4 s = 8 × 14.07 s | 8.1 s | clip (`omni-ref2va-frames.png`); tables (17 PDD knots, 165 MB) in 182 s |
| `regen-2k` from the 768p draft | 118 s | | 3 × 97.7 s at 2560×1440 (+ 103 s prepare: draft encode, upscale, tables) | 21.1 s | ran, but the frames carry a 16-pixel grid (`regen-2k-frames.png`, below) |

The reference image is presented to Qwen3-VL at a 2048 short edge
(`REFERENCE_IMAGE_SHORT_EDGE`, the released contract), which is 7176
vision tokens: the Ref2VA prompt is 7209 rows and the DiT sequence grows by
that much, hence 18.3 s per dense forward against 10.7 s for T2VA, and the
host text encoder takes two minutes. Omni's 14 s per VSA forward is on the
same long sequence (and bf16 weights).

### Found and fixed

1. **61-frame requests** (my scripts): both 61-frame attempts were refused
   by the geometry guard; H3's floor is 124 frames (Pika's 120).
2. **Qwen3-VL vision tower on CUDA** (`h3-bin` → `h3-bin4`): the per-frame
   rope tables were copied from f32 tables into bf16 scratch; the CUDA
   backend keeps the dtypes apart and asserted (`output-5-omni-first-try.txt`),
   so every image-conditioned run (FL2VA, Ref2VA, Omni) died after load. One
   frame's table is now uploaded once (every frame shares the coordinates).
3. **Fast decoder weights** (`output-6-regen-first-try.txt`): the release's
   `vae_fast/` file fuses `to_qkv`; the loader read `to_q/k/v`. Fixed, then
   found irrelevant: see 5.
4. **Pika renoise** (`pika-fl2va.log`): `flow_step` reads the renoise noise
   at the rows' offset, the pipeline allocated it for the target rows only;
   with a keyframe or a reference the offset is non-zero and the CUDA bounds
   assert fired (the CPU would have sliced out of range). Fixed in 9a5bf77
   with a tiny-pipeline regression test; **not yet re-run on a GPU**.
5. **Regen-2K decoder**: `vae_fast/fast_h3_vae_noncausal_24l.safetensors`
   is the Sol-H3-VAE decoder (`blocks.N.attn.to_qkv`), a 24-block model with
   16×16-token spatial × 5-latent temporal attention windows, half-window
   shifts on odd layers, five global rows per block and window-local RoPE
   (ComfyUI `comfy/ldm/minimax/vae.py`, `NonCausalViT3DDecoder`). Run through
   the full-attention decoder it paints the 16-pixel grid in
   `regen-2k-frames.png`. FastVideo's `open_h3_regenerate_2k` pipeline does
   not use it: it decodes with `base_model/vae` (36 blocks) and trims 124
   frames to 121 (`stages/finalize.py`). `h3 regen2k` now does the same
   (commit after 9a5bf77); the Sol decoder is not implemented. **Not yet
   re-run on a GPU**; the 2K decode through the full decoder will cost more
   than the 21 s measured here.

Cost: the box ran 23:27–02:43 UTC (3.3 h, $1.54/h) and pulled ~420 GB; the
account went from ~$8.3 to $2.72, which is why the Pika and Regen-2K re-runs
wait for credit.
