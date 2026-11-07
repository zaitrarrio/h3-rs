# h3-rs design

MiniMax H3 (33 B, video + audio, 50 Omni-Transformer blocks) and every recipe
of the FastVideo MiniMax H3 cookbook, in Rust, on consumer GPUs. The
methodology is fastwan-rs's: one small fused op vocabulary, a CPU reference
that defines every op, GPU backends checked against it, host-side
conditioning, precomputation of everything that does not depend on the
latent, and weight formats that fit the card.

## What runs where

| piece | where | why |
|---|---|---|
| Qwen3-VL conditioner (50 of 64 layers, vision tower with DeepStack) | backend, language layers streamed from the mmapped checkpoint one at a time | 64 GB of bf16 never fits next to the DiT; a prompt is encoded once (`h3 encode` galleries skip it entirely) |
| AdaLN tables | computed once per generation by streaming each block's `[96768, 2688]` weight through the backend, or loaded from a file | the modulation is a pure function of the timestep ladder; 13 B parameters stay off the device |
| Token refiner (text) | backend, once per prompt | timestep-free |
| 50 transformer blocks | backend, resident (native / int8 / int4 / NVFP4) or streamed per step from pre-quantized host copies | the `plan` command decides from the card's memory |
| VAE encoder / decoder, audio VAE, TAEH3, latent upscaler | backend, tiled | spatial tiles of 256 px and temporal chunks of 7 latent frames, stitched on the host exactly as the reference blends them |
| Audio `pre_block` attention projection | host f32 | 0.1 % of the work; the reference keeps it fp32 |
| Samplers, packing, row timesteps | host | geometry in f64 as the reference |

## The op vocabulary

`h3_core::Backend`: `linear` (with act epilogues and weight-only int8 / 4-bit /
NVFP4), `swiglu`, `rms_norm` and `residual_rms_norm` with a row-indexed
modulation table, `gate_residual`, `layer_norm`, `head_split` (per-head RMSNorm
+ partial rotate-half RoPE), `attention` (GQA, causal), `vsa_attention`,
`patchify` / `unpatchify`, `flow_step`, `axpby`, `channel_affine`, `conv2d`,
`conv3d`, `conv1d`, `conv_transpose1d`, `replicate_pad1d`, `group_norm`,
`snake`, `trilinear_resize`, `pixel_shuffle`, `tanh_clamp`, `clamp`. Every op
has a CPU implementation that is the contract, a WGSL kernel and a CUDA
kernel, and a conformance case in `h3-model/src/check.rs` (odd shapes, offsets,
every epilogue), plus whole-model cases (tiny DiT dense and VSA at each weight
format, tiny VAEs, TAEH3, upscaler).

Fusions: the H3 block is five kernels between the GEMMs (modulated norm,
head split, attention, residual + gated add + modulated norm, gated add); the
FFN runs in token chunks so its `[S, 28672]` hidden never has to fit; the VSA
gate projection reuses the attention scratch; the packed sequence is assembled
by writing each modality's projection straight into its rows (`Run`s) instead
of gathering.

## Variants

| variant | checkpoint | sampler | attention | extra |
|---|---|---|---|---|
| `h3` | MiniMaxAI/MiniMax-H3 `transformer` / `transformer_ref` | 50 uniform points (49 Euler forwards), shifts 12 / 3 | dense | T2VA, FL2VA (stretch first / cover last keyframe, 0.999 noise aug), Ref2VA (ordered refs, continuation) |
| `fasth3-v1` | FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree | rungs 999/749/500/250 | VSA 0.9 tile 64 | the LoRA repo variant merges `.diff` / `.set_weight` / rank-64 pairs |
| `fasth3-v2` | FastVideo-FastH3-8-Step-V2 | 8 rungs, shift 10 / 3 | VSA 0.8 tile 64 | |
| `fasth3-trim` | FastVideo-FastH3-Trim-8-Step-NVFP4 | as V2 | VSA 0.8 | 42 blocks, `adaln_basis` rank 16, NVFP4 linears (FlashInfer 128×4 scale swizzle undone at load), 26-block fast decoder |
| `fasth3-omni` | FastVideo-FastH3-Omni-8-Step-V1 (`transformer_ref`) | PDD: 32 fine intervals, node indices 0,4,…,32; fused heads per block, initial noise ×0.999 | VSA 0.9 tile 128, reference regions at keep rate 0.1 | Ref2VA only |
| `pika-flex` | Pika-Labs/Pika-H3-Flex adapters over the base | renoise knots 1000/750/500/250/100/40, noise from seed + 1 | dense | 480p / 768p buckets, both keyframes cover-cropped |
| `regen-2k` | Open-H3-Regenerate-2K | sigmas 0.909375, 6/7, 2/3, 0 (Euler), video clock only, silent audio conditions | VSA optional | latent upscaler to 90×160, rank-576 LoRA, Sharp curve AdaLN branches (`adaln_t_table.bin` is fetched separately), 24-block fast decoder, output trimmed to 121 frames |

A checkpoint's `fastvideo_inference.json` overrides the preset's sampler and
VSA numbers.

## Memory

`h3 plan` sizes a run: resident block weights at the chosen format (bf16 42 GB,
int8 21 GB, int4 12 GB, NVFP4 11 GB for the 50-block model), the AdaLN tables
(f32, entries × 50 × 96768), activations at the packed length (1344×768×124 is
~38 k rows), the FFN chunk, and the decoder's peak. When the blocks do not fit,
they are streamed per step from pre-quantized host copies (two resident at a
time); when the tables do not fit, they are uploaded per step.

## What is verified, what is not

- `cargo test --workspace`: core geometry and schedules against values
  derived from the reference sources; the CPU backend's op identities (dense
  VSA = attention, GQA = repeated heads, patchify round trips, chunked TAEH3 =
  whole decode); tiny-model shape and finiteness runs; the conformance suite on
  itself and on WebGPU (lavapipe).
- CUDA and WebGPU on an RTX A6000: 90 / 90 checks each; real clips from
  FastH3-8-Step-V2 and FastH3-Trim at 1344×768×124 with sound, matching
  their prompt (`docs/runs/2026-11-06-vast-a6000-first-gpu`).
- Parity with PyTorch: `scripts/reference/make_fixtures.py` specifies the
  fixtures; it has not been run (no PyTorch in the authoring environment), so
  the parity tests currently skip. The clips show the pipeline is right; the
  fixtures would show how right (a PSNR against the reference sample).
- Speed: A6000 numbers in that run directory; nothing on a 4090 / 5090 /
  AMD / Apple yet.

## Next

1. The VSA fine stage (29 % of a CUDA block after
   `docs/runs/2026-11-07-vast-l40s-vsa-phases`): instruction-issue and
   occupancy bound at 12 warps per SM. Next: 32 rows per warp with Q
   fragments in shared memory (not re-read from L1), a four-block
   configuration, `pool` fused into `head_split`.
2. `h3 quantize` the trees once (load was 280 s with int8 quantization at
   load) and the prompt gallery for demos (290 s cold prompt encode).
3. Decode: 11.9 s for 124 frames on the PRO 6000 Max-Q, now GPU-bound
   (10.7 s of kernels; the host side went from ~10 s to ~1.2 s in
   `docs/runs/2026-11-07-vast-pro6000-decode`). What is left is the
   decoder's GEMM efficiency on ~1.3 k-row tile batches (208 TFLOP/s vs
   fastvideo-rs's 397) and bf16 vs MXFP8 block linears. Audio decode is
   0.24 s (the 36 k copies are gone).
4. MXFP8 is in (`--weights mxfp8`: denoise at parity with fastvideo-rs on
   the 600 W PRO 6000, `docs/runs/2026-11-07-vast-pro6000-server-mxfp8`).
   The FFN-up FP8 GEMM is split along N (cuBLASLt ran `n=28672` at 355
   TFLOP/s, its halves run at 775: 4.83 s per forward, ~38.6 s per clip
   projected, to be measured as a clip). Next on that path: MXFP8 for the
   decoder's block linears (their 36 × 28-tile
   GEMMs are 73 % of the decode), a stored MXFP8 checkpoint (`h3 quantize`;
   quantizing at load costs 140 s of host time), and a Blackwell FP4
   tensor-core GEMM for the NVFP4 linears (now 4-bit resident, expanded to
   bf16 per GEMM; fastvideo-rs's FP4 VSA fine stage is its last 1.3×).
5. Generate the parity fixtures; fix what they find.
6. WebGPU: 16× behind CUDA on the same card; f16 activations and a
   tensor-core-friendly GEMM tiling are the two levers (the 4 GiB buffer /
   2 GiB binding limits also cap clips above ~38k tokens until activations
   are chunked more widely).
7. Block streaming with overlapped uploads (now serial); a 24 GB card run.
