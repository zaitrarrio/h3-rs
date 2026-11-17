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
| `regen-2k` | Open-H3-Regenerate-2K | sigmas 0.909375, 6/7, 2/3, 0 (Euler), video clock only, silent audio conditions | VSA optional | latent upscaler to 90×160, rank-576 LoRA, Sharp curve AdaLN branches (`adaln_t_table.bin` is fetched separately), decoded by the base model's 36-block VAE as in FastVideo's pipeline (the release's `vae_fast/` Sol decoder is a windowed-attention model this crate does not implement), 124 frames trimmed to 121 |

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
- Parity with PyTorch: `scripts/reference/make_fixtures.py` was run on
  2026-11-14 (CPU, FastVideo b4faeae, transformers 5.23.0) and the four
  parity tests pass at 2e-3: the tiny DiT forward, both VAEs' encode and
  decode, and the Qwen3-VL conditioner layer by layer. The first run found
  only script-side mistakes (API shape, a wrong packed layout, FastVideo's
  internal key names, transformers 5's `mm_token_type_ids`) and one real
  divergence of definition: transformers' recorded `hidden_states[k]`
  exclude the DeepStack injection that follows block k − 1, while
  FastVideo's own H3 text stack (and ours) returns the injected state; the
  test adds the injection to the expectation, and the real conditioner's
  layer 50 is past the last DeepStack index either way. A PSNR against a
  reference clip at full size is still open (it needs the real weights on
  both sides).
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
   projected). The stored MXFP8 checkpoint, the MXFP8 decoder and the
   chunk-overlapped decode are in: on the Server Edition the V2 clip is
   38.9 s denoise + 5.6 s decode against fastvideo-rs's 45.8 s MXFP8
   denoise (`docs/runs/2026-11-08-vast-pro6000-server-best`). Next on that
   path: a Blackwell FP4
   tensor-core GEMM for the NVFP4 linears (now 4-bit resident, expanded to
   bf16 per GEMM; fastvideo-rs's FP4 VSA fine stage is its last 1.3×), and
   the VSA fine stage itself, 35 % of an MXFP8 forward. INT8 Q·Kᵀ
   (`--vsa-qk8`) took the stage to 30.6 ms per call and the clip denoise
   to 37.2 s on the Server Edition
   (`docs/runs/2026-11-08-vast-pro6000-server-variants`); the int8 body
   is instruction bound like the bf16 one, so the next step there is
   INT8 / FP8 P·V (SageAttention2's second half), which also halves the
   V staging.
5. ~~Generate the parity fixtures; fix what they find.~~ Done 2026-11-14: generated, four parity tests green (see above).
6. WebGPU: 16× behind CUDA on the same card; f16 activations and a
   tensor-core-friendly GEMM tiling are the two levers (the 4 GiB buffer /
   2 GiB binding limits also cap clips above ~38k tokens until activations
   are chunked more widely).
7. ~~Block streaming with overlapped uploads~~: done (prefetch on a side
   stream, byte uploads, page-locked weights with a pinned bounce for the
   small copies; 5090 record: the forward is compute-bound now). On a
   24 GB card (RTX 4090, `docs/runs/2026-11-14-vast-4090-24gb`) V2 at
   1344×768×124 streams in 19.1 GB with the GPU waiting 0.6 ms per block
   over PCIe 4: denoise 107 s, 125.7 s with decode, bf16 weights (the
   MXFP8 path is emulated on Ada and only costs decode time).
7b. INT8 dense attention (`--qk8`): the VSA fine stage's recipe
   (per-64-row-tile int8 Q and mean-shifted K, `mma.s8` scores, bf16
   P·V) inside the dense `attn_mma2` kernel, with the CPU oracle and the
   WebGPU emulation. On the 5090 the attention call goes 181 → 112 ms
   and the dense base step 13.2 → 9.95 s; ComfyUI's Kitchen int8 step is
   6.8 s, which puts its attention near 80 ms per call: FP8 P·V and a
   tighter pipeline are the remaining dense work. The base model, Pika
   and Regen-2K are all dense.
7c. FP8 P·V (`--pv8`, SageAttention2's second half) in both kernels: the
   dense `fw_attention_pv8` and the VSA `fw_vsa_fine3g`. The point of the
   design is that the CPU can reproduce the kernel's arithmetic bit for
   bit where it matters: V is smoothed by its per-channel mean and rounded
   to E4M3 with one scale per (KV) head, so a channel offset does not eat
   the 3-bit mantissa; the softmax row maximum is rounded up to an
   integer in log2 units, so the kernel's running rescales are exact
   powers of two and the E4M3 rounding of P at a chunk equals the
   oracle's rounding at the final maximum (E4M3 rounding commutes with
   powers of two inside the normal range); P is scaled by 2^6 so every
   weight down to 2^-12 of the row's largest is normal; the denominator
   sums the unrounded P. V is transposed once per call into `[d][keys]`
   with each 32-key chunk in the mma B-fragment key order (and, for VSA,
   in tile-slot order), so the P·V half reads V with one `ldmatrix.x2`
   per eight channels and no shared-memory transpose: one k32 FP8 mma
   replaces two k16 bf16 ones and the V staging halves. `cvt.rn.satfinite
   .e4m3x2` packs P (sm_89+; the host oracle's `e4m3_satfinite` is its
   twin, the same one MXFP8 uses). On a 5090
   (`docs/runs/2026-11-13-runpod-5090-fp8-pv`): the dense call 119.5 →
   82.1 ms on one pod (the ~80 ms ComfyUI's step implies), the dense base
   step 10.35 → 8.66 s (7.84 on the faster pod kind) against ComfyUI's
   6.8 s, the VSA fine stage 34.2 → 27.6 ms per call and the V2 forward
   6.39 → 6.11 s (denoise 49.2 s against the 45.2 s sampler); 22–25 dB
   against the INT8-only clips, the 8-bit band. Attention is 52 % of the
   dense step now and the rest of the gap is the MXFP8 GEMM side (478
   TFLOP/s with per-call activation quantisation) plus the small kernels
   between them. The dense first steps of a VSA run and the dense layers
   now go through the dense kernel plus the gated pooled branch (a dense
   plan is defined that way in the CPU reference), where the all-tiles
   fine stage had cost ~2× for the same softmax. The checker's lesson:
   an E4M3 quantiser is sensitive to
   whether its input was already bf16, so the pv8 cases feed both sides
   bf16-representable values, and the oracle reports the outputs whose P
   rounding sits within a few f32 ulps of a boundary as slack, so an
   exp2 that differs from libm's in the last bits does not fail a kernel.
8. Variants: all of them run (Pika Flex and Regen-2K on RunPod,
   `docs/runs/2026-11-12-runpod-pro6000-variants`). The RunPod volume
   (750 GB) holds every variant's weights, the AdaLN tables and a stored
   MXFP8 V2 (`docs/runs/2026-11-12-runpod-volume-fill`); the fastest
   recipe reproduces from it (`…-runpod-pro6000-v2-omni-trim`). The
   mmap loader pages the network volume at 0.2 GB/s while `cp` reads it
   at 0.8–1.9 GB/s, so pods stage their trees onto NVMe first (70 s for
   the V2 recipe, then load 47 s + prompt 86 s against 122 + 195 s;
   `docs/runs/2026-11-13-runpod-pro6000-stage`); kernel readahead advice
   in the loader measured slower on FUSE and stayed opt-in. A device or
   pre-encoded (`h3 encode`) prompt path is still the next host cost.
   Ref2VA / Omni prompts are 7.2 k rows and the host
   Qwen3-VL takes two minutes: a device text encoder is the next host
   cost. Optional: the Sol-H3-VAE windowed decoder (`vae_fast/`, 24
   blocks, ComfyUI decodes it in ~⅓ of the full decoder's time) needs a row
   gather / scatter op and block-major rows.
