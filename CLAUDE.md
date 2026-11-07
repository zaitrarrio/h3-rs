# CLAUDE.md: h3-rs

State and invariants only; the reasoning is in `docs/design.md`.

- One op vocabulary (`h3_core::Backend`), three implementations. A new op
  means: CPU reference, WGSL kernel, CUDA kernel, and a case in
  `h3-model/src/check.rs`, all in the same change.
- The CPU backend is ground truth. GPU code is trusted only after
  `h3 check --backend <b>` passes on that hardware. Both pass on an RTX
  A6000 (driver 570.195.03): CUDA 90/90 at 3e-2, WebGPU 90/90 at 1e-2
  (`docs/runs/2026-11-06-vast-a6000-first-gpu`). The WebGPU suite also runs
  on lavapipe in CI, including a forced-64-KiB-binding pass that exercises
  the row-chunked GEMM / head_split paths; CUDA kernels are NVRTC-compiled
  in `cargo test -p h3-cuda`. Neither has run on AMD / Apple.
- No speed claim without a measurement on named hardware (card, driver,
  power limit) recorded under `docs/runs/`. 1344×768×124, VSA 0.8, per
  block / per real FastH3-V2 forward: RTX A6000 470 ms / 25.1 s (int8), L40S
  230 ms / 11.9 s, RTX PRO 6000 Max-Q (300 W, sm_120) 185 ms / 9.6–10.1 s
  (bf16 and int8 alike; Trim NVFP4 8.2–8.6 s). Video decode 11.9 s for
  124 frames (GPU-bound, 1.53 s per chunk; was 20.5 s with half host-side),
  audio decode 0.24 s. The Max-Q runs the denoise power-capped at ~1.4 GHz
  (3.09 GHz boost): its per-block number is 197 ms sustained vs 185 ms in a
  cold bench, and fastvideo-rs's 600 W Server Edition is not comparable
  without that caveat. WebGPU 7.6 s/block on the A6000. `H3_PROFILE=1`
  prints a kernel table per stage of `generate`
  (`docs/runs/2026-11-07-vast-pro6000-clips`, `…-pro6000-decode`).
  **RTX PRO 6000 Server Edition (600 W, 2.2–2.4 GHz under load, the card
  class fastvideo-rs measures on)**: bf16 126 ms / block, 6.3 s per forward,
  V2 clip denoise 50.6 s, decode 8.9 s, audio 0.19 s; `--weights mxfp8`
  110 ms / block, denoise **45.0 s** (fastvideo-rs's MXFP8 recipe: 45.8 s),
  peak 41 GB (`docs/runs/2026-11-07-vast-pro6000-server-mxfp8`).
- MXFP8 (`--weights mxfp8`, `quant::Mxfp8`): W8A8, E4M3 codes with one E8M0
  scale per 32 input columns in cuBLASLt's `VEC32_UE8M0` swizzle
  (`mx_scale_offset`, rows padded to 128, groups to 4), the E4M3 / E8M0
  twins bit-exact with Sol-H3 / fastvideo-rs. `WeightQuant::for_block`
  keeps blocks 0–1 and the last three in bf16. The CPU reference rounds the
  bf16 activations the same way (`Mxfp8::round_rows`) before its f32 GEMM;
  WebGPU (`mx_act.wgsl`) and pre-Blackwell CUDA (`H3_CUDA_NO_FP8=1` too)
  emulate exactly, sm_100+ runs cuBLASLt FP8 with the bias (+ tanh-GELU)
  epilogue. The first call on each shape times cuBLASLt's candidates
  (`H3_CUDA_LT_NTH=i` forces one); the `n=28672` FFN-up shape runs at 355
  TFLOP/s with every candidate (the others at ~750): splitting it along N
  is the open lever. No stored MXFP8 form yet (`h3 quantize` refuses it).
- Decode host path: tiles are decoded one ahead of their download, the
  stitch / chunk finish / 8-bit conversion are row-parallel (rayon) and
  tested against the sequential reference; the audio upsampler's crop is
  the transposed conv's `pad` (symmetric) with ×ratio in the filter. Keep
  `cargo test -p h3-model --release -- --ignored host_decode_chunk_timing`
  under ~0.2 s per chunk.
- CUDA attention kernels (all 97/97 on an L40S): dense attention runs v2
  (`attn_mma2_body`, 32 rows per warp, 128-row blocks; `H3_CUDA_ATTN_V1=1`
  for v1). The VSA fine stage runs `vsa_fine3_body` variant d (16 rows per
  warp, 32-key chunks, K/V double-buffered a chunk ahead, Q fragments
  resident, swizzled 32 KB staging, 168 registers, three blocks per SM);
  `H3_CUDA_VSA_FINE=v1|a|b|c` select the others, `H3_CUDA_VSA_PAIR=1` the
  two-tiles-per-block body, `H3_CUDA_VSA_COARSE1=1` the one-query-tile
  coarse kernel. The fine stage is instruction-issue and occupancy bound,
  not memory bound: sharing gathers across tiles and halving shared-memory
  reads gained nothing, the instruction diet (hardware bf16 cvt, scale in
  the exp FMA, mask skipped on full chunks, ldmatrix K, conditional
  rescale) and the third block per SM took it from 87 to 66.5 ms per call
  (`docs/runs/2026-11-07-vast-l40s-vsa-phases`). Read the SASS before
  guessing: `H3_PTX_OUT=f cargo test -p h3-cuda dump_ptx`, then `ptxas -v`
  (pip `nvidia-cuda-nvcc-cu12`) and `nvdisasm` (NVIDIA's apt `.deb`).
  Q fragments must never be re-read from L1 per mma (2.5× slower).
- NVFP4 on CUDA stays 4-bit on the device (codes + E4M3 scales, expanded
  into the bf16 scratch per GEMM like int8 / int4): the 42-block Trim runs in
  24.3 GB peak on an L40S; expanding at load needs 36 GB and OOMs a 48 GB
  card. WebGPU still expands NVFP4 to f16 pairs at load.
- Rentals: filter offers on `compute_cap ≥ 800` (a V100 was rented once by
  mistake) and arm the destroy timer on the box (`nohup setsid … sleep …
  DELETE /api/v0/instances/<id>/` with the instance key) plus a final
  self-destroy command: a timer in the controlling session dies with it.
- WebGPU binds any activation over `max_storage_buffer_binding_size` (2 GiB
  on NVIDIA; the fused QKV is 3.3 GB at 38k tokens) as aligned row
  sub-ranges in `linear` / `head_split`; the FFN chunk is capped by
  `Backend::max_buffer_elems`. Any new op that touches a `[tokens, ≥ 14k]`
  activation needs the same treatment.
- Checkpoint formats: the FastVideo HF trees (`transformer/` shards +
  `fastvideo_inference.json`, which overrides the preset) and ComfyUI's
  single-file H3 layout (`weights::ComfySource`, detected by
  `video_patch_proj` + `blocks.0.attn.qkv_proj`): fused `qkv_proj` row
  thirds, `mlp.fc1` is `[gate | value]` (Diffusers: `[value | gate]`),
  `comfy_quant` NVFP4 = codes U8 with the **first code in the high nibble**
  + E4M3 scales in the 128×4 block layout + `weight_scale_2`, and
  `adaln_t_table [grid, rank]` instead of the time embedder (lerp at
  `t·(grid−1)`, no SiLU). NVFP4 block scales (FlashInfer, Comfy) are torchao's
  `to_blocked`: `Nvfp4::swizzled_index`; the old formula was the identity
  and loaded garbage scales. FlashInfer / compressed-tensors put the first
  code in the low nibble (`Nvfp4.first_high`). A stored format decides what
  the device gets (`--weights` cannot re-quantize an NVFP4 file).
- The reference is FastVideo's `pipelines/basic/minimax_h3` (+ `pika_h3_flex`,
  `open_h3_regenerate_2k`) and the released configs. Checkpoint contracts that
  are easy to get wrong: channel-major patch features (`patchify`), the
  `value, gate` halves of every SwiGLU `fc_in`, per-head `norm_q/norm_k`
  `[128]`, rotate-half RoPE over the first 96 of 128 channels, AdaLN row
  `timestep_idx·3 + tag` with slot order shift/scale/gate (attn) then (mlp),
  `norm_out.linear` giving `shift, scale`, keyframe rows at `0.999·clean +
  0.001·noise` with the "last" anchor at `origin + span − 5/3`, `x0 = x + σ·v`.
- Weight keys are the Diffusers names, checked against the manifests in
  `h3_core::weights`. A missing key is an error, never a default. Check gate /
  weight presence with `has_linear`, never `contains("….weight")` (quantized
  and NVFP4 files store other names; `nvfp4_module` maps `ff.net.0.proj →
  ff.fc_in` etc.).
- Variants are presets in `h3_core::variant`; a checkpoint's
  `fastvideo_inference.json` overrides the preset. Base and Pika Ref2VA use
  `transformer_ref`. Adapters (Pika LoRA, Regen-2K LoRA, FastH3 V1 diffs) are
  merged in f32 at load, before quantization.
- AdaLN: the 13 B modulation weights are never resident. Tables are computed
  per generation by streaming each block's weight through the backend once,
  or loaded from an `h3 adaln-tables` file; `h3 quantize --skip-adaln` then
  writes a DiT without them.
- VSA-H3 (`h3_core::vsa`): dense prefix tiles never straddle segments;
  per-region top-k with ties to the lower index; the gate branch is
  `softmax(all tiles) · V̄ · gate`; reference videos are regions at keep rate
  0.1 (Omni), continuation clips stay dense. The CPU oracle is
  `h3_cpu::vsa_reference`.
- Parity fixtures (`scripts/reference/make_fixtures.py`) have not been
  generated: the container had no PyTorch. The parity tests skip without
  them and must be run before any parity claim.
- WGSL rules (from fastwan-rs): named vec4 accumulators, never arrays indexed
  in loops; zero loop-local accumulators explicitly; every declared binding
  statically used. Kernels and their CUDA twins share names (`fw_<op>`).
- `h3 serve` binds 127.0.0.1, answers CORS for one `--allow-origin` only, and
  rejects non-JSON or foreign-origin generate requests; `web/demo.html` is
  compiled into the binary (rebuild after editing).
- Tests: `cargo test --workspace` (the WebGPU suite needs a Vulkan driver:
  `mesa-vulkan-drivers`, `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`,
  `H3_ALLOW_SOFTWARE=1`); the CUDA compile test needs libnvrtc on
  `LD_LIBRARY_PATH` (`pip install nvidia-cuda-nvrtc-cu12`).
- The tiny configs (`*Config::tiny*`) form one pipeline: `h3 serve --synthetic`
  and the checker run them end to end, and `config::tests::tiny_models_agree`
  pins their channel counts to each other. Change them together.
