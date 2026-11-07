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
  power limit) recorded under `docs/runs/`. 1344×768×124, VSA 0.8: RTX A6000
  470 ms/block CUDA (25.1 s per real forward at int8; FastH3-V2 clip: denoise
  201 s, decode 44 s), 7.6 s/block WebGPU; L40S 254 ms/block (Trim NVFP4
  clip: 11.9 s per forward, denoise 95 s, decode 22 s). `H3_PROFILE=1`.
- CUDA attention kernels: two bodies, both 95/95 on an L40S. v2
  (`attn_mma2_body`, 32 query rows per warp, 128-row blocks) is the default
  for dense attention (11 % faster on the L40S); v1 (16 rows per warp) stays
  the VSA fine default because v2 was 29 % slower on 64-token tiles and
  equal on 128-token ones: the fine stage is bound by the gathered K/V
  loads, not shared memory (`docs/runs/2026-11-07-vast-l40s-nvfp4-vsa2`).
  `H3_CUDA_ATTN_V1=1` / `H3_CUDA_VSA_V2=1` flip them. The next VSA
  experiments are a deeper cp.async pipeline and two query tiles per block.
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
