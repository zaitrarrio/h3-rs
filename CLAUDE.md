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
  power limit) recorded under `docs/runs/`. RTX A6000, 1344×768×124, VSA 0.8:
  470 ms/block CUDA (25.1 s per real forward at int8; FastH3-V2 clip: denoise
  201 s, decode 44 s), 7.6 s/block WebGPU. Profile with `H3_PROFILE=1`; the
  VSA fine stage is 34 % of a CUDA block, GEMMs 56 % at ~80 % of peak.
- WebGPU binds any activation over `max_storage_buffer_binding_size` (2 GiB
  on NVIDIA; the fused QKV is 3.3 GB at 38k tokens) as aligned row
  sub-ranges in `linear` / `head_split`; the FFN chunk is capped by
  `Backend::max_buffer_elems`. Any new op that touches a `[tokens, ≥ 14k]`
  activation needs the same treatment.
- Checkpoint formats: the FastVideo HF trees load (`transformer/` shards +
  `fastvideo_inference.json`, which overrides the preset). The ComfyUI
  single-file NVFP4 Trim (`blocks.N.attn.qkv_proj.weight` U8 + `weight_scale`
  E4M3 [out, in/16] + `weight_scale_2`, `adaln_t_table [1025, 16]`) does not:
  it needs a key map and an unswizzled-scale NVFP4 variant.
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
