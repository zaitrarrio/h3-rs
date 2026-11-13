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
  110 ms / block and clip denoise **45.0 s** before the FFN-up split
  (fastvideo-rs's MXFP8 recipe: 45.8 s), 96.5 ms / block after it (clip not
  re-measured), peak 41 GB (`docs/runs/2026-11-07-vast-pro6000-server-mxfp8`).
  **RTX 5090 (32 GB, 600 W on RunPod)** against ComfyUI's template on
  the same card class, prompt, canvas and seed (the h3-speedups page,
  `docs/runs/2026-11-13-runpod-5090-vs-comfy`): the V2 recipe does not
  fit resident (OOM at 31.7 GB), streamed with page-locked weights it
  runs 6.45 s per forward, denoise 52.0 s + decode 6.1 s against
  ComfyUI's 54.1 s total / 45.2 s sampler (7 % behind, on a pod whose
  pageable forward was 7.2 s; a faster pod ran the pageable path at
  6.6 s); FastH3 v1 4-step 30.1 s against 33.3 s (ahead); the 15 s clip
  (362 frames) 208.9 s against 213.6 s (ahead) at 31.2 GB peak; the raw
  model at 25 dense euler steps 13.2 s per step bf16, 9.95 s with
  `--qk8`, against ComfyUI's 6.8 s.
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
  TFLOP/s with every candidate (the others at ~750), so GEMMs with
  `n ≥ 24576` run as two row-slice GEMMs (`H3_CUDA_MX_SPLIT=p`,
  `H3_CUDA_MX_SPLIT_MIN=n`): 32.9 → 15.1 ms, the forward 110 → 96.5 ms per
  block (4.83 s) on the 600 W Server Edition; splitting the QKV shape gains
  nothing. Stored form: `h3 quantize --weights mxfp8 [--skip-adaln]` writes
  `<key>.weight.mx8` (E4M3 codes) + `.mxs` (swizzled scales), edges plain
  (`for_layer_key`); the V2 tree becomes 25.6 GB in 117 s and loads in 104 s
  against 202 s for bf16. `--decoder-weights mxfp8` puts the video
  decoder's block linears in MXFP8: 51.3 dB against the bf16 decode (the
  gate fastvideo-rs used), decode 9.75 → 7.5–7.9 s on a 600 W card. A clip
  with the split, the stored file and the MXFP8 decoder on the Server
  Edition (`docs/runs/2026-11-08-vast-pro6000-server-best`, the
  comparable card: bf16 bench 6.24–6.28 s per forward): **denoise 38.9 s,
  decode 5.6 s, 45.0 s after load, 37.9 GB** against bf16's 50.0 / 8.2 /
  58.7 s and fastvideo-rs's 45.8 s denoise. `H3_MXFP8_ALL_BLOCKS=1`
  quantizes the edge blocks too (−1.2 s, same PSNR band; off: the recipe's
  edges are calibrated). The cuBLASLt candidate timing makes two runs
  differ at the rounding level (`H3_CUDA_LT_NTH=i` pins it). Measure on a
  Server Edition whose bf16 bench forward is ~6.3 s: a Workstation Edition
  box ran 7.0 s, the Max-Q 9.6–10.1 s, and numbers across them do not
  compare.
- Decode host path: tiles are decoded one ahead of their download, each
  chunk's stitch / finish runs on a thread while the next chunk's kernels
  proceed (`on_chunk` stays in order; the first profiled decode table
  therefore holds two chunks), the stitch / chunk finish / 8-bit conversion
  are row-parallel (rayon) and tested against the sequential reference; the audio upsampler's crop is
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
  coarse kernel. `--vsa-qk8` (INT8 Q·Kᵀ, SageAttention's recipe: K shifted
  by its per-head mean, one int8 scale per head × 64-row tile, `mma.s8`
  k32 steps, bf16 P·V; CPU oracle `vsa::qk8_round`, WebGPU emulates) runs
  `fw_vsa_fine3e` (`H3_CUDA_VSA_FINE=f` the 4-blocks-per-SM twin, slower):
  on the 600 W Server Edition the fine stage goes 35.4 → 29.2 ms per call
  plus 1.5 ms of helpers (two-stage K mean 0.44 ms, single-read quant
  1.08 ms), the MXFP8 forward 4.79 → 4.54 s, the V2 clip denoise 39.2 →
  **37.2 s** (total after load 45.4 → 43.6 s); 17.5 dB against bf16
  attention, the MXFP8-vs-bf16 band, same scene, no artifacts
  (`docs/runs/2026-11-08-vast-pro6000-server-variants`). Opt-in like
  `--weights mxfp8`. The fine stage is instruction-issue and occupancy bound,
  not memory bound: sharing gathers across tiles and halving shared-memory
  reads gained nothing, the instruction diet (hardware bf16 cvt, scale in
  the exp FMA, mask skipped on full chunks, ldmatrix K, conditional
  rescale) and the third block per SM took it from 87 to 66.5 ms per call
  (`docs/runs/2026-11-07-vast-l40s-vsa-phases`). Read the SASS before
  guessing: `H3_PTX_OUT=f cargo test -p h3-cuda dump_ptx`, then `ptxas -v`
  (pip `nvidia-cuda-nvcc-cu12`) and `nvdisasm` (NVIDIA's apt `.deb`).
  Q fragments must never be re-read from L1 per mma (2.5× slower).
  Dense attention has the same INT8 Q·Kᵀ path (`--qk8`, the `--vsa-qk8`
  flag; `Backend::attention_qk8`, `Attention::Dense { qk8 }`): one scale
  per 64-row tile of Q and of mean-shifted K (`vsa::attn_qk8_round` is the
  oracle; WebGPU `attn_qk8.wgsl` emulates; CUDA `fw_attention_qk8_{64,128}`
  stages int8 K in the `attn_mma2` structure, `fw_attn_qk8_quant` + the VSA
  K-mean kernels prepare the codes). 127 checks with its cases (CUDA on a
  5090 at 7e-3–2e-2 of the oracle). On the 5090 the dense base step goes
  13.2 → 9.95 s (attention call 181 → 112 ms, 72 % → 61 % of the step),
  the 25-step clip 330 → 249 s; PSNR 20.5 dB against the bf16-attention
  clip (same scene, the 8-bit band), ComfyUI's Kitchen int8 step is 6.8 s
  (`docs/runs/2026-11-13-runpod-5090-vs-comfy`, pod 3). Opt-in.
- `--stream-blocks` (small cards): block weights stay in host memory in
  their device form; the DiT uploads block l + 1 on the backend's side
  queue while block l computes (`Backend::prefetch_begin / end / wait /
  release`, a non-blocking CUDA stream with events ordering the frees; two
  blocks resident). Byte-coded weights upload as bytes: the old path
  re-encoded them on the CPU each step and copied on the compute stream
  (5090: V2 forward 15.4 → 6.6 s, V1 14.4 → 6.0 s, dense base 21.3 →
  13.2 s; clips byte-identical, `h3 check` has streamed twins of the DiT
  cases). Streamed weights are page-locked at load (`pin_host`:
  `cuMemHostRegister` on the whole pages *inside* each buffer only,
  page-rounded ranges locked neighbours and broke every later copy; a
  guard per range lives in the DiT ahead of its blocks and unlocks on
  drop) and the small uploads of a prefetch go through a 64 MB pinned
  bounce buffer, so no copy of a prefetch waits on the side stream: on the
  5090 the V2 forward 7.2 → 6.45 s and the 15 s clip's 27.2 → 23.8 s on
  one pod, 127 / 127 checks either way; a pinned staging ring (memcpy of
  the whole block) had measured nothing and was removed.
  `H3_CUDA_NO_PIN=1` compares. The forward is then compute-bound and
  5090 pods differ by ~8 % through their clocks under the 575 W limit
  (median 2445 MHz on the slower kind; the link runs PCIe 5 × 8).
  `--steps N` sets the base model's uniform Euler ladder (ComfyUI's
  25-step template); distilled variants refuse it.
- NVFP4 on CUDA stays 4-bit on the device (codes + E4M3 scales, expanded
  into the bf16 scratch per GEMM like int8 / int4): the 42-block Trim runs in
  24.3 GB peak on an L40S; expanding at load needs 36 GB and OOMs a 48 GB
  card. WebGPU still expands NVFP4 to f16 pairs at load.
- Rentals: filter offers on `compute_cap ≥ 800` (a V100 was rented once by
  mistake) and arm the destroy timer on the box (`nohup setsid … sleep …
  DELETE /api/v0/instances/<id>/` with the instance key) plus a final
  self-destroy command: a timer in the controlling session dies with it.
  RunPod (`scripts/runpod/`, same R2 command channel): the EU network
  volume `fv-weights-h3-ltx-500` (750 GB since 2026-11-12, EU-RO-1, shared
  with fastvideo-rs, add-only, `/workspace/weights/<tree>/` with
  `.complete` / `.fv-revision` / `.fv-source`) holds h3-base, h3-ref2va,
  h3-8step (V2), fasth3-omni-8step-v1, fasth3-trim-8step-nvfp4, the 4-step
  LoRA, pika-h3-flex, open-h3-regenerate-2k, the fox assets (`h3-rs/`),
  the V2 / Omni / base / Pika AdaLN tables
  (`h3-rs/tables/`) and the stored MXFP8 V2 (`h3-8step-mxfp8/`): 689 GB
  (`docs/runs/2026-11-12-runpod-volume-fill`). It is a FUSE network
  filesystem that `cp` reads at 0.8 GB/s per stream (1.9 GB/s with
  eight) but the mmap loader pages at 0.2 GB/s, so **stage the trees a
  run needs onto the container disk** (`H3_STAGE="TREE[:SUB,…] …"` at
  boot, `stage` in a command; size the disk at `rp.py create`): the V2
  recipe stages in 70 s and then loads in 47 s + 86 s prompt against
  122 s + 195 s from the volume, 114 s ahead from the first run
  (`docs/runs/2026-11-13-runpod-pro6000-stage`; `madvise` readahead in
  the loader measured slower there and is opt-in, `H3_MADVISE=1`).
  Fetches write the volume at ~300 MB/s; clip times are unaffected either
  way. Pods self-terminate through the
  REST API (the boot's `terminate`, which uploads the running command's
  output first; `H3_POD_HOURS` backstop). EU-RO-1 stock is intermittent
  for every card: hunt with a retry loop, never two pods on the volume at
  once.
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
  merged in f32 at load, before quantization. Run with real weights on the
  600 W Server Edition (`docs/runs/2026-11-08-vast-pro6000-server-variants`):
  base T2VA (49 × 10.7 s), FL2VA (49 × 12.7 s), Ref2VA (49 × 18.3 s: the
  reference image is 7176 Qwen3-VL tokens at the contract's 2048 short
  edge), Omni Ref2VA (8 × 14.1 s); on RunPod's Server Edition with the EU
  weights volume (`docs/runs/2026-11-12-runpod-pro6000-variants`): Pika
  Flex FL2VA (6 × 12.9 s) and Ref2VA (6 × 18.6 s), Regen-2K (3 × 99 s at
  2560×1440, base-VAE decode 30.7 s), all coherent; from the filled volume
  (`docs/runs/2026-11-12-runpod-pro6000-v2-omni-trim`) the V2 best recipe
  (stored MXFP8, MXFP8 decoder, `--vsa-qk8`, tables) gives denoise 37.9 s /
  decode 5.8 s / 44.3 s after load (load 129 s, prompt 190 s from the
  volume), Omni Ref2VA 8 × 14.1–14.4 s and Trim NVFP4 from the ComfyUI
  file 8 × 5.5 s (denoise 44.1 s, decode 8.2 s: NVFP4 is expanded per
  GEMM, no FP4 tensor-core path). Frames floor: 124
  (Pika 120). Pika Ref2VA needs `transformer_ref` beside the base
  components (`--model-dir` of symlinks when they live in two trees). Backends keep f32 tables and bf16
  activations apart: `copy` never mixes them (the vision tower's per-frame
  rope tables are uploaded, not copied). Renoise noise buffers span the
  conditioning rows (`flow_step` reads noise at the rows' offset).
  `h3 regen2k` decodes with the base VAE like FastVideo's pipeline; the
  release's `vae_fast/` Sol decoder (`blocks.N.attn.to_qkv`, 24 blocks,
  16×16×5 attention windows, shifted on odd layers, 5 global rows per block,
  window-local RoPE; ComfyUI `comfy/ldm/minimax/vae.py`) is not implemented
  and paints a 16-pixel grid through the full-attention decoder.
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
