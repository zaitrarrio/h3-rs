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
  `--qk8`, against ComfyUI's 6.8 s. With FP8 P·V (`--pv8`,
  `docs/runs/2026-11-13-runpod-5090-fp8-pv`, the slower 5090 pod kind:
  its bf16 base step is 13.9 s, pod-3 numbers in parentheses): V2 forward
  6.46 s (6.11), denoise 52.0 s (49.2) + decode 6.6 s against 54.1 /
  45.2 s; dense base step 8.66 s (7.84) against 6.8 s, the attention
  call 82 ms, 52 % of the step. **With the streaming fixes and the QKV
  split** (`docs/runs/2026-11-13-vast-5090-streaming-gemm`, a 600 W
  Vast 5090 of the faster kind, driver 595.91.07, clips byte-identical
  to the released binary's): V2 forward 6.26 → **5.32 s**, denoise
  **42.6 s** + decode 6.3 s = **48.9 s**, ahead of ComfyUI's 45.2 s
  sampler / 54.1 s total; every block MXFP8 as well 4.9 s per forward,
  39.2 / 45.5 s (the usual 22 dB band, off by default); dense base step
  8.41 → **7.72 s** against 6.8 s (7.34 s with every block MXFP8), the
  25-step denoise 193 s against the 169 s sampler. The GPU now idles
  ~0.2 s per V2 forward; what remains against ComfyUI's dense step is
  the MXFP8 GEMMs (565 TFLOP/s) and the five bf16 edge blocks. With the
  dense steps of a VSA run through the dense kernel
  (`docs/runs/2026-11-14-vast-5090-dense-steps`, a 575 W 5090, driver
  615.71.09): the raw model at ComfyUI's block-sparse row (keep 10 %, 5
  dense first steps) **130.9 / 123.9 s against 131.4 / 122.7 s**, at
  parity; 3 dense first steps 124.0 / 116.8 against 118.0 / 109.6 s; V2
  at ComfyUI's budget 44.7 s denoise (dense steps 8.1 s, sparse 4.6 s),
  still behind V2 at keep 20 % (42.2 s), which stays the recipe.
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
  large shapes may run as two row-slice GEMMs (`H3_CUDA_MX_SPLIT=p`): on
  the Server Edition the FFN-up 32.9 → 15.1 ms, the forward 110 →
  96.5 ms per block (4.83 s), the QKV shape no gain; on an RTX 5090 the
  QKV shape 23.0 → 13.2 ms too (V2 forward 5.70 → 5.32 s). The slices
  round like the whole GEMM (candidate pinned, clips byte-identical), so
  each shape's first call times whole against split, both tuned, and
  keeps the faster (`mx_choice`; `H3_CUDA_MX_SPLIT_MIN=n` forces the
  rule). On the 5090 it picks the split for the QKV and FFN-up shapes (22.2 → 12.5 ms, 31.1 → 17.7 ms) and only with a 10 % margin: a 0.2 ms shape that measured the same either way split in one run and not in another, and its slices do not round like its whole GEMM. A plain per-tensor-scaled FP8 GEMM of the same shape (timed under `H3_STALL=1` on the first call) runs 18.1 / 25.4 ms whole: the block scaling costs ~25 %, the split saves ~46 % with it kept, so a per-row FP8 weight format is not worth building (`docs/runs/2026-11-14-vast-5090-dense-steps`). Stored form: `h3 quantize --weights mxfp8 [--skip-adaln]` writes
  `<key>.weight.mx8` (E4M3 codes) + `.mxs` (swizzled scales), edges plain
  (`for_layer_key`); the V2 tree becomes 25.6 GB in 117 s and loads in 104 s
  against 202 s for bf16. `--decoder-weights mxfp8` puts the video
  decoder's block linears in MXFP8: 51.3 dB against the bf16 decode (the
  gate fastvideo-rs used), decode 9.75 → 7.5–7.9 s on a 600 W card. A clip
  with the split, the stored file and the MXFP8 decoder on the Server
  Edition (`docs/runs/2026-11-08-vast-pro6000-server-best`, the
  comparable card: bf16 bench 6.24–6.28 s per forward): **denoise 38.9 s,
  decode 5.6 s, 45.0 s after load, 37.9 GB** against bf16's 50.0 / 8.2 /
  58.7 s and fastvideo-rs's 45.8 s denoise; with `--vsa-qk8 --pv8` and the
  tables, on RunPod's Server Edition from the staged volume
  (`docs/runs/2026-11-14-runpod-6000s-probe-bug`, `fixcheck/`): **denoise
  36.6 s, decode 5.7 s, 42.9 s after load, 39.6 GB resident** (streamed
  37.1 s at 18.7 GB, byte-identical). `H3_MXFP8_ALL_BLOCKS=1`
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
  (`docs/runs/2026-11-08-vast-pro6000-server-variants`); with FP8 P·V as
  well the fine stage is 24.1 ms per call and the stored-file clip
  37.8 → 36.6 s denoise (`…-11-14-runpod-6000s-probe-bug`). Opt-in like
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
  FP8 P·V on top of either INT8 path (`--pv8`, implies the INT8 Q·Kᵀ;
  `Attention::{Dense, Vsa} { pv8 }`, `VsaSpec::with_pv8`; the recipe is
  `docs/design.md` 7c): V minus its per-head mean as E4M3 codes with one
  scale per KV head (`vsa::attn_v8_round`), the row maximum rounded **up
  to an integer in log2 units** so the kernels' running rescales are
  exact powers of two and round P like the oracle's final maximum,
  `p = exp2(s − ⌈m⌉ + 6)` rounded to E4M3 for the product and exact for
  the denominator (`vsa::attention_pv8_reference`, `PV8_SHIFT`); V is
  transposed once per call into `[d][keys]` in mma B-fragment order
  (`fw_attn_v8_quant` + `fw_attn_v8_max`, VSA in tile-slot order) so the
  P·V half is one k32 FP8 mma per 32 keys with `ldmatrix.x2` and no
  shared-memory transpose. Kernels `fw_attention_pv8_{64,128}` and
  `fw_vsa_fine3g_mma{64,128}` (`H3_CUDA_VSA_FINE=g`), sm_89+
  (`cvt.rn.satfinite.e4m3x2`); WebGPU emulates (`attn_pv8.wgsl`, the
  V codes from `attn_qk8.wgsl`, the VSA fine stage in `flash.wgsl`). 153 checks (CUDA: 143 / 143 on an RTX 4000 Ada, 153 / 153 after the
  streamed-equals-resident cases,
  sm_89, after the checker fix below; the dense pv8 cases at 3e-3). Head
  sizes without an FP8 kernel (not 64 / 128) run the host oracle on CUDA,
  dense and VSA alike (`CuVsa::host_pv8`); pre-sm_89 cards warn and run
  bf16 P·V. On the 5090
  (`docs/runs/2026-11-13-runpod-5090-fp8-pv`, the slower pod kind): the
  dense call 119.5 → **82.1 ms** (1.46×), the dense base step 10.35 →
  **8.66 s** (pod 3: 7.84 s; ComfyUI 6.8 s), the 25-step denoise 258 →
  216 s, 24.7 dB against bf16 attention (`--qk8` alone 22.1 dB); the VSA
  fine stage 34.2 → **27.6 ms** per call (+1.3 ms of V helpers), the V2
  forward 6.46 s (pod 3: 6.11), denoise 54.4 → **52.0 s** (51.5 →
  49.2), 21.7 dB against the `--vsa-qk8` clip; +576 MiB (V2) / +160 MiB
  (base) of V staging. Opt-in like `--qk8`. The checker gives the pv8
  cases bf16-representable inputs (the device rounds uploads to bf16,
  and E4M3 codes taken from f32 and from bf16 values differ one time in
  32) and the oracle reports where its P rounding sits within a few f32
  ulps of an E4M3 boundary, so a kernel is not charged for a flip its
  own exp2 produces; without both, the pv8 cases sit at 2–3.4e-2 of the
  oracle on a 5090 and one of them crosses 3e-2.
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
  `H3_CUDA_NO_PIN=1` compares. A 24 GB RTX 4090 (450 W, PCIe 4 ×16,
  driver 595.99.02) streams V2 at 1344×768×124 in 19.1 GB, the GPU
  waiting 0.6 ms per block: forward 13.4 s, denoise 107 s, 125.7 s with
  decode, bf16 weights (MXFP8 is emulated on sm_89 and costs 2.6 s of
  decode for nothing); 170 / 170 checks there plain, `H3_STALL=1` and
  under stress (`docs/runs/2026-11-14-vast-4090-24gb`). The prefetch fence must outlive
  `prefetch_wait`: `prefetch_begin` synchronises it on the host before
  the bounce buffer is reused, and taking it early let the next block's
  small copies overwrite slots still queued on the copy stream (clips
  black on three hosts, byte-identical on one, 29317cb; the 4090's PCIe 4
  link gains nothing from pinning: 14.8 s either way). The fences are a
  FIFO (`H3_PREFETCH_DEPTH=2` keeps two prefetches in flight) and each is
  popped exactly once in `prefetch_wait`; keeping the only entry made
  every block after the first wait on the previous block's fence
  (3eb1d82 → c0a807e: clips differed between depth 1 and depth 2). The
  checker compares every streamed tiny-DiT forward with its resident
  twin **exactly** (`streamed == resident`), and `H3_STREAM_STRESS=1`
  zeroes, delays 5 ms and then copies each prefetch so a wrong wait reads
  zeros: a build with the FIFO bug fails those cases under stress and
  passes without it (`docs/runs/2026-11-13-vast-5090-streaming-gemm`).
  Streamed native-precision blocks (the five bf16 edge blocks of the
  MXFP8 recipe) are kept on the host as bf16 bytes and page-locked
  (`HostWeight::Bf16`, `Backend::weight_bf16`, only where
  `Backend::bf16_weights`): as f32 they cost a host-side conversion and
  a synchronous pageable copy per block per step, ~1.6 s of a V2 forward
  and ~1.3 s of a dense base step on a 5090, while the GPU never waited
  on the fence (`H3_STALL=1` prints the compute-stream wait, the copy
  windows and a 56 GB/s pinned h2d probe). The forward is then
  compute-bound and
  5090 pods differ by ~8 % through their clocks under the 575 W limit
  (median 2445 MHz on the slower kind; the link runs PCIe 5 × 8; a Vast
  box at 550 W ran a median 2055 MHz and 15–20 % slower kernels).
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
  REST API, GraphQL when that fails (one pod could not resolve
  rest.runpod.io) (the boot's `terminate`, which uploads the running command's
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
  `transformer_ref`. Adapters (Pika LoRA, Regen-2K LoRA, FastH3 V1 diffs,
  LongLive-Plug): dense deltas (`set_weight`, `diff`) merge in f32 at load,
  before quantization; LoRA pairs on the block and refiner linears stay an
  unmerged branch, `y = W x + B (A x)` (`linear::LowRank`,
  `Backend::linear_add`), because a merge keeps only what survives the
  device weight's rounding: LongLive-Plug's |ΔW|/|W| ≈ 2·10⁻⁴ kept a third
  in bf16 and a tenth in MXFP8, and its MXFP8 clips turned stylised
  (`docs/runs/2026-11-15-runpod-6000s-longlive-plug`). `H3_LORA_MERGE=1`
  merges as before; the branch costs ~0.6 s of a 7.0 s plug forward on the
  Server Edition, and Pika's FL2VA clip moves 28.6 dB with it (its delta
  survived the merge far better). `h3-plug4` is the base with LongLive-Plug's four-step
  adapter (convert `generator_lora.pt` with
  `scripts/reference/convert_longlive_plug.py`) on Pika's renoise rule over
  linspace(1, 0, 5) shifted 12 / 3, no CFG; a variant's missing adapter is
  an error: on the Server Edition with `--weights mxfp8 --decoder-weights
  mxfp8 --pv8` it makes a photographic clip in 34.7 s after load (4 × 7.0 s
  dense forwards, decode 6.1 s; V2's best 43.3 s), FL2VA 42.3 s.
  `pika-plug4` is LongLive-Plug's transfer to Pika Flex: Pika's task LoRA
  and the plug's stacked as one branch of rank 256 (the checker holds two
  stacked branches to their merge on the CPU and to the CPU on each
  backend), the plug's four forwards, Pika's buckets; not yet run on real
  weights. Run with real weights on the
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
- LightX2V's MiniMax-H3 Turbo LoRAs (`h3-turbo4` / `-ref`, `h3-turbo8` /
  `-ref`, `lightx2v/Minimax-h3-Turbo`) and LongLive-Plug's CFG LoRA
  (`h3-cfg`, not combined with the few-step one, as its card advises) are
  presets over the base DiT with LightX2V's recipes (Euler on
  linspace(1, 0, n + 1), shifts 6 / 3 for 4-step v1.0, 12 / 3 otherwise).
  An adapter's pair scale follows LightX2V: a per-layer `<module>.alpha`
  tensor, else the file's `alpha` metadata, both over the pair's own rank,
  else the config's alpha / rank; times `--adapter-strength`. Turbo 8 v1.2
  also carries rank-16 pairs on every `adaln_proj.linear` and on
  `norm_out.linear`: the AdaLN tables run them as a branch, and a `--tables`
  file computed for another variant is refused when an adapter touches the
  modulation.
- Draft tier: `--variant draft` is `h3-plug4` (LongLive-Plug's 4-step
  distillation), `--tier 480p | 768p` sets any variant's canvas (832×480 /
  1344×768 at 16:9). A draft is refined with `h3 regen2k --draft`, which
  keeps its composition; a 480p draft is a different sample from the 768p
  one for the same seed, so never re-render a picked draft at another
  tier. On the 600 W Server Edition with `--tables` and the fast recipe: a
  480p draft in 8.6 s after load (TAEH3), 768p 30–35 s, Regen-2K 446 s
  (`docs/runs/2026-11-17-vast-pro6000-turbo-cfg-draft`). Without
  `--tables` every run spends ~46 s on the AdaLN tables.
- `--tier 288p` is H3's 480×288 (288×480 at 9:16, 288×288 at 1:1:
  `canvas_size` with short edge 288 and a 480×288 budget), on the 32-pixel
  grid like every tier; it replaced a 272p tier (480×272 cropped from
  480×288) on 2026-11-18. A `--width` / `--height` off the grid still
  renders the canvas rounded up to 32 and `generate` / `serve` centre-crop
  the decoded frames (`finish.rs`). `--upscale realesr-animevideov3.safetensors` runs Real-ESRGAN
  ×4 (`esrgan.rs`, fastwan-rs's network: convs with
  `ConvEpilogue::Prelu` on all three backends, `pixel_shuffle`, the 8-bit
  residual on the host; byte-identical to fastwan-rs's on the real
  weights) at any size, `--upscale-factor 2 | 3` Lanczos-resampling the ×4
  frames (default 4 up to 480p, 2 above). Regen-2K follows the draft's
  aspect (`regen_canvases`: snapped to a standard family within 3 %, the
  768p draft canvas and a 1440-short-edge output; 2560×1440 / 1440×2560).
  AdaLN tables record their transformer (`h3.transformer`): Ref2VA's
  `transformer_ref` has its own modulation, and a table from the other DiT
  is refused. Server Edition, fast recipe, `--tables`
  (`docs/runs/2026-11-17-vast-pro6000-ref2va-upscale`, 180 / 180 checks):
  a 480×288 draft 4.0 s (measured as 272p, the same canvas cropped) + ×4
  upscale 1.7 s (TAEH3: 2.8 s), 480p
  10.6 + 4.4 s to 3328×1920, 768p 34.7 + 18.5 s to 2688×1536 (×2: the
  host Lanczos of the ×4 frames is most of it). 288p is below H3's
  trained tiers: the landscape fox ghosted at that canvas (twice, also
  through `serve` on RunPod), the portrait did not; 480p is the smallest
  reliably clean tier. Ref2VA: draft 480p 19.4 s denoise
  (4.8 s per forward, 2.5× T2VA's), `h3-turbo4-ref` 46.4 s, `-turbo8-ref`
  92.7 s at 768p, all keeping the reference's subject.
  Regen-2K from 272p (480×288-canvas) drafts: 9:16 → 1440×2560 in 446.5 s, 16:9 →
  2560×1440 in 450.1 s (the 2K canvas sets the cost, whatever the draft).
- AdaLN: the 13 B modulation weights are never resident. Tables are computed
  per generation by streaming each block's weight through the backend once,
  or loaded from an `h3 adaln-tables` file; `h3 quantize --skip-adaln` then
  writes a DiT without them.
- VSA-H3 (`h3_core::vsa`): dense prefix tiles never straddle segments;
  per-region top-k with ties to the lower index; the gate branch is
  `softmax(all tiles) · V̄ · gate`; reference videos are regions at keep rate
  0.1 (Omni), continuation clips stay dense. The CPU oracle is
  `h3_cpu::vsa_reference`. A plan that keeps every tile (`TilePlan::
  is_dense`: dense layers, the dense first steps of a run) means dense
  attention over the natural order, with the dense 8-bit recipes when
  the spec asks, plus `pooled[tile(tok)] · gate`; every backend runs its
  dense kernel there (CUDA `fw_vsa_gate_add`, WebGPU elementwise op 20),
  the all-tiles fine stage having cost ~2× for the same softmax
  (`docs/runs/2026-11-14-vast-5090-dense-steps`).
- Parity with PyTorch: `scripts/reference/make_fixtures.py` (run on CPU
  against FastVideo b4faeae / transformers 5.23.0 / torch 2.12.0+cpu on
  2026-11-14; the recipe is in its docstring) writes the committed tiny
  fixtures, and `cargo test -p h3-model --test parity`, `-p h3-text --test
  parity` hold the DiT, both VAEs (encode and decode) and the Qwen3-VL
  conditioner (every layer up to its output, DeepStack included) within
  2e-3 of the reference. The tiny configs are pinned to the fixtures
  (audio VAE `latent_dim` 24 = 4 × 6 channels, Qwen3-VL vision patch 16,
  DeepStack indexes [0, 1], image / video / vision-start / end ids 62 /
  63 / 60 / 61); a tiny-config change means regenerating. Regenerate only
  when a reference library changes, and record the version.
- WGSL rules (from fastwan-rs): named vec4 accumulators, never arrays indexed
  in loops; zero loop-local accumulators explicitly; every declared binding
  statically used. Kernels and their CUDA twins share names (`fw_<op>`).
- `h3 serve` binds 127.0.0.1, answers CORS for one `--allow-origin` only, and
  rejects non-JSON or foreign-origin generate requests; `web/demo.html` is
  compiled into the binary (rebuild after editing).
  `serve` crops off-grid sizes like `generate` (`finish.rs` is shared) and,
  with `--upscale`, runs Real-ESRGAN on a finished clip when the request
  asks (`upscale: true | 2 | 3 | 4`): stage `upscaling`, frames kept as
  JPEGs (`/api/jobs/<id>/hd?frame=N`; raw ×4 frames of a 480p clip would be
  2.4 GB per job), the page decoding a 16-frame window of ImageBitmaps
  around the playhead. Its default factor (`finish::serve_factor`) is the
  largest whose output fits 2560×1440; `generate`'s is ×4 up to 480p.
  Checked on the synthetic engine in Chromium (CPU backend); not yet timed
  on a GPU. One request carries a task's inputs (no per-job `h3 generate`
  for a hosted worker): `task` (`t2va | fl2va | ref2va`, default inferred
  from the media, else t2va / the start task), `first_image` / `last_image`
  `{b64, mime: image/png | image/jpeg}`, `refs` `[{kind: image | video |
  audio, b64, mime?, audio_b64?}]` (video = Y4M, audio = WAV, a reference
  video ≥ 22 frames), `continue` `{b64: y4m, audio_b64?}`; decoded on the
  HTTP thread (400 with the reason; WAVs resampled to the resident audio
  VAE's rate, 32 kHz, the tiny model's 320 Hz: at 32 kHz a 0.3 s clip was
  ~1900 reference rows and 22 s per tiny step), sized and VAE /
  vision-encoded on the engine thread (`inputs.rs` is shared with
  `generate`, which now refuses a reference video under 22 frames instead
  of panicking in `trim_reference_frames`'s slice). Body cap 64 MiB
  with a media field, 64 KiB without, 413 above. `/api/info` has `tasks`
  (what the resident DiT takes: `--task ref2va` loads `transformer_ref` so
  that engine serves Ref2VA only; Pika's per-task adapter pins the start
  task; a gallery engine without `--encoder` has no vision tower, T2VA
  only, and with one it serves FL2VA / Ref2VA too; the synthetic
  engine all three on the tiny pipeline) and `inputs.max_body_bytes`; a
  job's status has `task` and `ref_count`; the page sends a First / Last
  frame pair when `tasks` lists fl2va. The encoders load only for the
  served tasks. `cargo test -p h3-cli` runs an FL2VA and a Ref2VA job
  through the synthetic CPU engine over HTTP.
- CUDA kernels load as SASS: NVRTC compiles `kernels.cu` for the device's
  own SM (`sass_arch`, `compile_cubin`, `nvrtcGetCUBIN`) and PTX for
  `compute_XY` only when NVRTC has no real target or the cubin failed,
  fastvideo-rs's `nvrtc_sass` rule: a cubin needs no PTX JIT, and NVRTC 13.4
  against driver R595 rejected PTX with CUDA_ERROR_UNSUPPORTED_PTX_VERSION
  (fastvideo-studio's first CUDA 13 image). `cargo test -p h3-cuda` compiles
  cubins for sm_89 / sm_120 with libnvrtc alone.
- Backends: `--backend cuda | wgpu | cpu` never falls back (the chosen
  one works or the run stops with its error); `--backend auto` (the
  default) takes the first that works of CUDA (if built in), WebGPU (if
  built in), the CPU. `Cuda::new` returns an error when libcuda is missing
  (cudarc itself panics), which is what lets `auto` move on in a container
  started without the NVIDIA runtime.
- Container (`Dockerfile`, `.github/workflows/container.yml` →
  `ghcr.io/zaitrarrio/h3-rs:{sha-…,branch,latest}`): `h3` built with
  `--no-default-features --features cuda` on debian:bookworm-slim plus only
  libcublas / libcublasLt / libnvrtc 12.8 (675 MB compressed, cuBLASLt is
  most of it); the driver comes from the host (R570+ for CUDA 12.8).
  Entrypoint `h3`: `serve --addr 0.0.0.0:7860 …` as a service, any
  subcommand as a CLI, `worker` on RunPod serverless. CI smoke-tests it on
  the GPU-less runner (CPU op checks, `auto` on a host with no
  driver, a worker job, the synthetic service over a
  mapped port) before pushing. A local `docker build` inside the agent
  sandbox needs its proxy CA in the build stages (a scratch copy of the
  Dockerfile; never commit that).
  Validated on RunPod (`docs/runs/2026-11-18-runpod-container`, image
  `sha-3030e53`, public): `serve` on an RTX PRO 6000 pod (ready 255 s
  after create, a 480p draft through the proxy URL and from the hosted
  page) and `worker` on a serverless endpoint that got an RTX 4090
  (`check --backend cuda` 180 / 180, a streamed 480p `generate` uploaded
  to R2). A pod's start command is the image's CMD (after the `h3`
  entrypoint). Serverless `BLACKWELL_96` stayed throttled; pools are a
  comma list tried in order (`BLACKWELL_96,ADA_24`). Cold starts are the
  volume reads (load ~5 min, prompt ~5 min on serverless).
- Text encoder (`h3_text::Placement`, `crates/h3-cli/src/encoder.rs`):
  `--encoder auto | stream | host | resident | none` × `--encoder-weights
  bf16 | int8` (Q8 weight-only on the 50 language layers, 45.4 → 23.4 GB;
  `h3 quantize-encoder` writes `text_encoder_int8/`, picked up by default).
  Non-resident layers are read on a thread one layer ahead and uploaded on the
  side queue (`prefetch_*`). Reads go through `SafetensorsDir::read_into`
  (parallel `pread` into the final buffer), never the mmap: page faults made
  bf16 streaming 37 s per prompt against 6.5 s. Every placement gives the
  same bits per format; int8 against bf16 is ≤ 8e-3 relative, clips within the
  first-call cuBLASLt rounding band (14 dB). `auto` = resident if device free
  ≥ need + `H3_ENCODER_RESERVE_GB` (16), else host if `MemAvailable` ≥ need +
  16 GB, else stream; stream for one-prompt runs; `serve` opens the encoder
  last so free memory is real. `--prompt-cache` keys on prompt + tokenizer +
  encoder file headers + output layer + format (f32 rows, 32 in memory).
  `serve`: gallery, then cache, then encoder (`prompt_source` per job); a
  gallery without an encoder is gallery-only. Server Edition: bf16 resident
  opens in 35 s, ~0.1 s per prompt; int8 tree resident 7 s / 0.1–0.2 s
  (`docs/runs/2026-11-18-runpod-text-encoder`).
- `h3 worker` (`worker.rs`) speaks RunPod's serverless protocol itself
  (runpod-python's `rp_job` / `rp_http` / `rp_ping`: job-take with
  `job_in_progress`, job-done with `isStream=false`, pings with `job_id`,
  `Authorization: $RUNPOD_AI_API_KEY`), so the image has no Python. A job
  is `{"args": [...], "uploads": {"name": "PUT URL"}, "timeout_s"}`: `h3
  <args>` runs in a fresh per-job directory, uploads only name files inside
  it (no absolute paths or `..`), and the output carries the exit code,
  time, stdout / stderr tails and the files left. `serve` / `worker` are
  refused as jobs. `--test-input job.json` runs one job locally.
  `scripts/runpod/container.py` starts the image on a pod (service) or a
  serverless endpoint (CLI) through RunPod's GraphQL API (rest.runpod.io
  is unreachable from the agent sandbox). RunPod resource tags are
  dashboard-only and for organization accounts; the scripts name every
  resource `h3-rs-…` instead.
- Site: https://h3-video.gialabs.sh (also https://h3-rs.pages.dev; classic
  Cloudflare Pages project `h3-rs`, a proxied CNAME in the `gialabs.sh`
  zone), built by `web/build.py` into `web/site/` and deployed with
  `scripts/deploy-pages.sh`. The project was created with wrangler's
  `--force` (classic Pages, not Workers); later deploys need no flag.
- Diagnostic switches (`H3_PROFILE`, `H3_STALL`, `H3_TRACE`,
  `H3_STREAM_STRESS`, `H3_CUDA_LT_NTH`, …) must leave a clip's bytes alone
  apart from a pinned candidate: the plain-FP8 timing probe (first call per
  shape under `H3_PROFILE` / `H3_STALL`) wrote its unscaled output into the
  model and every diagnostic clip of a Server Edition round was garbage
  while the checker passed (`docs/runs/2026-11-14-runpod-6000s-probe-bug`).
  A GPU round runs `h3 check --backend cuda` plain and under each switch,
  and a clip with and without the switch must be byte-identical.
- Tests: `cargo test --workspace` (the WebGPU suite needs a Vulkan driver:
  `mesa-vulkan-drivers`, `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`,
  `H3_ALLOW_SOFTWARE=1`); the CUDA compile test needs libnvrtc on
  `LD_LIBRARY_PATH` (`pip install nvidia-cuda-nvrtc-cu12`).
- The tiny configs (`*Config::tiny*`) form one pipeline: `h3 serve --synthetic`
  and the checker run them end to end, and `config::tests::tiny_models_agree`
  pins their channel counts to each other. Change them together.
