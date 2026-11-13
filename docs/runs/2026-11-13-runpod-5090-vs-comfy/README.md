# 2026-11-13: an RTX 5090 against the "MiniMax H3 speed-ups, measured" page

The page (https://plox-1.github.io/h3-speedups/, generated 2026-10-18,
rows saved in `h3-speedups-reference.json`) times ComfyUI's official H3
template on one RTX 5090 (32 GB): "MiniMax H3, pruned int8 (template)",
Comfy Kitchen int8 attention, 1344×768, 124 frames at 24 fps, seed 1,
one prompt ("Samurai in the rain", 5.2 s, saved in the JSON) and a 15 s
four-shot prompt (362 frames). Its **total** is the warm render of the
whole workflow in a fresh ComfyUI (the cold render is listed too), its
**sampler** the KSampler alone; the baseline is the raw model at 25 euler
steps and every row is a speed-up against it. ComfyUI caches the prompt
conditioning between the cold and the warm render, so its totals carry no
text-encoder time.

h3-rs ran the same prompt, canvas, frame count and seed on RunPod
5090 pods (secure cloud EU-RO-1, $1.39/h, **600 W** power limit, driver
595.91.07, 3150 MHz max SM clock, 32 607 MiB), weights staged onto the
container disk (`H3_STAGE`), release binary of fb78a6b (`h3-bin7`, the
serial block streaming) for pod `nmvicb3scqaun8` (04:01–04:44 UTC) and
of a8fd30a (prefetched streaming) for pod 2. Our numbers:
`load` (weights to the device), `prompt` (Qwen3-VL on the device, 462
rows for this prompt: the step ComfyUI caches away), `denoise`, `decode`,
and `peak VRAM` from `nvidia-smi` sampled every 0.5 s. The comparable
pairs are **denoise vs sampler** and **denoise + decode vs total**.

## Why a 5090 is a different problem

The V2 recipe that leads on the 96 GB Server Edition (stored MXFP8
weights 25.6 GB resident, MXFP8 decoder, INT8 Q·Kᵀ VSA, tables) **does
not fit 32 GB** (`v2-resident`: OOM at 31.7 GB after load, log in
`output-1-v2-v1.txt`). On this card the block weights stream from host
memory each step (`--stream-blocks`), and the first pod measured the
serial streaming path that existed: every block's codes re-encoded on the
CPU and copied on the compute stream before its kernels could start.

## Pod 1 (fb78a6b, serial streaming): the samurai prompt, 124 frames

| | ComfyUI row | ComfyUI total / sampler | h3-rs config | load | prompt | **denoise** | decode | denoise + decode | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| FastH3-V2, 8 steps | `fasth3-v2-euler`: pruned int8, Kitchen int8 + VSA keep 10 %, first 20 % dense | **54.1 s / 45.2 s** (cold 62.5 s, 30.0 GB) | stored MXFP8, `--decoder-weights mxfp8 --vsa-qk8`, tables, VSA 0.8 (the released contract), `--stream-blocks` | 28.7 s | 69.9 s | **123.1 s** = 15.57, then 15.19–15.50 s × 7 | 6.28 s | 129.4 s | 17.5 GB |
| same, ComfyUI's attention budget | | | + `--vsa-sparsity 0.9 --dense-first-steps 2` | 26.9 s | 76.8 s | 130.2 s = 21.2, 21.3 (dense), then 14.6 s × 6 | 6.23 s | 136.4 s | 17.1 GB |
| FastH3 v1, 4 steps | `fasth3-v1-euler`: int8, VSA keep 10 %, first 20 % dense | **33.3 s / 23.9 s** | base + `vsa-datafree` LoRA, MXFP8 at load, `--vsa-qk8`, VSA 0.9, tables, streamed | 86.4 s | 76.5 s | **57.9 s** = 14.7, 14.4, 14.4, 14.4 s | 6.19 s | 64.1 s | 17.1 GB |
| baseline, raw H3, 25 euler steps | `baseline`: pruned int8, Kitchen int8, dense | **181.7 s / 169.0 s** (median of 4; cold 189.3 s) | `--variant h3 --steps 25` (the same uniform ladder), MXFP8 at load, dense, tables, streamed | 58.8 s | 66.4 s | **530.8 s** = 21.0–21.7 s × 25 | 6.21 s | 537.0 s | 15.1 GB |
| FastH3-Trim, ComfyUI NVFP4 file | (no row) | | `--variant fasth3-trim`, NVFP4 resident, `--vsa-qk8`, 26-block decoder | 85.0 s | 55.2 s | 63.6 s = 7.90–7.98 s × 8 | 12.35 s | 76.0 s | 26.9 GB |

Also on pod 1: a V1 run with both LoRAs of the tree merged by mistake
(`v1-both-adapters.log`, 4 × 14.0 s; the tree holds `dense-datafree/`
and `vsa-datafree/`, only the latter is the VSA checkpoint) and a
362-frame attempt that failed on a duplicated `--frames` flag in my
script (re-run on the follow-up pod). Frames: `*-frames.png` (0 / 40 /
80 / 120). Every clip follows the prompt (the arm-cross, the push-in,
rain on the face); like the page's V2 row, ours gives him a topknot.

Reading pod 1: the streamed forward costs ~9 s over the card's compute
(V2: 15.4 s per forward; the 600 W Server Edition runs it in 4.7 s with
resident weights, and a 5090 has ~0.8× its tensor throughput), and the
dense base forward 21.3 s where ComfyUI's int8 attention + int8 GEMMs do
a step in 6.8 s. Trim, the one recipe that fits resident, runs 7.95 s per
forward (5.5 s on the Server Edition) with its per-GEMM NVFP4 expansion.

## Pod 2 (a8fd30a): prefetched streaming, same prompt, same card class

Pod `2ly33bst3f26zj` (05:01–05:34 UTC, same offer class and driver).
7383401 made byte-coded weights (int8 / int4 / NVFP4 / MXFP8 codes and
scales) upload straight from the host bytes and the DiT prefetch block
l + 1 on a non-blocking side stream while block l computes (two blocks
resident, event-ordered frees); a8fd30a added streamed twins of the tiny
DiT cases to the checker. On this pod: `h3 check --backend cuda` **121 /
121** at 3e-2 (`check-cuda-5090.log`), and the streamed V2 clip from the
new binary is **byte-identical** to the old binary's on the same pod
(`cmp`, both with `H3_CUDA_LT_NTH=0` so cuBLASLt picks the same kernel).

| | ComfyUI total / sampler | h3-rs config (all streamed unless noted) | load | prompt | **denoise** | decode | **denoise + decode** | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| FastH3-V2, 8 steps | **54.1 s / 45.2 s** | stored MXFP8, MXFP8 decoder, `--vsa-qk8`, tables, VSA 0.8 | 27.2 s | 68.6 s | **54.9 s** = 6.70, then 6.80–6.92 s × 7 (second run 55.7 s) | 6.6 s | **61.5 s** | 18.0 GB |
| | | same binary, old serial path (`h3-bin-old`, fb78a6b) | 27.2 s | 65.6 s | 114.8 s = 14.3–14.4 s × 8 | 6.4 s | 121.2 s | 17.5 GB |
| | | ComfyUI's budget: `--vsa-sparsity 0.9 --dense-first-steps 2` | 26.3 s | 68.7 s | 62.8 s = 12.9, 12.6 (dense), then 6.20–6.22 s × 6 | 6.5 s | 69.2 s | 17.9 GB |
| FastH3 v1, 4 steps | **33.3 s / 23.9 s** | base + VSA LoRA, MXFP8 at load, `--vsa-qk8`, VSA 0.9 | 75.8 s | 68.6 s | **24.9 s** = 6.40, 6.13–6.17 s | 6.4 s | **31.3 s** | 17.9 GB |
| raw H3, 25 euler steps | **181.7 s / 169.0 s** | `--steps 25`, MXFP8 at load, dense, tables | 61.5 s | 68.9 s | **349.6 s** = 13.8–14.05 s × 25 | 6.5 s | 356.1 s | 15.9 GB |
| FastH3-V2, the 15 s prompt (362 frames) | **213.6 s / 185.6 s** (cold 225.5 s, 30.0 GB) | as the V2 row, `--ffn-chunk 16384` | 26.7 s | 569 rows, 67.5 s | **211.4 s** = 26.2–26.6 s × 8 | 19.4 s (+ 0.5 s audio) | **230.8 s** | **31.2 GB** |

Frames: `v2-new-frames.png`, `v1-vsa-new-frames.png`, `base25-new-frames.png`,
`v2-15s-frames.png` (frames 0 / 40 / 80 / 120: the night market, the
cook at the wok, the macaque's snatch). Clips coherent throughout.

## Where this leaves h3-rs on a 32 GB card

- **Streaming is no longer the story.** Prefetching took the V2 forward
  from 15.4 s to 6.9 s on the same card (the Server Edition runs it
  resident in 4.7 s; a 5090 at 600 W has ~0.8× its tensor throughput, so
  the remaining streaming cost is small and unmeasured here). The same
  change took the V1 forward 14.4 → 6.15 s and the dense base step 21.3
  → 14.0 s. Nothing about the output changed: identical bytes.
- **FastH3 v1, 4 steps: on par** with the page's row (31.3 s denoise +
  decode against ComfyUI's 33.3 s warm total; 24.9 s against the 23.9 s
  sampler).
- **FastH3-V2, 8 steps: 14 % behind** (61.5 s against 54.1 s; denoise
  54.9 s against the 45.2 s sampler). ComfyUI's row also keeps 10 % of
  the key tiles with two dense steps; at that budget we are slower
  (62.8 s), so the gap is in the kernels, not the attention budget: their
  Kitchen int8 attention for the dense steps and whatever its VSA engine
  does per key tile, against our bf16 dense attention and `fw_vsa_fine3e`.
- **The 15 s clip: 8 % behind** (230.8 s against 213.6 s; denoise
  211.4 s against 185.6 s), at 31.2 GB peak against their 30.0 GB: the
  FFN chunk kept it inside 32 GB, with nothing to spare.
- **The raw model at 25 dense steps: 2.1× behind** (14.0 s per step
  against 6.8 s). This is the dense attention: 38k tokens × 47 blocks in
  bf16 `attn_mma2` against ComfyUI's INT8 Kitchen attention. The INT8
  Q·Kᵀ recipe that already runs in the VSA fine stage (`--vsa-qk8`) is
  the obvious next kernel for the dense path, and what the base model,
  Pika and Regen-2K (all dense) would gain from most.
- The prompt step (Qwen3-VL on the device, 68 s for 462 rows) is outside
  the page's "total" because ComfyUI caches the conditioning between its
  cold and warm render; `h3 encode` does the same for us, and the Server
  Edition record's 189–278 s prompt steps were the network volume, not
  the encoder.

Cost: two pods, ~$1.8 (04:01–04:44, 05:01–05:34 at $1.39/h);
$27.70 → $25.68 on the account.

## Pod 3 (8c20073): INT8 dense attention, profiles, the staging ring

Pod `fagl3elupnvi43` (10:22–11:14 UTC, same offer class). Binary of
8c20073: INT8 Q·Kᵀ dense attention (`--qk8`), the pinned staging ring
behind `H3_CUDA_PIN=1`. (A pod before it, `r5yeb70y4nuekr`, lost every
run to fdd287c's `cuMemHostRegister` path: registering page-rounded Vec
ranges pinned neighbouring heap pages and later copies from buffers
straddling them failed with `CUDA_ERROR_INVALID_VALUE`; its log is
`output-3-…`'s sibling run and the fix is 8c20073.)

`h3 check --backend cuda`: **127 / 127** at 3e-2 (`check-cuda-5090-127.log`);
the six INT8 dense attention cases at 7e-3 to 2e-2 against the oracle.

| | ComfyUI total / sampler | h3-rs config | load | prompt | **denoise** | decode | **denoise + decode** | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| raw H3, 25 euler steps, **bf16 attention** | **181.7 s / 169.0 s** | `--steps 25`, MXFP8 at load, dense, tables, streamed | 60.1 s | 67.0 s | 330.0 s = 13.1–13.3 s × 25 | 6.2 s | 336.2 s | 15.9 GB |
| raw H3, 25 euler steps, **INT8 Q·Kᵀ** (`--qk8`) | | same + `--qk8` | 59.6 s | 66.8 s | **248.8 s** = 10.17, then 9.86–10.01 s × 24 | 6.2 s | **255.0 s** | 16.8 GB |
| FastH3-V2, 8 steps | **54.1 s / 45.2 s** | the V2 recipe, streamed (default path) | 50.6 s | 69.9 s | **53.2 s** = 6.86, then 6.59–6.63 s × 7 | 6.2 s | **59.3 s** | 18.0 GB |
| | | same, `H3_CUDA_PIN=1` (two runs) | 26.7 s | 66.4 s | 53.3 s, 53.2 s | 6.1 s | 59.4 s | 17.9 GB |
| | | ComfyUI's budget (VSA 0.9, 2 dense steps) | 26.3 s | 65.5 s | 59.9 s = 12.2, 11.9, then 5.96–5.98 s × 6 | 6.2 s | 66.1 s | 17.9 GB |
| FastH3 v1, 4 steps | **33.3 s / 23.9 s** | V1 VSA LoRA, streamed | 74.3 s | 67.4 s | **24.0 s** = 6.17, 5.92–5.95 s | 6.1 s | **30.1 s** | 17.9 GB |
| FastH3-V2, 15 s (362 frames) | **213.6 s / 185.6 s** | the V2 recipe, `--ffn-chunk 16384` | 26.3 s | 65.6 s | **199.6 s** = 25.3, then 24.8–25.0 s × 7 | 18.8 s | **218.4 s** | 30.8 GB |
| | | same, `H3_CUDA_PIN=1` | 26.3 s | 65.5 s | 199.4 s | 18.9 s | 218.2 s | 30.9 GB |

**INT8 dense attention**: the step goes 13.2 → 9.95 s (−25 %), the
25-step clip 330 → 249 s. Per kernel (`pod3-base2-*-profile.log`, synced
per op, two forwards): `attention (mma2, d128)` 181.3 ms per call, 72 %
of the step; `attention (int8 qk, d128)` 112.3 ms, 61 %, plus 0.5 ms per
call of quantisation and 0.37 ms of K mean. The int8 kernel runs the
Q·Kᵀ half at the int8 rate and P·V in bf16, so 1.6× on the attention is
in the band expected; ComfyUI's whole step (6.8 s) implies its attention
at ~80 ms per block-call, which is where FP8 P·V and a tighter pipeline
would go next. Output: `base25-qk8-frames.png` against
`base25-bf16-pod3-frames.png` are the same scene (PSNR between the two
clips 20.5 dB average, Y 18.8, U/V 38.5–38.8: the euler trajectory
diverges in texture, as every 8-bit rounding does here; no artifacts).
The GEMMs did not move (MXFP8 at 485–518 TFLOP/s on this card).

**V2 and the streaming residue**: the V2 denoise profile
(`pod3-v2-profile.log`) sums to 44.7 s of GPU op time over 8 forwards,
5.6 s per forward (`vsa fine (3e int8 qk)` 34.2 ms × 400 = 13.7 s, the
MXFP8 GEMMs 25.0 s at 518 TFLOP/s, the rest 6 s); the unprofiled forward
takes 6.6 s. The missing ~1 s per forward is time the compute queue sits
idle between blocks: the host spends ~50–100 ms per block copying 545 MB
of codes (pageable `cuMemcpyHtoDAsync`, or the memcpy into the pinned
ring: the ring changed nothing, 53.3 against 53.2 s) while the queue
holds ~115 ms of kernels. ComfyUI's sampler at 45.2 s is 5.65 s per step,
our pure GPU time. Removing the host copy (DMA straight from page-locked
weight memory, which 8c20073's predecessor tried the wrong way) is the
one thing left between the V2 numbers.

**15 s**: 218.4 s against 213.6 s (2 % behind), at 30.8 GB.

Cost: ~$1.3 for pod 3 plus ~$0.5 for the lost one; $25.68 → $23.15.

## Pod 4 (54256d1): page-locked DMA, first try

Pod `d9vt7lde0hjo86` (11:30–12:05 UTC; `output-5-pod4-pagelock-dma.txt`).
54256d1 registers only the whole pages inside each streamed buffer and
copies the two partial edge pages as pageable memory, so no neighbouring
allocation is ever locked. The clips ran, the measurement said nothing
moved, and the checker still failed:

| | denoise | decode | note |
|---|---|---|---|
| V2, page-locked (two runs) | 57.9 s, 57.5 s = 7.1–7.4 s × 8 | 6.6 s | |
| V2, `H3_CUDA_NO_PIN=1` | 58.3 s | 6.7 s | |
| V1 | 25.9 s | 6.5 s | |
| V2, 15 s | 218.8 s = 27.0–27.6 s × 8 | 20.3 s | 30.9 GB |
| raw H3, 25 steps, `--qk8` | 267.8 s = 10.5–10.9 s × 25 | | |

Two readings. First, this box is ~8 % slower than pod 3's across the
board (V2 7.2 against 6.6 s per forward, the dense INT8 step 10.7
against 9.95 s) with jitter the earlier pods did not show; `nvidia-smi`
before the run reported its PCIe link at **gen 1 × 8** (idle state or
not, the next pod samples the link, SM clock and power through the
clips). Second, page-locking did not move the needle (57.7 against
58.3 s), which means the host's copy time is not what idles the GPU
either: every small upload in the prefetch (norms, the partial edge
pages) still ends in a `cuStreamSynchronize` of the side stream, and an
in-order stream makes that wait cover the large DMAs issued just before.
The checker panicked at its first upload with `CUDA_ERROR_INVALID_VALUE`
again (`h3 check` loads several streamed DiTs in one process and the
locked pages of a dropped one stayed registered): 0c29fac turns the
locks into guards the DiT drops before its buffers, and gates the path
behind `H3_CUDA_PIN=1` until the suite passes with it.
