# 2026-11-13: RTX 5090 on Vast.ai, the remaining gap to ComfyUI: streaming stalls, GEMM side, attention budgets

Follow-up to `2026-11-13-runpod-5090-fp8-pv`, whose per-kernel profiles
left three things unexplained or untried on the 5090: ~1.0 s per V2
forward and ~0.85 s per dense base step that is in no kernel (the
streamed forward may wait on block uploads), the GEMM side (five bf16
edge blocks, the QKV shape at 388 TFLOP/s where the split FFN-up runs at
690), and ComfyUI's own block-sparse rows on the raw model, which we had
never run at their budget.

Same weights as the ComfyUI rows in the sense that matters: ComfyUI's
"pruned int8" files (`Comfy-Org/MiniMax-H3`, headers read on 2026-11-13)
are the same 50-block model with the AdaLN weights replaced by a rank-8
table (what our `--tables` do), all 50 blocks int8 with one scale per
output row, bf16 token refiner; ours are FastVideo's HF trees in MXFP8
(one E8M0 scale per 32 columns) with blocks 0–1 and the last three in
bf16, VSA keep 20 % where ComfyUI keeps 10 % with two dense steps.

Instrumentation added for this round (3eb1d82): `H3_STALL=1` records
event pairs around every `prefetch_wait` on the compute stream and
around every prefetch's copies on the copy stream (no per-op syncs, so
the overlap is the real one) and prints, per stage, how long the GPU
waited for streamed weights and the copy rate; a 256 MB pinned /
pageable host-to-device probe at startup; `H3_PREFETCH_DEPTH=2` keeps
two blocks in flight (three resident).

## Box

Vast.ai offer 54227877, instance 60557523 (Texas, US), one **RTX 5090**
on **PCIe 5 ×16** (the RunPod pods of the previous records ran ×8),
AMD Ryzen 9 7950X3D, 95 GB RAM, 1.5 TB disk, driver 595.91.07 (the same
as the RunPod pods), $0.80/h. Weights pulled from the R2 mirror
(`weights/h3-8step/<sha>/` 148 GB with the text encoder, and
`weights/h3-base/<sha>/transformer/` 130 GB; the base run links the
VAEs, text encoder and tokenizer from the V2 tree, which are the same
files). V2 quantized to MXFP8 at load (no stored file on the mirror),
AdaLN tables computed on the box.

`nvidia-smi` on the box: power limit **550 W** (default and max 600 W,
RunPod's pods report 575 W), PCIe gen 5 ×16 current and max (gen 1 while
idle), max SM clock 3180 MHz, 92 GB RAM, 32 threads. The boot pulled
198 GB in 885 s (1.8 Gbps) and lost one V2 transformer shard to a
truncated transfer (`FAILED` in the boot log; the loader refused the
file as "incomplete metadata"), which the repair command refetched.
`h3 check --backend cuda`: 143 / 143, the FP8 cases at 2–3e-3.

## What the box runs like

Slower than the RunPod 5090 pods on identical code, with the same driver
and a wider link: the V2 forward 7.9 s against 6.1–6.5 s, the dense
base step 10.2 s against 7.8–8.7 s, the INT8+FP8 attention call 91 ms
against 82 ms, the MXFP8 FFN-up GEMM 22.3 ms against 18.2 ms. The
clocks are the difference: during the denoise this card runs at a
**median 2055 MHz** (mean 2291, 2010–3015, 478 W mean draw;
`stall/clocks-v2.log`) where the RunPod pods ran a median 2445 MHz under
their 575 W limit, 15–20 % apart like the kernels. The numbers below
compare against each other on this box, and only the ratios carry to
other 5090s.

## Item 1: the non-kernel time was the bf16 edge blocks' uploads

Every run here printed its prefetch accounting; the first binary's
events had timing disabled (cudarc's default, fixed in 922127c), so the
first numbers come from the per-kernel profile and from switching the
edge blocks' format, and the event numbers from the follow-up command.

`WeightQuant::for_block` keeps blocks 0–1 and the last three in bf16.
When the DiT streams, those five blocks were kept on the host as **f32**
(`HostWeight::F32`, 1.5 GB each) and every step `Backend::weight`
converted them to bf16 on the CPU and copied the 770 MB from pageable
memory with a synchronous `memcpy_htod` (the page-locked DMA path only
covered byte-coded weights, the thing bb79796 fixed for the quantized
blocks); the host sat in that copy instead of launching the next block's
kernels. Quantizing those five blocks too (`H3_MXFP8_ALL_BLOCKS=1`)
removes the path entirely and shows its cost:

| dense base, 2 euler steps, step 2 | step | V2, 8 steps, forward 2–8 | forward |
|---|---|---|---|
| as released (edges bf16, f32 on the host) | 10.18 s | | 7.91–7.96 s |
| `H3_MXFP8_ALL_BLOCKS=1` | **8.64 s** | | **5.91–5.98 s** |
| `H3_CUDA_MX_SPLIT_MIN=21504` (QKV row split) | 9.90 s | | 7.59–7.65 s |
| both | **8.12 s** | | **5.56–5.64 s** |

With all blocks MXFP8 the profiled step's kernels sum to 8.0 s against
8.12 s wall (`gemm/base2-both-profile.log`): nothing is left outside the
kernels. That is 15 % of the dense step and 25 % of the V2 forward,
far more than the five blocks' GEMM difference (~0.35 s), so it was the
upload path, not the arithmetic. **But quantizing the edge blocks is not
the fix**: on the base model the clip stays in the band (base25, 25
steps: 10.1 → 8.33 s per step, 20.3 dB against the bf16-edge clip, same
scene), while on the DMD-distilled V2 the edges are calibrated and the
clip changes character: `gemm/v2-allblocks.grid.png` is an anime
rendering of the samurai (14.8 dB, Y 10.6), `v2-both.grid.png` a
different photoreal composition (17.1 dB). The Server Edition record had
called all-blocks "same PSNR band" on one clip; this prompt says
otherwise. The fix that keeps the recipe is to stream the bf16 blocks as
bf16 bytes through the page-locked path (`HostWeight::Bf16`,
`Backend::weight_bf16`, BF16PIN_COMMIT): the follow-up command measures
it below.

The QKV row split (`H3_CUDA_MX_SPLIT_MIN=21504`) is worth 3–4 % on its
own on this card (the `n=21504` shape ran at 388 TFLOP/s unsplit on the
RunPod pod; split it runs at 16.4 ms per call here against the FFN-up's
22.3 ms); it changes the sample at the rounding level only (20.7 dB
against the unsplit clip, the cuBLASLt-candidate band).

Prefetch depth 2 (`H3_PREFETCH_DEPTH=2`) was slower on this box with the
f32-host edge blocks (V2 8.24 s against 7.92 s; base 10.42 against
10.18): the host sync in `prefetch_begin` now waits for the newer of two
copies before issuing the next, so the pageable edge-block copies held
the host longer. Its clips are **byte-identical** to depth 1's on CUDA
with the cuBLASLt candidate pinned (`H3_CUDA_LT_NTH=0`, V2 and base),
so the FIFO fences are right; without the pin two runs of the same
configuration differ at the rounding level and the DMD sampler turns
that into a 21–22 dB clip difference (`fix/`, the first pair).

### The event accounting (`stall/`, timing-enabled events)

| | h2d probe, 256 MB | compute stream waited on the prefetch fence | copy windows | kernels (profile) | wall |
|---|---|---|---|---|---|
| V2, 8 forwards | pageable 19.0 GB/s, **pinned 56.1 GB/s** | **1 ms** over 400 blocks | 33.4 s for 191 GB, 5.7 GB/s | 6.14 s / forward | 7.92 s / forward |
| base, 2 steps | same | 0 ms over 100 blocks | 15.3 s for 43 GB, 2.8 GB/s | ≈ 8.0 s / step | 10.1 s / step |

The link is fast (56 GB/s pinned) and the GPU **never waits for the
fence**: by the time the compute stream reaches `prefetch_wait` the
copies are long done. The 1.8 s per V2 forward and 2 s per base step
that are in no kernel are the **host**: for each f32-host edge block,
`Backend::weight` ran a single-threaded f32 → bf16 pass over 385 M
values and then a synchronous pageable copy, with no kernels queued
behind it, so the GPU idled until the host came back; the copy windows
read slow for the same reason (they bracket those host passes). The
quantized blocks (byte copies, page-locked DMA) were never the problem,
and PCIe ×8 against ×16 does not matter at these sizes.

### The fix: streamed native blocks as bf16 bytes (6fbed45, 64bb520)

`HostWeight::Bf16` / `Backend::weight_bf16`: when the backend stores
bf16 anyway (`Backend::bf16_weights`, CUDA), a streamed native-precision
block is converted once at load and page-locked like the quantized
ones, so its upload is the same DMA; the f32 reference keeps f32 (its
streamed and resident paths stay bit-identical, `cpu_agrees_with_itself`).
Half the host memory for those blocks, too. Measured on this box
(`bf16pin/`), the cuBLASLt candidate pinned so the clip can be compared
byte for byte with the f32-host path:

| | per forward (V2) / per step (base) | GPU waited on the fence | copy windows |
|---|---|---|---|
| f32-host edge blocks (the released path) | 7.92 s / 10.1 s | 1 ms / 0 ms | 5.7 / 2.8 GB/s |
| bf16 host path, depth 1 | 6.32 s / 8.83 s | 8 ms / 6 ms | 10.9 / 3.5 GB/s |
| bf16 host path, depth 2 | **6.33 s** (byte-identical to the released path) | 1 ms | 11.4 GB/s |
| bf16 host path + QKV split, depth 1 | 6.01 s / 8.50 s | 4 / 3 ms | 13.6 / 3.9 GB/s |
| bf16 host path + QKV split, 25 steps | 8.57 s per step, denoise 214.1 s (was 253.6) | | |

20 % off the V2 forward and 13 % off the base step with no change to
the arithmetic, which the depth-2 run proves: its clip is byte for byte
the released path's. The depth-1 clips were **not**, and that found a
second bug: the FIFO fences of 3eb1d82 kept the front entry when it was
the only one, so with one prefetch in flight every block after the first
waited on the *previous* block's fence and could start before its own
weights had landed. With the slow f32-host path the host was always
late enough for the copies to be done; with fast uploads (the bf16 path,
or all blocks MXFP8) the race was live: the all-blocks clips of this
round with their "anime" look and the depth-1 bf16 clips all carry the
same first-frame luma signature, and two depth-1 runs reproduced the
same wrong clip bit for bit. Fixed in c0a807e (each fence popped
once; the host sync takes the newest fence by its own handle). The
timing of the depth-1 rows above stands (the depth-2 run, correct, took
the same 6.33 s); **the all-blocks quality claims of this round do
not**, and the edge-block question is re-run on the second box below.

Making the checker see such a bug took two steps (`check-ada/`, RTX
2000 Ada pods `5bjtum4fm1o6bw` and before it `fqluhaxbg03dhg`,
`dn0o831opl8rum`): a spin on the copy stream ahead of each prefetch
changed nothing, because the stream-ordered allocator handed every
block the address a correct earlier forward had left its weights at;
`H3_STREAM_STRESS=1` now zeroes each prefetch's destination, spins
~5 ms, then copies (the early read is zeros), and the checker compares
every streamed tiny-DiT forward with its resident twin **exactly**
(`… streamed == resident`, 153 cases): against the CPU the buggy build
had read 2.1–2.5e-2, inside the 3e-2 CUDA tolerance. With that, a
one-off build with the FIFO bug re-applied fails all ten streamed cases
under stress (`check-bug-stress.log`, 2.1–2.5e-2 against resident) and
still passes without it; the fixed binary passes 153 / 153 plain, under
stress, and under stress at depth 2.

## The second box: everything again with the fences fixed

Vast offer 60356253, instance 60580981 (United Arab Emirates), RTX 5090
at its **600 W** default limit, PCIe 5 ×16, driver 595.91.07, AMD Ryzen
9 9950X, 123 GB RAM, pulled 161 GB in 1287 s with 17 truncated transfers
the setup command refetched; 143 / 143 plain and under
`H3_STREAM_STRESS=1` (`box2/check*.log`; the binary predates the exact
streamed cases). A fast 5090: its released-path V2 forward is 6.26 s and
its kernels match the RunPod pods' faster kind (fine stage 27.1 ms, MXFP8
GEMMs 491 TFLOP/s). Every run pins the cuBLASLt candidate
(`H3_CUDA_LT_NTH=0`) so clips compare byte for byte.

**FastH3-V2, 8 steps, 1344×768×124** (`box2/v2-*.log`), the released
f32-host path (the binary of the FP8 record) against c0a807e:

| | per forward | denoise | denoise + decode | GPU waited on fences | clip |
|---|---|---|---|---|---|
| released path (`v2-ref`) | 6.22–6.29 s | 50.1 s | 56.4 s | 0 ms | reference |
| fixed fences + bf16 host path (`v2-fix`) | **5.65–5.71 s** | **45.6 s** | **51.9 s** | 117 ms / 400 blocks | **byte-identical** |
| + QKV row split (`H3_CUDA_MX_SPLIT_MIN=21504`) | **5.29–5.33 s** | **42.6 s** | **48.9 s** | 118 ms | **byte-identical** |
| + depth 2 instead of the split | 5.67–5.73 s | 45.7 s | 52.0 s | 0 ms | byte-identical |
| all 50 blocks MXFP8 (`H3_MXFP8_ALL_BLOCKS=1`), no split | 5.30–5.34 s | 42.7 s | 49.0 s | 63 ms | 22.3 dB, same scene (`v2-fix-allblocks.grid.png`) |
| all 50 blocks MXFP8 + QKV split (`v2-fix-both`) | **4.86–4.91 s** | **39.2 s** | **45.5 s** | 63 ms | the all-blocks clip, bit for bit |
| ComfyUI's budget on the fixed path (keep 10 %, 2 dense first steps, split) | 9.8 s dense / 4.75 s sparse | 48.1 s | 54.4 s | | |
| ComfyUI `fasth3-v2-euler` (page) | 5.65 s | 45.2 s sampler | 54.1 s total | | |

So on a 600 W 5090 the V2 recipe is now **ahead of ComfyUI's row on both
of the page's numbers**: denoise 42.6 s against the 45.2 s sampler (6 %),
denoise + decode 48.9 s against the 54.1 s total (10 %), with the clip
bit for bit the one the released binary made. The split changes no bits
either (the two row-slice GEMMs round like the whole one, candidate
pinned), so it is free; depth 2 buys nothing once the host is out of the
way; and quantizing the edge blocks, which the fence race had painted as
an anime clip, is in fact the usual 22 dB band on this prompt, same speed
as the split, and left off because the split gets there without touching
the recipe's calibrated edges. The profile of the fixed path (`v2-fix-
prof.log`): 44.0 s of kernels over 8 forwards, 5.51 s per forward
against 5.70 s wall, so 0.19 s per forward (3 %) is outside the kernels
now, from 1.8 s; with the split (`v2-fix-split-prof.log`) 40.7 s of
kernels, 5.09 s against 5.32 s wall, the QKV GEMM 23.0 → 13.2 ms per
call and the MXFP8 GEMMs at 565 TFLOP/s. With every block MXFP8 as
well, the forward is 4.9 s: denoise 39.2 s and 45.5 s with decode,
13 % and 16 % ahead of ComfyUI's row, at the usual 22 dB from the
calibrated-edge clip; the recipe's own edges with the split stay the
default. ComfyUI's attention budget is still not a win for us (48.1 s):
its two dense steps go through the all-tiles VSA path at 9.8 s where
the dense INT8+FP8 kernel would do them in ~5.3 s, which is the one
routing change left on that row (then ~39 s). What remains per forward: the VSA fine stage 1.36 s
(27.1 ms × 50), the MXFP8 GEMMs 2.6 s (QKV 23.0 ms per call unsplit,
FFN-up 17.6 ms split, FFN-down 8.9, out 4.3, gate 4.4), the five bf16
edge blocks' GEMMs 0.67 s, head_split / quant / SwiGLU / norms / coarse /
pool / residual ~1.0 s.

**Raw H3, dense, 25 uniform euler steps** (`box2/base*.log`; ComfyUI's
`baseline` row 181.7 s total / 169.0 s sampler, 6.8 s per step):

| | per step | denoise | denoise + decode | clip |
|---|---|---|---|---|
| released path (2-step run) | 8.41 s | | | reference |
| fixed fences + bf16 host path | 8.05–8.11 s | 202.5 s | 208.9 s | byte-identical (2-step run) |
| + QKV row split | **7.71–7.75 s** | **193.0 s** | **199.4 s** | **byte-identical** |
| + all 50 blocks MXFP8 | 7.33–7.37 s | 183.5 s | 189.9 s | 29.3 dB against the bf16-edge clip, same scene |
| ComfyUI `baseline` (page) | 6.8 s | 169.0 s sampler | 181.7 s total | |

The dense base is at 1.13× ComfyUI's step with the recipe's bf16 edges
(1.08× with every block MXFP8, the closest thing to ComfyUI's own
all-int8 file), from 1.15–1.27× before this round and 2.0× in bf16
attention. Its profile is attention 4.1 s (82–91 ms × 50), MXFP8 GEMMs
2.6–3.1 s, the bf16 edge blocks 0.6 s, small kernels 0.65 s per step:
what is left against ComfyUI is the GEMM side and the edge blocks,
not the attention and not the host any more.

### The split as a per-shape choice

Since the split is pure speed on the shapes that matter, the backend now
decides it per shape on the first call: whole and split, each with its
tuned candidate, timed back to back (`mx_choice`; `H3_CUDA_MX_SPLIT_MIN=n`
still forces the old rule). On this box (`box2/v2-auto*.log`,
`base2-auto.log`, the decision lines under `H3_STALL=1`):

| shape (n × k, 38 172 rows) | whole | ×2 | choice |
|---|---|---|---|
| QKV 21504 × 5376 | 22.2 ms | **12.5 ms** | ×2 |
| FFN-up 28672 × 5376 | 31.1 ms | **17.7 ms** | ×2 |
| 16384 × 2048, 1797 rows | 0.21 ms | 0.20 ms | ×2 once, whole once |

The forward lands at 5.31–5.38 s, the forced split's number. The first
version took any speed-up, and the 0.2 ms shape, which measures the
same either way, then split in one run and not in another; its split
does **not** round like its whole GEMM (the clip differed from the
reference), where the two big shapes' splits do. The decision now needs
a 10 % win, which on this card is exactly the forced-split
configuration measured above (byte-identical to the released path);
that margin version was built after the box was gone and has not run
at full size itself.

## Item 5: ComfyUI's block-sparse rows, and V2 at its budget

Run before the bf16 fix below, so every step here still carries the
edge blocks' upload stall; the ratios between rows are what to read.

| 25 euler steps, raw H3, keep 10 % of key tiles | ComfyUI total / sampler | h3-rs denoise + decode / denoise | per step: dense / sparse | PSNR vs the dense 25-step clip |
|---|---|---|---|---|
| 5 dense first steps (`--attention vsa --vsa-sparsity 0.9 --dense-first-steps 5 --pv8`) | 131.4 / 122.7 s | 204.6 / 197.6 s | 12.0 s / 6.85 s | 21.4 dB |
| 3 dense first steps | 118.0 / 109.6 s | 195.4 / 188.6 s | 12.1 s / 6.9 s | 21.0 dB |
| V2, 8 steps, first 2 dense (`--vsa-sparsity 0.9 --dense-first-steps 2`) | 54.1 / 45.2 s | 76.9 / 69.9 s | 12.5–13.3 s / 7.35 s | 18.4 dB vs our V2 clip |

Both base clips keep the scene (`budget/*.grid.png`). Two things stand
out. The "dense" first steps of a VSA run go through the all-tiles VSA
state (the gate branch needs it), i.e. the bf16 fine kernel over every
tile, at 12–13 s against the dense INT8+FP8 kernel's 10.1 s step on the
same box; routing those steps through the dense attention kernel (plus
the gate branch) is the cheap fix and would put V2 at ComfyUI's budget
ahead of V2 at ours (2 × ~8 + 6 × 7.35 = 60 s against 63.8 s here). And
at keep 10 % our sparse step is 6.85 s where ComfyUI's row implies
~4.4 s ((122.7 − 5 × 6.8) / 20): the GEMM side (≈ 2.8 s per step at
this box's 530 TFLOP/s) plus the stall were most of it; the fine stage
at 10 % is a small part of the step.

RESULTS_TBD

## What the round changed

- `HostWeight::Bf16` / `Backend::weight_bf16` / `Backend::bf16_weights`
  (6fbed45, 64bb520): streamed native-precision blocks are bf16 bytes
  on the host, page-locked, uploaded as DMA. The 1.6–1.8 s per V2
  forward that was in no kernel is gone (0.2 s left).
- The fence FIFO bug of 3eb1d82 fixed (c0a807e); `H3_STREAM_STRESS=1`
  and the exact `streamed == resident` cases (153 checks) make such a
  bug fail the checker (shown on an Ada with a one-off buggy build).
- `H3_STALL=1`: per-stage prefetch accounting and an h2d probe.
- The MXFP8 row split decided per shape on first use (9bb1a3f + the
  margin), `H3_CUDA_MX_SPLIT_MIN=n` to force.
- Numbers on a 600 W 5090 against ComfyUI's page: V2 42.6 s denoise /
  48.9 s with decode against 45.2 / 54.1 s (ahead); the dense base
  7.72 s per step against 6.8 s (1.13×, from 1.15–1.27× and 2.0× in
  bf16). Every block MXFP8 (off by default, 22 dB band): 39.2 / 45.5 s
  and 7.34 s.
- Open: route the dense first steps of a VSA run through the dense
  INT8+FP8 kernel (V2 at ComfyUI's budget would then be ~39 s); the
  GEMM side against ComfyUI's int8 (565 TFLOP/s here).

Cost: two Vast 5090 boxes (Texas 54227877 ≈ 3.5 h at $0.80/h, UAE
60356253 ≈ 2 h at $0.67/h), five RunPod Ada kernel-check pods at
$0.24–0.69/h for ~10 min each: about $6.5.

