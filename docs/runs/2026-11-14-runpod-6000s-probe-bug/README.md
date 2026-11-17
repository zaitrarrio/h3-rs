# 2026-11-14: the current tree on a Server Edition; a diagnostic probe that wrote into the model

Purpose: measure the current tree's V2 recipe on fastvideo-rs's card class
(FP8 P·V, the per-shape GEMM split, the dense-plan path and the streaming
fixes were all unmeasured there since `2026-11-08-vast-pro6000-server-best`
and `2026-11-12-runpod-pro6000-v2-omni-trim`), with the stored MXFP8
checkpoint and the AdaLN tables from the RunPod volume, staged onto the
container disk. Three RunPod pods in EU-RO-1 (a Vast "RTX PRO 6000 S" box
rented first showed another tenant's job at 89 GB / 100 % and a 450 W cap
at boot and was destroyed unused), each **RTX PRO 6000 Blackwell Server
Edition, 600 W, driver 595.91.07**, AMD EPYC 9555, median SM clock
2257–2287 MHz under load.

## Round 1: valid timings, garbage clips (`round1-garbage/`)

Binary 881e276 (the dense-steps round's), `H3_STALL=1` on every clip. The
checker passed (168 / 168 plain and under `H3_STREAM_STRESS=1`) and the
times were what the tree promised, but every clip was saturated colour
streaks with no fox in them (`v2-best.grid.png`), resident and streamed
alike, and the two recipes were 22 dB from each other as if they were
proper clips.

| fox, 1344×768×124, seed 1 | denoise | decode | total after load | peak |
|---|---|---|---|---|
| stored MXFP8, `--vsa-qk8 --pv8`, resident | 36.74 s (5.16, 4.47–4.53 s per forward) | 5.93 s | 43.24 s | 39.6 GB |
| the same without `--pv8` | 38.03 s | 5.77 s | 44.37 s | 38.9 GB |
| the same `--stream-blocks` | 37.09 s (compute stream waited 118 ms in 400 blocks, copies 17.2 GB/s) | 5.84 s | 43.50 s | **18.7 GB** |
| ComfyUI's budget (keep 10 %, first 2 dense) | 38.68 s (7.67, 6.99, then 3.99–4.01 s) | 5.78 s | 44.99 s | 40.3 GB |

The first-call decision lines: QKV 21504 × 5376 whole 12.46 ms, ×2 12.44 ms
→ **whole** (the Server Edition's cuBLASLt runs that shape well), plain FP8
11.65 ms; FFN-up 28672 × 5376 whole 32.62, ×2 15.20 → ×2, plain FP8 15.60 ms.

## The cause (f740ec6, f4934ac)

Not the weights, the tables, the prompt or the hardware: the **plain-FP8
timing probe** added in 90ab8f4 for the "how much is the block scaling"
question. It runs on a shape's first call, only under `H3_STALL=1` /
`H3_PROFILE=1`, with the real operands and **the real output buffer** (the
unscaled E4M3 codes: garbage of enormous magnitude), after the whole and
split plans have been timed. The recompute that follows ran only when the
split decision differed from the initial `parts`, so a shape that kept
whole returned the probe's output on its first call. On the Server Edition
the QKV shape keeps whole: the first block's QKV of step 1 was garbage and
the clip never recovered. On the 5090 (`2026-11-14-vast-5090-dense-steps`)
the QKV and FFN-up shapes split, so the DiT was untouched, and only the
decoder's 16384 × 2048 shape kept whole: one FFN GEMM call of the first
decode chunk, which is the grey patch in the top-left corner of frame 0 of
that record's V2 grids. The timings of both records stand (the same kernels
run); their clips are what the text above says.

The same recompute condition hid a second, smaller flaw with the diagnostics
off: the timing runs leave the **split** plan's output in y, so a shape that
kept whole returned the split's rounding on its first call and the whole
plan's afterwards. That is why the current binary's fox differed from the
11-09 binary's at 19 dB in the bisect below (no garbage, the usual band of a
one-GEMM rounding change through eight steps) while every binary before
5677a4a was byte-identical to it. y now always comes from the chosen plan
(and the probe, when it ran, is always followed by that recompute).

## Bisect (`bisect/`, the same stored file, tables and fox prompt, `--vsa-qk8`, no diagnostics)

| binary | clip | against the 11-09 binary's |
|---|---|---|
| 11-09 (h3-bin5: the code of the good 2026-11-12 clip) | fox, luma 116.65 | reference |
| 11-13 04:31, 12:07, 17:30, 21:39; 11-14 00:08 (streaming, page-locking, INT8 / FP8 attention, stall accounting) | fox | **byte-identical** ×5 |
| 11-14 04:08 (5677a4a, the per-shape split decision) | fox, luma 115.34 | 19.1 dB (the first-call rounding above) |
| current (881e276), `H3_TRACE=2`, no `H3_STALL` | fox, luma 115.34 | 19.1 dB, the same clip as 04:08 |
| current, samurai prompt | the samurai | (another scene) |
| current, bf16 tree quantized at load | fox, luma 117.05 | 15.8 dB (a different cuBLASLt candidate at load-time quantization: the band) |
| current, tables rebuilt by it | fox, luma 115.04 | 18.6 dB (the rebuilt tables differ from the 11-12 file at the rounding level) |

Without the diagnostic switches the current binary makes a proper fox
(`cur.grid.png`), which is why the 5090 rounds, whose clips were checked by
eye and by PSNR between budgets, never showed the probe: their shapes split.

## Verification with the fix (`fixcheck/`, `check-5090/`)

The checker had not seen the probe because none of its MXFP8 shapes reached
the per-shape split decision (`n ≥ 16384`, `rows ≥ 1024`): the two
"split-able" 130 × 64 × 512 cases passed under `H3_STALL=1` on the old
binary too. Two cases now use 1024 × 64 × 16384: the first call against the
CPU, and the second call exactly equal to the first. On an RTX 5090 pod
(`check-5090/`, no volume) the new checker built on the old CUDA code
**fails both under `H3_STALL=1` and `H3_PROFILE=1`** (rel 2.3e5: the
unscaled codes) and passes plain; the fixed tree passes 170 / 170 plain and
under `H3_STALL=1`, `H3_PROFILE=1` and `H3_STREAM_STRESS=1`.

On the third Server Edition pod (f4934ac, the fix without the new cases):
168 / 168 in all four modes, and the clips:

| fox, stored MXFP8, tables, resident unless said | denoise | decode | total after load | peak | clip |
|---|---|---|---|---|---|
| 11-09 binary, `--vsa-qk8` (the 2026-11-12 recipe) | 37.48 s (4.82, 4.64–4.68 s) | 5.83 s | 43.86 s | 38.9 GB | reference fox |
| fixed, `--vsa-qk8` | 37.78 s (5.12 first: the whole-against-split timing, then 4.64–4.69 s) | 5.91 s | 44.26 s | 38.9 GB | fox, 19.1 dB from the reference |
| fixed, `--vsa-qk8`, `H3_STALL=1` | 38.09 s | 6.05 s | 44.71 s | | **byte-identical** to the line above |
| **fixed, `--vsa-qk8 --pv8` (the best recipe)** | **36.55 s** (4.96, 4.48–4.53 s) | **5.73 s** | **42.85 s** | 39.6 GB | fox (`best.grid.png`), 18.4 dB from the qk8 clip |
| the same, `H3_STALL=1` | 36.79 s | 6.02 s | 43.37 s | | **byte-identical** |
| the same, `H3_PROFILE=1` | 36.88 s | 7.18 s (synced per op) | | | **byte-identical** |
| the same, `--stream-blocks` | 37.11 s (compute stream waited 117 ms in 400 blocks; copies 17.2 GB/s) | 5.90 s | 43.59 s | **18.7 GB** | **byte-identical** |
| the same at ComfyUI's budget (keep 10 %, first 2 dense) | 38.65 s (7.66, 6.96, 3.99–4.02 s) | 5.70 s | 44.89 s | 40.3 GB | 13.7 dB from the best clip |

So the diagnostic switches and streaming leave the clip's bytes alone, and
the best recipe on the gold card is now **denoise 36.6 s, decode 5.7 s,
42.9 s after load** (stored MXFP8, MXFP8 decoder, INT8 Q·Kᵀ + FP8 P·V,
tables; load 31 s staged) against 37.9 / 5.8 / 44.3 s on 2026-11-12 with
INT8 Q·Kᵀ alone and fastvideo-rs's 45.8 s denoise. FP8 P·V is worth 1.2 s
(3.3 %) here; the fine stage runs 24.1 ms per call (27.3 on the 5090).

The 19.1 dB between the fixed binary's clip and the 11-09 binary's on the
same inputs is not the first-call flaw (f4934ac did not move it: 5677a4a's
clip, 881e276's and the fixed binary's are the same clip): the per-shape
decision tunes both plans on the first call where the old code ran the
first call untuned, so one GEMM of the first forward rounds differently.
Both are proper foxes; the new behaviour is the self-consistent one (the
first call now uses the plan every later call uses). The 2026-11-13 claim
that the slices "round like the whole GEMM, clips byte-identical" was
measured on the 5090's released binary and still holds there.

Per-kernel profile of the best recipe (`best-prof.log`, 8 forwards): the
fine stage 9.65 s (24.1 ms × 400), MXFP8 GEMMs 16.2 s at 659 TFLOP/s
(FFN-up ×2 16.3 ms, QKV whole 12.4, FFN-down 8.2, gate 4.1, out 4.1), the
five bf16 edge blocks' GEMMs 3.16 s (28.5 / 21.6 / 14.4 / 7.2 / 7.2 ms
per call: 0.39 s per forward against 0.22 s if they were MXFP8), and
~0.88 s per forward of memory-bound kernels (head_split 1.22 s, mx quant
1.04, swiglu 0.91, coarse 0.82, rms_norm 0.81, gate_residual 0.48, pool
0.47, the two 8-bit quantisations 0.87, k mean 0.33, v max 0.14): 19 % of
a 4.5 s forward on this card, 17 % on the 5090.

Cost: three Server Edition pods (14 + 50 + 23 min at $2.69/h) and a 5090
check pod (6 min at $1.39/h), about $4.
