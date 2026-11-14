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

## Verification with the fix (`fixcheck/`)

_To be filled from the third pod: the previous binary's checker under
`H3_STALL=1` (the split-able MXFP8 cases should fail), the fixed binary's
checker plain and under `H3_STALL=1`, `H3_PROFILE=1`, `H3_STREAM_STRESS=1`,
the fixed binary's fox byte-identical to the 11-09 binary's, byte-identical
with and without `H3_STALL=1`, and the recipe's numbers._
