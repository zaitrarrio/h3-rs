# 2026-11-07: VSA phase by phase on an L40S

Four L40S rentals (vast 59517729, 59519700, 59522311 in Japan, driver
575.57.08; 59567904 in Türkiye, driver 595.91.07; 350 W, $0.59–0.60/h,
~10 min each, self-destroying). 1344×768×124 (37 966 tokens, VSA 0.8, tile
64) unless stated; `bench --layers 4 --steps 1` with `H3_PROFILE=1`,
numbers per call (one block). Raw output: `output.txt`; `ptxas -v` reports
for every kernel before and after the instruction work: `ptxas-before.txt`,
`ptxas-after.txt`.

## Phases of a VSA block (before this work)

| phase | per block | of block |
|---|---|---|
| pool (tile means of Q, K, V) | 2.8 ms | 1.1 % |
| coarse: scores, per-region top-k, gate branch | 6.4 ms | 2.5 % |
| fine: attention over the selected tiles | 82–88 ms | 33 % |
| everything else (GEMMs 56 %, norms, head_split, swiglu) | ~160 ms | |

## Coarse: four query tiles per block

`fw_vsa_coarse4` reads the pooled K̄ and V̄ rows once for four query rows.
6.4 → 3.5 ms (1.8×), 97 / 97 checks. Default (`H3_CUDA_VSA_COARSE1=1`
for the old kernel; nb > 2048 falls back to it).

## Fine: what it is not bound by

| experiment | fine, ms / call | reading |
|---|---|---|
| v1 (16 rows / warp, 64-key chunks) | 88.2 | baseline (this box) |
| paired tiles: two query tiles per 8-warp block over the union of their selections, warps skip unselected key tiles | 84.8–92.2 | **gather sharing: no gain** |
| 32 rows / warp (v2) | 105.5 | 2-warp blocks: worse |
| `H3_CUDA_VSA_DBG=1` sequential keys instead of the gather | 89.4 | **the gather costs nothing** |
| `=32` no token-index prefetch | 89.8 | prefetch worth 1.6 ms |
| `=2` Q·Kᵀ mma removed | 67.2 | QK = 21 ms |
| `=4` P·V mma removed | 71.4 | PV = 17 ms |
| `=8` exp2 removed | 83.7 | exp = 4.5 ms |
| `=16` K/V loads removed | 70.0 | loads = 18 ms |
| `=6` all mma removed | **59.5** | two thirds of the kernel is not tensor work |
| `=7` all mma removed, sequential keys | 59.9 | |

Not shared memory (the v2 A/B), not the gather, not the loads: with every
mma removed the kernel still takes 59.5 of 88 ms.

`ptxas -v` on the NVRTC PTX then showed why: `fw_vsa_fine_mma128` uses 205
registers and 34.9 KB of shared memory, so exactly two 4-warp blocks fit an
SM, and its loop issues ~4000 PTX instructions per 64-key chunk against 128
mma. The kernel is instruction-issue bound. A third of those instructions
were `bf()` / `pack_bf16`: a software round-to-nearest-even (11 integer
instructions per value, 32 values per chunk) where sm_80+ has
`cvt.rn.bf16x2.f32`. The rest: a multiply per score for the softmax scale,
a byte load + compare + select per key for the padding mask on every chunk,
64-bit index arithmetic per gathered key.

## Fine: the third body and the occupancy experiments

| variant | regs | smem | blocks / SM | 1344×768 | 832×480 | Omni tile 128 |
|---|---|---|---|---|---|---|
| v1 | 205 | 34.9 KB | 2 | 87.9 | 22.5 | 64.6 |
| 3a: 32-key chunks, Q re-read, 4 blocks asked | 128 (spills) | 17.4 KB | 4 | 238.1 | 52.5 | 159.3 |
| 3b: 32-key chunks, K/V double-buffered, Q resident | 193 | 34.9 KB | 2 | **86.7** | **20.0** | **60.4** |
| 3c: double-buffered + Q re-read, 3 asked | 163 | 34.9 KB | 2 | 204.1 | 49.0 | 138.1 |

Q fragments must stay in registers (3a / 3c re-read them from L1 per mma
and lose 2.5×; 3a also spills). Double buffering with the next chunk
issued a whole chunk ahead (3b) is worth 1–11 %. All variants 97 / 97.

## Fine: instruction diet (fourth rental, old and new binary on one box)

`cvt.rn.bf16(x2)` for every bf16 conversion, the scale folded into the
exp's FMA, a warp vote that skips the per-key mask on fully valid chunks,
32-bit gather indices: 4027 → 2689 PTX instructions per chunk (v1),
2934 → 1671 (3b); registers 205 → 196 / 193 → 189, no spills. `fw_rms`'s
row cache moved from a 96-byte stack frame into registers (unrolled loop).
Results: see the table at the end of `output.txt` (filled in by the run).
