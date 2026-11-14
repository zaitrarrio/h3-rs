# 2026-11-14: the dense steps of a VSA run through the dense kernel; plain FP8 against MXFP8

Follow-up to `2026-11-13-vast-5090-streaming-gemm`, items 1 and 2 of its
closing list. Two changes measured on one more Vast 5090:

1. **A dense plan is dense attention plus the gated pooled branch**
   (90ab8f4). The dense layers and the "dense first steps" of a VSA run
   (ComfyUI's attention budget) went through the all-tiles fine stage,
   which computes the same softmax at about twice the dense kernel's
   cost (9.8 s against 5.3 s per V2 step on the previous box). The CPU
   reference now defines a plan that keeps every tile as dense attention
   over the sequence in its natural order, with the dense 8-bit recipes
   (`attn_qk8_round`, `attention_pv8_reference`) when the spec asks,
   plus `pooled[tile(tok)] · gate`; CUDA runs its dense kernel and
   `fw_vsa_gate_add`, WebGPU its dense attention and elementwise op 20.
   Five check cases per tile configuration (168 in all); lavapipe at
   1e-7 of the reference.
2. **How much of an MXFP8 GEMM is the block scaling.** Under
   `H3_STALL=1` the first call on each large shape also times a plain
   FP8 plan (per-tensor scales, same operands, output discarded) next to
   the whole and split MXFP8 plans, to decide whether a per-row FP8
   weight format (ComfyUI's int8 file is per-row) could be worth
   building.

## Box

Vast offer 43110775, instance 60656633: RTX 5090 at a **575 W** limit,
PCIe 5 ×16, **driver 615.71.09** (newer than the 595.91.07 of every
earlier 5090 box; kernel numbers may not compare exactly), AMD EPYC 9654,
weights pulled in 590 s with no lost transfer, V2 and base-25 tables
built on the box. `h3 check --backend cuda`: **168 / 168**, plain and
under `H3_STREAM_STRESS=1`; the fifteen dense-plan cases at 2–9e-3 of
the reference (`check.log`).

## Results

All on this box, `H3_STALL=1`, the automatic split (QKV and FFN-up ×2),
1344×768×124, seed 1. The page's numbers are ComfyUI total / sampler.

| | ComfyUI | denoise + decode / denoise | per step | PSNR |
|---|---|---|---|---|
| FastH3-V2, 8 steps, keep 20 % (our budget) | 54.1 / 45.2 s | **49.4 / 42.2 s** | 5.11–5.16 s | |
| FastH3-V2, ComfyUI's budget: keep 10 %, first 2 dense | 54.1 / 45.2 s | 51.8 / 44.7 s | dense **8.1–9.0 s** (was 9.8–13.3), sparse 4.60 s | 18.2 dB vs ours |
| raw H3, 25 steps, keep 10 %, 5 dense first (`core-blocksparse` row) | 131.4 / 122.7 s | **130.9 / 123.9 s** | dense 7.63 s, sparse 4.25 s | 26.4 dB vs dense |
| raw H3, 25 steps, keep 10 %, 3 dense first (`sparse-solattn` row) | 118.0 / 109.6 s | 124.0 / 116.8 s | dense 7.61 s, sparse 4.24 s | 23.2 dB vs dense |
| raw H3, 25 dense steps (`baseline`) | 181.7 / 169.0 s | 197.8 / 191.0 s | 7.52–7.65 s | reference |

The dense steps of a VSA run now cost what a dense step costs (V2's at
8.1 s is the V2 forward with the dense call in place of the fine stage,
not a 5.3 s step: that earlier figure was the base model's), which is
what makes the block-sparse rows comparable: the raw model with five
dense steps lands **on ComfyUI's row to within 1 %** (130.9 against
131.4 s total, 123.9 against 122.7 s sampler) from 1.6× behind on the
previous box, and the three-dense-steps row 5 % behind. V2 at ComfyUI's
budget is 44.7 s, 1 % inside its 45.2 s sampler, but still behind V2 at
its own budget (42.2 s): two dense steps at 8.1 s cost more than six
sparse steps save, so keep 20 % stays the recipe. Clips keep the scene
(`*.grid.png`); the PSNRs are the usual bands between attention budgets.

### Plain FP8 against MXFP8 (item 2)

The first call's decision lines (`v2.log`), 38 172 rows:

| shape | MXFP8 whole | MXFP8 ×2 (chosen) | plain FP8, per-tensor scale, whole |
|---|---|---|---|
| QKV 21504 × 5376 | 24.3 ms | **12.9 ms** | 18.1 ms |
| FFN-up 28672 × 5376 | 31.9 ms | **17.5 ms** | 25.4 ms |
| 16384 × 2048, 1797 rows | 0.19 ms | 0.20 ms (whole kept) | 0.28 ms |

Dropping the block scaling buys ~25 % on a whole-shape GEMM, less than
the row split buys with the scaling kept (~46 %); a per-row FP8 split
GEMM would at best shave the same quarter off the split MXFP8 times,
about 0.15 s of a 5.1 s V2 forward. **Not worth a new weight format**;
the GEMM gap against ComfyUI is cuBLASLt's kernels at these shapes, and
the split already takes most of it. Item 2 is closed.

## Where this leaves the 5090 against the page

V2 ahead on both numbers (49.4 / 42.2 against 54.1 / 45.2 s); the raw
model's block-sparse rows at parity (5 dense steps) and 5 % behind (3
dense steps); the raw dense row 1.12× behind on the step (7.64 against
6.8 s), the one row where ComfyUI's all-int8 file and GEMM kernels still
show. Cost of this round: one box, ~1 h at $0.60/h.
