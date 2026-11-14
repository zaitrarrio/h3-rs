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

RESULTS_TBD
