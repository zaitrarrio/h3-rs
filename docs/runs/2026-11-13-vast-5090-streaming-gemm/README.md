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

## Results

RESULTS_TBD
