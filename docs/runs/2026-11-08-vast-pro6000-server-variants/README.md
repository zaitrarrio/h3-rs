# 2026-11-08/09: RTX PRO 6000 Server Edition (600 W): the official variants, and INT8 Q·Kᵀ

Vast instance 59800772 (offer 59376656): NVIDIA RTX PRO 6000 Blackwell
**Server Edition** (sm_120, 96 GB, **600 W**), driver 615.71.09, 256 CPU
cores, Michigan, $1.54/h. Weights: the MiniMax-H3 base tree + its
`transformer_ref` (265 GB from R2 in 487 s), then from Hugging Face in the
background: Pika-H3-Flex adapters (5.2 GB), Open-H3-Regenerate-2K (20 GB),
FastH3-Omni `transformer_ref` (66 GB), and for the INT8 Q·Kᵀ clip pair the
FastH3-V2 transformer. Raw outputs `output-N-*.txt` (trace lines and
presigned URLs stripped), per-clip logs `*.log`, frame grids `*-frames.png`.

Binaries: `h3-bin` = `498adb4^` (the validation round), `h3-bin2` = 498adb4
(INT8 Q·Kᵀ behind `--vsa-qk8`), `h3-bin3` = dc4e0c8 (two-stage K mean,
single-read quantizer).

Conformance: `h3 check --backend cuda` **109 / 109** (h3-bin) and
**115 / 115** with the six `vsa … qk8` cases (h3-bin2), tolerance 3e-2.

## Base MiniMax-H3 (`--variant h3`, bf16, dense attention, 49 Euler forwards)

AdaLN tables (`h3 adaln-tables --variant h3`, 99 timesteps): 960 MB in
197 s.

| | load | prompt (Qwen3-VL on the host) | denoise | total |
|---|---|---|---|---|
| T2VA 1344×768×124, seed 7, the fox prompt | 83.7 s | 96.1 s | **524.7 s** (49 × 10.70–10.72 s) | 12 m 00 s wall |

`base-t2va-frames.png` (frames 0 / 20 / 40 / 60): a coherent fox at dawn
in a snowy pine forest, so the base sampler (49 Euler forwards on the `h3` preset)
runs end to end on real weights. The host-side text encoder
is now the second-largest cost of a base clip after the denoise.

The first FL2VA and Ref2VA attempts asked for 61 frames and were refused by
the pipeline's own geometry guard (`H3 generates 5–15 s at 24 fps; aligned
num_frames 73 is outside 124–362`): the floor is 124 frames. They were
re-run at 124 frames (below).
