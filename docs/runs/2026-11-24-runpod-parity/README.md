# 2026-11-24: full-width DiT parity, a cold load through pread, the fast decoder's profile, and a pod that terminated itself

Pod `snt7u97qgy54qy`: RTX PRO 6000 Blackwell Server Edition (sm_120),
600 W, driver 595.112.07, EU-RO-1. It was created by `scripts/runpod/pod.py`
after 10 minutes of out-of-stock retries. The binary is df3d775 (+ the
`regen2k --out` fix), built with `--features cuda`. 193 / 193 CUDA checks.

## The pod terminated itself

The pod was created with **no account key** in its env, so `RUNPOD_API_KEY`
is RunPod's pod-scoped one (`H3_RUNPOD_KEY` unset). The last command's
`terminate` (REST unresolvable, then GraphQL) removed the pod: `pod-status`
answers `null`. This is the first time a pod of this project has terminated
itself (`2026-11-24-runpod-validate` has the diagnosis).

## Full-width DiT parity against FastVideo

- **Reference:** FastVideo a7eb340, torch 2.12.0+cu130, f32 on the GPU, SDPA
  attention. `scripts/reference/full_parity.py` loads h3-base's real
  `transformer/` for the embedders, the text refiner, blocks 0–1 and the head:
  62 tensors from 14 shards.
- **Packed sequence:** 2 × 8 × 8 latents (48 video rows of 96), 5 audio
  latents (10 rows of 32) and 12 text rows (one padding).
- **Ours:** `h3 parity-dit` loads the same blocks and runs the same sequence.

| backend | result |
|---|---|
| CUDA (bf16 activations) | **parity ok**: within 3e-2 of the reference on video and audio |
| CPU (f32) | **parity ok**: within 2e-3 |

The exact errors were cut by the command's `tail`: a rerun records them. This
holds the real widths: 56 heads × 128, 5376 hidden, 14336 FFN, the
96-of-128 RoPE split, the AdaLN tables from the real modulation weights. Until
now that had only been held by clips; the committed fixtures are tiny.

The first two tries (`2026-11-24-runpod-validate`) used CPU torch and failed:
FastVideo picks NCCL whenever NVML sees a GPU, whatever
`CUDA_VISIBLE_DEVICES` says. CUDA torch on the GPU is the way.

## A cold load through parallel pread: no gain

The stored MXFP8 V2 (`transformer.safetensors`, 25.6 GB) was read from the
volume with `H3_PREAD`'s path (then the default), as the first read of that
file on this pod: load **107.9 s**. The previous pod's first read of the same
file through the mmap took 89.1 s. These are two pods, so not a controlled
comparison, but there is no gain to keep. The pread path is now opt-in
(`H3_PREAD=1`), and staging remains the answer.

## The fast decoder at 2K, per op

`H3_PROFILE=1 h3 vae-roundtrip --skip-base`, 2560×1440×124 (37×90×160
latents), bf16. Each op is timed between syncs, so the sum (7.57 s) is below
the 11.9 s decode wall time: the rest is the head's downloads and host work.

| op | time | share | calls |
|---|---|---|---|
| FFN up GEMM (n 16384, k 2048) | 2141 ms | 28.3 % | 792 |
| `box_rows` (one bf16 per thread) | 1317 ms | 17.4 % | 8922 |
| FFN down GEMM (n 2048, k 8192) | 1114 ms | 14.7 % | 792 |
| window / global attention (mma2, d64) | 850 ms | 11.2 % | 1240 |
| QKV GEMM (n 6144) | 798 ms | 10.5 % | 192 |
| SwiGLU | 451 ms | 6.0 % | 792 |
| to_out GEMM | 258 ms | 3.4 % | 24 |
| norms, head_split, residuals | ~590 ms | 7.7 % | |

- **GEMMs:** 4.33 s at 398 TFLOP/s.
- **Window gathers:** `box_rows` moved one bf16 per thread. It now moves 8
  (one 16-byte `uint4`) when the row width is a multiple of 8, the decoder's
  64 (`fw_box_rows16`, checker cases at width 16). That kernel is not yet
  timed on a GPU.
