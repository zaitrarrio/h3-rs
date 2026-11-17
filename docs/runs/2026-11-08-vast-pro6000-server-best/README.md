# 2026-11-08: RTX PRO 6000 Server Edition (600 W): the best configuration, measured

Vast instance 59667020 (offer 59376660): NVIDIA RTX PRO 6000 Blackwell
**Server Edition** (sm_120, 96 GB, **600 W**, boost 2430 MHz), driver
615.71.09, 256 CPU cores, 755 GB RAM, Michigan, $1.54/h; 50 min,
self-destroyed. Binary: release of `Decode: each chunk's host work runs on
a thread…` (`--features cuda`). Weights: the FastH3-V2 tree (138 GiB) in
254 s. Raw outputs `output-1-check-bench.txt`, `output-2-clips.txt`; the
profiled run `v2-best-profiled.log`; `clocks-best.log`; `quantize.log`.

**Comparable card**: the bf16 bench forward is 6.24–6.28 s here against
6.29–6.34 s on the Server Edition of `2026-11-07-vast-pro6000-server-mxfp8`
(the Workstation Edition of `2026-11-08-vast-pro6000-ws-mxfp8-stored` was
7.0 s). Under the denoise: 2175–2392 MHz, median 2272, at the 600 W cap,
peak device memory 37.9 GB. This is the card class fastvideo-rs's gold
numbers come from.

Conformance: **109 / 109** with the FP8 path and 109 / 109 emulated
(`H3_CUDA_NO_FP8=1`): the `snake beta` fix of the previous record holds.

## Bench (synthetic weights, 50 blocks, 37 966 tokens, per forward)

| | forward | per block |
|---|---|---|
| bf16 | 6.24–6.28 s | 125 ms |
| `--weights mxfp8` (blocks 2–46, split FFN-up) | 4.75–4.76 s | 95 ms |
| + `H3_MXFP8_ALL_BLOCKS=1` | 4.59 s | 92 ms |

## Clips (V2, seed 7, the fox prompt, 1344×768×124, AdaLN tables)

| | (a) bf16 DiT, bf16 decoder | (b) stored MXFP8 DiT, MXFP8 decoder, overlapped decode | (c) = (b) + all blocks MXFP8 |
|---|---|---|---|
| denoise, 8 forwards | 50.0 s (6.21–6.27 s) | **38.9 s** (4.81–4.85 s after a 5.03 s first step that times the cuBLASLt candidates) | 37.6 s (4.65–4.69 s) |
| video decode | 8.15 s | **5.56 s** | 5.79 s |
| audio decode | 0.19 s | 0.19 s | 0.19 s |
| total after load | 58.7 s | **45.0 s** | 44.0 s |
| load | 76 s | **37 s** | 41 s |
| PSNR vs (a) | | 16.7 dB (Y 12.2) | 16.8 dB; 18.9 dB vs (b) |

`h3 quantize --weights mxfp8 --skip-adaln`: 25.6 GB in **46.9 s** on this
256-core host (117 s on 16 cores).

- **Decode** is 5.56 s: 7 chunks of 0.80 s GPU work with the host stitch
  of chunk i now overlapped with chunk i + 1's kernels (it was 7.54 s with
  the same kernels). The first profiled "video decode" table carries two
  chunks' calls because the first `Frames` event now arrives after the
  second chunk's kernels are queued.
- **All blocks MXFP8** buys 1.2 s of denoise; its clip sits in the same
  band against bf16 (16.8 dB) and 18.9 dB from the recipe's clip. Kept
  behind `H3_MXFP8_ALL_BLOCKS=1`: the recipe's edges are fastvideo-rs's and
  Sol-H3's calibrated choice, and the gain is 3 %.
- Denoise profile of (b), 38.67 s in ops for 38.86 s wall: VSA fine
  13.79 s (**35.7 %**, 34.5 ms per call), FFN-up ×2 5.78 s (16.0 ms), QKV
  4.38 s (12.2 ms), FFN-down 2.94 s, out / gate 2.92 s, the five bf16 edge
  blocks 3.11 s, head_split 1.22 s, activation quantization 1.04 s, swiglu
  0.91 s, coarse + top-k 0.83 s, rms_norm 0.80 s. GEMMs 669 TFLOP/s
  overall (the FP8 ones ~760).

## Against fastvideo-rs (same card class)

| | fastvideo-rs | h3-rs (b) |
|---|---|---|
| FastH3-V2 denoise, VSA 0.8, MXFP8 recipe | 45.8 s | **38.9 s** |
| V2-NVFP4 + FP4 attention (its "fastest", gate-failing tier) | 31.9 s | — |
| 36-block video decode | 4.7 s MXFP8 / 6.3 s bf16 | **5.6 s** (MXFP8 linears) |
| peak device memory | 61.0 GiB | 37.9 GB |

The denoise is 1.18× faster than fastvideo-rs's with the same recipe and
quantization edges; the decode is between its two decoders. What is left
on the denoise is the bf16 VSA fine stage (36 %); fastvideo-rs's remaining
1.4× to its fastest tier comes from FP4 weights and an FP4 fine stage,
both of which fail its own quality gate.
