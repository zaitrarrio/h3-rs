# 2026-11-07: RTX PRO 6000 Server Edition (600 W), MXFP8 block linears

The first run on fastvideo-rs's gold GPU class. Vast instance 59622767
(offer 47550078): NVIDIA RTX PRO 6000 Blackwell **Server Edition** (sm_120,
96 GB, **600 W** power limit, boost 2430 MHz), driver 595.71.05, 16 CPU
cores, Virginia, $1.54/h; 1 h 15 min, self-destroyed. Image
`nvidia/cuda:12.8.1-runtime-ubuntu24.04`; binaries: release builds of
`MXFP8 block linears…` (`h3-bin`) and `CUDA MXFP8: time cuBLASLt's
heuristic candidates…` (`h3-bin2`), `--features cuda`. Weights: the
FastH3-V2 tree (138 GiB) in 336 s. Raw outputs: `output-1-check-bench.txt`,
`output-2-clips.txt`, `output-4-autotune.txt`; full profiled runs
`v2-mxfp8-profiled.log`, `v2-mxfp8-tuned-profiled.log`; `clocks-*.log` are
`nvidia-smi` every 2 s.

## Conformance

`h3 check --backend cuda`: **107 / 107** at 3e-2 with the FP8 tensor-core
path, 107 / 107 with the bf16 emulation (`H3_CUDA_NO_FP8=1`), both
binaries. The ten `linear mxfp8` cases land at rel 2.2–4.3e-3 against the
CPU W8A8 reference (bf16 output rounding; the other bf16-output linears sit
in the same band).

## The card

Under the denoise the Server Edition holds **2197–2422 MHz, median
2220–2340**, at its 600 W cap (`SW power cap` set in 25 of 30 samples,
48–68 °C). The Max-Q of the previous records held 1380–1987 MHz (median
1413) at 300 W: the same chip runs the same bf16 forward 1.57× faster here
(126 vs 197 ms per block). Every earlier gap to fastvideo-rs that was
attributed to "the card" was this.

## Bench (synthetic weights, 1344×768×124, 37 966 tokens, 50 blocks)

| | per forward | per block | GEMMs |
|---|---|---|---|
| bf16 | 6.29–6.34 s | 126 ms | 408 TFLOP/s |
| mxfp8, cuBLASLt's first heuristic pick | 5.65–5.67 s | 113 ms | 4 shapes at ~780 TFLOP/s, n = 28672 at 355 |
| mxfp8, candidates timed on first use + 8-wide quantizer | 5.52–5.53 s | 110.5 ms | as above |
| mxfp8 emulated in bf16 (`H3_CUDA_NO_FP8=1`, 8 layers) | 127 ms / block | | = bf16 |

The FFN-up GEMM (`n=28672 k=5376`, 11.7 TFLOP) runs at **33 ms = 355
TFLOP/s in FP8, slower than its bf16 28.4 ms**, and every one of cuBLASLt's
four candidates for it does (`H3_CUDA_LT_NTH=0..3`: 32.9–33.0 ms), while
`n=21504` runs at 12.2 ms (720 TFLOP/s) and `n=5376 k=14336` at 8.1 ms
(727). So the autotune gains nothing on this shape and the 8-wide
quantizer took the activation quantization from 1.27 to 0.58 ms per call
(2.3 % of the forward). The remaining lever on this shape is to split it
along N: see below.

## The N-split (second instance, bench only)

Vast instance 59631383 (offer 59376736): the same Server Edition at 600 W,
driver 615.71.09, 128 CPU cores, Nevada, $1.20/h, 25 min, no weights
(synthetic bench), binary = release of `CUDA MXFP8: optional N-split…`.
Raw output: `output-5-nsplit-bench.txt`. `H3_CUDA_MX_SPLIT=p` runs an FP8
GEMM with `n ≥ H3_CUDA_MX_SPLIT_MIN` as `p` GEMMs over row slices of the
weight (whole 128-row scale tiles), writing column slices of the same y.

| FFN-up `n=28672 k=5376` (11.7 TFLOP) | ms / call | TFLOP/s |
|---|---|---|
| whole | 32.9 | 355 |
| ×2 (two 14336-row halves) | **15.1** | **775** |
| ×4 | 15.1 | 775 |
| ×7 | 15.3 | 765 |

The QKV GEMM `n=21504` does not move (11.2 ms whole, 11.3 ×2, 11.1 ×3);
`n=7168` ×4 is 3.87 vs 3.75 ms. Default: split 2 on `n ≥ 24576`, nothing
else. Full depth, 50 blocks, 37 966 tokens: **4.80–4.83 s per forward
(96.5 ms per block)** against 5.51–5.53 s with the split off and 6.29–6.34
s in bf16; a V2 clip's 8 forwards project to ~38.6 s against the 45.0 s
measured without the split (not re-measured as a clip on this instance:
no weights). Conformance on this box: 108 / 109, the one failure `snake
beta` at rel 3.01e-2 against the 3e-2 tolerance on driver 615.71.09 (it
passed at 595.71.05 and on the Max-Q); the split-able MXFP8 cases pass
with the split forced (`H3_CUDA_MX_SPLIT_MIN=256`, rel 3.6 / 4.4e-3).

## Clips (V2, seed 7, the fox prompt, 1344×768×124)

| | bf16 | mxfp8 (first pick) | mxfp8 (tuned binary) |
|---|---|---|---|
| denoise, 8 forwards | **50.6 s** (6.26–6.38 s) | 45.7 s (5.67–5.75 s) | **45.0 s** (5.55–5.62 s after a 5.85 s first step that times the candidates) |
| video decode | 8.9 s (7 × 1.17 s GPU) | 8.8 s | 8.8 s |
| audio decode | 0.19 s | 0.19 s | 0.19 s |
| total after load | 60.3 s | 55.2 s | 54.5 s |
| peak device memory | 56.2 GB | 41.0 GB | 41.0 GB |
| load (the 138 GiB tree, page-cache cold / warm) | 148 s | 307 s (quantizing 45 blocks on the host) | 289 s |

`fox-v2-bf16-frames.png` / `fox-v2-mxfp8-frames.png`: frames 0 / 40 / 80 /
120. The MXFP8 clip is the same scene, composition and motion with the fox
and camera a few pixels off; `ffmpeg psnr` against bf16 is 17.3 dB (Y
12.7) for the first pick and 16.6 dB for the tuned binary (a different
cuBLASLt algorithm rounds differently; the two MXFP8 clips differ from
each other too). int8 vs bf16 was 18.5 dB on the Max-Q: the DMD sampler
amplifies any weight rounding into texture while keeping the picture.

Denoise profile, tuned MXFP8 (`v2-mxfp8-tuned-profiled.log`, 44.55 s in
ops for 44.96 s wall): VSA fine 13.66 s (30.7 %, 34.1 ms per call), FFN-up
FP8 GEMM 11.80 s (26.5 %), the other FP8 GEMMs 10.2 s, bf16 GEMMs of the
five edge blocks 3.16 s, head_split 1.22 s, activation quantization 1.04 s,
swiglu 0.90 s, coarse + top-k 0.82 s, rms_norm 0.81 s. With the FFN-up at
the other shapes' rate the denoise would be ~38.5 s.

## Against fastvideo-rs on the same card class

fastvideo-rs `lab/gold` / `docs/techniques.md`, RTX PRO 6000 Blackwell
Server Edition, 1344×768×124, 8 forwards:

| | fastvideo-rs | h3-rs (this run) |
|---|---|---|
| FastH3-V2 denoise, VSA 0.8, MXFP8 linears | 45.8 s | **45.0 s** (bf16: 50.6 s) |
| V2-NVFP4 + FP4 attention denoise | 31.9 s | — |
| 36-block video decode | 6.3 s bf16 / 4.7 s MXFP8 | 8.9 s bf16 |
| peak device memory | 61.0 GiB | 41.0 GB (mxfp8) / 56.2 GB (bf16) |

Denoise is at parity with the same recipe (blocks 2–46 in MXFP8). The
decode is 1.4× behind their bf16 decoder and 1.9× behind their MXFP8 one:
our decoder GEMMs run at 285 TFLOP/s on ~1.3 k-row tile batches against
their 397, and the decoder's linears are still bf16 here.
