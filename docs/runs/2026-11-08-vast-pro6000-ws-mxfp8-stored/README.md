# 2026-11-08: RTX PRO 6000 Workstation Edition (600 W): stored MXFP8, MXFP8 decoder, the split clip

Vast instance 59660209 (offer 58645173): NVIDIA RTX PRO 6000 Blackwell
**Workstation Edition** (sm_120, 96 GB, **600 W**, boost 2430 MHz), driver
615.71.10, Czechia, $1.52/h; the host took 3 h to hand the box over, then
55 min of work, self-destroyed. Binary: release of `Stored MXFP8
checkpoints, parallel MXFP8 quantization, --decoder-weights`
(`--features cuda`). Weights: the FastH3-V2 tree (138 GiB) in 557 s. Raw
outputs `output-1-check.txt`, `output-2-clips.txt`; the profiled run
`v2-mx8-stored-profiled.log`; `clocks-mx8.log`; `quantize.log`.

**This box is slower than the Server Edition of
`2026-11-07-vast-pro6000-server-mxfp8`**: the same bf16 forward is 7.0 s
here against 6.3 s there (SM clock 2002–2497 MHz, median 2100, at the 600
W cap, against 2220–2340). Compare rows within this record, not across.

Conformance: 108 / 109 both with the FP8 path and emulated; the one miss,
`snake beta` at rel 3.01e-2, is the checker's random stream shifted by the
new MXFP8 cases (fixed in the next commit by giving the snake case
bf16-representable parameters; lavapipe then matches to 4e-7).

## The stored MXFP8 checkpoint

`h3 quantize --variant fasth3-v2 --weights mxfp8 --skip-adaln`: the 65 GiB
transformer tree → **25.6 GB** single file in **116.6 s** (903 tensors;
blocks 0–1 and 47–49 plain f16, the rest `.weight.mx8` + `.weight.mxs`).
Loading it: **104 s** against 202 s for the bf16 tree on this host (the
earlier 289–307 s with quantization at load).

## Clips (V2, seed 7, the fox prompt, 1344×768×124, AdaLN tables)

| | (a) bf16 DiT, bf16 decoder | (b) bf16 DiT, `--decoder-weights mxfp8` | (c) stored MXFP8 DiT (split FFN-up), MXFP8 decoder |
|---|---|---|---|
| denoise, 8 forwards | 56.3 s (6.96–7.09 s) | 56.4 s | **44.4 s** (5.49–5.56 s after a 5.72 s first step that times the cuBLASLt candidates) |
| video decode | 9.75 s | **7.90 s** | **7.54 s** |
| audio decode | 0.17 s | 0.17 s | 0.17 s |
| total after load | 67.1 s | 65.4 s | **53.0 s** |
| load | 202 s | 205 s | **104 s** |
| peak device memory | | | 37.9 GB |
| PSNR vs (a) | | **51.3 dB** (Y 48.1) | 16.6 dB (Y 12.1) |

- **The MXFP8 decoder is the bf16 decoder to 51 dB**: the frames are
  indistinguishable (`fox-bf16-dit-mxfp8-decoder-frames.png` against
  `fox-bf16-frames.png`), the same number fastvideo-rs gated its MXFP8
  decoder on (PSNR 51, LPIPS 0.001). The decode drops 19 % (9.75 → 7.90
  s). Per chunk 0.80 s of GPU work (was 1.17 s in bf16): the four FP8
  GEMMs 0.47 s at 512 TFLOP/s, attention 0.13 s, activation quantization
  54 ms over 4032 calls; the other ~0.3 s per chunk is the host stitch and
  the chunk boundaries.
- **The MXFP8 DiT** changes the sample as before (16.6 dB against bf16,
  same scene and motion, `fox-mx8-stored-frames.png`): the DMD sampler
  amplifies weight rounding into texture.
- Denoise profile of (c), 44.03 s in ops: VSA fine 15.3 s (34.8 %, 38.3 ms
  per call on this slower box), FFN-up ×2 6.83 s (19.0 ms per call), QKV
  6.05 s, FFN-down 3.45 s, out / gate 3.37 s, the five bf16 edge blocks
  3.62 s, activation quantization 0.98 s. The split FFN-up now costs 1.13×
  the QKV GEMM for 1.33× its FLOPs, i.e. the same rate.
- (c) and its profiled twin (d) are not byte-identical any more: the first
  call on each FP8 shape times the candidates and the winner can differ
  between runs (the clips differ at the rounding level). `H3_CUDA_LT_NTH=i`
  pins the choice.

## Against fastvideo-rs

On this slower box the bf16 → MXFP8 ratio is 0.79 (56.3 → 44.4 s); applied
to the Server Edition's 50.6 s bf16 clip that is ~40 s against
fastvideo-rs's 45.8 s MXFP8 recipe on that card class (its V2-NVFP4 + FP4
attention tier: 31.9 s). The decode gap closes to 7.5 s against their
4.7 s MXFP8 decode, with the host stitch and the 1.3 k-row tile batches as
the remaining difference.
