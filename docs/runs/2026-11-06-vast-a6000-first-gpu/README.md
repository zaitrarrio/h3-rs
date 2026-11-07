# 2026-11-06: first GPU run, RTX A6000 (vast.ai)

Vast instance 59484310 (offer 45242021): NVIDIA RTX A6000 48 GB (sm_86),
driver 570.195.03, power limit 300 W, 48 CPU cores, 282 GB RAM, 7.4 Gbit/s
down, Delaware, $0.54/h all-in; image `nvidia/cuda:12.8.1-runtime-ubuntu24.04`.
Binary: release build of f643150 (`--features cuda`, so WebGPU too), four
revisions during the run (bin1 = 3b8a94b, bin3 = CUDA kernel fixes, bin4 =
WebGPU chunked bindings). Raw output of every command: `output.txt`.
Weights: the R2 mirror of FastVideo-FastH3-8-Step-V2 (138 GiB, with the
MiniMax text encoder and tokenizer) and of FastVideo-FastH3-Trim-8-Step
(41 GiB bf16 tree), 149 GB in 468 s + 41 GB in 428 s.

## Correctness

| | result |
|---|---|
| `h3 check --backend cuda`, scalar attention / conv (`H3_CUDA_SCALAR_*=1`) | 90 / 90 at 3e-2 |
| `h3 check --backend cuda`, tensor-core kernels | 90 / 90 at 3e-2 |
| `h3 check --backend wgpu` (NVIDIA Vulkan 570.195.03) | 90 / 90 at 1e-2 |
| FastH3-8-Step-V2, int8, 1344×768×124, seed 7, T2VA | a coherent fox clip with sound (`fox-v2-int8-frames.png`, frames 0 / 40 / 80 / 120) |
| FastH3-Trim-8-Step (bf16 tree → int8 at load), same prompt and seed | a coherent fox clip with sound (`fox-trim-int8-frames.png`) |

The first binary passed every CUDA check on its first execution on a GPU,
before any kernel had run on hardware. The end-to-end clips came out of the
first real-weight run: tokenizer, Qwen3-VL (50 layers streamed from the
mmapped 64 GB checkpoint), token refiner, AdaLN tables, 50 / 42 blocks with
VSA, the ViT / fast decoders and the audio VAE all agree with the released
pipeline closely enough to produce the prompt. No PSNR against PyTorch yet
(the parity fixtures still have to be generated on a machine with torch).

The R2 copy of the Trim **NVFP4** file (`fastvideo_fasth3_trim_8step_nvfp4.safetensors`)
is ComfyUI's single-file format (fused `blocks.N.attn.qkv_proj`, `mlp.fc1/fc2`,
`weight` U8 [out, in/2] + `weight_scale` E4M3 [out, in/16] + `weight_scale_2`
scalar, `adaln_t_table [1025, 16]`), not the FastVideo HF layout the loader
expects: `adaln-tables` refused it with 542 missing tensors. Loading that
format is follow-up work (a key map plus an unswizzled-scale NVFP4 variant).

## Speed (CUDA)

`h3 bench --variant fasth3-v2 --layers 4 --steps 2` at 1344×768×124
(37 966 tokens, 36 k of them video), synthetic bf16 weights, VSA sparsity 0.8
tile 64, bin3:

| | per block | ×50 blocks |
|---|---|---|
| VSA | **470 ms** | 23.5 s / forward |
| dense | 815 ms | 40.8 s / forward |

Profile (per block, bin3): VSA fine 159 ms (34 %), the five GEMMs 265 ms
(56 %, 121 TFLOP/s on the big ones), VSA coarse + top-k 13 ms, rms_norm
5.1 ms × 2.25, head_split 2.45 ms × 4 (was 7.06 ms before the warp-per-head
kernel), swiglu 5.4 ms. The A6000's bf16 tensor peak is ~155 TFLOP/s dense,
so the GEMMs are at ~80 %; the VSA fine stage at ~53 TFLOP/s effective is
the kernel to work on next.

Real clips (bin3, `--weights int8`, `--tables` precomputed):

| | FastH3-V2 (50 blocks) | FastH3-Trim (42 blocks) |
|---|---|---|
| load (mmap + int8 quantization of the bf16 tree, 48 cores) | 280 s | 241 s |
| prompt (Qwen3-VL, 33 rows) | 290 s (cold page cache) | 159 s (warm) |
| denoise, 8 forwards | **201 s** (24.7–25.3 s each) | **162 s** (19.6–20.5 s) |
| decode (ViT decoder / Trim fast decoder, 124 frames) | 44.4 s | 36.4 s |
| audio decode | 0.43 s | 0.43 s |
| AdaLN tables, once per ladder (`h3 adaln-tables`) | 433 s → 174.6 MB | 298 s → 146.7 MB |

The denoise matches the bench (25.1 s/forward real vs 23.5 s extrapolated;
the real run adds the text rows, the heads and the int8 dequant). Load and
prompt dominate a one-off clip: `h3 quantize` once removes the load-time
quantization, and the prompt time is disk-bound (64 GB read at 220 MB/s
cold); a gallery (`h3 encode`) removes it entirely.

## Speed (WebGPU, same card, NVIDIA Vulkan 570.195.03)

`h3 bench --backend wgpu --layers 2 --steps 1`, bin4: **7.6 s per block**
(≈ 380 s per 50-block forward), 16× the CUDA time. f16-weight f32-math GEMM
28672×5376 at 649 ms (18 TFLOP/s); flash (VSA fine) 2.14 s per block;
`H3_WGPU_F16=1` 7.7 s (no gain, as on the fastwan 4090). The WebGPU path is
correct and runs the full geometry (after the chunked-binding fix) but is
not yet a way to make clips at this size.

## Fixes made during the run

1. `fw_rms`: double accumulation → f32 (no measurable change on this card:
   5.1 ms before and after, so the cost is elsewhere; kept for the
   GeForce parts where double is 1/32 rate).
2. `fw_head_split_warp`: warp-per-head RMS + RoPE for 32-aligned head dims;
   7.06 → 2.45 ms per call.
3. WebGPU: the fused QKV activation (3.3 GB) exceeds NVIDIA's 2 GiB storage
   binding; `linear` and `head_split` now bind aligned row sub-ranges and the
   FFN chunk is capped by the binding limit. A conformance test forces 64 KiB
   bindings to keep that path checked on lavapipe.

## Cost

2 h 55 min at $0.54/h plus ~190 GB of inbound transfer at $0.026/GB:
about $6.9 of credit. About two of those hours were idle after the work
finished (the controlling session restarted and lost its destroy timer):
arm the timer as a vast-side action, not a local process.
