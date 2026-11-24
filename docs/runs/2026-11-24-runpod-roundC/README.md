# 2026-11-24: round C — the round-A fixes on a GPU, fast VAE host spans (RunPod Server Edition)

Pod `ftojuk1fhcck8r`: RTX PRO 6000 Blackwell Server Edition (sm_120), 600 W,
2430 MHz, driver 595.112.07, EU-RO-1, staged weights. Binary 11074b7. It
terminated itself. Commands `cmd1`–`cmd3.sh` (helpers in `common.sh`), outputs
`out1`–`out3.txt`, per-run logs and memory samples (`*.mem`, MiB every 0.5 s).

## Checks

| backend | result |
|---|---|
| CUDA | **197 / 197** (W4A4 3.3–4.5e-3; int4 `streamed == resident` 0) |
| WebGPU (Vulkan, NVIDIA 595.112.07) | **197 / 197**: the two W4A4 cases that failed in round A are at 5.3e-7 / 1.4e-6 with `div_rn` |

## Streamed int4 on CUDA: packed, page-locked host weights

Settings: the draft variant, `--embeds`, `--tables`, seed 7, 124 frames.

| run | peak GPU | per step | denoise | round A |
|---|---|---|---|---|
| 480p int4 resident (`H3_CUDA_LT_NTH=0`) | 23.9 GB | 2.87 s | 11.5 s | 2.9 s |
| 480p int4 `--stream-blocks` (`H3_CUDA_LT_NTH=0`) | 11.0 GB | **2.89 s** | 11.6 s | 10.4–11.6 s |
| 480p int4 streamed + `--ffn-chunk 4096` | 10.1 GB | 3.02 s | 12.1 s | 10.7–11.7 s |
| **288p int4 streamed + `--ffn-chunk 4096`** | **8.9 GB** | **0.94 s** | 3.8 s (5.8 s after load) | 10.5–11.8 s |

- **Same speed:** streaming int4 now runs as fast as resident.
- **Same bytes:** the streamed clip is byte-identical to the resident one.

## WebGPU memory with the in-flight bound (`H3_WGPU_INFLIGHT` 4)

| run | peak GPU | during the denoise | per step | round A peak / plateau |
|---|---|---|---|---|
| 480p int4 resident | 47.3 GB | 27.4 GB | 54.2 s | 47.3 / 39.4 GB |
| 480p int4 streamed + ffn chunks | **16.5 GB** | 14.1–14.6 GB | 55.1 s | **54.5 GB**, swinging 11–54 GB |
| 288p int4 streamed + ffn chunks | 16.2 GB | 13.4–13.9 GB | 15.5 s | 16.2 / ~12–14 GB |

- **Streamed 480p:** the bound took the peak from 54.5 to 16.5 GB at the
  same speed. The streamed clip is byte-identical to the resident one (PSNR
  inf).
- **Resident 480p:** the plateau went from 39.4 to 27.4 GB.
- **Still open:** a transient at the start of the denoise is unchanged. It
  falls right after load (sample 184 of 724 at 480p resident, 164 at 288p)
  and gives the 47.3 / 16.2 GB peaks. Its cause is not found yet. At 288p
  the plateau is ~14 GB, so WebGPU does not fit 9 GB where CUDA does.

## Fast VAE decoder, 2K: where the wall time goes

`vae-roundtrip --skip-base`, fox at 2560×1440, 124 frames.

| | decode |
|---|---|
| validate round (bf16, unprofiled) | 11.85 / 12.08 s |
| **now** (16-byte `box_rows`, one-pass crop + denormalise) | **10.70 / 10.87 s** |

Host spans under `H3_PROFILE=1`, 10.06 s in all:
- forward: 6.86 s (synced: the kernels);
- heads and downloads: 1.58 s (bf16 pixels to the host and to f32);
- waiting on the conversion thread: 0.02 s;
- the caller's callback: 1.41 s, which is the round-trip tool cloning every
  chunk, not the decoder.

What is left on the host is the download of 2.7 GB of bf16 pixels. The rest
is the transformer's kernels (GEMMs 4.4 s).

At 480p (`generate --fast-vae`, a draft) the decode is 1.08 s:
- forward 0.86 s;
- heads and downloads 0.14 s;
- callback 0.04 s.
