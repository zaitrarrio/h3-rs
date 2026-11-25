# 2026-11-25: flash3h (f16 attention) and the staged-upload flush (Vast RTX PRO 6000 Workstation Edition)

Vast contract 62431393: RTX PRO 6000 Blackwell Workstation Edition, 600 W,
3090 MHz max, driver 615.71.09, Vulkan with `shaderFloat16` (Illinois);
~35 min. It destroyed itself from its last command. Binary 659df76 (PR
"WebGPU: flash3h (f16 attention, opt-in); submit staged uploads during
load"), plus that build's `conformance` test binary for the probe.

- **Commands:** `cmd2`–`cmd5.sh`; outputs `out2`–`out5.txt`, with per-run
  logs and memory samples.
- **Profiles:** `h3 bench` used synthetic weights at the real geometry: 50
  blocks, the h3-plug4 draft config, 124 frames, dense attention, two
  forwards each.

## Checks (NVIDIA Vulkan, not llvmpipe)

| mode | result |
|---|---|
| WebGPU f32 | 197 / 197 |
| WebGPU `H3_WGPU_F16=attn` (flash3h) | 197 / 197 |
| WebGPU `H3_WGPU_F16=1` (flash3h + linear3h) | 152 / 152 ops |
| CUDA | 152 / 152 ops |

## The staged-upload flush: confirmed

`wgpu_upload_staging_probe` uploads 6 GB (24 × 256 MB) with no compute, then
syncs. It reads nvidia-smi's used memory before and after. Each setting ran
twice, with identical results.

| `H3_WGPU_STAGE_FLUSH_MB` | base | after the uploads | after the sync |
|---|---|---|---|
| 0 (the old behaviour) | 262 MiB | **12 550 MiB** | 12 550 MiB |
| 256 (the default now) | 263 MiB | **6 656 MiB** | 6 399 MiB |

**Without the flush, every upload is held twice.** The bytes and their
staging copy both stay alive. Even the sync frees nothing, because with no
compute recorded nothing is ever submitted. That is the load of a `generate
--tables`, which explains its 40.7 GB at the end of load and its 47.3 GB
peak at the first step (`2026-11-24-runpod-roundC`).

**With the flush, the overhead is at most one 256 MB batch.**

A `generate` with real weights was not re-run here (no weights on the box).

## flash3h: no gain on NVIDIA

| run | f32 (flash3) | `H3_WGPU_F16=attn` (flash3h) |
|---|---|---|
| 480p int4, per forward | 26.4 / 26.7 s | 26.8 / 27.0 s |
| 480p attention, per call | 205 ms | 211 ms (+3 %) |
| 288p int4, per forward | 7.85 / 7.82 s | 7.83 / 7.89 s |
| 288p attention, per call | 29.2 ms | 30.1 ms (+3 %) |

**Packed f16 math is not faster than f32 here** through NVIDIA's Vulkan
driver. This matches fastwan-rs, where f16 paid in Chrome (attention 1.62×)
and on AMD (1.33×) but only −7 % per native step on an RTX 4090.

flash3h stays opt-in, for AMD / Apple / Intel, where it is untested here.

## `H3_WGPU_F16=1` was a 1.7× regression (fixed)

| run | per forward | FFN up | QKV |
|---|---|---|---|
| 480p native, f32 (linear4_f16 + flash3) | **24.5 / 24.7 s** | 113 ms | 83 ms |
| 480p native, `H3_WGPU_F16=1` (linear3h + flash3h) | 41.5 / 42.1 s | 249 ms | 186 ms |

**Cause:** the f16-math GEMM is `linear3h`, a 64×64 kernel, and it displaced
`linear4`.

**Fix:** `linear3h` now replaces only `linear3`, under 128 rows or with
`H3_WGPU_LINEAR=3`. The f16 conformance test selects `linear3` so it keeps
covering `linear3h`.

**`linear4h` stays unported.** f16 gained ~6 % over linear3 per GEMM, and
fastwan-rs measured linear4h at 1.11–1.24× in Chrome only.

This box ran the f32 kernels ~10 % faster than the
`2026-11-24-vast-pro6000ws-linear4-flash3` box (26.4 vs 29.3 s at 480p
int4), at the same card class and power limit, so compare within one box
only.
