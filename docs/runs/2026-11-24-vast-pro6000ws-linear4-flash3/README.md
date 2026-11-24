# 2026-11-24: linear4 + flash3 against linear3 + flash (Vast RTX PRO 6000 Workstation Edition)

Vast contract 62408578: RTX PRO 6000 Blackwell Workstation Edition, 600 W,
3090 MHz max, driver 615.78.08, Vulkan (Germany); ~40 min. It destroyed
itself from its last command. Binary eec1c0e (PR "WebGPU: port fastwan-rs's
linear4 and flash3 kernels").

`h3 bench` used synthetic weights at the real geometry: 50 blocks, the
h3-plug4 draft config, 124 frames, dense attention, two forwards each. New
and old kernels ran on the same box. The old kernels are selected with
`H3_WGPU_LINEAR=3 H3_WGPU_FLASH=1`. Commands are `cmd2`–`cmd4.sh`, outputs
`out2`–`out4.txt`, with per-run logs and memory samples.

## Checks (NVIDIA Vulkan, not llvmpipe)

| kernels | result |
|---|---|
| WebGPU, linear4 + flash3 | **197 / 197** (ops and tiny models) |
| WebGPU, linear3 + flash | 152 / 152 ops |
| CUDA | 152 / 152 ops |

## Per forward

| run | old (linear3 + flash) | **new (linear4 + flash3)** | speedup |
|---|---|---|---|
| 480p int4 | 53.1 / 54.3 s | **29.3 / 30.1 s** | 1.81× |
| 480p int4, unprofiled | — | 29.4 / 30.2 s | |
| 288p int4 | 15.3 / 15.5 s | **8.5 / 8.6 s** | 1.80× |
| 480p native (f16 weights) | 58.1 / 60.1 s on the profile box | **28.2 / 29.0 s** | ~2.06× |

- **The native row's old column** comes from another box of the same card
  class (`2026-11-24-vast-pro6000ws-wgpu-profile`, driver 615.71.09), not
  this one.
- **The profiler costs nothing:** the unprofiled 480p run times the same as
  the profiled one.
- **Peak memory is unchanged:** 21.6 GB (480p int4), 18.2 GB (288p int4),
  53.5 GB (480p native).
- **CUDA on the profile box:** 3.8 s per 480p int4 forward, so WebGPU is now
  ~7.8× behind it, where it was ~14× before.

## GPU time per call (timestamp queries)

| kernel | 480p old | 480p new | 288p old | 288p new |
|---|---|---|---|---|
| attention (`flash` → `flash3`) | 403 ms | **233 ms** (1.73×) | 59.6 ms | **32.9 ms** (1.81×) |
| FFN up 28672×5376 | 265 ms | **140 ms** (1.89×) | 97.6 ms | **52.3 ms** (1.87×) |
| QKV 21504×5376 | 199 ms | **104 ms** (1.91×) | 73.4 ms | **39.6 ms** (1.85×) |
| FFN down 5376×14336 | 135 ms | **74 ms** (1.82×) | 50.0 ms | **29.3 ms** (1.71×) |
| out 5376×7168 | 66 ms | **37 ms** (1.81×) | 24.9 ms | **14.7 ms** (1.69×) |
| elementwise + head_split + norm | 0.52 s | 0.52 s | 0.19 s | 0.19 s |

- **Throughput:** FFN up at 480p is 2·15100·28672·5376 = 4.65 TFLOP per call.
  That is 17.6 TFLOP/s with linear3 and **33.2 TFLOP/s** with linear4,
  against CUDA's ~260 TFLOP/s on the int4 GEMMs. Attention stays ~40 % of
  the step.
- **What's left:** the remaining gap to CUDA is tensor cores, which WGSL
  cannot reach.
- **f16 math:** not timed. `linear4h` is not ported, and `H3_WGPU_F16=1`
  still runs `linear3h` on the native-weight GEMMs.
