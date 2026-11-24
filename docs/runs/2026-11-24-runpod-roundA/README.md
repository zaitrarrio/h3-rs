# 2026-11-24: round A — box_rows16, FP4 candidate tuning, 8-block parity, regen2k memory, 4-bit drafts under 9 GB (RunPod Server Edition)

Pod `cvmauwewna7u7g`: RTX PRO 6000 Blackwell Server Edition (sm_120), 600 W,
2430 MHz, driver 595.112.07, EU-RO-1, EU weights volume. Binary ec0bf90. It
terminated itself with the pod-scoped key (`out6.txt`: `self-terminate`).
Commands `cmd1`–`cmd5.sh` (helpers in `common.sh`), outputs `out1`–`out5.txt`,
per-run logs and GPU memory samples (`*.mem`, `nvidia-smi` every 0.5 s, MiB).

## Checks

| backend | result |
|---|---|
| CUDA | **197 / 197** (box_rows width-16 cases ~2e-3, W4A4 3.3–4.5e-3) |
| WebGPU (Vulkan, NVIDIA 595.112.07) | **195 / 197**: `linear nvfp4 w4a4 129x1536x130` 3.19e-2, `… 1100x1536x200 GeluTanh` 1.20e-2 |

**The two WebGPU failures were WGSL division.** `nv_act.wgsl` divided with
`/`, which WGSL allows to be 2.5 ulp off, and NVIDIA's driver divides through
a reciprocal. One ulp in `2688 / amax` or `encode / scale` flips an E2M1 code
at a rounding boundary.

- **Reproduced on lavapipe:** replacing `a / b` with `a * (1 / b)` gives the
  same two numbers exactly, plus a third case at 1.3e-2.
- **Fix:** the kernel now divides with `div_rn`, a long division of the
  significands, correctly rounded like the oracle's `/` and CUDA's
  `__fdiv_rn`. It also builds E4M3 powers of two with a bitcast instead of
  `exp2`.
- **Test:** bit-exact against float32 on 199,513 operand pairs, ties
  included.
- **After the fix:** the W4A4 cases are at 3e-7 – 1.4e-6 on lavapipe,
  197 / 197.
- **Still to do:** re-check on NVIDIA in the next round.

## Fast VAE decoder: 16-byte `box_rows`

2K (`vae-roundtrip --skip-base`, fox at 2560×1440, `H3_PROFILE=1`):

| | before (validate round) | now |
|---|---|---|
| `box_rows` | 1317 ms | **473 ms** (8922 calls, 0.053 ms each) |
| fast decode | 11.85 s | **11.38 s** |

Ops take 6.76 s of the 11.38 s. The GEMMs are 4.36 s at 395 TFLOP/s, and
attention takes 0.86 s.

That leaves ~4.6 s outside the kernels (host side). Round B's target.

## FP4 GEMMs time their cuBLASLt candidates

FastH3-Trim, ComfyUI file, `H3_NVFP4_A4=1`, 1344×768×124, `--embeds`:

| | load | per forward | denoise | peak |
|---|---|---|---|---|
| tuned (new) | 20.4 s | 3.14–3.16 s | 25.37 s | 26.0 GB |
| candidate 0 (`H3_CUDA_LT_NTH=0`, the old behaviour) | 18.5 s | 3.14–3.17 s | 25.27 s | 26.8 GB |

- **Speed:** the timing found nothing faster than the heuristic's first
  candidate on this card.
- **Load:** the parallel repack into cuBLASLt's layout took W4A4 load from
  25.7 s (validate round) to 18.5–20.4 s.
- **Rounding:** the tuned run picked another candidate for some shapes. The
  clips are the same scene at 14.8 dB Y: candidate rounding, as with MXFP8.
- **Kept:** the tuning stays, since it can pay on other cards.

## Full-width parity: 2 and 8 blocks of h3-base

The reference was FastVideo a7eb340 with CUDA torch, f32, on the pod's GPU
(`full_parity.py`, then `h3 parity-dit`). It covered the embedders, the text
refiner, the first N blocks and the head. 48 video rows, 10 audio rows.

| blocks | our CPU (video / audio) | our CUDA (video / audio) |
|---|---|---|
| 2 | 7.0e-7 / 1.2e-6 | 3.5e-3 / 4.3e-3 |
| 8 | 8.2e-7 / 1.1e-6 | 5.4e-3 / 5.0e-3 |

The CPU matches the reference to f32 rounding. CUDA sits at its bf16
activation level and grows little with depth.

## regen2k frees the denoise state before decoding

| | peak GPU | total | decode |
|---|---|---|---|
| validate round | 92.5 GB | 406.8 s | 9.4 s |
| now | **80.5 GB** | 416.0 s (prepare 110 s, 3 × 99 s) | 9.1 s |

## H3 drafts under ~9 GB: 4-bit weights, streamed blocks, FFN chunks

Settings: the draft variant (LongLive-Plug 4-step), `--embeds`, `--tables`,
seed 0, 124 frames. Peak is the card's used memory.

| backend, tier | weights | streaming | peak GPU | per step | denoise | decode |
|---|---|---|---|---|---|---|
| CUDA 480p | bf16 | – | 48.7 GB | 2.82 s | 11.3 s | 3.6 s |
| CUDA 480p | int4 | – | 23.9 GB | 2.89 s | 11.6 s | 3.7 s |
| CUDA 480p | int4 | `--stream-blocks` | 10.9 GB | 10.4–11.6 s | 44.1 s | 3.7 s |
| CUDA 480p | int4 | + `--ffn-chunk 4096` | 10.1 GB | 10.7–11.7 s | 44.8 s | 3.7 s |
| CUDA 288p | int4 | + `--ffn-chunk 4096` | **8.9 GB** | 10.5–11.8 s | 44.2 s | 1.8 s |
| WebGPU 480p | int4 | – | 47.3 GB | 54.3 s | 217 s | 51.2 s |
| WebGPU 480p | int4 | + `--ffn-chunk 4096` | 54.5 GB | 55.7 s | 223 s | 51.7 s |
| WebGPU 288p | int4 | + `--ffn-chunk 4096` | 16.2 GB | 15.9–17.4 s | 65.9 s | 25.7 s |

**Quality (`int4-cmp.jpg`).** Rows, top to bottom: CUDA bf16, CUDA int4,
WebGPU int4. All three are clean fox clips. int4 against bf16 is 9.3 dB Y,
which is the usual 4-bit "another sample", not breakage.

### Two problems, both fixed since (not yet re-measured)

**CUDA int4 streaming was 4× slower than resident.** The step took the same
~11 s at 288p as at 480p, so uploads set it, not compute. Each prefetch:
- `Backend::weight_q4` packed the block's codes on the host;
- it uploaded them from pageable memory;
- streamed int4 weights were never page-locked (`HostWeight::pin` skipped
  Q4).

Fix: streamed blocks keep int4 as `Q4Packed`, the device form (two codes per
byte plus the interleaved scale/zero table). It is page-locked, so a
prefetch is a DMA (`Backend::weight_q4_packed`), and the host copy is half
the size.

**WebGPU held a whole step of buffers.** It submitted every 64 dispatches but
only polled the device at a download. The host ran a full step ahead of the
GPU, and wgpu frees a dropped buffer only after its submission is done and
the device is polled. That kept every per-op temporary and every streamed
block alive until the step ended:
- a 39 GB plateau resident;
- swings between 11 and 54 GB with streaming (`w480-int4-stream-ffn.mem`).

Fix: after each submission the backend waits for the oldest once more than
`H3_WGPU_INFLIGHT` (4) are outstanding.

Both fixes pass the checker on lavapipe (197 / 197, including the int4
`streamed == resident` cases at 0) and the CPU (197 / 197).
