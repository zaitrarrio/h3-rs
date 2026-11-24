# 2026-11-24: validation round, W4A4 NVFP4, and why pods could not terminate themselves (RunPod Server Edition)

Pod `7jiirx2ukhp1i0`: RTX PRO 6000 Blackwell Server Edition (sm_120), 600 W,
2430 MHz, driver 595.112.07, EU-RO-1, EU weights volume; ~55 min. It was
created with `scripts/runpod/pod.py` after one out-of-stock retry.

Binaries, in order of use:
- 895966a (commands 1–4);
- d3cf128, then a6bb007 (W4A4, `parity-dit`; commands 5–6);
- the pread loader (command 7);
- the `regen2k --out` fix (command 8).

Commands 5–8 were rewritten while the pod ran.

## Checks

| binary / backend | result |
|---|---|
| 895966a CUDA | **188 / 188** (fast VAE tiny case 1.3e-2) |
| d3cf128 CUDA (W4A4 cases) | **193 / 193**; `nvfp4 w4a4` 3.3–4.5e-3 of the oracle |
| 895966a WebGPU (Vulkan, NVIDIA 595.112.07, not llvmpipe) | **188 / 188**; fast VAE tiny case 8.4e-4 |

## Resident and streamed: byte-identical

Each run is a 480p draft from `--embeds` with the fast recipe.

| pair | result |
|---|---|
| `H3_CUDA_LT_NTH=0`, resident vs `--stream-blocks` | **byte-identical** (31.6 vs 10.6 GB peak, 7.52 vs 7.67 s denoise) |
| `H3_CUDA_LT_NTH=0`, resident vs resident | byte-identical |
| unpinned, resident vs resident | byte-identical |

`2026-11-24-runpod-pika-plug4` found its unpinned resident and streamed
drafts 13.6 dB apart. The streamed path itself computes the same bytes. The
difference came from the unpinned candidate timing: each GEMM shape times
cuBLASLt's candidates on its first call, and a streaming run times them while
weights are uploading, so it can pick other candidates. Pin the candidate to
compare runs.

## Fast VAE decoder

**2K profile, `vae-roundtrip --skip-base`, warm.**
- bf16: 11.85 / 12.08 s.
- `--decoder-weights mxfp8`: 10.28 / 10.08 s.

The GEMMs are not most of the time. The per-op table was missing (the
round-trip tool did not print it; fixed since).

**WebGPU (Vulkan) at 768p, 68 frames (17 latents).**
- Released decoder: 52.9 s.
- Fast decoder: **15.0 s**, 33.6 dB from the released decoder's output.

The 2K guard was not exercised: encoding at 2K on WebGPU would take too long
for this round.

**`regen2k` defaults to the tree's `vae_fast/`.**
- The first try failed: the `--released-vae` change had moved `--out`'s
  attribute, which made it positional. Fixed, with a parse test.
- The rerun: total **406.8 s** against 488.4 s with the released decoder
  (`2026-11-12-runpod-pro6000-variants`).
  - prepare 107 s;
  - 3 × 97 s steps;
  - decode **9.4 s** against 30.7 s.
- Peak GPU **92.5 GB** of 97.9: tight.
- The strip (`regen-strip.jpg`) is clean.

## `serve` on a GPU (first timing of `--upscale`)

The draft variant from the gallery, fast recipe, `--upscale
realesr-animevideov3.safetensors`. Each line: three jobs on one engine,
fast VAE first, then the released one.

| job | decode (fast / released VAE) | upscale (fast / released run) | job total |
|---|---|---|---|
| 480p, ×2 → 1664×960 | 1.05 / 2.65 s | 5.41 / 5.08 s | 14.5 / 15.7 s |
| 480p, ×4 → 3328×1920 | 1.01 / 2.71 s | 5.24 / 4.84 s | 14.0 / 15.4 s |
| 768p, auto (×2) → 2688×1536 | 2.60 / 6.30 s | 19.3 / 17.5 s | 50.7 / 52.5 s |

- **Readiness:** the engine is ready 92 s after start with the fast decoder
  (it loads in 4.5 s), 78 s without it.
- **Upscale cost:** includes the JPEG encoding of every frame, which
  `generate --upscale` does not do (4.4 s / 18.5 s there).

## NVFP4 W4A4 on the FP4 tensor cores (`H3_NVFP4_A4=1`)

FastH3-Trim, ComfyUI single file, staged. 1344×768×124, `--embeds`, seed 7.

| | load | denoise (8 forwards) | decode | peak GPU |
|---|---|---|---|---|
| W4A16 (weights expanded per GEMM, as before) | 17.3 s | 43.3 s (5.36–5.45 s) | 8.4 s | 25.8 GB |
| **W4A4** | 25.7 s | **24.7 s** (3.08–3.11 s) | 8.6 s | 27.5 GB |

- **Speed:** 1.75× per forward.
- **Load:** the 8 s more of load is the host repack into cuBLASLt's layout.
- **Quality:** W4A4 against W4A16 is 14.6 dB Y. Both are clean and show the
  same scene and motion, a different sample (`trim-cmp.jpg`, W4A16 above).
- **Diagnostics rule:** under `H3_PROFILE=1` the clip is byte-identical.
- **Profile gap:** the profile did not count the FP4 GEMMs as GEMMs. Fixed
  since.

## Loading from the volume: mmap vs parallel pread (inconclusive)

The stored MXFP8 V2 (`transformer.safetensors`, 25.6 GB, one file), read
from the volume, not staged. In order:

| run | load |
|---|---|
| mmap (first read) | 89.1 s |
| pread | 26.7 s |
| mmap again | 30.4 s |
| pread again | 28.7 s |

`/proc/sys/vm/drop_caches` is read-only in the container, so only the first
run read cold. The pread number after it, and the warm mmap at 30.4 s, show
the cache rather than the reader. A clean comparison needs a different tree
for each reader; it is not done. The outputs are byte-identical.

## Why pods could not terminate themselves

Every 2026-11 pod's `terminate` failed: REST is unresolvable from the pods,
and GraphQL answered `{"error":{}}`. Commands 10–11 found why:

| key (from inside the pod) | `query { myself { id } }` |
|---|---|
| our account key (bearer or `?api_key=`) | **401** `{"error":{}}` |
| the pod-scoped `RUNPOD_API_KEY` RunPod injects | 200, `UNAUTHORIZED` for `myself` |
| no key | 200, `myself: null` |

`podTerminate` with the **pod-scoped** key returned `{"data":{"podTerminate":null}}`,
and the pod was gone.
- **Before:** the scripts passed the account key as `RUNPOD_API_KEY`, which
  replaced the pod-scoped one. RunPod rejects account keys from inside pods.
- **Now:** pods get no account key at all (`pod.py`, `rp.py`), and the boot's
  `terminate` uses the injected key.

## Full-width parity: not run

FastVideo decides its platform through NVML, so on a GPU host it initialises
NCCL even with a CPU-only torch and `CUDA_VISIBLE_DEVICES=`. Both tries
stopped at "Distributed package doesn't have NCCL built in". The next round
runs the reference with CUDA torch on the GPU. The tooling itself was
checked locally on the tiny model (`parity-dit --tiny`, 4.6e-7 against
FastVideo a7eb340).
