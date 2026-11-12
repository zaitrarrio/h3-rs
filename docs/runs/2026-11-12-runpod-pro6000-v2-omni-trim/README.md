# 2026-11-12: RunPod RTX PRO 6000 Server Edition from the filled volume: V2 best config, Omni Ref2VA, Trim NVFP4

After the volume fill (`../2026-11-12-runpod-volume-fill`): pod
`out33km54ble4e` (secure cloud, EU-RO-1, $2.69/h this time): NVIDIA RTX
PRO 6000 Blackwell Server Edition (sm_120, **600 W**), driver 595.91.07,
image `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, the network volume
`fv-weights-h3-ltx-500` (750 GB) at `/workspace`. Binary: release of
4c12af4 (`--features cuda`), the same `h3-bin` as the Pika / Regen round.
19:39–19:59 UTC. The Server Edition had no stock in EU-RO-1 from 16:32 to
19:39; the fill ran on a 4090 meanwhile and this pod started the moment
one appeared (`scripts/runpod/` hunt loop).

## The volume after this round

`output-1-verify.txt`: every fetched file present at its size
(`missing: 0`); `output-5-volume-report.txt`: every tree `.complete`.
The fill's second step, whose log was lost, had finished all three of its
products (timestamps from `ls -la`):

| path | size | written |
|---|---|---|
| `h3-rs/tables/v2.tables` (FastH3-V2's 8 DMD knots) | 175 MB | 17:58 |
| `weights/h3-8step-mxfp8/transformer.safetensors` (`h3 quantize --weights mxfp8 --skip-adaln` of V2; `.complete` / `.fv-revision` `3da2ddfe…-mxfp8` / `.fv-source` `h3-rs`) | 25.6 GB | 18:01 |
| `h3-rs/tables/omni.tables` (Omni's 17 PDD knots) | 165 MB | 18:07 |

**689 GB** of 750 in use; `h3-rs/` (tables 1.4 GB, assets 195 MB, the
`h3/` symlink model directory) beside the weight trees.

## Clips (1344×768×124, seed 7, the fox prompt; all from the volume)

| variant, config | load | prompt rows (Qwen3-VL, host) | denoise | decode | status |
|---|---|---|---|---|---|
| **FastH3-V2, the fastest measured recipe**: stored MXFP8 (`--transformer …/h3-8step-mxfp8/transformer.safetensors --weights mxfp8`), `--decoder-weights mxfp8`, `--vsa-qk8`, `--tables v2.tables` | 129.0 s | 33 in 189.8 s | **37.94 s** = 4.88, then 4.70–4.74 s × 7 | **5.79 s** | **44.27 s after load**; clip `v2-best-frames.png`, same scene as the Vast runs |
| FastH3-Omni Ref2VA (bf16, PDD, VSA 0.9 tile 128), one image reference, `--tables omni.tables` | 243.2 s | 7209 in 278.2 s | 114.39 s = 14.12, then 14.22–14.37 s × 7 | 8.14 s | clip `omni-ref2va-frames.png`: the fox of the reference frame, coherent |
| FastH3-Trim, ComfyUI NVFP4 single file | | | | | first attempt failed on my glob (`output-4-trim-wrong-glob.txt`: the file is `diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors`, one level down); re-run below |

Against the Vast Server Edition numbers for the same recipe
(`../2026-11-08-vast-pro6000-server-variants`, quantized at load:
denoise 37.18 s, decode 5.83 s, 43.56 s after load) this pod is 2 %
slower across the board: the band between two Server Edition boxes
(cuBLASLt candidate timing, clocks). Omni: 112.4 s = 8 × 14.07 s on Vast,
114.4 s here. The recipe's numbers hold from the stored file on a network
volume; what the volume changes is the time before the first step:

- V2 load 129 s against 83.6 s on Vast's NVMe, for 25.6 GB of stored
  MXFP8 plus the VAEs: ~0.25 GB/s effective, the slowest read of the
  session (the fill wrote at 300 MB/s).
- The prompt step is now dominated by the Qwen3-VL text encoder streaming
  from the volume: 190 s for 33 rows (96 s on Vast), 278 s for 7209 rows
  (122 s). Pre-encoded prompts (`h3 encode`) remove it entirely.
- Omni load 243 s (90.7 s on Vast): 66 GB of bf16 `transformer_ref`.

## Trim NVFP4 (follow-up pod)

(filled in below once the follow-up pod has run)
