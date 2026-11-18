# 2026-11-18: the h3 container on RunPod, as a service (pod) and as a CLI (serverless)

Image `ghcr.io/zaitrarrio/h3-rs:sha-3030e53` (public), built and
smoke-tested by `.github/workflows/container.yml` (run 5, green): `h3`
with the CUDA + CPU backends on debian:bookworm-slim plus libcublas,
libcublasLt and libnvrtc 12.8, 675 MB compressed. Weights from the
`fv-weights-h3-ltx-500` network volume (EU-RO-1): `h3-base`, the converted
LongLive-Plug adapter and its tables (`h3-rs/plug4/`). Pod first, then the
serverless endpoint, never both on the volume at once. Driven by
`scripts/runpod/container.py` (RunPod GraphQL; the REST host is not
reachable from the agent sandbox).

## Service: `h3 serve` on a pod

Pod `hbgeuoaizdx04x`, RTX PRO 6000 Blackwell Server Edition (sm_120),
start command

    serve --addr 0.0.0.0:7860 --backend cuda --variant draft
          --model-dir /workspace/weights/h3-base --variant-dir /workspace/h3-rs/plug4
          --tables /workspace/h3-rs/plug4/plug4.tables
          --weights mxfp8 --decoder-weights mxfp8 --pv8 --allow-origin https://h3-video.gialabs.sh

(RunPod passes the start command as the image's CMD, after the `h3`
entrypoint). Reached through RunPod's proxy at
`https://<pod>-7860.proxy.runpod.net/`:

| | |
|---|---|
| engine ready (pull + load from the volume) | 255 s after create (load 234 s) |
| job 1, 832×480×124, seed 7 | prompt 208 s (Qwen3-VL streamed from the volume, first use), denoise 4 × 1.92–2.06 s, decode 2.7 s |
| job 2, 480×272×124 (canvas 480×288, cropped), seed 3 | prompt 87 s (page cache warm), denoise 4 × 0.57–0.62 s, decode 1.3 s |
| job 3 from the hosted page (`h3-video.gialabs.sh/demo/` pointed at the pod), 832×480, seed 11 | prompt 73 s, denoise 7.8 s, decode 2.6 s, no page errors (`hosted-site-driving-pod.jpg`) |
| `GET /` | 200, the studio |
| `POST /api/generate` with `Origin: https://evil.example` | 403 |
| CORS preflight from `https://h3-video.gialabs.sh` | 204 with `Access-Control-Allow-Origin` |

`service-480p.jpg` (job 1, frames 0 / 60 / 120): a photographic fox
walking through snow at dawn. `service-272p.jpg` (job 2): ghosting, the
known 272p limit (below H3's trained sizes), not the container. The pod
ran ~20 min and was terminated (a first pod was terminated by mistake
within a minute of its start and recreated); the service has no authentication, so a
pod's proxy URL is for short sessions only.

## CLI: `h3 worker` on a serverless endpoint

Endpoint `m2mafv82p2iq2f` (template `dockerArgs: worker`, 0..1 workers,
the volume at `/runpod-volume`, execution timeout 40 min). GPU pools
`BLACKWELL_96` alone stayed `throttled` (no capacity) for 15 min; with
`ADA_24` added and the endpoint scaled 0 → 1 a worker started on an
**RTX 4090 (sm_89)**. Jobs are `h3` argument lists; outputs go to
presigned R2 PUT URLs.

| job | result |
|---|---|
| `check --backend cuda --report check.json` | **180 / 180** (`serverless-check-4090.json`), 11.0 s; queue delay 899 s (capacity + image pull) |
| `generate --variant draft … --tier 480p --weights mxfp8 --stream-blocks --pv8 --seed 7 --out clip.y4m` | exit 0 in 626 s: load 296.7 s, prompt 283.1 s (both from the volume), then 27.7 s: denoise 4 × 5.15–5.20 s, decode 6.8 s, audio 0.17 s; `clip.y4m` (148.6 MB) and `clip.wav` uploaded |

The 4090 has 24 GB, so the generate job streams the DiT blocks
(`--stream-blocks`, MXFP8 host copy ~26 GB, computed in bf16 on sm_89) and
uses the bf16 decoder. `serverless-480p-4090.jpg` (frames 0 / 60 / 120):
the same scene and composition as the service's seed-7 clip, 13.8 dB
between them (MXFP8 on Blackwell's FP8 tensor cores with the MXFP8
decoder against the emulated, streamed path with the bf16 decoder: the
8-bit band). The endpoint was deleted afterwards; its template can only be
deleted once RunPod has billed the last worker (~90 min).

## What this shows

- The image runs unchanged as a service (`serve`), a CLI job (`check`,
  `generate`) and a RunPod serverless worker (`worker`, no Python), on
  sm_120 and sm_89.
- Cold starts are dominated by reading the weights and the text encoder
  from the network volume (5 + 5 min on serverless). A real serverless
  deployment should bake the prompt embeddings (`h3 encode` → `--embeds` /
  `serve --gallery`), keep a stored MXFP8 DiT and the tables, and stage
  them onto the worker's disk or use FlashBoot; a service keeps them
  resident after the first load.
