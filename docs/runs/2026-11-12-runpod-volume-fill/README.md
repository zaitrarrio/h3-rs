# 2026-11-12: filling the grown RunPod volume (750 GB) with V2, Omni and Trim NVFP4

The EU network volume `fv-weights-h3-ltx-500` (RunPod, EU-RO-1, shared with
fastvideo-rs) was grown from 500 to 750 GB (`PATCH /networkvolumes/bz0ofkfx45
{"size": 750}`, HTTP 200; a volume can only grow) so the trees the previous
round could not fit have a home. The Server Edition had no stock in EU-RO-1
for two hours (every `POST /pods` answered "no instances currently
available"), so the fill ran on the cheapest card that appeared: pod
`v0x3uauu2vkbny`, **NVIDIA GeForce RTX 4090** (450 W, driver 595.91.07),
$0.89/h, `maxDownloadSpeedMbps` 5188, 17:39–~19:30 UTC. Nothing here is a
speed claim about the model; the fetch rate is a property of the volume.

## Fetch (`output-1-fetch.txt`)

76 files, 214.1 GiB, 12 parallel `curl`s from R2 (`fv-weights`) and
Hugging Face, size-checked and idempotent (`scripts/runpod/fetch_plan.py`):
**752 s, ~300 MB/s onto the network volume**, `missing: 0`. Marked with
`.complete` / `.fv-revision` / `.fv-source`:

| tree | source | size |
|---|---|---|
| `weights/h3-8step/` (FastH3-V2: transformer 14 shards, text encoder, VAEs, tokenizers) | R2 `weights/h3-8step/3da2ddfe…` | 138 GB |
| `weights/fasth3-omni-8step-v1/` (the full HF repo) | HF, `main` as of 2026-11-12 | 66 GB |
| `weights/fasth3-trim-8step-nvfp4/` (ComfyUI single file) | R2 `weights/fasth3-trim-8step-nvfp4/f66c13dc…` | 12 GB |

Volume after the fetch: **665 GB** of 750 (`du -sh /workspace`).

## Tables and the stored MXFP8 V2 (`cmd-2-tables-quantize.sh`)

The second command computed `h3-rs/tables/v2.tables` (`adaln-tables
--variant fasth3-v2`), `h3 quantize --weights mxfp8 --skip-adaln` of V2 to
`weights/h3-8step-mxfp8/transformer.safetensors` (+ markers) and
`h3-rs/tables/omni.tables`, then terminated the pod. **Its output was
lost**: the script's final `terminate` deleted the pod before the boot
loop uploaded the command's log (`boot.log` ends at `== cmd 2 17:52:32`).
41caca6 makes `terminate` upload the in-flight output first. The next
Server Edition run re-checks each of the three products and computes
whatever is missing, so the volume state is verified there
(`docs/runs/2026-11-12-runpod-pro6000-v2-omni-trim`).

## Notes

- The fetch-only part needs no particular GPU; splitting it from the
  measurement run let it proceed while the Server Edition was out of
  stock. Two pods must never write the volume at once: the hunt script
  waits for the fill before creating the measurement pod.
- RunPod's S3 API (`s3api-eu-ro-1.runpod.io`) would fill the volume with
  no pod at all, but needs an S3 API key made in the console; the account
  API key is refused (`No active S3 API key exists`).
- Cost: ~$1.7 of pod time; the volume is now $52.50/month at $0.07/GB.
