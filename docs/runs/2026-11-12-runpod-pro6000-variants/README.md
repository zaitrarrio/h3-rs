# 2026-11-12: RunPod RTX PRO 6000 Server Edition, the EU weights volume: Pika Flex and Regen-2K after their fixes

RunPod pod `jt5y7a5g9mvf45` (secure cloud, EU-RO-1, $2.49/h): NVIDIA RTX
PRO 6000 Blackwell Server Edition (sm_120, 96 GB, **600 W**, boost
2430 MHz), driver 595.91.07, 128 vCPUs, 2.2 TB RAM, image
`nvidia/cuda:12.8.1-runtime-ubuntu24.04`, the network volume
`fv-weights-h3-ltx-500` (500 GB, shared with fastvideo-rs) at `/workspace`.
Binary: release of 4c12af4 (`--features cuda`), the one the Vast round's
fixes went into. Driven through `scripts/runpod/` (same R2 command channel
as `scripts/vast/`). Raw outputs `output-N-*.txt`, clip logs `*.log`,
frame grids `*-frames.png`.

Conformance: `h3 check --backend cuda` **115 / 115** (tolerance 3e-2).

## The volume

Before this run (fastvideo-rs's fill of 2026-10-13, `.complete` trees):
h3-base 206 GB (the MiniMax-H3 tree plus fastvideo-rs's `text_encoder_fp8`
and two prequantized MXFP8 DiTs), h3-ref2va 65 GB (`transformer_ref`),
FastH3-4-step-Preview-v1-LoRA 6.4 GB, ltx25 117 GB, ltx25-dev 36 GB and
small trees: ~433 GB of the 500 GB quota. Added by this run (add-only,
with `.complete` / `.fv-revision` markers), ~21 GB:

| path | content |
|---|---|
| `weights/pika-h3-flex/` | `fl2va/`, `ref2va/` adapters, `processor/`, manifests (5.2 GB) |
| `weights/open-h3-regenerate-2k/` | `lora/`, `latent_upsampler/`, `config.json` (14 GB; not `base_model/` or `vae_fast/`) |
| `h3-rs/tables/pika.tables`, `base.tables` | AdaLN tables: Pika's 13 renoise knots (126 MB), base H3's 99 timesteps (960 MB) |
| `h3-rs/assets/` | the fox draft (`fox.y4m`, 124 frames 1344×768), `first.png`, `last.png`, `adaln_t_table.bin` |
| `h3-rs/h3/` | a model directory of symlinks: `h3-base`'s components plus `h3-ref2va/transformer_ref` (base and Pika Ref2VA need both) |

FastH3-V2 (`h3-8step`, 66 GB transformer), FastH3-Omni (65 GB), the Trim
NVFP4 file (11 GB) and a stored MXFP8 V2 (26 GB) do **not** fit: the
volume has ~45 GB left. Growing it (or a second volume; a pod mounts one)
is the owner's call.

The volume is a network filesystem (`mfs`, `maxDownloadSpeedMbps` 5900 on
this host): weight-streaming passes read it at ~0.4–0.5 GB/s, so the
AdaLN tables took 576 s (Pika, 145 s on Vast's local NVMe) and 439 s
(base, 197 s), and a Pika load 490 s against 83 s. The clips themselves
are unaffected. The Hugging Face fetch ran at ~170 MB/s (18.6 GiB in
110 s).

## Clips (1344×768×124, seed 7, the fox prompt)

| variant, task | load | prompt rows (host) | denoise | decode | status |
|---|---|---|---|---|---|
| `pika-flex` FL2VA, 768p bucket, first + last keyframe | 490 s | 2065 in 225 s | **77.1 s** = 6 × 12.75–12.93 s | 8.1 s | clip (`pika-fl2va-frames.png`): the fox approaches between the keyframes, no artifacts |
| `pika-flex` Ref2VA, one image reference | | | | | first attempt: `transformer_ref` must sit beside the base components (`/w/h3-base` has none); re-run with `h3-rs/h3/` below |
| `regen-2k` from the 768p draft, base-VAE decode at 2560×1440 | 397 s | | 3 × 98.8–99.6 s (+ 160 s prepare: draft encode, upscale, tables) | **30.7 s** (36-block decoder, 7 chunks of 100 k rows) | 121 clean frames (`regen-2k-frames.png`, `regen-2k-crop-640x360.png` at 1:1): no 16-pixel grid, needles and fur resolved |

The Pika FL2VA clip is the first conditioned renoise run after 9a5bf77
(the noise rows): the Vast round's `renoise needs noise` panic is gone.
Regen-2K through the base VAE (4c12af4) replaces the Vast round's grid
with a clean 2K frame; the decode costs 30.7 s against the broken fast
decoder's 21 s.
