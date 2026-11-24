# 2026-11-24: pika-plug4 on real weights, and the draft tier's VRAM (RunPod Server Edition)

Pod `1vnbktx5qehz09`: RTX PRO 6000 Blackwell Server Edition (sm_120), 600 W,
2430 MHz max SM clock, driver 595.112.07, EU-RO-1, with the EU weights volume.
The run took ~25 min. The binary was built from `main` at 0149732. The pod
staged h3-base (55 s), h3-ref2va's `transformer_ref` (21 s) and
pika-h3-flex (8 s) onto its container disk.

The pod's own `terminate` could not resolve rest.runpod.io, and the GraphQL
fallback returned `{"error":{}}`. It was terminated from the agent
(`container.py pod-terminate`) right after the results were uploaded.

## Conformance

`h3 check --backend cuda`: **183 / 183** (tolerance 3e-2).

## pika-plug4 against Pika Flex

Every clip is 768p (1344×768×124), seed 7, the fox prompt and assets
(`first.png` / `last.png`; Ref2VA takes `first.png` as an image reference),
with `--weights mxfp8 --decoder-weights mxfp8 --pv8`. The AdaLN tables are
computed per run (no `--tables` for these variants).

| clip | forwards | denoise | AdaLN tables | decode | peak GPU |
|---|---|---|---|---|---|
| Pika FL2VA | 6 × 8.1–8.6 s | 49.4 s | 52.8 s | 6.0 s | 39.7 GB |
| **pika-plug4 FL2VA** | 4 × 8.2–8.8 s | **33.6 s** | 45.9 s | 6.2 s | 41.9 GB |
| Pika Ref2VA | 6 × 11.5–12.1 s | 69.8 s | 44.7 s | 5.9 s | 42.0 GB |
| **pika-plug4 Ref2VA** | 4 × 11.7–12.3 s | **47.3 s** | 39.6 s | 8.0 s | 43.9 GB |

- **Step cost.** The plug's branch is stacked on Pika's task LoRA (one
  branch of rank 256). It costs ~0.1 s per forward over Pika's own.
- **Prompt encoding.** 2065 rows in 12.9 s for FL2VA; 7209 rows (with the
  reference image) in 54–55 s.
- **Quality.** Every clip is clean and photographic (`*-strip.jpg`, one
  frame every 24). pika-plug4 keeps the scene, the subject and the
  keyframes, but it is a different sample: the camera framing drifts
  differently. PSNR against Pika's clip is therefore low even where the
  frames look alike: FL2VA Y 11.4 dB (per frame 32 dB at the first
  keyframe, 13 dB mid-clip, 26 dB at the last), Ref2VA Y 11.5 dB.

## Draft tier without the text encoder

`h3 encode` wrote the fox prompt's embedding (19 s). The drafts then run
`--variant draft --embeds … --tables plug4.tables` with the fast recipe, so
the text encoder is never loaded.

| draft | blocks | denoise | decode | total after load | peak GPU |
|---|---|---|---|---|---|
| 480p (832×480) | resident | 7.63 s (4 × 1.87–2.02) | 2.62 s | 10.5 s | 31.4 GB |
| 480p | `--stream-blocks` | 7.77 s (4 × 1.90–2.08) | 2.65 s | 10.7 s | **10.6 GB** |
| 288p (480×288) | resident | 2.30 s (4 × 0.56–0.62) | 1.26 s | 3.8 s | 28.8 GB |
| 288p | `--stream-blocks` | 2.58 s (4 × 0.62–0.71) | 1.27 s | 4.1 s | **7.5 GB** |

These are the smallest H3 footprints measured so far. The streamed V2 at
768p takes 18.7 GB on the Server Edition and 19.1 GB on a 4090.

**Streaming is nearly free here.** On the 600 W card it costs 2–12 % of the
denoise.

**The 288p fox ghosts again.** The strip shows a double outline of the fox,
as `docs/runs/2026-11-17-vast-pro6000-ref2va-upscale` found: 480p stays the
smallest reliably clean tier.

**Resident against streamed is not byte-identical in this round.** The
480p pair is 13.6 dB Y apart. Both runs use MXFP8, and each run times
cuBLASLt's candidates on its first call per shape, so two runs differ at the
rounding level. Through 4 renoised steps that becomes CLAUDE.md's ~14 dB
"first-call rounding band". The checker's exact streamed == resident cases
pass on this card. A pinned pair (`H3_CUDA_LT_NTH=0`) was not run, so the
streamed path at real size is consistent with this explanation but not yet
proven identical.
