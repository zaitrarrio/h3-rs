# 2026-11-24: the fast (Sol-H3) VAE decoder on a GPU (RunPod Server Edition)

Pod `nwk3bol0iparme`: RTX PRO 6000 Blackwell Server Edition (sm_120),
600 W, 2430 MHz max SM clock, driver 595.112.07, EU-RO-1, EU weights volume.
The binary is this branch (4e9fc4d + docs), built with `--features cuda`.

The first pass of this round used a binary built without `--features cuda`,
so every GPU step failed at once. The pod's `terminate` could not reach
rest.runpod.io (as in `2026-11-24-runpod-pika-plug4`), so the pod stayed up;
a CUDA binary was queued as command 5 and reran everything. The pod was then
terminated from the agent.

## Weights

`Efficient-Large-Model/Open-H3-Regenerate-2K`
`vae_fast/fast_h3_vae_noncausal_24l.safetensors` (6.47 GB, 297 f32 tensors)
was fetched from Hugging Face in 51 s and SHA-256-checked
(`8658f1e8…`). It is now on the volume at
`weights/open-h3-regenerate-2k/vae_fast/`.

## Conformance

`h3 check --backend cuda`: **188 / 188** (tolerance 3e-2).

| case | rel |
|---|---|
| `box_rows` (four cases) | 2.0–2.5e-3: bf16 upload rounding; the op itself is a copy |
| fast vae tiny decode 7×5×7 (windows, globals, shifts) | 1.3e-2 |

## Decoders on the same latents (`h3 vae-roundtrip`)

`fox.y4m` (124 frames, 1344×768) is encoded with the H3 VAE (tile 256, the
mode of the posterior) and decoded by both decoders. For 2K the clip is
Lanczos-upscaled to 2560×1440 first.

- **Released decoder:** spatial tiles of 256 px (`generate`'s default).
- **Fast decoder:** one pass over the whole clip.
- **PSNR:** RGB over all 124 frames.
- **Times:** the second of two decodes. The first is within 0.1–0.5 s.

| canvas | weights | released decoder | fast decoder | speed-up | released vs input | fast vs input | fast vs released |
|---|---|---|---|---|---|---|---|
| 768p (37×48×84 latents) | bf16 | 9.39 s | **3.13–3.32 s** | 2.9× | 33.16 dB | 31.61 dB | 34.52 dB |
| 768p | `--decoder-weights mxfp8` | 6.58 s | **2.86 s** | 2.3× | 33.16 dB | 31.55 dB | 34.38 dB |
| 2K (37×90×160) | bf16 | 33.97 s | **11.30–11.69 s** | 3.0× | 34.03 dB | 33.49 dB | 39.65 dB |

**Quality.** Sana's README reports ~30 dB against the released decoder;
here it is 34.5 dB at 768p and 39.7 dB at 2K. The fast decoder is 1.6 dB
further from the input than the released one at 768p, and 0.5 dB at 2K.

**Crops.** `rt768-crop.jpg` and `rt2k-crop.jpg` show frame 60, 640×360,
with input, released and fast from top to bottom. The three are visually
the same, with no window seams. The strips (one frame every 24) are clean
and show no seams at the 5-latent block boundaries.

**Peak GPU.** 8.1–8.2 GB at 768p and 16.6 GB at 2K, the same with and
without the released decoder (`*-fastonly.log`). The H3 VAE *encoder*
(50 s at 768p, 187 s at 2K) sets the peak in this tool, so the decoder's
own footprint is below these numbers.

**Load.** The fast decoder loads in 3.9–4.5 s, f32 from local disk with the
`to_qkv` / `w1` permutes.

## In a clip (`generate --fast-vae`)

The draft variant (`h3-plug4`) at 768p, from `--embeds` with
`--tables plug4.tables`, bf16 weights, seed 7. Bf16 weights keep the
denoise deterministic: both runs' steps match to 0.01 s, and the latents
are the same.

| decoder | decode | total after load | peak GPU |
|---|---|---|---|
| released | 8.65 s | 53.0 s | 53.5 GB |
| **fast** | **2.71 s** | **47.1 s** | 51.9 GB |

The two clips are Y 33.8 dB apart (38.2 dB averaged over Y, U and V). Both
are clean (`d768-*-strip.jpg`).
