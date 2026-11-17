# 2026-11-17: Ref2VA drafts and Turbo-ref, 480×272, Real-ESRGAN at every tier, Regen-2K at any aspect (RTX PRO 6000 Server Edition)

Vast instance 61099445 (RTX PRO 6000 Blackwell Server Edition, Nevada,
driver 580.173.02, 600 W limit), H3 base tree from R2 (`h3-base`, with
`transformer_ref`, 196 GB). Prompt: the fox, encoded once with `h3 encode`
(100 s) and passed as `--embeds`; seed 7. "fast" is `--weights mxfp8
--decoder-weights mxfp8 --pv8`. Every draft runs with `--tables` (computed
once per DiT: `h3 adaln-tables` 3 min 13 s for `transformer`, 3 min 12 s
for `transformer_ref`, load included). Load is 69–80 s per process and is
not in the totals below.

## Conformance

`h3 check --backend cuda`: 180 / 180, with the new cases: conv2d with the
PReLU epilogue (`3→64`, `64→48`), the tiny Real-ESRGAN network (×2, float
output of `Esrgan::network` against the CPU), and the LoRA branch on the
AdaLN tables.

## Drafts at every tier, upscaled (`--variant draft`, fast, `--tables`)

| run | canvas → output | denoise | decode | total | upscale | peak VRAM |
|---|---|---|---|---|---|---|
| 272p 16:9 | 480×288 → crop 480×272 → ×4 1920×1088 | 2.42 s (4 × 0.57–0.72) | 1.32 s | **4.02 s** | 1.66 s (32 frames / batch) | 29.6 GB |
| 272p 9:16 | 288×480 → crop 272×480 → ×4 1088×1920 | 2.34 s | 1.32 s | 3.92 s | 1.64 s | 29.6 GB |
| 272p 16:9, TAEH3 (`--preview`) | 480×288 → 480×272 | 2.33 s | 0.22 s | **2.82 s** | | 26.6 GB |
| 480p | 832×480 → ×4 3328×1920 | 7.67 s (4 × 1.88–2.03) | 2.64 s | 10.62 s | 4.42 s (10 / batch) | 31.8 GB |
| 768p | 1344×768 → ×2 2688×1536 | 28.03 s (4 × 6.87–7.31) | 6.34 s | 34.74 s | 18.45 s (4 / batch) | 37.8 GB |

Totals include 0.19 s of audio decode. The upscale time is the network on
the GPU, the 8-bit residual on the host, and for ×2 / ×3 a Lanczos
resample of the ×4 frames on the host: at 768p the ×4 intermediate is
5376×3072 per frame and the resample dominates; a GPU resample (or ×2
weights) would take most of those 18 s away.

`upscaled-272-480-768.jpg` (frames 0 / 60 / 120 of the 272p, 480p and
768p drafts, all scaled to the same width): 480p and 768p are clean
photographic foxes. **272p is out of H3's distribution** (its smallest
trained tier is 480p): the landscape draft has ghosting (a doubled fox
in frame 0, a smeared one in frame 60), while the portrait came out clean
(`upscaled-272p-portrait-f60.jpg`, frame 60 at 1088×1920). Treat 272p as a
thumbnail tier; 480p is the smallest that is reliably clean. `d272-taeh3.jpg`
is the same 272p draft through TAEH3.

## Ref2VA (one reference image, a new scene)

Reference: `ref2va-reference.jpg` (a frame of an earlier fox clip,
uploaded as a PNG); prompt "The red fox from the reference image sits
beside a crackling campfire at night in a snowy pine forest; firelight
flickers on its fur and embers drift upward." Fast recipe:

| run | canvas | forwards | per forward | denoise | AdaLN | decode | total |
|---|---|---|---|---|---|---|---|
| `draft` (LongLive-Plug by transfer), `--tables` (ref) | 832×480 | 4 | 4.74–5.12 s | 19.39 s | 0 (table) | 2.65 s | 34.19 s |
| `h3-turbo4-ref` (LightX2V 4-step v0.1) | 1344×768 | 4 | 11.40–11.96 s | 46.35 s | 47.10 s | 6.18 s | 111.84 s |
| `h3-turbo8-ref` (LightX2V 8-step v1.0, 768p) | 1344×768 | 8 | 11.40–11.96 s | 92.68 s | 45.59 s | 6.20 s | 156.72 s |

The rest of each total is the reference image's conditioning (Qwen3-VL on
the 2048-short-edge reference, ~12 s). A Ref2VA forward costs 2.5× a T2VA
one at 480p (4.8 vs 1.9 s) and 1.7× at 768p (11.5 vs 6.9 s): the reference
adds its tokens to every attention. The Turbo-ref runs had no table file
(one per variant: Turbo 8 carries AdaLN pairs).

`ref2va-draft-turbo4-turbo8.jpg` (rows: draft 480p, Turbo 4, Turbo 8;
frames 0 / 60 / 120): all three keep the reference's fox (coat, white
chest, black legs) and build the asked scene. The draft holds one pose
by the fire; Turbo 4 and 8 walk the fox in and sit it down.

A T2VA table on a Ref2VA run is refused before any work: "`/w/t2v.tables`
was computed from transformer, this run uses transformer_ref: run `h3
adaln-tables` with the same --task".

## Regen-2K from 272p drafts (any aspect)

`h3 regen2k --draft` from the two un-upscaled 272p drafts (rendered again
for this, 3.9 s each), base-VAE decode, 3 steps:

| draft | canvases | prepare | steps | decode | total |
|---|---|---|---|---|---|
| 272×480 (9:16) | draft canvas 768×1344 → 1440×2560 | 104.3 s | 103.0 / 103.6 / 103.6 s | 31.9 s | 446.5 s |
| 480×272 (16:9) | draft canvas 1344×768 → 2560×1440 | 107.3 s | 103.3 / 103.6 / 103.6 s | 32.2 s | 450.1 s |

Both wrote 121 frames at the draft's aspect (`regen_canvases`), at the
same cost as from a 768p draft (446 s, the earlier round): Regen-2K's time
is set by its 2K canvas, not by the draft's size. **No images**: the
round's archive step globbed in the wrong directory (`tar -C $O r2k-*`
expands the glob before `-C` applies) and the box destroyed itself as
scheduled, so the 2K frames were lost; only the log above survives. The
composition-keeping claim rests on the earlier round's
`regen2k-from-drafts.jpg` (480p and 768p drafts).
