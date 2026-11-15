# 2026-11-15: LongLive-Plug's four-step adapter on the base H3 (`--variant h3-plug4`)

[LongLive-Plug](https://nvlabs.github.io/LongLive/LongLive-Plug/) distils a
backbone once into separate LoRAs; for MiniMax-H3 it releases a few-step
adapter (`Efficient-Large-Model/LongLive-Plug-MiniMax-H3-few-step`, revision
b3686f4, rank 128 / alpha 128 on the 312 attention and FF linears of the 50
blocks and the two refiner blocks, a DMD student trained on T2VA) and a CFG
adapter (not used here; its README says not to combine the two yet). Its
page claims the adapters transfer to the backbone's other tasks without
retraining.

## What h3-rs runs

- `scripts/reference/convert_longlive_plug.py` turns the release's
  `generator_lora.pt` (a torch pickle) into `generator_lora.safetensors`
  after checking its SHA-256 and the release loader's contract (shifts
  12 / 3, rank 128, f32, the 312 targets). The opt-in test
  `longlive_plug_adapter_maps_onto_h3` maps every pair onto the H3
  manifest.
- `--variant h3-plug4`: the base DiT with the adapter merged in f32 at load,
  and the release's sampler, which is diffusers' `MiniMaxH3Scheduler` grid
  for `num_inference_steps=5` (linspace(1, 0, 5), shifted 12 for video and
  3 for audio; pinned bit for bit in `plug4_grid_matches_diffusers`) with a
  predict-x0 / fresh re-noise step: Pika's renoise rule, so the preset is
  `Sampler::Renoise([1000, 750, 500, 250])`. No CFG (the release forbids a
  guidance scale). Four forwards.
- A variant whose adapter is missing is now an error: the four-step sampler
  on unadapted weights would make noise, not a clip.

Converted adapter and tables are on the RunPod volume
(`/workspace/h3-rs/plug4/`: `generator_lora.safetensors`
sha256 4f8175d5…, `plug4.tables`, `plug4-ref.tables`).

## Round 1 (`round1/`): RunPod Server Edition, EU-RO-1

RTX PRO 6000 Blackwell Server Edition, 600 W, driver 595.91.07, AMD EPYC
9555, median SM clock 2280–2310 MHz under load; the base tree staged from
the volume (220 GB in 175 s), `transformer_ref` from `h3-ref2va`. Binary
7a74532. `h3 check --backend cuda` **170 / 170** plain and under
`H3_STALL=1`, `H3_PROFILE=1`, `H3_STREAM_STRESS=1`. The four-step tables
took 3 min per transformer.

1344×768×124, the fox prompt and seed 7 unless said ("fast" = `--weights
mxfp8 --decoder-weights mxfp8 --pv8`, MXFP8 quantized at load after the
merge, dense INT8 Q·Kᵀ + FP8 P·V attention):

| run | forwards | denoise | decode | after load | peak | clip |
|---|---|---|---|---|---|---|
| `h3-plug4`, bf16 weights, bf16 attention | 4 × 10.65–10.78 s | 42.9 s | 8.6 s | 51.8 s | 51.6 GB | a natural fox at dawn, coherent (`plug-bf16.grid.png`) |
| `h3-plug4`, fast | 4 × 6.49–6.94 s | **26.5 s** | 6.0 s | **32.8 s** | 35.7 GB | coherent but a different, oversaturated, painterly close-up (`plug-fast.grid.png`): 9.2 dB from the bf16 clip |
| the same, `H3_STALL=1` | | 26.7 s | 6.0 s | 33.0 s | | **byte-identical** to the line above |
| `h3-plug4` FL2VA (transfer), fast | 4 × 7.63–8.19 s | 31.1 s | 5.8 s | 40.3 s | 37.2 GB | follows the keyframes, saturated fox |
| `h3-plug4` Ref2VA (transfer), `transformer_ref`, fast | 4 × 10.89–11.55 s | 44.3 s | 5.8 s | 61.7 s | | follows the reference, saturated |
| `h3-plug4`, the samurai, seed 1, fast | 4 × 6.63–7.15 s | 27.1 s | 6.1 s | 33.5 s | 35.8 GB | coherent samurai, temple, gesture; saturated, no rain |
| `h3` base, 49 forwards, fast | 49 × 6.49–6.58 s | 322.5 s (+ 51.9 s AdaLN, no tables) | | 380.8 s | 35.7 GB | realistic fox at sunset (`base49-fast.grid.png`) |
| `fasth3-v2` best recipe (stored MXFP8, VSA, `--vsa-qk8 --pv8`) | 8 × 4.50–4.98 s | 36.6 s | 6.1 s | 43.3 s | 39.6 GB | realistic fox (`v2-best.grid.png`) |

Load: 72–89 s (the bf16 tree read, the adapter merged, MXFP8 quantized);
prompt 75–90 s (the host text encoder; Ref2VA 7209 rows in 119 s).

So the plug delivers the four-forward clip it promises and the transfer
claim holds for FL2VA and Ref2VA (both coherent and conditioned), and with
the fast recipe it is the shortest clip h3-rs makes: 32.8 s after load
against V2's 43.3 s. But its look depends on the numerics in a way the base
and V2 do not: on the same 8-bit path they stay photographic, the plug
turns stylised. A dense forward is 6.5 s against V2's 4.5 s (no VSA gate
in the base, so dense attention), so the 4 forwards save less than the step
count suggests.

## Round 2: which part of the fast recipe does it

_In progress: the same fox / seed with bf16 weights + `--pv8`, bf16 + `--qk8`,
MXFP8 weights + bf16 attention, MXFP8 on every block, and the bf16 path
again for run-to-run determinism, each against the bf16 clip._
