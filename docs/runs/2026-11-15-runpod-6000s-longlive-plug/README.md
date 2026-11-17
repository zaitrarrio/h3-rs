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

## Round 2 (`round2/`): which part of the fast recipe does it

Second pod (same card class and driver), binary 7a74532, the fox at seed 7,
each clip against round 1's bf16 clip (PSNR between the two x264 encodes):

| run | denoise | luma | against the bf16 clip | look |
|---|---|---|---|---|
| bf16 weights, `--pv8` (INT8 Q·Kᵀ + FP8 P·V) | 31.2 s (4 × 7.8 s) | 86–87 | **21.2 dB** | the same natural scene (`plug-pv8.grid.png`) |
| bf16 weights, `--qk8` | 34.8 s | 86–87 | **22.0 dB** | the same scene |
| `--weights mxfp8`, bf16 attention | 38.2 s | 110–111 | **10.5 dB** | the stylised close-up (`plug-mx.grid.png`) |
| `--weights mxfp8`, every block (`H3_MXFP8_ALL_BLOCKS=1`) | 37.5 s | 106 | 10.9 dB | stylised |
| bf16 again | 42.9 s | 88 | **byte-identical** (the mp4s too) | the round-1 clip |

The 8-bit attention keeps the scene within the usual band; the MXFP8
weights alone flip it, with or without the bf16 edge blocks. The base and
V2 take the same MXFP8 weights without changing scene, so the difference
lies in the adapter.

## The cause: a merged delta below the weights' rounding

Measured on blocks 18–20 of the real base weights and the converted
adapter (CPU, torch 2.12):

| | ratio |
|---|---|
| ‖ΔW‖ / ‖W‖ (the adapter's delta against the weight) | 1–2·10⁻⁴ |
| MXFP8 rounding of W, ‖Q(W) − W‖ / ‖W‖ | 2.7·10⁻² |
| what an MXFP8 merge keeps of Δ: ‖Q(W+Δ) − Q(W) − Δ‖ / ‖Δ‖, cosine with Δ | 1.5–1.9, cosine 0.07–0.10 |
| what a **bf16** merge keeps of Δ: ‖bf16(W+Δ) − W − Δ‖ / ‖Δ‖, cosine with Δ | 0.89–0.98, cosine 0.24–0.49 |
| an unmerged branch with bf16 factors, ‖B·A − Δ‖ / ‖Δ‖ | 1.5–2.3·10⁻³ |

The DMD student's per-element change is far below one bf16 step of the
weight it is added to, so round-to-nearest mostly snaps it back: the
bf16 "quality anchor" of round 1 ran a third of the adapter, the MXFP8 one
a tenth. The release runs the LoRA as 312 fp32 branches and never merges;
WebGPU's f16 weights lose it the same way, and only the CPU backend's f32
weights merge exactly.

So h3-rs now keeps LoRA pairs as an unmerged branch on the block and
refiner linears (857cbe1): `y = W x + B (A x)` with the pair's factors in
the device's native precision (fused QKV as one block-structured pair),
through a new `Backend::linear_add` op (CUDA: the bf16 GEMM with beta = 1;
CPU and WebGPU: linear into a scratch plus axpby). Dense deltas
(`set_weight`, `diff`) still merge; a stored quantized file keeps its codes
under a LoRA; `H3_LORA_MERGE=1` restores the merge. Checker (177 cases):
`linear_add`, the tiny DiT with a rank-3 LoRA on every block and refiner
linear, native and MXFP8, resident and streamed, against the CPU; and on
the CPU, branch against merge at 1.4·10⁻⁷ with the adapter moving the
output by 24 %. CPU and lavapipe green.

## Round 3 (`round3/`): the branch on the Server Edition

Third pod, binary 857cbe1, same card class and driver. `h3 check --backend
cuda` **177 / 177** plain and under `H3_STALL=1`, `H3_PROFILE=1` and
`H3_STREAM_STRESS=1` (the LoRA-branch cases at the CUDA tolerance).
The fox at seed 7 unless said; "fast" as above:

| run | denoise | decode | after load | peak | clip |
|---|---|---|---|---|---|
| `H3_LORA_MERGE=1`, bf16 (the old merge) | 42.9 s | 8.5 s | 51.7 s | 51.6 GB | **byte-identical** to round 1's bf16 clip |
| **branch, bf16** | 44.8 s (4 × 11.1–11.3 s) | 8.6 s | 53.8 s | 53.5 GB | a natural fox walking at the camera through young pines (`plug-bf16-branch.grid.png`); 11.2 dB from the merged clip: the full adapter takes another trajectory |
| **branch, fast** | **28.4 s** (4 × 7.0–7.4 s) | 6.1 s | **34.7 s** | 37.6 GB | the same scene as the bf16 branch (`plug-fast-branch.grid.png`): **17.3 dB** luma / 19.0 average, luma 90 against 89; no longer stylised |
| branch, MXFP8 weights only | 40.1 s | 8.9 s | 49.2 s | 39.0 GB | the same scene, 16.4 dB |
| branch, fast, `H3_PROFILE=1` | 28.9 s | 7.5 s | | | **byte-identical** to the fast clip |
| branch, fast, the samurai (seed 1) | 29.0 s (4 × 7.1–7.6 s) | 6.0 s | 35.3 s | | photographic: the gesture, the push-in and the close-up the prompt asks for (`plug-samurai-branch.grid.png`) |
| branch, fast, FL2VA (transfer) | 33.2 s | 5.7 s | 42.3 s | 40.0 GB | follows the keyframes (`plug-fl2va-branch.grid.png`) |
| Pika Flex FL2VA, bf16, branch | 81.4 s (6 forwards) | 8.5 s | 93.3 s | 54.8 GB | |
| Pika Flex FL2VA, bf16, `H3_LORA_MERGE=1` | 76.5 s | 8.5 s | 88.4 s | 52.9 GB | **28.6 dB** from the branch clip: Pika's adapter survives the merge far better, so its earlier clips were close |

The branch's cost (`plug-fast-prof.log`, 4 forwards): the up-projections
`gemm+acc` (rank-128 / 384 GEMMs with beta = 1) 1.63 s, the down-projections
~0.3 s, about 0.6 s per forward (6.5 → 7.0 s, +8 %); the FFN-up branch is
3.75 ms a call for a memory-bound read-modify-write of `[38k, 28672]`,
which a GEMM over `[W | B]` and `[x ; A x]` would fold into the main one.
Dense attention is half the plug's forward (71 ms a call: the base has no
VSA gate), which is why its four forwards (28.4 s) cost three quarters of
V2's eight (36.6 s).

## Where the plug stands

On the Server Edition, the four-step plug with the fast recipe and the
branch makes a correct, photographic 1344×768×124 clip in **34.7 s after
load** (denoise 28.4 s, decode 6.1 s), against FastH3-V2's best 43.3 s
(36.6 s denoise) and the 49-step base's 380.8 s, and it transfers to FL2VA
(42.3 s) and Ref2VA. It is the fastest clip h3-rs makes on this card.

Open: a PSNR against the release's own fp32-branch PyTorch run at full
size (needs the base weights in a PyTorch environment on a GPU); the
LongLive-Plug CFG adapter (its README says not to combine it with the
few-step one yet); folding the branch into the main GEMM; Regen-2K's
rank-576 LoRA under the branch (unmeasured, now a ~20 % larger DiT cost
there, `H3_LORA_MERGE=1` restores the old path).

Cost: four Server Edition pods (47, 22, 43 min and the volume staging) at
$2.69 / h, about $5.
