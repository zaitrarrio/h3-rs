# 2026-11-13: an RTX 5090 against the "MiniMax H3 speed-ups, measured" page

The page (https://plox-1.github.io/h3-speedups/, generated 2026-10-18,
rows saved in `h3-speedups-reference.json`) times ComfyUI's official H3
template on one RTX 5090 (32 GB): "MiniMax H3, pruned int8 (template)",
Comfy Kitchen int8 attention, 1344×768, 124 frames at 24 fps, seed 1,
one prompt ("Samurai in the rain", 5.2 s, saved in the JSON) and a 15 s
four-shot prompt (362 frames). Its **total** is the warm render of the
whole workflow in a fresh ComfyUI (the cold render is listed too), its
**sampler** the KSampler alone; the baseline is the raw model at 25 euler
steps and every row is a speed-up against it. ComfyUI caches the prompt
conditioning between the cold and the warm render, so its totals carry no
text-encoder time.

h3-rs ran the same prompt, canvas, frame count and seed on RunPod
5090 pods (secure cloud EU-RO-1, $1.39/h, **600 W** power limit, driver
595.91.07, 3150 MHz max SM clock, 32 607 MiB), weights staged onto the
container disk (`H3_STAGE`), release binary of fb78a6b (`h3-bin7`, the
serial block streaming) for pod `nmvicb3scqaun8` (04:01–04:44 UTC) and
of a8fd30a (prefetched streaming, below) for the follow-up. Our numbers:
`load` (weights to the device), `prompt` (Qwen3-VL on the device, 462
rows for this prompt: the step ComfyUI caches away), `denoise`, `decode`,
and `peak VRAM` from `nvidia-smi` sampled every 0.5 s. The comparable
pairs are **denoise vs sampler** and **denoise + decode vs total**.

## Why a 5090 is a different problem

The V2 recipe that leads on the 96 GB Server Edition (stored MXFP8
weights 25.6 GB resident, MXFP8 decoder, INT8 Q·Kᵀ VSA, tables) **does
not fit 32 GB** (`v2-resident`: OOM at 31.7 GB after load, log in
`output-1-v2-v1.txt`). On this card the block weights stream from host
memory each step (`--stream-blocks`), and the first pod measured the
serial streaming path that existed: every block's codes re-encoded on the
CPU and copied on the compute stream before its kernels could start.

## Pod 1 (fb78a6b, serial streaming): the samurai prompt, 124 frames

| | ComfyUI row | ComfyUI total / sampler | h3-rs config | load | prompt | **denoise** | decode | denoise + decode | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| FastH3-V2, 8 steps | `fasth3-v2-euler`: pruned int8, Kitchen int8 + VSA keep 10 %, first 20 % dense | **54.1 s / 45.2 s** (cold 62.5 s, 30.0 GB) | stored MXFP8, `--decoder-weights mxfp8 --vsa-qk8`, tables, VSA 0.8 (the released contract), `--stream-blocks` | 28.7 s | 69.9 s | **123.1 s** = 15.57, then 15.19–15.50 s × 7 | 6.28 s | 129.4 s | 17.5 GB |
| same, ComfyUI's attention budget | | | + `--vsa-sparsity 0.9 --dense-first-steps 2` | 26.9 s | 76.8 s | 130.2 s = 21.2, 21.3 (dense), then 14.6 s × 6 | 6.23 s | 136.4 s | 17.1 GB |
| FastH3 v1, 4 steps | `fasth3-v1-euler`: int8, VSA keep 10 %, first 20 % dense | **33.3 s / 23.9 s** | base + `vsa-datafree` LoRA, MXFP8 at load, `--vsa-qk8`, VSA 0.9, tables, streamed | 86.4 s | 76.5 s | **57.9 s** = 14.7, 14.4, 14.4, 14.4 s | 6.19 s | 64.1 s | 17.1 GB |
| baseline, raw H3, 25 euler steps | `baseline`: pruned int8, Kitchen int8, dense | **181.7 s / 169.0 s** (median of 4; cold 189.3 s) | `--variant h3 --steps 25` (the same uniform ladder), MXFP8 at load, dense, tables, streamed | 58.8 s | 66.4 s | **530.8 s** = 21.0–21.7 s × 25 | 6.21 s | 537.0 s | 15.1 GB |
| FastH3-Trim, ComfyUI NVFP4 file | (no row) | | `--variant fasth3-trim`, NVFP4 resident, `--vsa-qk8`, 26-block decoder | 85.0 s | 55.2 s | 63.6 s = 7.90–7.98 s × 8 | 12.35 s | 76.0 s | 26.9 GB |

Also on pod 1: a V1 run with both LoRAs of the tree merged by mistake
(`v1-both-adapters.log`, 4 × 14.0 s; the tree holds `dense-datafree/`
and `vsa-datafree/`, only the latter is the VSA checkpoint) and a
362-frame attempt that failed on a duplicated `--frames` flag in my
script (re-run on the follow-up pod). Frames: `*-frames.png` (0 / 40 /
80 / 120). Every clip follows the prompt (the arm-cross, the push-in,
rain on the face); like the page's V2 row, ours gives him a topknot.

Reading pod 1: the streamed forward costs ~9 s over the card's compute
(V2: 15.4 s per forward; the 600 W Server Edition runs it in 4.7 s with
resident weights, and a 5090 has ~0.8× its tensor throughput), and the
dense base forward 21.3 s where ComfyUI's int8 attention + int8 GEMMs do
a step in 6.8 s. Trim, the one recipe that fits resident, runs 7.95 s per
forward (5.5 s on the Server Edition) with its per-GEMM NVFP4 expansion.

## Follow-up pod (a8fd30a): prefetched streaming

(filled in below from the second pod)
