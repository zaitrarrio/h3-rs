# 2026-11-13: FP8 P·V (SageAttention2's second half) on an RTX 5090

Follow-up to `2026-11-13-runpod-5090-vs-comfy`: that record left the
dense base step at 9.95 s against ComfyUI's 6.8 s (its attention call
112 ms, 61 % of the step) and the V2 clip's denoise at 52.0 s against
ComfyUI's 45.2 s sampler. This round adds FP8 P·V to the INT8 Q·Kᵀ
attention on both paths (`--pv8`, dense `fw_attention_pv8_{64,128}` and
the VSA fine stage `fw_vsa_fine3g`; the recipe is `docs/design.md` 7c),
measures it on the same card class, prompt, canvas and seed, and on the
way found and fixed a race in the page-locked streaming path that had
been in the tree since 673d23b.

All pods: RunPod secure cloud EU-RO-1, the `fv-weights-h3-ltx-500`
volume, driver 595.91.07, weights staged onto a 250 GB container disk
(`h3-base/transformer`, the stored MXFP8 V2, the V2 text encoder and
VAEs: 170 GB). RTX 5090 pods report a **575 W** power limit and 32 607
MiB; the 4090 450 W.

## Pods

| pod | card | when (UTC) | what happened |
|---|---|---|---|
| `nrsb6g6rfq87d3` | 5090, $1.39/h | 15:06–17:10 | my boot staged the whole 220 GB `h3-base` tree; the disk filled, the command loop could not even write its fetched command. Stopped it to reboot onto an empty disk; the host gave the GPU away in those seconds and `start` was refused. ~$2.9 for nothing. |
| `qgw7fyo402fro1` | 5090 | 17:23–17:28 | the uploaded binary had been built without `--features cuda` (every build since 13:22 had). Ran the script in a minute, self-terminated. |
| `cfgk81b4wo999q` | 5090 | 17:38–18:30 | the full script (`pod3-black/`): 141 / 143 checks, all timings below, **every clip black with full-scale audio**, PSNRs infinite because all five clips were identical. |
| `1vinrib42j65l9` | 5090 | 18:41–18:58 | debug 1 (`debug/dbg1-*`): `H3_TRACE=1` per-block magnitudes, and the same recipe on a binary of 673d23b (the commit of the pod-5 clips of the previous record): **black too**, so not a regression of this round. |
| `mickhwtcdsw49f` | 4090, $0.89/h | 19:05–19:32 | debug 2 (`debug/dbg2-*`): one switch per run. `H3_CUDA_LT_NTH=0` black, `H3_CUDA_NO_FP8=1` black, **`H3_CUDA_NO_PIN=1` a real clip** (`dbg2-v2-nopin.grid.png`), `H3_TRACE=2` finite through step 1, non-finite from step 2 (the per-block syncs change the timing). |
| see below | 5090 | | the full script again with the fix (29317cb). |

## The streaming race (29317cb)

`prefetch_wait` took the fence event when it made the compute stream
wait on it, so the next `prefetch_begin` found no fence, skipped its
host-side wait and reset the 64 MB pinned bounce buffer while the
previous block's small copies (norm weights, the unlocked edge pages of
every weight) could still be queued behind a release wait on the copy
stream. The next block's bytes overwrote them, a few of a block's
weights came from the block after it, and the hidden state went
non-finite a step or two later (`dbg1-v2-trace.log`: the first forward
finite through block 49 with max |h| 1.0e7, step 0 latents max 5.3, step
1 all NaN; `dbg2-v2-trace2.log`: step 1 still finite, step 2 NaN). How
far the host runs ahead of the GPU decides whether a slot is reused in
time, which is why pod 5 of the previous record produced byte-identical
clips with the same code and three hosts today never produced a frame.
The fence now stays until the next `prefetch_end` replaces it; the host
blocks in `prefetch_begin` until the previous block's copies have run,
at most two blocks behind the compute stream. `H3_CUDA_NO_PIN=1` was
never affected (pageable copies are waited for).

On the 4090 (PCIe 4) the pinned and unpinned V2 forwards were the same
14.8–14.9 s, so page-locking buys nothing there; its gain was measured on
the 5090's PCIe 5 ×8 link (7.2 → 6.45 s in the previous record).

## The two causal checks (6bdc2c2)

`attention qk8+pv8 70x70 causal` and `33x33 causal` sat at 5.1e-2 of the
oracle on the 5090 while the other 141 checks passed (the non-causal
pv8 cases at 1.9–2.1e-2, the VSA pv8 cases at 0.4–2.3e-2, the tiny DiTs
at 5e-3). P is rounded to E4M3 in steps of 1/8, so one rounding flip on
a row with one key is 6 % of that row, and the checker's metric is the
max-abs difference over the max-abs reference. A kernel whose f32 scores
differ from the oracle's in the last bits produces a few such flips per
case; perturbing the oracle's own scores by 1e-5 trips the 33x33 shape
in 1 of 40 seeds and 1e-4 the 70x70 one, while the non-causal shapes
stay under 2e-2. The causal cases now have 32 more keys than queries, so
every row averages at least 33; the mask is exercised as before.

## Results (pod 3 timings, confirmed on the fixed pod below)

RESULTS_TBD
