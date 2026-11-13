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
| `whlz3nu2tryo3i` | 5090 | 20:21–21:10 | the full script again with the fix (29317cb + 6bdc2c2, `pod-fixed/`): real clips, finite PSNRs, 142 / 143 checks (below). The slower of the two 5090 pod kinds: its bf16 base step is 13.87 s against pod 3's 12.85 s, 8 % apart on the same code, the spread the previous record saw between pods (clocks were not logged on either). |

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

## Results

Pod 3 produced black frames but its kernels ran the same work on the
same inputs (the race corrupts a few weights, not the schedule), and the
fixed pod reproduces every timing 6–10 % slower across the board,
including the bf16 rows that this round did not touch: the two pods are
the two 5090 kinds of the previous record. The tables give the fixed
pod's numbers (`pod-fixed/*.log`, `.mem` for the 0.5 s VRAM samples);
pod 3's are in parentheses where they differ.

**FastH3-V2, 8 steps, 1344×768×124, seed 1**, stored MXFP8 + MXFP8
decoder + AdaLN tables + `--stream-blocks`, VSA 0.8, both with
`--vsa-qk8`:

| | load | prompt | **denoise** | per forward | decode | audio | **denoise + decode** | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| INT8 Q·Kᵀ (`--vsa-qk8`) | 28.3 s | 64.8 s | **54.4 s** (51.5) | 7.05, then 6.69–6.82 s (6.39–6.43) | 6.41 s | 0.19 s | **60.8 s** (57.7) | 18 016 MiB |
| + FP8 P·V (`--pv8`) | 26.7 s | 64.1 s | **52.0 s** (49.2) | 6.77, then 6.45–6.50 s (6.09–6.12) | 6.62 s | 0.19 s | **58.7 s** (55.3) | 18 592 MiB |
| ComfyUI `fasth3-v2-euler` (page) | | | 45.2 s sampler | 5.65 s | | | 54.1 s total | 30.0 GB |

PSNR of the `--pv8` clip against the `--vsa-qk8` clip: **21.7 dB**
(Y 17.2, U 32.5, V 34.8; `psnr.log`), the same band as INT8 Q·Kᵀ
against bf16 attention (17.5 dB in the previous record) and as MXFP8
against bf16 weights; the grids (`v2-*.grid.png`) are the same scene,
the same motion, no artifacts. Per-kernel (`v2-pv8-profile.log`, the
profiler's per-op syncs cost 1.2 s per forward): the fine stage is
**27.6 ms per call** (3g, 400 calls, 25.2 % of the ops) against the
INT8-Q·Kᵀ-only 3e's 34.2 ms in the previous record's profile, plus
0.98 ms of V quantisation and 0.33 ms of V max per call (the Q/K
quantisation, 1.02 ms, is in both). The MXFP8 GEMMs are the other
half: 25.9 s of the 43.7 s of ops at 500 TFLOP/s.

**Raw H3, 25 uniform euler steps, dense, 124 frames**, MXFP8 at load +
MXFP8 decoder + tables + `--stream-blocks` (ComfyUI's `baseline` row:
pruned int8 weights, Kitchen int8 attention, 181.7 s total / 169.0 s
sampler = 6.8 s per step):

| attention | **denoise** | per step | decode | **denoise + decode** | peak VRAM | PSNR vs bf16 attention |
|---|---|---|---|---|---|---|
| bf16 `attn_mma2` | 346.0 s (322.3) | 13.86, then 13.66–13.88 s (12.82–12.90) | 6.42 s | 352.4 s (328.4) | 15 804 MiB | |
| INT8 Q·Kᵀ (`--qk8`) | 258.3 s (237.7) | 10.43, then 10.17–10.38 s (9.46–9.51) | 6.39 s | 264.7 s (243.8) | 16 476 MiB | 22.1 dB |
| + FP8 P·V (`--pv8`) | **215.9 s** (197.5) | 8.77, then 8.49–8.67 s (7.83–7.85) | 6.37 s | **222.3 s** (203.5) | 16 636 MiB | **24.7 dB** (22.2 dB against `--qk8`) |

The attention call (`base2-*-profile.log`, 2 steps under the profiler,
100 calls over the two profiled steps, 38k × 38k × 128 each): bf16 181.3 ms
(previous record) → INT8 Q·Kᵀ **119.5 ms** on this pod (112.3 on the
previous record's pod) → INT8 Q·Kᵀ + FP8 P·V **82.1 ms** (81.6 on
pod 3), 1.46× the INT8-only call on the same pod, with 0.93 ms of V
quantisation and 0.34 ms of V max per call. That is the ~80 ms per call
the previous record inferred for ComfyUI's Kitchen attention from its
6.8 s step. Attention is now 52 % of the dense step's ops (was 61 %,
72 % in bf16); the rest is the MXFP8 GEMMs at 478 TFLOP/s (6.2 s of the
15.6 s of ops over two steps: 39 % of the step) and 1.3 s of
norms / rope / SwiGLU / quantisation, so the dense step is 8.66 s on this
pod against ComfyUI's 6.8 s: 1.27× behind (1.15× on pod 3), from 1.52×
with INT8 Q·Kᵀ and 2.0× in bf16. The remaining gap is on the GEMM side
(ComfyUI's int8 GEMMs against our MXFP8 ones with the per-call
activation quantisation and the `×2` FFN-up split) and in the
non-overlapped small kernels, not in attention any more.

Memory: FP8 P·V adds the E4M3 V staging and its scales, 576 MiB on the
V2 clip and 160 MiB on the base; the base's 25-step peak is 16 636 MiB;
the 32 GB card runs every row streamed with room to spare.

`--pv8` stays opt-in like `--qk8` / `--vsa-qk8`: the 8-bit attention
path is a speed recipe, and every variant of it moves the sample for a
seed by the same ~20 dB band; the V2 best recipe on a 5090 is the
previous record's plus `--pv8`.

## The check on this pod: 142 / 143

`attention qk8+pv8 h2 kv2 130x65 d64 causal=false` came in at 3.37e-2
of the oracle (tolerance 3e-2); it was 1.87e-2 on pod 3, before 6bdc2c2
moved the random stream of the cases after the reshaped causal ones.
The band itself (every pv8 case sits at 2–3e-2 where the qk8 cases sit
at 1e-2 and bf16 attention at 5e-3) is not the kernel's: the checker
hands both sides f32 inputs, the device rounds them to bf16, and the
oracle quantises V to E4M3 from the f32 values while the kernel
quantises the bf16 ones. bf16 moves a value by up to 2⁻⁹ and E4M3's
rounding boundaries are 2⁻⁴ apart, so one V element in 32 lands on a
different code; on the row's dominant key, in a channel of the top
binade, one such flip is 0.1 × 0.25 / max |o| ≈ 1–3e-2 of this metric.
The fix feeds the pv8 cases bf16-representable inputs, so
both sides quantise the same values, and lets the oracle report where
its P rounding sits within a few f32 ulps of an E4M3 boundary, so a
kernel whose scores differ from the oracle's in the last bits is not
charged for the flip. It also makes the oracle's int8 code arithmetic the kernels' (`x · (1/s)`
rather than `x / s`: one code in a million differed, and a code flip
moves a score by ~3e-3, enough to re-round P). With it the CPU and
lavapipe checks are 143 / 143 (lavapipe's pv8 cases at 1e-7 of the
oracle, bit-faithful emulation); the CUDA check ran again on the change, on a
kernel-check pod without the volume (`rp.py create … - -`): **143 / 143
on an RTX 4000 Ada** (sm_89, driver 595.91.07, 130 W; RunPod
`76c8crm2zg1yyh`, EUR-IS-2, $0.28/h, 21:34–21:45 UTC;
`check-ada/output-1.txt`). The dense pv8 cases went from 2.1–3.4e-2 to
2.8–3.1e-3 and the d64 / d128 VSA pv8 cases from 0.2–2.3e-2 to
2.1–3.1e-3 of the oracle: the kernels were faithful all along, the
checker was not. The d32 VSA pv8 cases stayed at 0.5–2.3e-2, because
there is no FP8 fine kernel below d64 and that path had ignored `pv8`
(bf16 P·V against the FP8 oracle, inside the tolerance by luck); it now
runs the host oracle on CUDA like the dense path does
(`CuVsa::host_pv8`); the CUDA check of that tree is the next entry below.

CUDA_RECHECK2_TBD

Cost of the round: eight pods, ~$6.5 (five 5090 pods at $1.39/h, 5 min
to 2 h each, the 4090 at $0.89/h for 27 min, the kernel-check Ada pods at
$0.28/h for ~10 min each); the pod that failed to restart was the
expensive one.
