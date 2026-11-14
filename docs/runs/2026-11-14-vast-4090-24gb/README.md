# 2026-11-14: FastH3-V2 on a 24 GB card (RTX 4090, Vast)

The open "small card" question from `docs/design.md` item 7: the 5090
peaked at 15.9–18 GB streamed at 124 frames, so a 24 GB card should run
the V2 recipe with `--stream-blocks`. Two Vast RTX 4090 boxes, the second
one is the record.

## Boxes

- **60675502** (California, driver 615.71.09, **260 W** power limit, EPYC
  7742): the advertised 5 Gb/s link delivered ~23 MB/s, 95 GB in 73 min
  with 12 lost transfers; the setup command's refetch loop was cut off by
  the boot's 90-minute per-command budget and a transformer shard stayed
  truncated (`incomplete metadata, file not fully covered`). Only the
  checker ran: 168 / 168 (the binary before the split-decided cases),
  plain and under `H3_STREAM_STRESS=1`, on sm_89. Destroyed.
- **60692904** (the Netherlands, offer 58067846, driver 595.99.02,
  **450 W**, PCIe 4 ×16, AMD EPYC, 24 564 MiB): 138 GB (the whole
  `h3-8step` tree, text encoder included) in 990 s with no lost transfer;
  V2 AdaLN tables built on the box in 6 m 09 s; **170 / 170** checks
  plain, under `H3_STALL=1` and under `H3_STREAM_STRESS=1`. Binary
  2ec8e13 (the probe fix and the split-decided cases). Commands split so
  that no single one holds the weights wait and the work.

## Results (samurai prompt, 1344×768×124, seed 1, `--stream-blocks`, INT8 Q·Kᵀ + FP8 P·V, tables)

| | bf16 weights, bf16 decoder | `--weights mxfp8 --decoder-weights mxfp8` (emulated GEMMs on sm_89) |
|---|---|---|
| peak device memory | **19.1 GB** | 19.9 GB |
| forward | 13.36–13.39 s | 13.33–13.37 s |
| denoise (8 forwards) | **107.0 s** | 106.8 s |
| video decode | **17.4 s** | 20.0 s |
| total after load | **125.7 s** | 128.0 s |
| load / prompt (host text encoder) | 189 s / 188 s | 184 s / 189 s |
| prefetch | compute stream waited 249 ms in 400 blocks; copies 6.4 GB/s pageable-class on PCIe 4 | 272 ms; 3.8 GB/s |
| clip | the samurai (`v2-bf16.grid.png`) | 20.4 dB from the bf16 clip: the 8-bit band |

So **V2 runs on 24 GB** at 1344×768×124 with 4–5 GB to spare, and the
streaming is as free on a PCIe 4 ×16 link as on the 5090's PCIe 5: the
GPU waited 0.6 ms per block. On Ada there is no FP8 block-scaled GEMM, so
MXFP8 weights only save host memory (the device form is still bf16 per
GEMM through the emulation) and cost 2.6 s of decode: **bf16 weights are
the 24 GB recipe** on this generation; `--vsa-qk8 --pv8` run natively
(sm_89 has the INT8 and FP8 tensor cores).

Per-kernel profile of the bf16 streamed forward (`v2-bf16-prof.log`,
`H3_PROFILE=1`, 14.8 s per forward synced per op): the bf16 GEMMs 80.3 s
of 104.5 s in ops at **161 TFLOP/s** (FFN-up 73.4 ms, QKV 53.8, FFN-down
36.3, gate 18.6, out 18.5 ms per call), the fine stage 38.0 ms per call
(15.2 s, 14.6 %), the memory-bound kernels ~1.1 s per forward
(head_split 1.28 ms, swiglu 3.5, rms_norm 1.4, coarse 2.65, pool 1.9,
8-bit quantisations 3.0, gate_residual 1.5, k mean 0.67, v max 0.59 ms
per call). The 4090 is GEMM-bound at 77 % of the forward: the dense bf16
tensor-core rate is what a 24 GB clip costs, 2.1 min of denoise against
the 5090's 43 s and the Server Edition's 37 s.

Cost: the first box ~$2.4 (download at $0.016 / GB plus 2.5 h), the
second ~$0.9 (0.4 h at $0.46 / h plus 138 GB at $0.004 / GB).
