# 2026-11-18: Qwen3-VL text encoder placements, int8, prompt cache (RTX PRO 6000 Server Edition)

RunPod pod, RTX PRO 6000 Blackwell Server Edition (sm_120, 600 W, driver
595.112.07, 97.9 GB), 755 GB RAM, the `fv-weights-h3-ltx-500` volume
(EU-RO-1). `h3-base` staged onto the container disk first (144 GB in 76 s,
1895 MB/s), except the first row. Six text prompts (19–33 tokens), `h3
encode --backend cuda`. Binary c9f8411 (memory-mapped reads) and ab78342
(parallel `pread`, the only change). The 50 language layers are 45.4 GB
in bf16 and 23.4 GB in int8.

## Per placement

| `--encoder` / `--encoder-weights` | open | per prompt (c9f8411, mmap) | per prompt (ab78342, pread) |
|---|---|---|---|
| stream, bf16, straight from the network volume | 0.2 s | 166.0 s cold, then 34.2 s | not measured |
| stream, bf16 (staged copy) | 0.2 s | 36.6–43.7 s | **6.4–9.4 s** (6.5 typical) |
| host, bf16 (page-locked RAM copy) | 41.6 s → **7.0 s** with pread | 0.9–1.0 s | 0.9–1.0 s |
| resident, bf16 | 35.1 s | 0.0–0.1 s | |
| stream, int8 from the `quantize-encoder` tree | 0.1 s | 7.3–8.6 s | **3.4–3.6 s** |
| host, int8 tree | 6.5 s | 0.5–0.6 s | |
| resident, int8 tree | 7.0 s | 0.1–0.2 s | |
| resident, int8 quantized at load (no tree) | 103.7 s → 75.7 s with pread | 0.1–0.2 s | |

`h3 quantize-encoder` wrote the int8 tree (24 GB) in 111 s. Stream
placement reads layer l + 1 on a thread while layer l computes, and
uploads it on the side queue. With mmap reads it spent most of its time
in the kernel (bf16: 6 min 53 s of sys time for six prompts) on page
faults and a second copy. `pread` in 8 MB chunks on the rayon pool,
straight into the final buffer (a fused linear's parts back to back),
reads the 45.4 GB in ~6.5 s from the page cache, about 7 GB/s.

## Exactness

The embeddings were compared bit for bit with bf16 resident
(`prompt_embeds`, f16 rows of 5120):

- Every bf16 placement, read either way (mmap or pread), from the volume
  or the staged copy, and auto, gives **identical** bits.
- Every int8 placement (resident / host / stream, the stored tree or
  quantized at load, mmap or pread) gives identical bits to every other
  int8 placement.
- int8 against bf16: relative error ≤ 8.0·10⁻³ per prompt, every token's
  cosine ≥ 0.99974.

## `h3 serve` with `--encoder auto` and `--prompt-cache`

The draft variant (LongLive-Plug 4 forwards), `--weights mxfp8
--decoder-weights mxfp8 --pv8`, the plug4 tables, 832×480×124 at seed 7.
The encoder opens last, so auto sees the free memory after the DiT and
decoders are loaded: 68.4 GB.

| run | auto picks | encoder open | GPU used | job | encode | prompt source | total |
|---|---|---|---|---|---|---|---|
| int8 | Resident (23.4 GB + 16 reserve ≤ 68.4) | 10.0 s | 51.2 GB | fox | 0.19 s | encoder | 10.97 s |
| | | | | fox again | 0.2 ms | cache (memory) | 10.54 s |
| | | | | night market | 0.13 s | encoder | 10.79 s |
| bf16 | Resident (45.4 + 16 ≤ 68.4) | 41.6 s | 73.8 GB | fox | 0.11 s | encoder | 10.72 s |
| | | | | night market | 0.06 s | encoder | 10.62 s |
| int8, restarted | Resident | 10.8 s | | fox | 0.7 ms | cache (disk) | 10.70 s |

Engine ready 112 s (int8) / 126 s (bf16) / 96 s (int8, warm page cache)
after start. The prompt is no longer part of a job's time: before this
change `serve` streamed the encoder per prompt (73–208 s,
`docs/runs/2026-11-18-runpod-container`).

`serve-clips-bf16-int8.jpg` (frames 0 / 60 / 120, top to bottom: bf16
fox, int8 fox, int8 fox again, bf16 night market, int8 night market): the
same scene and composition for bf16 and int8. Clip PSNR, int8 against
bf16: fox 14.8 dB, night market 13.5 dB. That cannot be told apart from
run-to-run rounding. The same int8 embeddings from the memory cache, as
a process's second job, sit 13.8 dB from its first job. The restarted
process's first job, from the disk cache, is **byte-identical** to the
first process's first job. The difference is cuBLASLt's candidate timing
on each shape's first call (`H3_CUDA_LT_NTH` pins it), not the cache.

## Notes

- The pod's self-terminate failed: `rest.runpod.io` did not resolve on
  this pod. It was terminated through the GraphQL API from the agent
  sandbox. `make_boot.py`'s `terminate` now falls back to GraphQL.
- Default left as is: `--encoder auto` resolves to resident for `serve` /
  `encode` when the device has room, and to stream for one-prompt runs.
  The format stays bf16 unless `--encoder-weights int8` is given (the
  embeddings move ~1 %).
