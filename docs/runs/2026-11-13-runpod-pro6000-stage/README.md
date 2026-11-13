# 2026-11-13: stage the weights onto the pod's NVMe, or load from the network volume?

Question (the owner's): why load from RunPod's network volume at all
instead of copying to the pod's local disk first? Three pods, same
recipe (FastH3-V2, stored MXFP8 + MXFP8 decoder + `--vsa-qk8` + tables,
1344×768×124, seed 7, the fox prompt), binary 4c12af4 (`h3-bin`) and a
build of 9aaac21 for the loader test (`h3-bin6`):

| pod | card | RAM | when (UTC) | what |
|---|---|---|---|---|
| `aclrjlfkrun7et` | RTX PRO 6000 Server Edition, 600 W | 2.2 TB | 02:38–02:48 | baseline from the volume, first staging attempt (`output-1-first-pod.txt`) |
| `xfngssig824fpl` | RTX 4090 | 124 GB | 02:50–02:56 | cold copy rates only (24 GB cannot run the recipe; `output-2-4090-cold-stage.txt`) |
| `ltbtexhz8uxmeo` | RTX PRO 6000 Server Edition, 600 W | 755 GB | 03:00–03:34 | cold stage, then the recipe from NVMe twice and from the volume; the madvise loader A/B (`output-3-server-cold-stage-and-madvise.txt`) |

All with a 250 GB container disk (`rp.py create … 250`), the volume
mounted as `mfs#euro.runpod.net:9421` (FUSE), driver 595.91.07.

## The volume is fast; the loader's reads from it are not

Cold reads of files no process on the pod had touched:

| read pattern | 4090 box | Server Edition box |
|---|---|---|
| one `cp`, a 5.3 GB shard | 694 MB/s | 772 MB/s |
| `stage` of the 25.6 GB MXFP8 file (one stream) | 775 MB/s | 853 MB/s |
| `stage` of 77.8 GB in 17 files, 8 `cp` streams | 1.52 GB/s | **1.94 GB/s** |
| the loader (mmap, page faults), the 25.6 GB file | | 0.21 GB/s (121.5 s) |

The whole V2 recipe (103 GB: the stored DiT, the text encoder's 14
shards, VAEs, tokenizer) stages in **70 s**. The 2026-11-12 record's
"0.25–0.5 GB/s" was the loader's rate, not the volume's.

## Same pod, same recipe: from the volume vs from the staged copy

| | from the volume | staged copy (page cache holds what the copy wrote) |
|---|---|---|
| stage | | 70 s, once |
| load (stored MXFP8 DiT + VAEs) | 121.5 s | 46.7 s, then 49.8 s |
| prompt (Qwen3-VL, 33 rows; the shards are read here) | 195.4 s | 85.8 s, then 72.3 s |
| **time to the first denoise step** | **317 s** | **203 s** the first run, **120–135 s** after |
| denoise / decode | 37.7 s / 5.6 s | 37.7 s / 5.6–5.9 s |

Staging wins from the first load (114 s saved including the copy) and
saves ~190 s on every later load in the same pod. The NVMe numbers
match Vast's local-disk pods (load 37–84 s, prompt 96 s) within the
page-cache band. Clip times are identical: the volume never touched the
denoise.

## A loader-side fix did not work

9aaac21 asked the kernel to read ahead (`MADV_SEQUENTIAL` per map at
open, `MADV_WILLNEED` per tensor). From the volume, on the same pod:

| | load | prompt |
|---|---|---|
| madvise loader | 166.9 s, 146.7 s | 224.4 s, 75.9 s |
| same binary, advice off (control) | 118.0 s | 185.0 s |

Slower (the third run's 75.9 s prompt is the one unexplained number;
after three reads of the 65 GB of shards on a 755 GB box the page cache
is the likely reason, and it does not rescue the load). FUSE's readahead
does not turn into the large parallel requests `cp` gets. The advice is
kept behind `H3_MADVISE=1` with these numbers; the default loader is
unchanged. Reading tensors with explicit large `pread`s on worker
threads would be the next loader-side attempt, if one is ever needed:
staging makes it moot on RunPod.

## Consequences

- `scripts/runpod/make_boot.py`: `stage TREE [SUB...]` (eight parallel
  `cp` streams, directories pre-created, every file size-checked against
  the volume: the first version raced on `mkdir` and silently dropped
  the tokenizer) and `H3_STAGE="TREE[:SUB,SUB] …"` to stage at boot,
  before READY. Size the container disk for it.
- The staged copy lives on the container disk, which is gone with the
  pod; the volume stays the durable store. One pod, one stage.
- Cost: three pods, ~$2.1; $30.45 → $27.70 on the account.
