# 2026-11-25: worker cold start with per-job staging; WebGPU generate peak with the staged-upload flush (RunPod Server Edition)

Pod `m8ufs7eonva8r3`: RTX PRO 6000 Blackwell Server Edition (sm_120),
600 W, 2430 MHz, driver 595.112.07, EU-RO-1, the EU weights volume at
`/workspace`, 400 GB container disk. No trees were staged at boot, so the
volume was cold for the first command. Binary c4dc316 (`h3 worker` with
`stage`). It terminated itself. Commands are `cmd1`–`cmd3.sh`; the worker
jobs are `job-staged.json` and `job-volume.json`; outputs are
`out1`–`out3.txt`, the job results (`*.json`, `*.err`) and the
`generate` logs and memory samples.

## Cold start: `h3 worker` with a staged job

The job is the draft from the serverless validation
(`2026-11-18-runpod-container`): 480p, `--weights mxfp8 --stream-blocks
--pv8`, live prompt (bf16 Qwen3-VL), seed 7. It ran through `h3 worker
--test-input`. `stage` lists the h3-base transformer, vae, audio_vae,
text_encoder and tokenizer plus the plug4 adapter and tables, `H3_STAGE_FROM
/workspace` → `H3_STAGE_TO /root/stage`.

| run | stage | load | prompt | denoise + decode | job total |
|---|---|---|---|---|---|
| serverless worker, from the volume (2026-11-18) | — | 296.7 s | 283.1 s | 27.7 s | **626 s** |
| **cold, staged first** | **147.0 GB in 48.1 s (3.05 GB/s)**, 53 files | 74.9 s | 12.9 s | 11.8 s | **162 s** |
| warm worker, staged again | 0 bytes copied, 0.02 s | 62.6 s | 11.8 s | 11.7 s | **100 s** |
| from the volume, page cache now warm | — | 252.3 s | 118.7 s | 11.8 s | **397 s** |

- **Cold start: 3.9× faster.** The staged cold job takes 162 s against 626 s.
- **The baseline favours the volume.** Its rows read the volume after the
  staging run had warmed the page cache, so the volume had every advantage
  and still took 397 s.
- **Different hardware:** the 2026-11-18 serverless worker was an RTX 4090
  with its own disk. Denoise and decode are not comparable across the
  cards. Load and prompt are dominated by the volume either way.
- **Disk speed:** the container disk's speed on a serverless worker is not
  measured. A serverless round should confirm the copy rate there.

## WebGPU `generate`: the staged-upload flush on real weights

The draft variant, `--embeds` (gallery prompt), `--tables`, int4, seed 7,
WebGPU on NVIDIA Vulkan.

| run | peak GPU | load | per step | denoise | decode |
|---|---|---|---|---|---|
| 480p resident, the flush (default) | **27.3 GB** | 90.2 s | 41.0 s | 164.5 s | 33.2 s |
| 480p resident, `H3_WGPU_STAGE_FLUSH_MB=0` | 47.3 GB | 89.8 s | 41.0 s | 164.5 s | 33.1 s |
| 288p streamed + `--ffn-chunk 4096`, the flush | **11.3 GB** | 78.4 s | 11.3 s | 45.8 s | 16.6 s |

**The spike is gone.** The 480p peak went 47.3 → 27.3 GB at the same speed,
and the clip is byte-identical. The 288p streamed run held 16.2 GB in
`2026-11-24-runpod-roundC` and now holds 11.3 GB.

**The new kernels show in a full run.** `linear4` / `flash3` took the
WebGPU 480p step from 54.2 to 41.0 s (round C), and the released-VAE decode
from 51 to 33 s.

WebGPU is still about 2.4 GB above CUDA at 288p (8.9 GB). That gap is the
f32 activations.
