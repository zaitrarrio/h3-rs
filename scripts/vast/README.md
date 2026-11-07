# Running on Vast.ai without SSH

Ported from fastwan-rs; the same R2 command channel. The box polls for
numbered command scripts and uploads each one's output:

1. `make_boot.py BIN_KEY` writes the bootstrap to `h3-runs/<run>/boot.sh` and
   prints the instance's `onstart` line. It fetches the `h3` binary from
   `<run>/BIN_KEY`, then pulls every `prefix=dest` of `H3_WEIGHT_SETS` from the
   `fv-weights` mirror in the background (`/w/weights.done` when complete).
   Mirrored there: `weights/h3-8step/<sha>/` (FastH3-8-Step-V2 full tree with
   text encoder and tokenizer, 138 GiB), `weights/h3-base/<sha>/` (MiniMax-H3),
   `weights/h3-ref2va/`, `weights/fasth3-trim-8step-nvfp4/<sha>/` (11 GiB
   single file), `weights/auxiliary/tae/eu-20261011/taeh3.safetensors`.
2. Rent with `rent.py OFFER "ONSTART"` (`PUT /asks/<offer>/`, image
   `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, `runtype: ssh`, env
   `NVIDIA_DRIVER_CAPABILITIES=all`, disk ≥ 250 GB: the V2 tree is 138 GiB
   with the text encoder). It prints the instance API key: arm the destroy
   timer on the box with it as the first command (`nohup setsid bash -c
   "sleep N && curl -X DELETE …"`) and queue a self-destroy as the last.
3. Queue work with `rr.py cmd N file.sh`; read it back with `rr.py out N`.
4. Destroy (`DELETE /api/v0/instances/<id>/`; the v1 path answers "Not
   found" for a delete) and check the credit. Offers: `POST /api/v0/bundles/`
   with the query as the body; instances: `GET /api/v1/instances/`.

Watchdog: `sleep N && <destroy>`, never `sleep N; <destroy>`.

Env: `VAST_API_KEY`, `CLOUDFLARE_API_KEY` (token with R2 read/write),
`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_TOKEN_ID`, `H3_RUN`. Needs boto3.

What to run first on a fresh card, cheapest and most decisive first:

```
timeout 1200 /w/h3-bin check --backend cuda
timeout 1800 /w/h3-bin bench --variant fasth3-v2 --backend cuda --layers 4 --steps 2
timeout 3600 /w/h3-bin adaln-tables --variant fasth3-v2 --model-dir /w/h3 --variant-dir /w/h3/fasth3-v2 --out /w/v2.tables
timeout 7200 /w/h3-bin generate --variant fasth3-v2 --model-dir /w/h3 --variant-dir /w/h3/fasth3-v2 --tables /w/v2.tables --weights int4-hqq-g64 --prompt "..." --out /w/out.y4m
```

Record the card, driver and power limit with every number (`docs/runs/`).
First run: `docs/runs/2026-11-06-vast-a6000-first-gpu`.
