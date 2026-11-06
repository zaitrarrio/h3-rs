# Running on Vast.ai without SSH

Ported from fastwan-rs; the same R2 command channel. The box polls for
numbered command scripts and uploads each one's output:

1. `make_boot.py BIN_KEY` writes the bootstrap to `h3-runs/<run>/boot.sh` and
   prints the instance's `onstart` line. It pulls the mirrored
   `MiniMaxAI/MiniMax-H3` tree (`H3_WEIGHTS_PREFIX`) and `taeh3.safetensors`
   into `/w/h3`, then the `h3` binary from `<run>/BIN_KEY`.
2. Rent with `vast_api.py` (`PUT /asks/<offer>/`, image
   `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, `runtype: ssh`, env
   `NVIDIA_DRIVER_CAPABILITIES=all`, disk ≥ 250 GB: the base tree is 95 GB
   with the text encoder). Arm a destroy timer immediately.
3. Queue work with `rr.py cmd N file.sh`; read it back with `rr.py out N`.
4. Destroy (`DELETE /instances/<id>/`) and check the credit.

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
No run has been made yet: this directory is the plan, not a result.
