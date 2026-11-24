# RunPod runs

Same R2 command channel as `scripts/vast/` (`rr.py cmd/out/log/upload/download`),
with the weights on a RunPod network volume instead of a fresh download per
rental. `rp.py` talks to `rest.runpod.io/v1` with `$RUNPOD_API_KEY`;
`make_boot.py` writes the pod's bootstrap (links `/w/<tree>` to
`/workspace/weights/<tree>`, then runs the queued commands);
`fetch_plan.py` prints size-checked fetch lines for trees that are not on
the volume yet (R2 `fv-weights` or Hugging Face).

    export H3_RUN=runpod-$(date -u +%Y%m%d-%H%M)
    python3 scripts/vast/rr.py upload target/release/h3 h3-bin
    START=$(python3 scripts/runpod/make_boot.py h3-bin)
    python3 scripts/runpod/rp.py create h3-rs "NVIDIA RTX PRO 6000 Blackwell Server Edition" EU-RO-1 bz0ofkfx45 "$START"
    # from the agent sandbox (no rest.runpod.io): GraphQL, retrying while out of stock
    echo "$START" > start.txt
    python3 scripts/runpod/pod.py "NVIDIA RTX PRO 6000 Blackwell Server Edition" EU-RO-1 bz0ofkfx45 h3-rs start.txt
    python3 scripts/vast/rr.py cmd 1 first.sh    # … rr.py out 1

Stage what a run loads: `H3_STAGE="h3-8step-mxfp8 h3-8step:text_encoder,tokenizer,vae,audio_vae"`
before `make_boot.py` copies those onto the container disk at boot (the
V2 recipe's 103 GB in 70 s) and the loads then run at NVMe speed; from
the volume the mmap loader reads at 0.2 GB/s
(`docs/runs/2026-11-13-runpod-pro6000-stage`).

The volume `fv-weights-h3-ltx-500` (EU-RO-1) is shared with fastvideo-rs,
whose fetcher lays trees out as `/workspace/weights/<tree>/` with
`.complete` and `.fv-revision` markers; writes there are add-only (never
move or delete a tree). `stage TREE [SUB...]` (exported by the boot) copies a tree,
or only the named subdirectories with the rest symlinked back, onto the
container disk with eight parallel `cp` streams and repoints `/w/TREE`;
size the container disk for it at `rp.py create` (last argument, GB). A
command terminates the pod with `terminate`
(exported by the boot: REST, then GraphQL, logging which one answered; the
account key reaches the pod as `H3_RUNPOD_KEY`, because RunPod replaces a
`RUNPOD_API_KEY` env with its own pod-scoped key, which could not terminate
the pod); the boot's backstop terminates it after
`H3_POD_HOURS` (default 4) regardless.
