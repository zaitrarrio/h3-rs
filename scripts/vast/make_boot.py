"""Write and upload the rental's bootstrap script; print the onstart line.

    H3_RUN=vast-20261106-2330 H3_WEIGHT_SETS="weights/h3-8step/<sha>/=/w/h3,weights/auxiliary/tae/eu-20261011/=/w/tae" \
        python3 make_boot.py BIN_KEY

The bootstrap installs curl / Vulkan / X11-GL libs / ffmpeg, fetches the h3
binary from `<run>/BIN_KEY`, starts pulling every `prefix=dest` weight set of
`H3_WEIGHT_SETS` from the `fv-weights` bucket with presigned URLs in the
background (R2 egress is free; Vast bills the download per GB; `/w/weights.done`
appears when all are in; with the static `weights` CLI uploaded to
`<run>/bin/weights` (weights-rs's release binary) it is one ranged, parallel,
resumable transfer, otherwise one curl per file), then runs `<run>/cmd/N.sh` for N = 1..40 as they
appear, uploading `<run>/out/N.log` after each. Commands that need weights
wait for the marker. The X11/GL libraries matter: without them NVIDIA's
Vulkan ICD (libGLX_nvidia) fails to load and wgpu silently falls back to
llvmpipe.
"""
import os, shlex, sys
exec(open(os.path.join(os.path.dirname(__file__), "r2.py")).read())
run = os.environ["H3_RUN"]; B = os.environ.get("H3_BUCKET", "h3-runs"); EXP = 6 * 3600
# Mirror of the MiniMaxAI/MiniMax-H3 tree (vae/, audio_vae/, text_encoder/,
# tokenizer/) plus the variant's transformer under <SRC>/<variant>/; populate
# the bucket once with `huggingface-cli download` + `rr.py upload`.
SETS = [kv.split("=", 1) for kv in os.environ.get("H3_WEIGHT_SETS", "weights/minimax-h3/=/w/h3").split(",") if kv]
get = lambda b, k: s3.generate_presigned_url("get_object", Params={"Bucket": b, "Key": k}, ExpiresIn=EXP)
put = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket": B, "Key": k}, ExpiresIn=EXP)
files = []
for src, dst in SETS:
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket="fv-weights", Prefix=src):
        for o in page.get("Contents", []):
            rel = o["Key"][len(src):]
            files.append((get("fv-weights", o["Key"]), o["Size"], dst if rel == "" else dst.rstrip("/") + "/" + rel))
print(f"{len(files)} weight files", file=sys.stderr)
# The `weights` CLI (zaitrarrio/weights-rs) at <run>/H3_WEIGHTS_BIN, when
# uploaded: ranged downloads over 32 connections shared by every file (a
# 10 GB shard is no longer one curl stream), resumable. Without it the boot
# falls back to one curl per file.
WBIN = os.environ.get("H3_WEIGHTS_BIN", "bin/weights")
try:
    s3.head_object(Bucket=B, Key=f"{run}/{WBIN}")
    wbin_url = get(B, f"{run}/{WBIN}")
except Exception:
    wbin_url = None
print("weights CLI: " + ("yes" if wbin_url else f"no ({run}/{WBIN} not uploaded; curl per file)"), file=sys.stderr)
N = 40
L = ["#!/bin/bash", "set -u", "LOG=/root/boot.log", "exec >>$LOG 2>&1",
     f"LOGPUT={shlex.quote(put(run + '/boot.log'))}",
     'up() { curl -s -X PUT --upload-file "$1" "$2" >/dev/null; }',
     '(while true; do sleep 20; up $LOG "$LOGPUT"; done) &',
     'echo "== boot $(date -u)"', "nvidia-smi", "free -g; nproc; df -h /",
     "export DEBIAN_FRONTEND=noninteractive",
     "apt-get update -qq && apt-get install -y -qq curl ca-certificates libvulkan1 vulkan-tools ffmpeg "
     "libx11-6 libxext6 libegl1 libgl1 libglvnd0 >/dev/null && echo 'apt ok'",
     "mkdir -p /w",
     f"curl -sf -o /w/h3-bin {shlex.quote(get(B, run + '/' + sys.argv[1]))} && chmod +x /w/h3-bin && echo 'binary ok'",
     "cat > /root/files.txt <<'EOF2'"]
L += [f"{u} {z} {p}" for u, z, p in files]
L += ["EOF2",
      ("(T0=$(date +%s); "
       + (f"curl -sf -o /w/weights {shlex.quote(wbin_url)} && chmod +x /w/weights && "
          "{ /w/weights plan /root/files.txt -j 32 --progress 30 || echo 'FAILED weights plan'; } || " if wbin_url else "")
       + "awk '{print $1, $3}' /root/files.txt | xargs -P 8 -n 2 sh -c 'mkdir -p \"$(dirname \"$1\")\" && curl -sf --retry 5 -o \"$1\" \"$0\" || echo \"FAILED $1\"'; "
      'echo "weights $(du -sh /w | cut -f1) in $(( $(date +%s) - T0 ))s"; touch /w/weights.done) &'),
      'up $LOG "$LOGPUT"', "echo READY",
      "CMD=(" + " ".join(shlex.quote(get(B, f"{run}/cmd/{i}.sh")) for i in range(1, N + 1)) + ")",
      "OUT=(" + " ".join(shlex.quote(put(f"{run}/out/{i}.log")) for i in range(1, N + 1)) + ")",
      f"for i in $(seq 0 {N - 1}); do",
      '  until curl -sf -o /root/cmd.sh "${CMD[$i]}"; do sleep 10; done',
      '  echo "== cmd $((i+1)) $(date -u)"',
      '  (cd /w && timeout 5400 bash /root/cmd.sh) > /root/out.log 2>&1; echo "exit $?" >> /root/out.log',
      '  up /root/out.log "${OUT[$i]}"; up $LOG "$LOGPUT"', "done"]
path = "/tmp/h3-boot.sh"
open(path, "w").write("\n".join(L) + "\n")
s3.upload_file(path, B, run + "/boot.sh")
url = get(B, run + "/boot.sh")
print("export DEBIAN_FRONTEND=noninteractive; (apt-get update -qq && apt-get install -y -qq curl ca-certificates) >/root/pre.log 2>&1; "
      f"curl -sf -o /root/boot.sh '{url}' && nohup bash /root/boot.sh >/dev/null 2>&1 &")
