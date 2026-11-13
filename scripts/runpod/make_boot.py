"""Write and upload a RunPod pod's bootstrap script; print the start line.

    H3_RUN=runpod-20261112-1200 python3 make_boot.py BIN_KEY

The pod runs `bash -c "<start line>"` (rp.py create) on
nvidia/cuda:12.8.1-runtime-ubuntu24.04 with the weights volume at
/workspace. The bootstrap installs curl / Vulkan / X11-GL libs / ffmpeg,
fetches the h3 binary from `<run>/BIN_KEY` in the h3-runs bucket, links
/w/<tree> to /workspace/weights/<tree> for every tree there (`stage TREE
[SUB...]` in a command copies a tree, or parts of it, onto the container
disk and repoints the link: loads from the network volume run at
0.25-0.5 GB/s, the local disk is NVMe), then runs
`<run>/cmd/N.sh` for N = 1..40 as they appear (rr.py cmd N FILE),
uploading `<run>/out/N.log` after each (rr.py out N). Weights are fetched by
commands built with fetch_plan.py (idempotent, size-checked, add-only on the
volume). The boot log is `<run>/boot.log` (rr.py log). The pod never exits
on its own: a command terminates it through the REST API with the key in its
env (`terminate` below, which first uploads the command's output so far), and
`H3_POD_HOURS` (default 4) arms a backstop that does the same.
"""
import os, shlex, sys
exec(open(os.path.join(os.path.dirname(__file__), "..", "vast", "r2.py")).read())
run = os.environ["H3_RUN"]; B = os.environ.get("H3_BUCKET", "h3-runs"); EXP = 6 * 3600
hours = float(os.environ.get("H3_POD_HOURS", "4"))
get = lambda b, k: s3.generate_presigned_url("get_object", Params={"Bucket": b, "Key": k}, ExpiresIn=EXP)
put = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket": B, "Key": k}, ExpiresIn=EXP)
N = 40
L = ["#!/bin/bash", "set -u", "LOG=/root/boot.log", "exec >>$LOG 2>&1",
     f"LOGPUT={shlex.quote(put(run + '/boot.log'))}",
     'up() { curl -s -X PUT --upload-file "$1" "$2" >/dev/null; }',
     # A command that terminates the pod would lose its own output: upload the in-flight out.log first.
     'terminate() { [ -n "${OUT_URL:-}" ] && up /root/out.log "$OUT_URL"; up $LOG "$LOGPUT"; '
     'curl -sS -X DELETE -A h3-rs-pod/1 -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"; }',
     # stage TREE [SUB...]: copy /workspace/weights/TREE (or only its SUB directories, the rest
     # symlinked back to the volume) onto the container disk with 8 parallel cp streams, then point
     # /w/TREE at the copy. Idempotent (cp -n). Prints bytes and seconds: the volume's read rate.
     'stage() { t=$1; shift; src=/workspace/weights/$t; dst=/root/w/$t; mkdir -p "$dst"; T0=$(date +%s); '
     'if [ $# -eq 0 ]; then subs=$(cd "$src" && ls -A | tr "\\n" " "); else subs="$*"; for e in "$src"/* "$src"/.[!.]*; do '
     '[ -e "$e" ] && ln -sfn "$e" "$dst/$(basename "$e")"; done; fi; '
     'for sub in $subs; do [ -L "$dst/$sub" ] && rm -f "$dst/$sub"; (cd "$src" && find "$sub" -type d -exec mkdir -p "$dst/{}" \\; && '
     'find "$sub" -type f -print0 | xargs -0 -P 8 -n 2 cp --update=none --parents -t "$dst"); done; '
     'b=$(cd "$src" && du -sbL $subs | awk \'{s+=$1} END {print s}\'); dt=$(( $(date +%s) - T0 )); '
     'bad=$(cd "$src" && find $subs -type f -printf "%s %p\\n" | while read z f; do [ "$(stat -c %s "$dst/$f" 2>/dev/null)" = "$z" ] || echo "$f"; done | wc -l); '
     'echo "staged $t [$subs]: $b bytes in $dt s ($(( b / (dt > 0 ? dt : 1) / 1000000 )) MB/s), $bad files wrong"; ln -sfn "$dst" "/w/$t"; }',
     'export -f up terminate stage; export LOG LOGPUT',
     f'(sleep {int(hours * 3600)}; echo "backstop: terminating after {hours} h"; while :; do terminate; sleep 120; done) &',
     '(while true; do sleep 20; up $LOG "$LOGPUT"; done) &',
     'echo "== boot $(date -u) pod $RUNPOD_POD_ID"', "nvidia-smi", "free -g; nproc; df -h / /workspace",
     "export DEBIAN_FRONTEND=noninteractive",
     "apt-get update -qq && apt-get install -y -qq curl ca-certificates libvulkan1 vulkan-tools ffmpeg "
     "libx11-6 libxext6 libegl1 libgl1 libglvnd0 >/dev/null && echo 'apt ok'",
     "mkdir -p /w /workspace/weights /workspace/h3-rs",
     'for t in /workspace/weights/*/; do t=${t%/}; ln -sfn "$t" "/w/$(basename "$t")"; done; ls -la /w',
     f"curl -sf -o /w/h3-bin {shlex.quote(get(B, run + '/' + sys.argv[1]))} && chmod +x /w/h3-bin && echo 'binary ok'",
     'up $LOG "$LOGPUT"', "echo READY",
     "CMD=(" + " ".join(shlex.quote(get(B, f"{run}/cmd/{i}.sh")) for i in range(1, N + 1)) + ")",
     "OUT=(" + " ".join(shlex.quote(put(f"{run}/out/{i}.log")) for i in range(1, N + 1)) + ")",
     f"for i in $(seq 0 {N - 1}); do",
     '  until curl -sf -o /root/cmd.sh "${CMD[$i]}"; do sleep 10; done',
     '  echo "== cmd $((i+1)) $(date -u)"',
     '  export OUT_URL="${OUT[$i]}"; (cd /w && timeout 7200 bash /root/cmd.sh) > /root/out.log 2>&1; echo "exit $?" >> /root/out.log',
     '  up /root/out.log "${OUT[$i]}"; up $LOG "$LOGPUT"', "done",
     "echo 'all 40 commands ran'; terminate; sleep 600"]
path = "/tmp/h3-runpod-boot.sh"
open(path, "w").write("\n".join(L) + "\n")
s3.upload_file(path, B, run + "/boot.sh")
url = get(B, run + "/boot.sh")
print("export DEBIAN_FRONTEND=noninteractive; (apt-get update -qq && apt-get install -y -qq curl ca-certificates) >/root/pre.log 2>&1; "
      f"curl -sf -o /root/boot.sh '{url}' && bash /root/boot.sh; sleep 3600")
