B=/w/h3-bin; A=/workspace/h3-rs/assets
FOX="A red fox trots through fresh snow at dawn in a quiet pine forest; snow crunches softly under its paws and a light wind moves through the branches."
FAST="--weights mxfp8 --decoder-weights mxfp8 --pv8"
mkdir -p /root/res
# run NAME ARGS... : h3 generate to /root/res/NAME.y4m, GPU memory sampled every 0.5 s
run() { n=$1; shift; echo "== $n"
  (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > /root/res/$n.mem 2>/dev/null &); sleep 0.3
  ( time timeout 2400 $B generate --backend cuda --seed 7 --out /root/res/$n.y4m "$@" ) > /root/res/$n.log 2>&1; echo "exit $?"
  pkill -f "memory.used"; echo "peak GPU $(sort -n /root/res/$n.mem | tail -1) MiB"
  grep -vE "^\s*$" /root/res/$n.log | grep -iE "load|prompt|denoise|step|decode|total|adapter|tables|error|panic|real|forward|wrote" | tail -14 | cut -c1-200; }
strip() { ffmpeg -loglevel error -y -i /root/res/$1.y4m -vf "select=not(mod(n\,24)),scale=320:-1,tile=6x1" -frames:v 1 /root/res/$1-strip.jpg; ffmpeg -loglevel error -y -i /root/res/$1.y4m -c:v libx264 -crf 22 -pix_fmt yuv420p /root/res/$1.mp4; }
psnr() { ffmpeg -i /root/res/$1.y4m -i /root/res/$2.y4m -lavfi psnr -f null - 2>&1 | grep -o "PSNR.*" | sed "s/^/$1 vs $2: /"; }
# rt NAME ARGS... : h3 vae-roundtrip, GPU memory sampled every 0.5 s
rt() { n=$1; shift; echo "== $n"
  (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > /root/res/$n.mem 2>/dev/null &); sleep 0.3
  ( time timeout 3000 $B vae-roundtrip --backend ${RT_BACKEND:-cuda} --model-dir /w/h3-base "$@" ) > /root/res/$n.log 2>&1; echo "exit $?"
  pkill -f "memory.used"; echo "peak GPU $(sort -n /root/res/$n.mem | tail -1) MiB"
  grep -vE "^\s*$" /root/res/$n.log | tail -16 | cut -c1-200; }
# vstrip FILE OUT: one frame every 24, 320 wide
vstrip() { ffmpeg -loglevel error -y -i $1 -vf "select=not(mod(n\,24)),scale=320:-1,tile=6x1" -frames:v 1 $2; }
# crop3 IN BASE FAST OUT X Y: frame 60, a 640x360 crop of each, stacked
crop3() { ffmpeg -loglevel error -y -i $1 -i $2 -i $3 -filter_complex "[0:v]select=eq(n\,60),crop=640:360:$5:$6[a];[1:v]select=eq(n\,60),crop=640:360:$5:$6[b];[2:v]select=eq(n\,60),crop=640:360:$5:$6[c];[a][b][c]vstack=3" -frames:v 1 $4; }
same() { if cmp -s /root/res/$1.y4m /root/res/$2.y4m; then echo "$1 == $2 (byte-identical)"; else echo "$1 != $2"; psnr $1 $2; fi; }
# job W H UPSCALE NAME: one /api/generate on the local serve, polled to the end
job() { id=$(curl -s -H 'Content-Type: application/json' -d "{\"prompt\":\"$FOX\",\"width\":$1,\"height\":$2,\"frames\":124,\"seed\":7,\"upscale\":$3}" http://127.0.0.1:7860/api/generate | grep -oE '"id":"?[A-Za-z0-9-]+' | sed -E 's/"id":"?//');
  echo "== $4 job $id"; for i in $(seq 1 600); do s=$(curl -s http://127.0.0.1:7860/api/jobs/$id);
    if echo "$s" | grep -qE '"stage":"(done|error|failed)"' && ! echo "$s" | grep -q '"up_s":null' ; then break; fi
    if echo "$s" | grep -qE '"stage":"(error|failed)"|"up_error":"'; then break; fi; sleep 1; done; echo "$s" | cut -c1-900; }
# runb BACKEND NAME ARGS...: generate on any backend, GPU memory sampled
runb() { be=$1; n=$2; shift 2; echo "== $n ($be)"
  (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > /root/res/$n.mem 2>/dev/null &); sleep 0.3
  ( time timeout 3600 $B generate --backend $be --seed 7 --out /root/res/$n.y4m "$@" ) > /root/res/$n.log 2>&1; echo "exit $?"
  pkill -f "memory.used"; echo "peak GPU $(sort -n /root/res/$n.mem | tail -1) MiB"
  grep -vE "^\s*$" /root/res/$n.log | grep -iE "load|step|denoise|decode|total|error|panic|real|wgpu|weights" | tail -12 | cut -c1-220; }
