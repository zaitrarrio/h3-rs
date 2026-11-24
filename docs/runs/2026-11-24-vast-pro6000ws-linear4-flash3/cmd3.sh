B=/w/h3-bin; mkdir -p /root/res
prof() { n=$1; shift; echo "== $n"; (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > /root/res/$n.mem 2>/dev/null &); sleep 0.3
  ( time timeout 2400 env "$@" ) > /root/res/$n.log 2>&1; echo "exit $?"; pkill -f memory.used; echo "peak GPU $(sort -n /root/res/$n.mem | tail -1) MiB"
  grep -vE "^\s*$" /root/res/$n.log | grep -vE "^\s+[0-9]+:|at /|backtrace|^(user|sys)" | head -24 | cut -c1-200; }
P480="--variant h3-plug4 --tier 480p --forwards 2"
P288="--variant h3-plug4 --tier 288p --forwards 2"
prof new480-int4 H3_PROFILE=1 $B bench --backend wgpu $P480 --weights int4
prof old480-int4 H3_PROFILE=1 H3_WGPU_LINEAR=3 H3_WGPU_FLASH=1 $B bench --backend wgpu $P480 --weights int4
