B=/w/h3-bin; mkdir -p /root/res
prof() { n=$1; shift; echo "== $n"; (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > /root/res/$n.mem 2>/dev/null &); sleep 0.3
  ( time timeout 2400 env "$@" ) > /root/res/$n.log 2>&1; echo "exit $?"; pkill -f memory.used; echo "peak GPU $(sort -n /root/res/$n.mem | tail -1) MiB"
  grep -vE "^\s*$" /root/res/$n.log | grep -vE "^\s+[0-9]+:|at /|backtrace|^(user|sys)|x96 |x32 " | head -20 | cut -c1-200; }
P480="--variant h3-plug4 --tier 480p --forwards 2"
P288="--variant h3-plug4 --tier 288p --forwards 2"
prof f32-480-int4 H3_PROFILE=1 $B bench --backend wgpu $P480 --weights int4
prof attn-480-int4 H3_PROFILE=1 H3_WGPU_F16=attn $B bench --backend wgpu $P480 --weights int4
