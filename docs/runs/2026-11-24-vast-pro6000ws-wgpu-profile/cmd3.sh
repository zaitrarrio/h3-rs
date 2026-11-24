B=/w/h3-bin; mkdir -p /root/res
P="--variant h3-plug4 --weights int4 --forwards 2"
prof() { n=$1; shift; echo "== $n"; (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > /root/res/$n.mem 2>/dev/null &); sleep 0.3
  ( time timeout 2400 env "$@" ) > /root/res/$n.log 2>&1; echo "exit $?"; pkill -f memory.used; echo "peak GPU $(sort -n /root/res/$n.mem | tail -1) MiB"
  grep -vE "^\s*$" /root/res/$n.log | grep -vE "^\s+[0-9]+:|at /|backtrace" | head -70 | cut -c1-200; }
prof w480-int4-plain $B bench --backend wgpu --tier 480p $P
prof w480-int4-prof H3_PROFILE=1 $B bench --backend wgpu --tier 480p $P
prof c480-int4-prof H3_PROFILE=1 $B bench --backend cuda --tier 480p $P
