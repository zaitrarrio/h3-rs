. /root/common.sh
echo "== regen2k peak memory (denoise state freed before the decode)"
(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > /root/res/regen.mem 2>/dev/null &); sleep 0.3
( time timeout 3000 $B regen2k --backend cuda --regen-dir /w/open-h3-regenerate-2k --model-dir /w/h3-base --draft $A/fox.y4m --adaln-table $A/adaln_t_table.bin --prompt "$FOX" --out /root/res/regen.y4m ) > /root/res/regen.log 2>&1; echo "exit $?"
pkill -f "memory.used"; echo "peak GPU $(sort -n /root/res/regen.mem | tail -1) MiB"; grep -vE "^\s*$" /root/res/regen.log | tail -6
