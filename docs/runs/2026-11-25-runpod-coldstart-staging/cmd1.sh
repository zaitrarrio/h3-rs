. /root/common.sh
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,memory.total --format=csv; df -h / | tail -1
echo "== cold worker job, staged (the volume is cold)"
export H3_STAGE_FROM=/workspace H3_STAGE_TO=/root/stage
( time timeout 3000 $B worker --test-input /root/job-staged.json ) > /root/res/job-staged-cold.json 2> /root/res/job-staged-cold.err; echo "exit $?"
grep -E '"seconds"|"copied_bytes"|"bytes"|"files"|"exit_code"' /root/res/job-staged-cold.json | head -8
grep -E "staged|load|prompt|step|denoise|decode|real" /root/res/job-staged-cold.err | grep -vE "^\[out\] *$" | tail -14 | cut -c1-200
