. /root/common.sh
export H3_STAGE_FROM=/workspace H3_STAGE_TO=/root/stage
echo "== warm worker, staged again (nothing to copy)"
( time timeout 3000 $B worker --test-input /root/job-staged.json ) > /root/res/job-staged-warm.json 2> /root/res/job-staged-warm.err; echo "exit $?"
grep -E '"seconds"|"copied_bytes"|"exit_code"' /root/res/job-staged-warm.json | head -6
grep -E "staged|load|prompt|denoise|real" /root/res/job-staged-warm.err | tail -8 | cut -c1-200
echo "== the same job read from the volume (page cache now warm: optimistic for the volume)"
( time timeout 3000 $B worker --test-input /root/job-volume.json ) > /root/res/job-volume.json 2> /root/res/job-volume.err; echo "exit $?"
grep -E '"seconds"|"exit_code"' /root/res/job-volume.json | head -4
grep -E "load|prompt|denoise|real" /root/res/job-volume.err | tail -8 | cut -c1-200
