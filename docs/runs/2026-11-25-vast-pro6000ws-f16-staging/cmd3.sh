echo "== staging probe: 6 GB of uploads, no compute, then a sync (nvidia-smi MiB)"
chmod +x /w/probe
for mb in 0 256 0 256; do echo "-- H3_WGPU_STAGE_FLUSH_MB=$mb"; H3_WGPU_STAGE_FLUSH_MB=$mb timeout 600 /w/probe --ignored wgpu_upload_staging_probe --nocapture 2>&1 | grep -E "staging probe|panicked|error|skipping"; done
