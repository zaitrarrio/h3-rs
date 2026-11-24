. /root/common.sh
echo "== H3 drafts under ~9 GB: 4-bit weights, streamed blocks, FFN chunks; CUDA and WebGPU (Vulkan)"
E=$(ls /root/g/*.safetensors | head -1)
P="--variant draft --model-dir /root/m --variant-dir /workspace/h3-rs/plug4 --tables /workspace/h3-rs/plug4/plug4.tables --embeds $E"
runb cuda d480-bf16 $P --tier 480p
runb cuda d480-int4 $P --tier 480p --weights int4
psnr d480-bf16 d480-int4
runb cuda d480-int4-stream $P --tier 480p --weights int4 --stream-blocks
runb cuda d480-int4-stream-ffn $P --tier 480p --weights int4 --stream-blocks --ffn-chunk 4096
runb cuda d288-int4-stream-ffn $P --tier 288p --weights int4 --stream-blocks --ffn-chunk 4096
echo "== check wgpu"; timeout 1800 $B check --backend wgpu 2>&1 | grep -E "adapter|NVIDIA|llvmpipe|checks|FAIL" | head -4
runb wgpu w480-int4 $P --tier 480p --weights int4
psnr d480-int4 w480-int4
runb wgpu w480-int4-stream-ffn $P --tier 480p --weights int4 --stream-blocks --ffn-chunk 4096
runb wgpu w288-int4-stream-ffn $P --tier 288p --weights int4 --stream-blocks --ffn-chunk 4096
strip d480-bf16; strip d480-int4; strip w480-int4
