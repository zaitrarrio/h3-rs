. /root/common.sh
echo "== WebGPU int4 drafts with the in-flight bound"
E=$(ls /root/g/*.safetensors | head -1)
P="--variant draft --model-dir /root/m --variant-dir /workspace/h3-rs/plug4 --tables /workspace/h3-rs/plug4/plug4.tables --embeds $E"
runb wgpu w288-int4-stream-ffn $P --tier 288p --weights int4 --stream-blocks --ffn-chunk 4096
runb wgpu w480-int4 $P --tier 480p --weights int4
runb wgpu w480-int4-stream-ffn $P --tier 480p --weights int4 --stream-blocks --ffn-chunk 4096
psnr w480-int4 w480-int4-stream-ffn
strip w480-int4
