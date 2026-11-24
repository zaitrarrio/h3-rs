. /root/common.sh
echo "== CUDA int4 drafts: streamed now packed + page-locked"
E=$(ls /root/g/*.safetensors | head -1)
P="--variant draft --model-dir /root/m --variant-dir /workspace/h3-rs/plug4 --tables /workspace/h3-rs/plug4/plug4.tables --embeds $E"
( export H3_CUDA_LT_NTH=0; runb cuda d480-int4 $P --tier 480p --weights int4 )
( export H3_CUDA_LT_NTH=0; runb cuda d480-int4-stream $P --tier 480p --weights int4 --stream-blocks )
same d480-int4 d480-int4-stream
runb cuda d480-int4-stream-ffn $P --tier 480p --weights int4 --stream-blocks --ffn-chunk 4096
runb cuda d288-int4-stream-ffn $P --tier 288p --weights int4 --stream-blocks --ffn-chunk 4096
( export H3_PROFILE=1; runb cuda d480-fastvae $P --tier 480p --fast-vae /root/vf.safetensors ) ; grep "fast vae host" /root/res/d480-fastvae.log
