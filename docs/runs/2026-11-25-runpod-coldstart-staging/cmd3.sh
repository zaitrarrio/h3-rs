. /root/common.sh
echo "== WebGPU generate memory with the staged-upload flush (480p int4 resident; 0 = old behaviour)"
S=/root/stage; mkdir -p /root/m; for c in transformer vae audio_vae text_encoder tokenizer; do ln -sfn $S/weights/h3-base/$c /root/m/$c; done
( time timeout 900 $B encode --backend cuda --model-dir /root/m --prompt "$FOX" --out-dir /root/g ) 2>&1 | tail -2
E=$(ls /root/g/*.safetensors | head -1)
P="--variant draft --model-dir /root/m --variant-dir $S/h3-rs/plug4 --tables $S/h3-rs/plug4/plug4.tables --embeds $E"
runb wgpu w480-int4 $P --tier 480p --weights int4
( export H3_WGPU_STAGE_FLUSH_MB=0; runb wgpu w480-int4-noflush $P --tier 480p --weights int4 )
same w480-int4 w480-int4-noflush
runb wgpu w288-int4-stream-ffn $P --tier 288p --weights int4 --stream-blocks --ffn-chunk 4096
