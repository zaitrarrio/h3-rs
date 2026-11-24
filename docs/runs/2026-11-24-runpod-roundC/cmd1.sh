. /root/common.sh
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,memory.total --format=csv
echo "== check cuda"; timeout 900 $B check --backend cuda 2>&1 | grep -E "w4a4|int4.*streamed ==|checks|FAIL|Error"
echo "== check wgpu"; timeout 1800 $B check --backend wgpu 2>&1 | grep -E "NVIDIA|llvmpipe|w4a4|checks|FAIL|Error" | head -12
mkdir -p /root/m; for c in transformer vae audio_vae text_encoder tokenizer; do ln -sfn /w/h3-base/$c /root/m/$c; done
( time timeout 900 $B encode --backend cuda --model-dir /root/m --prompt "$FOX" --out-dir /root/g ) 2>&1 | tail -2
cp /w/open-h3-regenerate-2k/vae_fast/fast_h3_vae_noncausal_24l.safetensors /root/vf.safetensors
ffmpeg -loglevel error -y -i $A/fox.y4m -vf scale=2560:1440:flags=lanczos -pix_fmt yuv420p -strict -1 /root/fox2k.y4m
echo "== 2K fast decode, host spans (H3_PROFILE=1), then plain x2"
( export H3_PROFILE=1; rt prof2k --input /root/fox2k.y4m --fast-vae /root/vf.safetensors --skip-base --repeat 1 ) | grep -E "exit|peak|fast vae host|fast decode|ops; GEMMs"
rt plain2k --input /root/fox2k.y4m --fast-vae /root/vf.safetensors --skip-base --repeat 2 | grep -E "exit|peak|fast decode|fast vs"
