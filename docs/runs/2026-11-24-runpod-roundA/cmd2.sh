. /root/common.sh
E=$(ls /root/g/*.safetensors | head -1)
T="--variant fasth3-trim --model-dir /root/m --transformer /w/fasth3-trim-8step-nvfp4/diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors --width 1344 --height 768 --frames 124 --embeds $E"
( export H3_NVFP4_A4=1; run trim-a4-tuned $T )
( export H3_NVFP4_A4=1 H3_CUDA_LT_NTH=0; run trim-a4-cand0 $T )
psnr trim-a4-tuned trim-a4-cand0
strip trim-a4-tuned
