# h3-rs

MiniMax H3 (video + 32 kHz stereo audio) on desktop GPUs, in Rust, following
the approach of [fastwan-rs](https://github.com/zaitrarrio/fastwan-rs): one
small fused op vocabulary (`h3_core::Backend`), a CPU reference that defines
every op, WebGPU and CUDA backends checked against it, and the conditioning
and modulation precomputed so the 33 B model fits a consumer card.

Every recipe of the [FastVideo MiniMax H3 cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
is a `--variant`:

| `--variant` | checkpoint | forwards | attention |
|---|---|---|---|
| `h3` | `MiniMaxAI/MiniMax-H3` (T2VA, FL2VA, Ref2VA) | 49 | dense |
| `fasth3-v1` | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` | 4 | VSA 0.9 |
| `fasth3-v2` | `FastVideo/FastVideo-FastH3-8-Step-V2` | 8 | VSA 0.8 |
| `fasth3-trim` | `FastVideo/FastVideo-FastH3-Trim-8-Step` (42 blocks; bf16 HF tree or the ComfyUI single-file NVFP4) | 8 | VSA 0.8 |
| `fasth3-omni` | `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` (Ref2VA, PDD) | 8 | VSA 0.9, tile 128 |
| `pika-flex` | `Pika-Labs/Pika-H3-Flex` (FL2VA / Ref2VA LoRA, 480p / 768p) | 6 | dense |
| `regen-2k` | `Efficient-Large-Model/Open-H3-Regenerate-2K` (`h3 regen2k`) | 3 | dense / VSA |

The MLX recipes of the cookbook use the same checkpoints; on Apple GPUs the
WebGPU backend runs them through Metal.

| backend | runs on | precision | status |
|---|---|---|---|
| `cpu` | anywhere | f32 | the reference every backend is checked against (`cargo test --workspace`) |
| `wgpu` | NVIDIA, AMD, Intel, Apple (Vulkan / DX12 / Metal) | f32 activations, f16 weights (int8 / int4 weight-only) | 90 / 90 checks on an RTX A6000 (NVIDIA Vulkan) and on lavapipe in CI; runs the full 1344×768×124 geometry at 7.6 s per block on the A6000, 16× slower than CUDA |
| `cuda` | NVIDIA sm_80+ | bf16, cuBLASLt GEMMs, tensor-core attention and VSA | 90 / 90 checks on an RTX A6000; real clips from FastH3-V2 and FastH3-Trim |

Measured on a vast.ai **RTX A6000** (48 GB, 300 W, driver 570.195.03),
1344×768×124 (5.2 s), VSA 0.8, int8 weights, precomputed AdaLN tables
(`docs/runs/2026-11-06-vast-a6000-first-gpu`):

| | FastH3-V2, A6000, int8 | FastH3-V2, **RTX PRO 6000 Max-Q**, bf16 | FastH3-Trim NVFP4 (ComfyUI file), PRO 6000 | FastH3 4-step V1, PRO 6000, int8 |
|---|---|---|---|---|
| denoise | 201 s (25.1 s / forward) | 79.5 s (9.6–10.1 s / forward) | 67.8 s (8.2–8.6 s) | 35.0 s (4 forwards) |
| decode | 44 s | **11.9 s** (was 20.5) | 17.4 s | 20.4 s |
| audio decode | | **0.24 s** (was 0.46) | 0.46 s | 0.46 s |
| peak device memory | | 56.2 GB | 24.4 GB | 37.0 GB |

(`docs/runs/2026-11-07-vast-pro6000-clips`, decode and audio from
`…-pro6000-decode`; the Max-Q is the 300 W card and runs the denoise
power-capped at ~1.4 GHz, not its 3.09 GHz boost.)

On the **RTX PRO 6000 Server Edition (600 W)**, fastvideo-rs's gold GPU
(`docs/runs/2026-11-07-vast-pro6000-server-mxfp8`), the same V2 clip:

| | h3-rs bf16 | h3-rs `--weights mxfp8` | fastvideo-rs (MXFP8) |
|---|---|---|---|
| denoise | 50.6 s | **45.0 s** | 45.8 s |
| video decode | 8.9 s | 8.8 s | 4.7–6.3 s |
| peak device memory | 56.2 GB | **41.0 GB** | 61.0 GiB |

`--weights mxfp8` runs blocks 2–46 as W8A8 on the FP8 tensor cores
(cuBLASLt `VEC32_UE8M0`, sm_100+; emulated in bf16 elsewhere); the clip
keeps the scene (17 dB PSNR against bf16, as int8's 18.5 dB: the DMD
sampler turns weight rounding into texture).

Per block at that geometry: 470 ms VSA / 815 ms dense on the A6000, 230 /
354 ms on the L40S. The VSA phases on the L40S after the kernel work in
`docs/runs/2026-11-07-vast-l40s-vsa-phases`: fine stage 66.5 ms (was 87),
coarse + top-k 3.6 ms (was 6.4), pool 2.8 ms; the fine stage is still
~2.7× its tensor-core floor and the record says what to try next.
No PyTorch parity number yet (`scripts/reference/make_fixtures.py` needs a
machine with torch); the clips themselves match their prompts
(`docs/runs/2026-11-06-vast-a6000-first-gpu/*.png`).

## Use

```bash
cargo build --release                      # WebGPU + CPU
cargo build --release --features cuda      # + CUDA (no CUDA toolkit needed to build)
```

```bash
./target/release/h3 check --backend wgpu          # every kernel and tiny model vs the CPU reference
./target/release/h3 plan --variant fasth3-v2 --weights int4-hqq-g64 --vram-gib 24
./target/release/h3 bench --variant fasth3-v2 --layers 4 --steps 2   # synthetic weights, real geometry
```

### Download

```bash
huggingface-cli download MiniMaxAI/MiniMax-H3 \
    --include "vae/*" "audio_vae/*" "text_encoder/*" "tokenizer/*" "transformer/*" --local-dir h3
huggingface-cli download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir fasth3-v2
huggingface-cli download madebyollin/taehv taeh3.safetensors --local-dir h3      # optional preview decoder
```

The base tree is ~95 GB (text encoder 64 GB). A variant directory holds its
own `transformer/` (or adapters) and `fastvideo_inference.json`, which `h3`
reads to set the sampler and VSA.

### Fit the card

```bash
# AdaLN tables for the variant's timestep ladder: the 13 B modulation weights
# are then never read again (174 MB for the 8-step ladders).
h3 adaln-tables --variant fasth3-v2 --model-dir h3 --variant-dir fasth3-v2 --out v2.tables
# A 4-bit DiT without the AdaLN weights: ~12 GB instead of 66 GB.
h3 quantize --variant fasth3-v2 --model-dir h3 --variant-dir fasth3-v2 \
    --weights int4-hqq-g64 --skip-adaln --out fasth3-v2-int4.safetensors
```

ComfyUI single-file checkpoints (`fastvideo_fasth3_trim_8step_nvfp4.safetensors`
and the like) are read directly: `--transformer <file>`; their NVFP4 linears
are dequantized to bf16 on the device at load.

`--weights int8 | int4[-hqq][-gN][-ffn] | nvfp4` quantizes at load as well;
`--stream-blocks` keeps the block weights in host memory and uploads them per
step; `h3 plan` tells what fits.

### One clip

```bash
h3 generate --variant fasth3-v2 --model-dir h3 --variant-dir fasth3-v2 \
    --transformer fasth3-v2-int4.safetensors --tables v2.tables \
    --prompt "A red fox trots through fresh snow at dawn; snow crunches under its paws." \
    --width 1344 --height 768 --frames 124 --seed 7 --out fox.y4m
ffmpeg -i fox.y4m -i fox.wav -c:v libx264 -pix_fmt yuv420p -c:a aac fox.mp4
```

Keyframes: `--first-image a.png --last-image b.png` (FL2VA; `--variant pika-flex`
for Pika). References: `--task ref2va --ref image:a.png --ref video:clip.y4m,audio=clip.wav
--ref audio:song.wav` in presentation order; `--continue tail.y4m` for a
continuation clip. Omni: `--variant fasth3-omni`. The prompt is encoded by
the native Qwen3-VL port on the chosen backend, streaming its layers from the
mmapped checkpoint; `h3 encode` stores text-only prompts in a gallery so
`--embeds` and `serve --gallery` need no text encoder at all.

### Regenerate at 2K

```bash
huggingface-cli download Efficient-Large-Model/Open-H3-Regenerate-2K --local-dir regen
# adaln_t_table.bin is not redistributed: it is the `adaln_t_table` tensor of
# Comfy-Org/MiniMax-H3 (1025×8 f32); fetch it and pass the raw bytes.
h3 regen2k --regen-dir regen --model-dir h3 --draft fox.y4m --prompt "..." \
    --adaln-table adaln_t_table.bin --out fox-2k.y4m
```

### Prompt-to-clip demo

```bash
h3 serve --variant fasth3-v2 --model-dir h3 --variant-dir fasth3-v2 --tables v2.tables \
    --weights int4-hqq-g64      # then open http://127.0.0.1:7860/
```

`h3 serve` keeps the DiT and the decoders resident, encodes prompts with the
conditioner (or a `--gallery`), and serves `web/demo.html`: prompt, size,
length, seed; the steps and the decode land as they finish, frames play as
they arrive and the soundtrack plays with them. `--synthetic` runs a tiny
random model for trying the page without weights. The engine listens on
127.0.0.1 and answers CORS for one `--allow-origin` only.

## Layout

```
crates/h3-core    configs, packing (sequence layout, buckets, row timesteps), schedules (Euler, renoise, PDD, Regen),
                  RoPE tables, weights + manifests, quant (int8 / 4-bit / NVFP4), LoRA merging, VSA-H3 tiling,
                  memory plan, variant presets, the Backend trait
crates/h3-cpu     reference backend
crates/h3-wgpu    WebGPU backend (src/shaders/*.wgsl)
crates/h3-cuda    CUDA backend (src/kernels.cu, NVRTC + cuBLASLt)
crates/h3-model   H3 DiT, video VAE, audio VAE, TAEH3, latent upscaler, pipelines (T2VA / FL2VA / Ref2VA / PDD / renoise / Regen-2K), checker
crates/h3-text    Qwen3-VL conditioner (tokenizer, image / video preprocessing, presentation, model)
crates/h3-cli     the `h3` binary
web/              demo.html (served by `h3 serve`), landing page, site build
scripts/          reference fixtures (make_fixtures.py), vast.ai rental scripts, Pages deploy
docs/design.md    what runs where, what is fused, what is verified, what is next
```

`cargo test --workspace` runs everything (the WebGPU suite needs a Vulkan
driver: `apt install mesa-vulkan-drivers` gives lavapipe on a machine without a
GPU; set `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json` and
`H3_ALLOW_SOFTWARE=1`).

Weights are MiniMax's, FastVideo's, Pika's and NVLabs' under their own
licences; this repository holds none.
