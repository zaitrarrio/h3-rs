"""Reference fixtures from the real implementations, tiny random weights.

    pip install torch transformers diffusers safetensors
    git clone https://github.com/hao-ai-lab/FastVideo && pip install -e FastVideo
    python scripts/reference/make_fixtures.py

Writes small safetensors files the Rust tests compare against
(`cargo test -p h3-model --test parity`; the tests skip with a message when a
file is missing):

  crates/h3-model/tests/fixtures/h3_dit_tiny.safetensors     FastVideo MiniMaxH3Transformer3DModel (tiny)
  crates/h3-model/tests/fixtures/h3_vae_tiny.safetensors     FastVideo AutoencoderKLMiniMaxH3 (tiny): encode + decode
  crates/h3-model/tests/fixtures/h3_avae_tiny.safetensors    FastVideo MiniMaxH3AudioVAE (tiny): encode + decode
  crates/h3-text/tests/fixtures/qwen3vl_tiny.safetensors     transformers Qwen3VLForConditionalGeneration (tiny): hidden_states[2]

Each holds the model's weights (f32, their real key names) plus `input.*`
and `expected.*` tensors. The tiny shapes follow `H3Config::tiny()`,
`VideoVaeConfig::tiny()`, `AudioVaeConfig::tiny()` and `Qwen3VlConfig::tiny()`
in crates/h3-core/src/config.rs; keep both sides in step. Regenerate only
when a reference library changes, and record which version.

This script has not been run in the environment that wrote it (no PyTorch
there): it is the specification of the fixtures, and the first run should be
treated as a review of it.
"""
import pathlib, sys
import torch
from safetensors.torch import save_file

root = pathlib.Path(__file__).resolve().parents[2]
torch.manual_seed(0)


def jitter(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight") or name.endswith("scale1") or name.endswith("scale2"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif name.endswith("bias") or "alpha" in name or "beta" in name:
                p.copy_(0.1 * torch.randn_like(p))
            elif "adaln" in name or "norm_out.linear" in name:
                p.copy_(0.02 * torch.randn_like(p))
            else:
                fan_in = p.shape[1:].numel() if p.ndim > 1 else 1
                p.copy_(torch.randn_like(p) / fan_in**0.5)


def tensors(model, extra):
    sd = {k: v.detach().float().contiguous().clone() for k, v in model.state_dict().items()}
    sd.update({k: v.detach().float().contiguous() for k, v in extra.items()})
    return sd


# ---- H3 DiT (FastVideo) ---------------------------------------------------------
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel
from fastvideo.pipelines.basic.minimax_h3.packing import build_packed_sequence, build_row_timesteps

arch = MiniMaxH3ArchConfig(num_attention_heads=2, attention_head_dim=16, hidden_size=32, num_layers=2, num_refiner_layers=1,
                           ffn_dim=48, in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=24, freq_dim=16,
                           time_embed_hidden_dim=32, time_embed_dim=24, rope_freq_dim=2, rope_theta=10000.0,
                           norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5)
dit = MiniMaxH3Transformer3DModel(MiniMaxH3Config(arch_config=arch)).eval()
jitter(dit)
text_tags = torch.tensor([1, 1, 0, 1, 1, 1])
layout = build_packed_sequence(text_tags, 5, 64, 96, 5, (1, 2, 2), ("first",))  # 2 latent frames, 4×6 latent, 5 audio latents
video = torch.randn(layout.video_indices.numel(), 4 * 4)
audio = torch.randn(layout.audio_indices.numel(), 6)
text = torch.randn(6, 24)
unique, inverse = build_row_timesteps(layout, video_timestep=0.25, audio_timestep=0.7,
                                      condition_video_timestep=0.999, condition_audio_timestep=1.0)
with torch.no_grad():
    v, a = dit(hidden_states=video[None], audio_hidden_states=audio[None], encoder_hidden_states=text[None],
               timestep=unique, timestep_indices=inverse, token_tags=layout.token_tags,
               position_ids=layout.position_ids, video_indices=layout.video_indices,
               audio_indices=layout.audio_indices, text_indices=layout.text_indices)
save_file(tensors(dit, {"input.video_rows": video, "input.audio_rows": audio, "input.text": text,
                        "input.text_tags": text_tags.float(), "expected.video": v[0], "expected.audio": a[0]}),
          root / "crates/h3-model/tests/fixtures/h3_dit_tiny.safetensors")
print("dit", v.shape, a.shape)

# ---- video VAE --------------------------------------------------------------------
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEArchConfig, MiniMaxH3VideoVAEConfig
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3

varch = MiniMaxH3VideoVAEArchConfig(in_channels=3, latent_channels=4, block_out_channels=(8, 8, 16, 16, 16, 16),
                                    layers_per_block=1, spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
                                    temporal_downsample_factors=(1, 2, 2, 1, 1, 1), norm_num_groups=4,
                                    decoder_num_layers=2, decoder_num_attention_heads=2, decoder_attention_head_dim=16,
                                    decoder_num_register_tokens=2, decoder_ffn_mult=2,
                                    latents_mean=(0.1, -0.2, 0.3, 0.0), latents_std=(1.1, 0.9, 1.3, 1.0))
vae = AutoencoderKLMiniMaxH3(MiniMaxH3VideoVAEConfig(arch_config=varch)).eval()
jitter(vae)
px = torch.rand(1, 3, 17, 32, 48)
z = torch.randn(1, 4, 2, 2, 3)
with torch.no_grad():
    moments = vae.encode_pixels(px).latent_dist.parameters
    dec = vae.decode(z).sample
save_file(tensors(vae, {"input.pixels": px[0], "expected.moments": moments[0], "input.latent": z[0], "expected.pixels": dec[0]}),
          root / "crates/h3-model/tests/fixtures/h3_vae_tiny.safetensors")
print("vae", moments.shape, dec.shape)

# ---- audio VAE --------------------------------------------------------------------
from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEArchConfig, MiniMaxH3AudioVAEConfig
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE

aarch = MiniMaxH3AudioVAEArchConfig(encoder_dim=4, encoder_rates=(2, 5), latent_dim=16, latent_channels=4, decoder_dim=8,
                                    decoder_rates=(5, 2), decoder_kernel_sizes=(9, 4), num_attention_heads=2,
                                    resblock_kernel_sizes=(3, 7), resblock_dilation_sizes=((1, 3), (1, 3)),
                                    sampling_rate=320, latents_mean=(0.0, 0.1, -0.1, 0.2), latents_std=(1.0, 1.5, 0.8, 1.2))
avae = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(arch_config=aarch)).eval()
jitter(avae)
wave = torch.sin(torch.arange(95) * 0.3)[None, None] * 0.5
za = torch.randn(1, 4, 6)
with torch.no_grad():
    post = avae.encode(wave).latent_dist
    out = avae.decode(za).sample
save_file(tensors(avae, {"input.wave": wave[0, 0], "expected.mean": post.mean[0], "expected.logs": post.logs[0],
                         "input.latent": za[0], "expected.wave": out[0, 0]}),
          root / "crates/h3-model/tests/fixtures/h3_avae_tiny.safetensors")
print("avae", post.mean.shape, out.shape)

# ---- Qwen3-VL conditioner (transformers) ------------------------------------------
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

qcfg = Qwen3VLConfig(
    text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, rope_theta=5e6,
                     rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default"}),
    vision_config=dict(depth=2, hidden_size=16, intermediate_size=24, num_heads=2, patch_size=16, spatial_merge_size=2,
                       temporal_patch_size=2, out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0, 1],
                       in_channels=3),
    image_token_id=62, video_token_id=63, vision_start_token_id=60, vision_end_token_id=61)
q = Qwen3VLForConditionalGeneration(qcfg).eval()
jitter(q)
ids = torch.tensor([[5, 17, 60, 62, 62, 62, 62, 61, 9, 1, 8]])  # 2 text, one 64×64 image (4 merged tokens), 3 text
pixels = torch.randn(16, 3 * 2 * 16 * 16)
grid = torch.tensor([[1, 4, 4]])
with torch.no_grad():
    hs = q.model(input_ids=ids, pixel_values=pixels, image_grid_thw=grid, output_hidden_states=True).hidden_states[2][0]
save_file(tensors(q, {"input.ids": ids[0].float(), "input.pixels": pixels, "expected.hidden2": hs}),
          root / "crates/h3-text/tests/fixtures/qwen3vl_tiny.safetensors")
print("qwen3vl", hs.shape)
