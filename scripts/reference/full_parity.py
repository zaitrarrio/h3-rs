"""Full-width DiT parity: FastVideo's MiniMaxH3Transformer3DModel on the real
checkpoint's first N blocks (with the embedders, the text refiner and the
head) for a small packed sequence, in f32, dumped for `h3 parity-dit`.

The committed tiny fixtures (`make_fixtures.py`) hold the structure at toy
sizes; this holds the real widths, head counts, RoPE split and weights.

    python3 -m venv venv && . venv/bin/activate
    pip install torch==2.12.0 "transformers>=5.15" diffusers safetensors einops numpy scipy cloudpickle imageio remote_pdb ftfy psutil
    git clone https://github.com/hao-ai-lab/FastVideo
    PYTHONPATH=$PWD/FastVideo python scripts/reference/full_parity.py \\
        --transformer /w/h3-base/transformer --layers 2 --out dit_full.safetensors --device cuda
    h3 parity-dit --backend cuda --transformer /w/h3-base/transformer --fixture dit_full.safetensors

`--tiny` uses `make_fixtures.py`'s tiny arch with the weights of a file in
Diffusers names (the committed `h3_dit_tiny.safetensors`): it checks this
script and `h3 parity-dit --tiny` end to end without the checkpoint.
"""
import argparse, json, os, pathlib
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--transformer", required=True, help="transformer/ directory (or one .safetensors with --tiny)")
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--latent", default="2,8,8", help="latent frames, height, width")
ap.add_argument("--audio", type=int, default=5, help="audio latents")
ap.add_argument("--text", type=int, default=12, help="text rows")
a = ap.parse_args()
torch.manual_seed(0)

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
for k, v in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29573", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
    os.environ.setdefault(k, v)
from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.forward_context import set_forward_context
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel
from fastvideo.pipelines.basic.minimax_h3.packing import build_packed_sequence, build_row_timesteps

maybe_init_distributed_environment_and_model_parallel(1, 1)


def diffusers_name(k):
    """FastVideo's internal names -> the checkpoint's (as in make_fixtures.py)."""
    return (k.replace("time_embedder.fc_in.", "time_embedder.linear_1.").replace("time_embedder.fc_out.", "time_embedder.linear_2.")
            .replace(".attn.to_out.", ".attn.to_out.0.").replace(".ff.fc_in.", ".ff.net.0.proj.").replace(".ff.fc_out.", ".ff.net.2."))


src = pathlib.Path(a.transformer)
if a.tiny:
    arch = MiniMaxH3ArchConfig(num_attention_heads=2, attention_head_dim=16, hidden_size=32, num_layers=a.layers, num_refiner_layers=1,
                               ffn_dim=48, in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=24, freq_dim=16,
                               time_embed_hidden_dim=32, time_embed_dim=24, rope_freq_dim=2, rope_theta=10000.0,
                               norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5)
    files = [src]
else:
    arch = MiniMaxH3ArchConfig()
    cfg = json.loads((src / "config.json").read_text())
    for k, v in cfg.items():
        if k in arch.__dataclass_fields__ and not k.startswith("_"):
            setattr(arch, k, tuple(v) if isinstance(v, list) else v)
    arch.num_layers = a.layers
    arch.__post_init__()
    files = sorted(src.glob("*.safetensors"))
print("arch", {k: getattr(arch, k) for k in ("num_attention_heads", "attention_head_dim", "hidden_size", "num_layers",
                                            "ffn_dim", "in_channels", "audio_in_channels", "text_dim", "rope_freq_dim")})

model = MiniMaxH3Transformer3DModel(MiniMaxH3Config(arch_config=arch), hf_config={}).eval()
where = {}
for f in files:
    with safe_open(str(f), "pt") as h:
        for k in h.keys():
            where[k] = f
sd = model.state_dict()
loaded = {}
for k, p in sd.items():
    dk = diffusers_name(k)
    if dk not in where:
        raise SystemExit(f"the checkpoint has no `{dk}` (model key `{k}`)")
    with safe_open(str(where[dk]), "pt") as h:
        t = h.get_tensor(dk).float()
    if tuple(t.shape) != tuple(p.shape):
        raise SystemExit(f"`{dk}`: checkpoint {tuple(t.shape)}, model {tuple(p.shape)}")
    loaded[k] = t
model.load_state_dict(loaded)
model = model.to(a.device, torch.float32)
print(f"loaded {len(loaded)} tensors from {len(files)} file(s)")

lt, lh, lw = (int(v) for v in a.latent.split(","))
tags = torch.ones(a.text, dtype=torch.long)
tags[a.text // 3] = 0                      # one padding row, as in the tiny fixture
layout = build_packed_sequence(tags, lt, lh, lw, a.audio, tuple(arch.patch_size), ("first",))
pt, ph, pw = arch.patch_size
video = torch.randn(layout.video_indices.numel(), arch.in_channels * pt * ph * pw)
audio = torch.randn(layout.audio_indices.numel(), arch.audio_in_channels)
text = torch.randn(a.text, arch.text_dim)
ts = (0.25, 0.7, 0.999, 1.0)
unique, inverse = build_row_timesteps(layout, video_timestep=ts[0], audio_timestep=ts[1],
                                      condition_video_timestep=ts[2], condition_audio_timestep=ts[3])
dev = lambda x: x.to(a.device) if torch.is_tensor(x) else x
with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
    v, au = model(hidden_states=dev(video)[None], audio_hidden_states=dev(audio)[None], encoder_hidden_states=dev(text)[None],
                  timestep=dev(unique), timestep_indices=dev(inverse), token_tags=dev(layout.token_tags),
                  position_ids=dev(layout.position_ids), video_indices=dev(layout.video_indices),
                  audio_indices=dev(layout.audio_indices), text_indices=dev(layout.text_indices))
out = {"input.geometry": torch.tensor([lt, lh, lw, a.audio, a.layers], dtype=torch.float32),
       "input.timesteps": torch.tensor(ts, dtype=torch.float32), "input.text_tags": tags.float(),
       "input.text": text, "input.video_rows": video, "input.audio_rows": audio,
       "expected.video": v[0].float().cpu(), "expected.audio": au[0].float().cpu()}
save_file({k: t.contiguous() for k, t in out.items()}, a.out)
print("wrote", a.out, "video", tuple(v.shape), "audio", tuple(au.shape))
