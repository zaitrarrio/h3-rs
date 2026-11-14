"""LongLive-Plug's MiniMax-H3 few-step adapter as a PEFT safetensors file.

    python scripts/reference/convert_longlive_plug.py DIR

DIR is the `Efficient-Large-Model/LongLive-Plug-MiniMax-H3-few-step` download
(revision b3686f4: `generator_lora.pt`, `EXPORT_VALIDATION.json`). The release
ships the student LoRA as a torch pickle,

    {"schema_version": 1, "step": int, "config": {...},
     "student_lora": {"<module>.lora_A.weight": [128, in] f32,
                      "<module>.lora_B.weight": [out, 128] f32, ...}}

with Diffusers module names (312 targets: every attention and FF linear of the
50 blocks and the 2 refiner blocks, rank 128, alpha 128, so the merge scale is
1). This writes `DIR/generator_lora.safetensors` with the same 624 tensors
under the same names, which `h3_core::lora::Adapter` reads as a PEFT LoRA,
after checking the pickle against the release's recorded SHA-256 and the
four-step contract the release's own loader enforces (shifts 12 / 3, rank 128,
f32, the 312 targets). Needs torch and safetensors (any CPU build).
"""
import hashlib, json, pathlib, sys

import torch
from safetensors.torch import save_file

d = pathlib.Path(sys.argv[1])
src = d / "generator_lora.pt"
want = json.loads((d / "EXPORT_VALIDATION.json").read_text())["export"]
h = hashlib.sha256()
with src.open("rb") as f:
    for block in iter(lambda: f.read(64 << 20), b""):
        h.update(block)
if h.hexdigest() != want["sha256"]:
    sys.exit(f"{src}: sha256 {h.hexdigest()} is not the release's {want['sha256']}")

payload = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
cfg = payload["config"]
assert payload.get("schema_version") == 1, "unknown checkpoint schema"
assert (float(cfg["video_shift"]), float(cfg["audio_shift"])) == (12.0, 3.0), "not the 12 / 3 four-step contract"
assert (cfg["lora_rank"], cfg["lora_alpha"]) == (128, 128), "rank / alpha changed"
state = payload["student_lora"]
targets = cfg["lora_targets"]
assert len(targets) == 312 and set(state) == {f"{t}.lora_{x}.weight" for t in targets for x in "AB"}, "target set changed"
for k, v in state.items():
    assert v.dtype == torch.float32 and v.ndim == 2, k
    assert (v.shape[0] if ".lora_A." in k else v.shape[1]) == 128, k
out = d / "generator_lora.safetensors"
save_file({k: v.contiguous() for k, v in state.items()}, str(out),
          metadata={"format": "pt", "source": "LongLive-Plug-MiniMax-H3-few-step generator_lora.pt",
                    "source_sha256": want["sha256"], "step": str(payload["step"]), "lora_rank": "128", "lora_alpha": "128"})
print(f"{out}: {len(state)} tensors, step {payload['step']}, "
      f"{sum(v.numel() for v in state.values())} parameters")
