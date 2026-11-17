# 2026-11-17: LightX2V Turbo, LongLive-Plug CFG and the draft tier on an RTX PRO 6000 Server Edition

Vast instance 61078272 (RTX PRO 6000 Blackwell Server Edition, Nevada,
driver 580.173.02, 600 W limit, 279 GB RAM), H3 base tree from R2
(`h3-base/42ed227…`, 140 GB in 293 s). Prompt: the fox (`FOX` in the
round's `common.sh`), encoded once with `h3 encode` (91 s) and passed
as `--embeds`; seed 7, 1344×768×124, T2VA unless noted. "fast" is
`--weights mxfp8 --decoder-weights mxfp8 --pv8`. No `--tables`: every run
computes its AdaLN tables (~46 s, listed apart from the denoise).

## Conformance

`h3 check --backend cuda`: 177 / 177 plain and under `H3_PROFILE=1`,
including the LoRA-branch cases with AdaLN and `norm_out` pairs (6.2e-3).

## Turbo (lightx2v/Minimax-h3-Turbo) and the CFG adapter

| run | forwards | denoise | per forward | decode | peak VRAM |
|---|---|---|---|---|---|
| `h3-turbo4` bf16 (FL2V v1.0, shift 6 / 3) | 4 | 45.0 s | 11.1–11.4 s | 8.5 s | 54.0 GB |
| `h3-turbo4` fast | 4 | 28.0 s | 6.9–7.3 s | 6.3 s | 37.8 GB |
| `h3-turbo8` bf16 (v1.2, per-layer alphas, AdaLN pairs) | 8 | 90.9 s | 11.1–11.5 s | 8.6 s | 54.0 GB |
| `h3-turbo8` fast | 8 | 55.9 s | 6.9–7.3 s | 6.1 s | 37.8 GB |
| `h3-turbo8` merged (`H3_LORA_MERGE=1`) | 8 | 87.0 s | 10.7–11.0 s | 8.6 s | 52.2 GB |
| `h3` base fast | 49 | 320.3 s | 6.4–6.8 s | | 35.9 GB |
| `h3-cfg` fast (LongLive-Plug CFG adapter) | 49 | 344.9 s | 6.9–7.3 s | | 37.8 GB |
| `h3-turbo4` FL2VA bf16, first frame of the base clip | 4 | 49.5 s | 12.2–12.5 s | 8.7 s | 54.9 GB |

- Turbo 4 is 11× less denoise than the 49-step base at the same recipe
  (28 vs 320 s). Each clip is a coherent photographic fox
  (`turbo-grids.jpg`: turbo4 bf16, turbo8 bf16, turbo4 fast, turbo8
  merged).
- The fast recipe moves Turbo clips like any MXFP8 run (turbo4 20.7 dB,
  turbo8 18.2 dB against bf16): the same scene.
- Merging Turbo 8 into the weights instead of the branch is 4 % faster per
  forward but makes another scene (16.7 dB against the branch): the
  merge loses part of the delta, as for LongLive-Plug's few-step adapter.
- The CFG adapter costs 8 % per forward over the base (one conditional
  forward instead of real CFG's two) and makes a brighter, differently
  composed clip (luma 81 → 90, 17.4 dB against the base,
  `base-cfg-fl2va-grids.jpg`: base, CFG, Turbo4 FL2VA).
- The AdaLN tables take 45–50 s per run without `--tables`: as long as a
  whole Turbo 4 denoise. A served Turbo / draft tier must load
  precomputed tables (`h3 adaln-tables --variant …`; Turbo 8's must come
  from its own variant, the CLI refuses another's).

Not run: the Ref2VA Turbo presets (need `transformer_ref`, 66 GB more).
