//! The released H3 variants (haoailab.com/FastVideo/cookbook/minimax-h3) as
//! presets: which checkpoint components, which sampler, which attention, and
//! which adapters. A `fastvideo_inference.json` (`fasth3-inference-contract-v1`)
//! found next to a checkpoint overrides the preset's sampler and VSA numbers.

use crate::{
    Error, Result,
    config::H3Config,
    schedule::{self, Schedule, Shifts, StepRule},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Text → video + audio.
    T2va,
    /// First / last keyframe → video + audio.
    Fl2va,
    /// Ordered image / video / audio references → video + audio.
    Ref2va,
    /// 768p draft → 2K (Open-H3-Regenerate-2K).
    Regen2k,
}

impl std::str::FromStr for Task {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "t2va" | "t2v" | "text" => Task::T2va,
            "fl2va" | "i2v" | "keyframes" => Task::Fl2va,
            "ref2va" | "r2v" | "references" => Task::Ref2va,
            "regen2k" | "regen" | "2k" => Task::Regen2k,
            o => {
                return Err(format!(
                    "unknown task `{o}` (t2va | fl2va | ref2va | regen2k)"
                ));
            }
        })
    }
}

/// How the timesteps of a run are chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum Sampler {
    /// DMD rungs in 1/1000 of the clean time, one forward per rung.
    Rungs(Vec<u32>),
    /// `points` uniformly spaced timesteps (`points − 1` forwards).
    Uniform(usize),
    /// Pika's renoise schedule: knots in 1/1000, one forward per knot.
    Renoise(Vec<u32>),
    /// Parallel Decoding Distillation: fine-grid indices of the fused heads.
    Pdd {
        indices: Vec<usize>,
        pdd_steps: usize,
    },
    /// Regen-2K refinement on explicit sigmas.
    Sigmas(Vec<f64>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VsaPreset {
    pub sparsity: f64,
    pub tile: usize,
    /// Keep rate of reference-video regions (Omni: 0.1).
    pub ref_keep_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdapterSpec {
    /// Path inside the variant's repository.
    pub path: String,
    pub alpha: Option<f32>,
    pub rank: Option<usize>,
    /// Which task the adapter serves (Pika ships one per task).
    pub task: Option<Task>,
    /// Multiplier on the adapter's LoRA pairs (1 for every release so far).
    pub strength: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub name: String,
    pub repo: String,
    pub summary: String,
    /// Subdirectory of the DiT weights (`transformer` / `transformer_ref`).
    pub transformer_dir: String,
    /// Repository the base components (VAEs, text encoder) come from when the
    /// variant ships only the DiT.
    pub base_repo: String,
    pub config: H3Config,
    pub sampler: Sampler,
    pub shifts: Shifts,
    pub vsa: Option<VsaPreset>,
    pub adapters: Vec<AdapterSpec>,
    pub tasks: Vec<Task>,
    /// The checkpoint stores NVFP4 block linears.
    pub nvfp4: bool,
    /// Pika's 480p / 768p bucket rules instead of the release canvas.
    pub pika_buckets: bool,
    /// Omni: keep rate of reference regions.
    pub reference_regions: bool,
}

const BASE: &str = "MiniMaxAI/MiniMax-H3";

/// LongLive-Plug's few-step adapter as `scripts/reference/convert_longlive_plug.py`
/// writes it next to the release's `generator_lora.pt`.
pub const PLUG4_ADAPTER: &str = "generator_lora.safetensors";

/// LightX2V's MiniMax-H3 Turbo adapters (`lightx2v/Minimax-h3-Turbo`): the
/// files and recipes of LightX2V's `configs/minimax_h3/distill` (Euler,
/// strength 1, alpha from the file: per-layer `.alpha` tensors, else its
/// `alpha` metadata).
pub const TURBO_REPO: &str = "lightx2v/Minimax-h3-Turbo";
pub const TURBO4_FL2V: &str = "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors";
pub const TURBO4_REF2V: &str = "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors";
pub const TURBO8_FL2V: &str = "minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors";
pub const TURBO8_REF2V: &str = "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors";

impl Variant {
    pub fn all() -> Vec<Variant> {
        vec![
            Variant {
                name: "h3".into(),
                repo: BASE.into(),
                summary: "MiniMax H3 base: 49 Euler forwards, dense attention; T2VA, FL2VA and Ref2VA (transformer_ref)".into(),
                transformer_dir: "transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Uniform(schedule::H3_BASE_POINTS),
                shifts: Shifts::H3,
                vsa: None,
                adapters: vec![],
                tasks: vec![Task::T2va, Task::Fl2va, Task::Ref2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "h3-plug4".into(),
                repo: "Efficient-Large-Model/LongLive-Plug-MiniMax-H3-few-step".into(),
                summary: "LongLive-Plug few-step LoRA over the base DiT: 4 forwards, predict-x0 and fresh re-noise on linspace(1, 0, 5) shifted 12 / 3, no CFG; T2VA (trained), FL2VA and Ref2VA (transfer)".into(),
                transformer_dir: "transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                // diffusers' MiniMaxH3Scheduler(num_inference_steps=5) with the
                // release's fresh-noise step: the grid is linspace(1, 0, 5), i.e.
                // the knots 1000 / 750 / 500 / 250 on the base clock, and the step
                // is Pika's renoise rule (`schedule::StepRule::Renoise`).
                sampler: Sampler::Renoise(schedule::PLUG4_KNOTS.to_vec()),
                shifts: Shifts::H3,
                vsa: None,
                adapters: vec![AdapterSpec {
                    path: PLUG4_ADAPTER.into(),
                    alpha: Some(128.0),
                    rank: Some(128),
                    task: None, strength: 1.0,
                }],
                tasks: vec![Task::T2va, Task::Fl2va, Task::Ref2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "h3-turbo4".into(),
                repo: TURBO_REPO.into(),
                summary: "LightX2V MiniMax-H3 Turbo 4-step v1.0 LoRA over the base DiT: 4 Euler forwards on linspace(1, 0, 5) shifted 6 / 3, no CFG; FL2VA (trained) and T2VA".into(),
                transformer_dir: "transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Uniform(5),
                shifts: Shifts { video: 6.0, audio: 3.0 },
                vsa: None,
                adapters: vec![AdapterSpec { path: TURBO4_FL2V.into(), alpha: None, rank: None, task: None, strength: 1.0 }],
                tasks: vec![Task::T2va, Task::Fl2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "h3-turbo4-ref".into(),
                repo: TURBO_REPO.into(),
                summary: "LightX2V MiniMax-H3 Turbo Ref2V 4-step v0.1 LoRA over transformer_ref: 4 Euler forwards shifted 12 / 3, alpha 8".into(),
                transformer_dir: "transformer_ref".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Uniform(5),
                shifts: Shifts::H3,
                vsa: None,
                adapters: vec![AdapterSpec { path: TURBO4_REF2V.into(), alpha: None, rank: None, task: None, strength: 1.0 }],
                tasks: vec![Task::Ref2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "h3-turbo8".into(),
                repo: TURBO_REPO.into(),
                summary: "LightX2V MiniMax-H3 Turbo 8-step v1.2 LoRA (block, refiner, AdaLN and norm_out pairs, per-layer alphas): 8 Euler forwards shifted 12 / 3, no CFG; FL2VA and T2VA".into(),
                transformer_dir: "transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Uniform(9),
                shifts: Shifts::H3,
                vsa: None,
                adapters: vec![AdapterSpec { path: TURBO8_FL2V.into(), alpha: None, rank: None, task: None, strength: 1.0 }],
                tasks: vec![Task::T2va, Task::Fl2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "h3-turbo8-ref".into(),
                repo: TURBO_REPO.into(),
                summary: "LightX2V MiniMax-H3 Turbo Ref2V 8-step v1.0 LoRA over transformer_ref: 8 Euler forwards shifted 12 / 3, alpha 8".into(),
                transformer_dir: "transformer_ref".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Uniform(9),
                shifts: Shifts::H3,
                vsa: None,
                adapters: vec![AdapterSpec { path: TURBO8_REF2V.into(), alpha: None, rank: None, task: None, strength: 1.0 }],
                tasks: vec![Task::Ref2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "h3-cfg".into(),
                repo: "Efficient-Large-Model/LongLive-Plug-MiniMax-H3-cfg".into(),
                summary: "LongLive-Plug CFG LoRA over the base DiT: CFG 3 distilled into the conditional forward (the base 49-step Euler ladder, one forward per step); not combined with the few-step LoRA, as its release advises".into(),
                transformer_dir: "transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Uniform(schedule::H3_BASE_POINTS),
                shifts: Shifts::H3,
                vsa: None,
                adapters: vec![AdapterSpec { path: "adapter_model.safetensors".into(), alpha: Some(128.0), rank: Some(128), task: None, strength: 1.0 }],
                tasks: vec![Task::T2va, Task::Fl2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "fasth3-v1".into(),
                repo: "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree".into(),
                summary: "FastH3 4-step V1: DMD rungs 999/749/500/250, VSA 0.9 tile 64".into(),
                transformer_dir: "transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Rungs(schedule::FASTH3_V1_RUNGS.to_vec()),
                shifts: Shifts::H3,
                vsa: Some(VsaPreset { sparsity: 0.9, tile: 64, ref_keep_rate: None }),
                adapters: vec![],
                tasks: vec![Task::T2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "fasth3-v2".into(),
                repo: "FastVideo/FastVideo-FastH3-8-Step-V2".into(),
                summary: "FastH3 8-step V2: eight DMD rungs, video shift 10, VSA 0.8 tile 64".into(),
                transformer_dir: "transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Rungs(schedule::FASTH3_V2_RUNGS.to_vec()),
                shifts: Shifts::FASTH3_V2,
                vsa: Some(VsaPreset { sparsity: 0.8, tile: 64, ref_keep_rate: None }),
                adapters: vec![],
                tasks: vec![Task::T2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "fasth3-trim".into(),
                repo: "FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4".into(),
                summary: "FastH3 Trim: 42 blocks, rank-16 AdaLN, NVFP4 block linears, 26-block VAE decoder; V2 sampler".into(),
                transformer_dir: "transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::fasth3_trim(),
                sampler: Sampler::Rungs(schedule::FASTH3_V2_RUNGS.to_vec()),
                shifts: Shifts::FASTH3_V2,
                vsa: Some(VsaPreset { sparsity: 0.8, tile: 64, ref_keep_rate: None }),
                adapters: vec![],
                tasks: vec![Task::T2va],
                nvfp4: true,
                pika_buckets: false,
                reference_regions: false,
            },
            Variant {
                name: "fasth3-omni".into(),
                repo: "FastVideo/FastVideo-FastH3-Omni-8-Step-V1".into(),
                summary: "FastH3 Omni: Ref2VA with Parallel Decoding Distillation (32 heads, 8 forwards), VSA 0.9 tile 128, reference keep rate 0.1".into(),
                transformer_dir: "transformer_ref".into(),
                base_repo: BASE.into(),
                config: H3Config::fasth3_omni(),
                sampler: Sampler::Pdd { indices: (0..=8).map(|i| 4 * i).collect(), pdd_steps: 32 },
                shifts: Shifts::H3,
                vsa: Some(VsaPreset { sparsity: 0.9, tile: 128, ref_keep_rate: Some(0.1) }),
                adapters: vec![],
                tasks: vec![Task::Ref2va],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: true,
            },
            Variant {
                name: "pika-flex".into(),
                repo: "Pika-Labs/Pika-H3-Flex".into(),
                summary: "Pika H3 Flex: rank-128 LoRA per task over the base DiT, renoise knots 1000/750/500/250/100/40, 480p and 768p buckets".into(),
                transformer_dir: "transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Renoise(schedule::PIKA_KNOTS.to_vec()),
                shifts: Shifts::H3,
                vsa: None,
                adapters: vec![
                    AdapterSpec { path: "fl2va/adapter.safetensors".into(), alpha: Some(128.0), rank: Some(128), task: Some(Task::Fl2va) , strength: 1.0},
                    AdapterSpec { path: "ref2va/adapter.safetensors".into(), alpha: Some(128.0), rank: Some(128), task: Some(Task::Ref2va) , strength: 1.0},
                ],
                tasks: vec![Task::Fl2va, Task::Ref2va],
                nvfp4: false,
                pika_buckets: true,
                reference_regions: false,
            },
            Variant {
                name: "regen-2k".into(),
                repo: "Efficient-Large-Model/Open-H3-Regenerate-2K".into(),
                summary: "Open-H3-Regenerate-2K: 768p draft → 2560×1440 through a 3-D latent upscaler, rank-576 LoRA and Sharp curve branches, 3 Euler forwards".into(),
                transformer_dir: "base_model/transformer".into(),
                base_repo: BASE.into(),
                config: H3Config::h3(),
                sampler: Sampler::Sigmas(schedule::REGEN_SIGMAS_3.to_vec()),
                shifts: Shifts::H3,
                vsa: None,
                adapters: vec![AdapterSpec {
                    path: "lora/open_h3_regenerate_2k_lora.safetensors".into(),
                    alpha: None,
                    rank: Some(576),
                    task: Some(Task::Regen2k), strength: 1.0,
                }],
                tasks: vec![Task::Regen2k],
                nvfp4: false,
                pika_buckets: false,
                reference_regions: false,
            },
        ]
    }

    pub fn by_name(name: &str) -> Result<Variant> {
        let n = name.trim().to_ascii_lowercase().replace('_', "-");
        let alias = match n.as_str() {
            "base" | "minimax-h3" => "h3",
            // The draft tier: LongLive-Plug's 4-step distillation (with
            // `--tier 480p` for the fast setting), refined by `h3 regen2k` or
            // re-rendered by a full variant once a draft is picked.
            "plug4" | "plug" | "longlive-plug" | "h3-4step" | "draft" => "h3-plug4",
            "turbo" | "turbo4" | "lightx2v" => "h3-turbo4",
            "turbo8" => "h3-turbo8",
            "plug-cfg" | "cfg" => "h3-cfg",
            "v1" | "fasth3-4-step" | "4-step" => "fasth3-v1",
            "v2" | "fasth3-8-step" | "8-step" => "fasth3-v2",
            "trim" | "nvfp4" => "fasth3-trim",
            "omni" | "pdd" => "fasth3-omni",
            "pika" | "pika-h3-flex" | "flex" => "pika-flex",
            "regen" | "regenerate-2k" | "2k" => "regen-2k",
            o => o,
        };
        Variant::all()
            .into_iter()
            .find(|v| v.name == alias)
            .ok_or_else(|| {
                Error::Config(format!(
                    "unknown variant `{name}`; one of: {}",
                    Variant::names().join(", ")
                ))
            })
    }

    pub fn names() -> Vec<String> {
        Variant::all().into_iter().map(|v| v.name).collect()
    }

    /// Number of transformer forwards per clip.
    pub fn forwards(&self) -> usize {
        match &self.sampler {
            Sampler::Rungs(r) | Sampler::Renoise(r) => r.len(),
            Sampler::Uniform(p) => p - 1,
            Sampler::Pdd { indices, .. } => indices.len() - 1,
            Sampler::Sigmas(s) => s.len() - 1,
        }
    }

    /// The video and audio schedules of the preset (PDD and Regen carry
    /// their own structures: [`schedule::PddPlan`] / [`schedule::regen_sigmas`]).
    pub fn schedules(&self) -> Result<(Schedule, Schedule)> {
        match &self.sampler {
            Sampler::Rungs(r) => Ok((
                Schedule::from_rungs(r, self.shifts.video, StepRule::Euler)?,
                Schedule::from_rungs(r, self.shifts.audio, StepRule::Euler)?,
            )),
            Sampler::Renoise(k) => Ok((
                Schedule::from_rungs(k, self.shifts.video, StepRule::Renoise)?,
                Schedule::from_rungs(k, self.shifts.audio, StepRule::Renoise)?,
            )),
            Sampler::Uniform(p) => Ok((
                Schedule::uniform(*p, self.shifts.video)?,
                Schedule::uniform(*p, self.shifts.audio)?,
            )),
            Sampler::Pdd { indices, .. } => {
                Ok(schedule::PddPlan::new(indices, self.shifts)?.schedules())
            }
            Sampler::Sigmas(s) => {
                let v: Vec<f32> = s.iter().map(|&x| x as f32).collect();
                Ok((
                    Schedule::from_sigmas(v, StepRule::Euler)?,
                    Schedule::from_sigmas(vec![1.0, 0.0], StepRule::Euler)?,
                ))
            }
        }
    }

    /// Override sampler and attention numbers from a checkpoint's
    /// `fastvideo_inference.json`.
    pub fn apply_contract(&mut self, v: &serde_json::Value) -> Result<()> {
        let schema = v
            .get("schema_version")
            .or_else(|| v.get("schema"))
            .and_then(|s| s.as_str())
            .unwrap_or("");
        if !schema.starts_with("fasth3-inference-contract") {
            return Err(Error::Config(format!(
                "unknown inference contract schema `{schema}`"
            )));
        }
        if let Some(r) = v.get("dmd_denoising_steps").and_then(|x| x.as_array()) {
            self.sampler = Sampler::Rungs(
                r.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as u32)
                    .collect(),
            );
        }
        if let (Some(idx), Some(steps)) = (
            v.get("pdd_step_indices").and_then(|x| x.as_array()),
            v.get("pdd_steps").and_then(|x| x.as_u64()),
        ) {
            self.sampler = Sampler::Pdd {
                indices: idx
                    .iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect(),
                pdd_steps: steps as usize,
            };
            self.config.pdd_steps = Some(steps as usize);
        }
        if let Some(s) = v.get("video_scheduler_shift").and_then(|x| x.as_f64()) {
            self.shifts.video = s as f32;
        }
        if let Some(s) = v.get("audio_scheduler_shift").and_then(|x| x.as_f64()) {
            self.shifts.audio = s as f32;
        }
        if v.get("attention_backend").and_then(|x| x.as_str()) == Some("VIDEO_SPARSE_ATTN_H3") {
            let sparsity = v
                .get("vsa_sparsity")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.8);
            let tile = v
                .get("vsa_tile_size")
                .and_then(|x| x.as_u64())
                .unwrap_or(64) as usize;
            let ref_keep_rate = v.get("vsa_ref_keep_rate").and_then(|x| x.as_f64());
            if crate::vsa::tile_shape(tile).is_none() {
                return Err(Error::Config(format!(
                    "vsa_tile_size {tile} is not 64, 128 or 256"
                )));
            }
            self.vsa = Some(VsaPreset {
                sparsity,
                tile,
                ref_keep_rate,
            });
            self.reference_regions = ref_keep_rate.is_some();
        }
        if let Some(c) = v.get("transformer_component").and_then(|x| x.as_str()) {
            self.transformer_dir = c.to_string();
        }
        if let Some(t) = v
            .get("model_type")
            .or_else(|| v.get("task"))
            .and_then(|x| x.as_str())
        {
            if let Ok(task) = t.parse::<Task>() {
                if !self.tasks.contains(&task) {
                    self.tasks.insert(0, task);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_resolve_and_count_forwards() {
        assert_eq!(Variant::by_name("v1").unwrap().forwards(), 4);
        assert_eq!(Variant::by_name("fasth3-v2").unwrap().forwards(), 8);
        assert_eq!(Variant::by_name("trim").unwrap().forwards(), 8);
        assert_eq!(Variant::by_name("omni").unwrap().forwards(), 8);
        assert_eq!(Variant::by_name("pika").unwrap().forwards(), 6);
        assert_eq!(Variant::by_name("h3").unwrap().forwards(), 49);
        assert_eq!(Variant::by_name("regen").unwrap().forwards(), 3);
        let plug = Variant::by_name("longlive-plug").unwrap();
        assert_eq!((plug.name.as_str(), plug.forwards()), ("h3-plug4", 4));
        assert_eq!(plug.schedules().unwrap().0.rule, StepRule::Renoise);
        assert_eq!(plug.adapters.len(), 1);
        assert!(Variant::by_name("wan").is_err());
        for v in Variant::all() {
            let (vs, as_) = v.schedules().unwrap();
            assert_eq!(vs.steps(), v.forwards(), "{}", v.name);
            assert!(as_.steps() <= v.forwards());
        }
    }

    #[test]
    fn contract_overrides_preset() {
        let mut v = Variant::by_name("fasth3-v1").unwrap();
        let c: serde_json::Value = serde_json::from_str(
            r#"{"schema_version":"fasth3-inference-contract-v1","dmd_denoising_steps":[999,874,749,624,500,375,250,125],
                "video_scheduler_shift":10.0,"audio_scheduler_shift":3.0,"attention_backend":"VIDEO_SPARSE_ATTN_H3",
                "vsa_sparsity":0.8,"vsa_tile_size":64}"#,
        )
        .unwrap();
        v.apply_contract(&c).unwrap();
        assert_eq!(v.forwards(), 8);
        assert_eq!(v.shifts, Shifts::FASTH3_V2);
        assert_eq!(v.vsa.unwrap().sparsity, 0.8);
        let omni: serde_json::Value = serde_json::from_str(
            r#"{"schema":"fasth3-inference-contract-v1","model_type":"ref2va","pdd_step_indices":[0,4,8,12,16,20,24,28,32],
                "pdd_steps":32,"attention_backend":"VIDEO_SPARSE_ATTN_H3","vsa_sparsity":0.9,"vsa_tile_size":128,
                "vsa_ref_keep_rate":0.1,"transformer_component":"transformer_ref","video_scheduler_shift":12.0}"#,
        )
        .unwrap();
        let mut o = Variant::by_name("h3").unwrap();
        o.apply_contract(&omni).unwrap();
        assert_eq!(o.transformer_dir, "transformer_ref");
        assert!(o.reference_regions);
        assert_eq!(o.forwards(), 8);
        assert_eq!(o.config.pdd_steps, Some(32));
    }
}
