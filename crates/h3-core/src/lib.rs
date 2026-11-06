//! `h3-core`: everything about MiniMax H3 that is not a GPU.
//!
//! The DiT, the two VAEs, the samplers and the conditioner in `h3-model` /
//! `h3-text` are written once against [`backend::Backend`]; the CPU
//! reference, WebGPU and CUDA backends implement that one op vocabulary.
//! Nothing here allocates device memory.

pub mod backend;
pub mod config;
pub mod export;
pub mod lora;
pub mod packing;
pub mod plan;
pub mod quant;
pub mod rng;
pub mod rope;
pub mod schedule;
pub mod timestep;
pub mod variant;
pub mod vsa;
pub mod weights;

pub use backend::{Act, Backend, Conv1dShape, Conv3dShape, ConvEpilogue, ConvShape, ConvT1dShape, FlowStep, Gate, HeadNorm, ModRef, ModTable, Modulation, Rope};
pub use variant::{Task, Variant};
pub use config::{AudioVaeConfig, H3Config, Qwen3VlConfig, TaeH3Config, UpscalerConfig, VideoVaeConfig};
pub use packing::{Geometry, Layout};
pub use schedule::Schedule;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("weights: {0}")]
    Weights(String),
    #[error("geometry: {0}")]
    Geometry(String),
    #[error("config: {0}")]
    Config(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;
