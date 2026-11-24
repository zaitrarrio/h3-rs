//! The MiniMax H3 models and pipelines, written once against
//! [`h3_core::Backend`]: the Omni-Transformer DiT, the video and audio VAEs,
//! the TAEH3 preview decoder, the Regen-2K latent upscaler, the samplers of
//! every released variant, and the backend conformance checker.

pub mod audio_vae;
pub mod check;
pub mod dit;
pub mod esrgan;
pub mod fast_vae;
pub mod linear;
pub mod pipeline;
pub mod regen;
pub mod taeh3;
pub mod upscaler;
pub mod vae;

pub use dit::{Attention, H3Dit};
pub use pipeline::{Event, Generation, Pipeline, Prompt, Request};
