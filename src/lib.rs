//! # MiniNN
//! A minimalist deep learnig crate for rust.

pub mod core;
pub mod layers;
pub mod utils;

mod recorders;

pub mod prelude {
    //! In this module you can find the most commonly used types and functions.
    pub use crate::nn;

    pub use crate::{
        core::*,
        layers::*,
        recorders::Recorder,
        utils::{MSGPackFormatting, MetricsCalculator, NNUtil},
    };
    pub use mininn_derive::*;
}
