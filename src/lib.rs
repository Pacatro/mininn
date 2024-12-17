//! # MiniNN
//! A minimalist deep learnig crate for rust.

pub mod core;
pub mod layers;
pub mod registers;
pub mod utils;

pub mod constants {
    //! In this module you can find the most commonly used constants.
    pub use crate::layers::DEFAULT_DROPOUT_P;
    pub use crate::utils::DEFAULT_MOMENTUM;
}

pub mod prelude {
    //! In this module you can find the most commonly used types and functions.
    pub use crate::register;
    pub use crate::{
        core::*,
        layers::{Activation, Dense, Dropout, Flatten, Layer, TrainLayer},
        registers::Register,
        utils::{
            Act, ActCore, ActivationFunction, Cost, CostCore, CostFunction, MSGPackFormatting,
            MetricsCalculator, NNUtil, Optimizer,
        },
    };
    pub use mininn_derive::*;
}
