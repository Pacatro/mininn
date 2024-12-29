//! # MiniNN
//! A minimalist deep learnig crate for rust.

pub mod core;
pub mod layers;
pub mod utils;

mod registers;

pub mod constants {
    //! In this module you can find the most commonly used constants.
    pub use crate::layers::DEFAULT_DROPOUT_P;
    pub use crate::utils::DEFAULT_MOMENTUM;
}

pub mod prelude {
    //! In this module you can find the most commonly used types and functions.
    pub use crate::nn;
    pub use crate::register;
    pub use crate::{
        core::*,
        layers::*,
        registers::Register,
        utils::{
            Act, ActCore, ActivationFunction, Cost, CostCore, CostFunction, MSGPackFormatting,
            MetricsCalculator, NNUtil, Optimizer,
        },
    };
    pub use mininn_derive::*;
}
