//! # MiniNN
//! A minimalist deep learnig crate for rust.

mod nn;

pub mod error;
pub mod layers;
pub mod utils;

pub use nn::NN;

pub mod prelude {
    pub use crate::{
        error::{MininnError, NNResult},
        layers::*,
        utils::*,
        NN,
    };
}
