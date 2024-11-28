//! # MiniNN
//! A minimalist deep learnig crate for rust.

mod error;
mod nn;

pub mod layers;
pub mod utils;

pub use error::*;
pub use nn::{NNMode, NN};

pub mod prelude {
    //! In this module you can find the most commonly used types and functions.
    pub use crate::{
        error::*,
        layers::*,
        utils::*,
        {NNMode, NN},
    };
}
