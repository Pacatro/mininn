//! # MiniNN
//! A minimalist deep learnig crate for rust.

mod error;
mod nn;

pub mod layers;
pub mod utils;

pub use error::*;
pub use nn::NN;

pub mod prelude {
    pub use crate::{error::*, layers::*, utils::*, NN};
}
