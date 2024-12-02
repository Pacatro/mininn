//! # MiniNN
//! A minimalist deep learnig crate for rust.

pub mod core;
pub mod layers;
pub mod registers;
pub mod utils;

pub mod prelude {
    //! In this module you can find the most commonly used types and functions.
    pub use crate::{core::*, layers::*, registers::*, utils::*};
}
