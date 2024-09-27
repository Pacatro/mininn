mod nn;

pub mod layers;
pub mod utils;

pub(crate) type NNResult<T> = Result<T, Box<dyn std::error::Error>>;

pub use nn::NN;

pub mod prelude {
    pub use crate::{
        NN,
        utils::*,
        layers::*,
    };
}
