mod nn;

pub mod layers;
pub mod utils;

pub use nn::{NN, NNResult};

pub mod prelude {
    pub use crate::{
        NN,
        utils::*,
        layers::*,
        NNResult
    };
}
