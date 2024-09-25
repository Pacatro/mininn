mod nn;
mod save_config;

pub mod layers;
pub mod utils;

pub use nn::NN;

pub mod prelude {
    pub use crate::{
        NN,
        utils::{ActivationFunc, Cost, MetricsCalculator},
        layers::*,
    };
}
