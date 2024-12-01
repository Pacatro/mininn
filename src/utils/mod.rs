//! Functions and types for the neural network.
//!
//! Here you can find:
//!
//! * Types for activation functions, cost functions, and optimizers.
//! * A `LayerRegister` struct for managing the layers in a neural network.
//! * A `MetricsCalculator` struct for calculating metrics like accuracy, precision, recall, and F1-score.

mod activation_func;
mod cost;
mod layer_register;
mod metrics;
mod optimizer;

pub(crate) use optimizer::OptimizerType;

pub use activation_func::{ActivationFunc, ActivationFunction};
pub use cost::{Cost, CostFunction};
pub use layer_register::LayerRegister;
pub use metrics::MetricsCalculator;
pub use optimizer::*;
