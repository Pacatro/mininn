//! This module contains the register for activations, costs and layers.

mod activation_register;
mod cost_register;
mod layer_register;

pub(crate) use activation_register::ACT_REGISTER;
pub(crate) use cost_register::COST_REGISTER;
pub(crate) use layer_register::LAYER_REGISTER;

pub use activation_register::register_activation;
pub use cost_register::register_cost;
pub use layer_register::register_layer;
