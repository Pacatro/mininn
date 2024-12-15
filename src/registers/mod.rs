//! This module contains the register for activations, costs and layers.

mod global_register;
mod register;

pub(crate) use global_register::REGISTER;

pub use register::Register;
