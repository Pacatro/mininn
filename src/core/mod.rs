//! Core module of the library.
//!
//! This module contains the core components of the library, including the `NN` struct and its associated methods.
//!
//! The `NN` struct represents a neural network, which is a container of layers that can be trained and used for
//! various tasks, such as classification, regression, and pattern recognition.
//!
mod error;
mod nn;
mod train_config;

pub use error::*;
pub use nn::*;
pub use train_config::TrainConfig;
