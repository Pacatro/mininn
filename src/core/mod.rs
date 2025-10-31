//! Core module of the library.
//!
//! This module contains the core components of the library, including the `NN` struct and its associated methods.
//!
//! The `NN` struct represents a neural network, which is a container of layers that can be trained and used for
//! various tasks, such as classification, regression, and pattern recognition.
//!
//! ## Activation functions
//!
//! The library provides a set of predefined activation functions that can be used in neural networks.
//! These functions are represented by the [`Act`] enum and can be used to apply specific
//! activation functions to the input data during the forward pass of a neural network.
//!
//! | Activation function        | Definition                                  |
//! |----------------------------|---------------------------------------------|
//! | [`Act::Step`]              | `step(x) = 1 if x > 0 else 0`               |
//! | [`Act::Sigmoid`]           | `sigmoid(x) = 1 / (1 + exp(-x))`            |
//! | [`Act::ReLU`]              | `ReLU(x) = x if x > 0 else 0`               |
//! | [`Act::Tanh`]              | `tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))` |
//! | [`Act::Softmax`]           | `softmax(x) = exp(x) / sum(exp(x))`         |
//!
//! ## Cost functions
//!
//! The library provides a set of predefined cost functions that can be used in the training process.
//! These functions are represented by the [`Cost`] enum and can be used to measure the difference between
//! the predicted and actual values during the training process.
//!
//! | Cost function | Description                                                                                                      |
//! |---------------|------------------------------------------------------------------------------------------------------------------|
//! | [`Cost::MSE`] | Mean Squared Error. This cost function measures the average squared difference between the predicted and actual values. |
//! | [`Cost::MAE`] | Mean Absolute Error. This cost function measures the average absolute difference between the predicted and actual values. |
//! | [`Cost::BCE`] | Binary Cross-Entropy. This cost function measures the average difference between the predicted and actual values, weighted by the binary cross-entropy loss function. |
//! | [`Cost::CCE`] | Categorical Cross-Entropy. This cost function measures the average difference between the predicted and actual values, weighted by the categorical cross-entropy loss function. |
//!
//! ## Optimizers
//!
//! The library provides a set of predefined optimizers that can be used in the training process.
//! These optimizers are represented by the [`Optimizer`] enum and can be used to update the weights and biases
//! of the neural network during the training process.
//!
//! | Optimizer   | Description                                                                                                      |
//! |-------------|------------------------------------------------------------------------------------------------------------------|
//! | [`Optimizer::SGD`]       | Stochastic Gradient Descent. This optimizer updates the weights and biases of the neural network using the gradient of the loss function with respect to the weights and biases. |
//! | [`Optimizer::Momentum`]  | Momentum. This optimizer updates the weights and biases of the neural network using the gradient of the loss function with respect to the weights and biases, but with a momentum term that helps accelerate the learning process. |
//! | [`Optimizer::Adam`]      | Adam. This optimizer updates the weights and biases of the neural network using the gradient of the loss function with respect to the weights and biases, but with a momentum term that helps accelerate the learning process and a learning rate that adjusts the step size of the gradient descent. |
//!
mod act;
mod cost;
mod error;
mod nn;
mod optimizer;
mod train_config;

pub(crate) use optimizer::OptimizerType;

pub use act::{Act, ActCore, ActivationFunction};
pub use cost::{Cost, CostCore, CostFunction};
pub use error::*;
pub use nn::*;
pub use optimizer::*;
pub use train_config::TrainConfig;
