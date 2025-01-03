//! Utility functions and types for the library.
//!
//! This module contains various utility functions and types that are used throughout the library.
//! These include activation functions, cost functions, metrics calculators, and optimizers.
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
//! | [`Cost::MSE`]         | Mean Squared Error. This cost function measures the average squared difference between the predicted and actual values. |
//! | [`Cost::MSE`]         | Mean Squared Error. This cost function measures the average squared difference between the predicted and actual values. |AE`         | Mean Absolute Error. This cost function measures the average absolute difference between the predicted and actual values. |
//! | [`Cost::MSE`]         | Mean Squared Error. This cost function measures the average squared difference between the predicted and actual values. |BCE`         | Binary Cross-Entropy. This cost function measures the average difference between the predicted and actual values, weighted by the binary cross-entropy loss function. |
//! | [`Cost::MSE`]         | Mean Squared Error. This cost function measures the average squared difference between the predicted and actual values. |CCE`         | Categorical Cross-Entropy. This cost function measures the average difference between the predicted and actual values, weighted by the categorical cross-entropy loss function. |
//!
//! ## Optimizers
//!
//! The library provides a set of predefined optimizers that can be used in the training process.
//! These optimizers are represented by the [`Optimizer`] enum and can be used to update the weights and biases
//! of the neural network during the training process.
//!
//! | Optimizer | Description                                                                                                      |
//! |-----------|------------------------------------------------------------------------------------------------------------------|
//! | `GD`      | Gradient Descent. This optimizer updates the weights and biases of the neural network using the gradient of the loss function with respect to the weights and biases. |
//! | `Momentum`| Momentum. This optimizer updates the weights and biases of the neural network using the gradient of the loss function with respect to the weights and biases, but with a momentum term that helps accelerate the learning process. |
//! | `Adam`    | Adam. This optimizer updates the weights and biases of the neural network using the gradient of the loss function with respect to the weights and biases, but with a momentum term that helps accelerate the learning process and a learning rate that adjusts the step size of the gradient descent. |
//!
mod act;
mod cost;
mod formatting;
mod metrics;
mod nn_util;
mod optimizer;

pub(crate) use optimizer::OptimizerType;

pub use act::{Act, ActCore, ActivationFunction};
pub use cost::{Cost, CostCore, CostFunction};
pub use formatting::MSGPackFormatting;
pub use metrics::MetricsCalculator;
pub use nn_util::NNUtil;
pub use optimizer::*;
