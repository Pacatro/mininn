//! Utility functions and types for the library.
//!
//! This module contains various utility functions and types that are used throughout the library.
//! These include activation functions, cost functions, metrics calculators, and optimizers.
//!
//! ## Activation functions
//!
//! The library provides a set of predefined activation functions that can be used in neural networks.
//! These functions are represented by the [`ActivationFunc`] enum and can be used to apply specific
//! activation functions to the input data during the forward pass of a neural network.
//!
//! | Activation function | Description                                                                                                      |
//! |---------------------|------------------------------------------------------------------------------------------------------------------|
//! | `STEP`              | Applies the step function to the input. This function maps the input to 0 if it is negative, and 1 if it is positive. |
//! | `SIGMOID`           | Applies the sigmoid function to the input. This function maps the input to a value between 0 and 1, which is the probability of the input being 1. |
//! | `RELU`              | Applies the rectified linear unit (ReLU) function to the input. This function maps the input to 0 if it is negative, and the input itself if it is positive. |
//! | `TANH`              | Applies the hyperbolic tangent function to the input. This function maps the input to a value between -1 and 1, which is the ratio of the input to the hyperbolic tangent of the input. |
//! | `SOFTMAX`           | Applies the softmax function to the input. This function maps the input to a probability distribution over the possible values of the input.|
//!
//! ## Cost functions
//!
//! The library provides a set of predefined cost functions that can be used in the training process.
//! These functions are represented by the [`Cost`] enum and can be used to measure the difference between
//! the predicted and actual values during the training process.
//!
//! | Cost function | Description                                                                                                      |
//! |---------------|------------------------------------------------------------------------------------------------------------------|
//! | `MSE`         | Mean Squared Error. This cost function measures the average squared difference between the predicted and actual values. |
//! | `MAE`         | Mean Absolute Error. This cost function measures the average absolute difference between the predicted and actual values. |
//! | `BCE`         | Binary Cross-Entropy. This cost function measures the average difference between the predicted and actual values, weighted by the binary cross-entropy loss function. |
//! | `CCE`         | Categorical Cross-Entropy. This cost function measures the average difference between the predicted and actual values, weighted by the categorical cross-entropy loss function. |
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
mod activation_func;
mod cost;
mod metrics;
mod optimizer;

pub(crate) use optimizer::OptimizerType;

pub use activation_func::{ActivationFunc, ActivationFunction};
pub use cost::{Cost, CostFunction};
pub use metrics::MetricsCalculator;
pub use optimizer::*;
