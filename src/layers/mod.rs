//! Contains the different types of layers that can be used in a neural network.
//!
//! A neural network is composed of multiple layers, each of which performs a specific task
//! to transform the input data into the desired output.
//!
//! All layers structs implement the [`Layer`] trait, which provides a common interface for
//! accessing and manipulating the layer's methods.
//!
//! ## Types of layers
//!
//! | Layer          | Description                                                                                                      |
//! |----------------|------------------------------------------------------------------------------------------------------------------|
//! | `Dense`        | Fully connected layer where each neuron connects to every neuron in the previous layer. It computes the weighted sum of inputs, adds a bias term, and applies an optional activation function (e.g., ReLU, Sigmoid). This layer is fundamental for transforming input data in deep learning models. |
//! | `Activation`   | Applies a non-linear transformation (activation function) to its inputs. Common activation functions include ReLU, Sigmoid, Tanh, and Softmax. These functions introduce non-linearity to the model, allowing it to learn complex patterns. |
//! | `Flatten`      | Flattens the input into a 1D array. This layer is useful when the input is a 2D array, but you want to treat it as a 1D array. |
//! | `Dropout`      | Applies dropout, a regularization technique where randomly selected neurons are ignored during training. This helps prevent overfitting by reducing reliance on specific neurons and forces the network to learn more robust features. Dropout is typically used in the training phase and is deactivated during inference. |
//! | `BatchNorm`    | Normalizes the input data by subtracting the mean and dividing by the standard deviation. This layer is useful for stabilizing the learning process and improving the convergence of the model. |
//! | `Conv`         | Applies convolutional operations to the input data. Convolutional layers are commonly used in image recognition tasks, where they help the model learn spatial patterns and features. |
//!
mod activation;
mod batchnorm;
mod conv;
mod dense;
mod dropout;
mod flatten;
mod layer;

pub use activation::Activation;
// pub use batchnorm::BatchNorm;
// pub use conv::Conv;
pub use dense::Dense;
pub use dropout::{Dropout, DEFAULT_DROPOUT_P};
pub use flatten::Flatten;
pub use layer::Layer;
