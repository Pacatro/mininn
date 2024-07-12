use std::fmt::Debug;

use ndarray::Array2;

/// Defines the behavior for layers in a neural network.
/// 
/// Each layer must implement the `Debug` trait.
pub trait BaseLayer: Debug {
    /// Performs the forward pass of the layer.
    /// 
    /// ## Arguments
    /// 
    /// - `input`: The input data as an [`Array2<f64>`]
    /// 
    /// ## Returns
    /// 
    /// The output data as an [`Array2<f64>`]
    /// 
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64>;

    /// Performs the backward pass of the layer.
    /// 
    /// ## Arguments
    /// 
    /// - `output_gradient`: The gradient of the loss with respect to the output of this layer.
    /// - `learning_rate`: The learning rate for updating the layer's parameters.
    /// 
    /// ## Returns
    /// 
    /// The gradient of the loss with respect to the input of this layer
    /// 
    fn backward(&mut self, output_gradient: Array2<f64>, learning_rate: f64) -> Array2<f64>;
}
