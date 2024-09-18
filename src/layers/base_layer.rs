use std::{any::Any, fmt::Debug};

use ndarray::{Array1, ArrayView1};

/// Defines the behavior for layers in a neural network.
/// 
/// Each layer must implement the `Debug` trait.
pub trait BaseLayer: Debug + Any {
    /// Performs the forward pass of the layer.
    /// 
    /// ## Arguments
    /// 
    /// - `input`: The reference to the input data as an [`Array2<f64>`]
    /// 
    /// ## Returns
    /// 
    /// The output data as an [`Array1<f64>`]
    /// 
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64>;

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
    fn backward(&mut self, output_gradient: ArrayView1<f64>, learning_rate: f64) -> Array1<f64>;

    fn as_any(&self) -> &dyn Any;
}
