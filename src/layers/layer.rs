use std::{any::Any, error::Error, fmt::Debug};
use hdf5::H5Type;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::utils::ActivationFunc;

/// Represents the type of a layer in a neural network.
///
/// `LayerType` is used to distinguish between different kinds of layers in a neural network. 
/// Each variant corresponds to a specific type of layer commonly used in machine learning models.
///
/// This enum is serialized and deserialized using the `serde` library, and supports HDF5 serialization
/// via the `H5Type` trait for compatibility with HDF5 file formats.
///
/// ## Variants
///
/// - `Dense`: Represents a fully connected (dense) layer.
/// - `Activation`: Represents an activation layer (e.g., ReLU, Sigmoid, etc.).
/// - `Pooling`: Represents a pooling layer (e.g., MaxPooling, AveragePooling).
/// - `Conv2D`: Represents a 2D convolutional layer.
///
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Serialize, Deserialize, H5Type)]
#[repr(u8)]
pub enum LayerType {
    Dense = 0,
    Activation = 1,
    Pooling = 2,
    Conv2D = 3,
}

/// Defines the behavior for layers in a neural network.
/// 
/// Each layer must implement the `Debug` trait to allow for the debugging of its state, and the `Any` trait
/// to enable downcasting for dynamic typing at runtime. Layers that implement this trait can participate in both 
/// the forward and backward passes of a neural network's training process.
/// 
pub trait Layer: Debug + Any {
    /// Returns the number of inputs of the layer
    fn ninputs(&self) -> usize;

    /// Returns the number of outputs of the layer
    fn noutputs(&self) -> usize;

    /// Returns the weights of the layer
    fn weights(&self) -> ArrayView2<f64>;
    
    /// Returns the biases of the layer
    fn biases(&self) -> ArrayView1<f64>;
    
    /// Returns the activation function of the layer
    fn activation(&self) -> Option<ActivationFunc>;
    
    /// Sets a new activation function for the layer
    ///
    /// ## Arguments
    /// 
    /// - `activation`: The new activation fucntion of the layer
    /// 
    fn set_activation(&mut self, activation: Option<ActivationFunc>);
    
    /// Set the weights of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `weights`: The new weights of the layer
    /// 
    fn set_weights(&mut self, weights: &Array2<f64>);
    
    /// Set the biases of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `biases`: The new biases of the layer
    /// 
    fn set_biases(&mut self, biases: &Array1<f64>);

    /// Performs the forward pass of the layer.
    /// 
    /// The forward pass is responsible for computing the output of the layer given the input data. 
    /// This method is usually invoked when passing data through the network during inference or training.
    ///
    /// ## Arguments
    /// 
    /// - `input`: A reference to the input data as an [`Array1<f64>`]. 
    ///   The input is typically a 1-dimensional array of floating-point numbers (f64).
    /// 
    /// ## Returns
    /// 
    /// - The output data as an [`Array1<f64>`]. The transformation depends on the type of layer 
    ///   (e.g., activation, dense, convolutional, etc.), and the specific operations applied.
    /// 
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64>;

    /// Performs the backward pass of the layer.
    /// 
    /// The backward pass computes the gradient of the loss function with respect to the input of this layer, 
    /// which is necessary for updating the parameters of the layer during backpropagation.
    /// 
    /// ## Arguments
    /// 
    /// - `output_gradient`: The gradient of the loss function with respect to the output of this layer. 
    ///   This is typically computed by the subsequent layer during backpropagation.
    /// - `learning_rate`: The learning rate used to adjust the layer's parameters. It controls how much 
    ///   the weights of the layer are updated during training.
    /// 
    /// ## Returns
    /// 
    /// - The gradient of the loss function with respect to the input of this layer. 
    ///   This is passed to the preceding layer to continue the backpropagation process.
    /// 
    fn backward(&mut self, output_gradient: ArrayView1<f64>, learning_rate: f64) -> Result<Array1<f64>, Box<dyn Error>>;

    /// Returns a reference to the layer as an `Any` type.
    /// 
    /// This method allows downcasting the layer to its concrete type, enabling dynamic behavior 
    /// when the type of the layer is not known at compile time.
    /// 
    /// ## Returns
    /// 
    /// - A reference to the layer as a trait object of type `Any`.
    ///   This can be used to downcast the layer to its concrete type using `downcast_ref`.
    /// 
    fn as_any(&self) -> &dyn Any;

    /// Returns the type of the layer as a `LayerType` enum.
    fn layer_type(&self) -> LayerType;

    fn to_json(&self) -> String;

    fn from_json(json: &str) -> Box<dyn Layer> where Self: Sized;
}