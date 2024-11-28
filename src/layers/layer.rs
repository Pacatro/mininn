use ndarray::Array1;
use std::{any::Any, fmt::Debug};

use crate::{error::NNResult, nn::NNMode, utils::Optimizer};

/// Defines the behavior for layers in a neural network.
///
/// Each layer must implement the `Debug` trait to allow for the debugging of its state, and the `Any` trait
/// to enable downcasting for dynamic typing at runtime. Layers that implement this trait can participate in both
/// the forward and backward passes of a neural network's training process.
///
pub trait Layer: Debug + Any {
    /// Returns the type of the layer.
    ///
    /// This method allows identification of the specific type of layer without needing to downcast.
    ///
    /// ## Returns
    ///
    /// - The type of this layer.
    ///
    fn layer_type(&self) -> String;

    /// Serializes the layer to a JSON string representation.
    ///
    /// This method is useful for saving the layer's state and configuration to a file or database.
    ///
    /// ## Returns
    ///
    /// - A `String` containing the JSON representation of the layer.
    ///
    fn to_json(&self) -> NNResult<String>;

    /// Deserializes a JSON string into a new instance of the layer.
    ///
    /// This method is used to reconstruct a layer from its JSON representation, typically when
    /// loading a saved model.
    ///
    /// ## Arguments
    ///
    /// - `json`: A string slice containing the JSON representation of the layer.
    ///
    /// ## Returns
    ///
    /// - A `Box<dyn Layer>` containing the deserialized layer.
    ///
    fn from_json(json: &str) -> NNResult<Box<dyn Layer>>
    where
        Self: Sized;

    /// Returns a reference to the layer as an `Any` type.
    ///
    /// This method allows downcasting the layer to its concrete type, enabling dynamic behavior
    /// when the type of the layer is not known at compile time.
    ///
    /// ## Returns
    ///
    /// - A reference to the layer as a trait object of type `Any`.
    /// This can be used to downcast the layer to its concrete type using `downcast_ref`.
    ///
    fn as_any(&self) -> &dyn Any;

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
    fn forward(&mut self, input: &Array1<f64>, mode: &NNMode) -> NNResult<Array1<f64>>;

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
    /// - `optimizer`: The optimizer used to update the layer's parameters. It determines the specific
    ///   algorithm used to minimize the loss function during training.
    ///
    /// ## Returns
    ///
    /// - The gradient of the loss function with respect to the input of this layer.
    ///   This is passed to the preceding layer to continue the backpropagation process.
    ///
    fn backward(
        &mut self,
        output_gradient: &Array1<f64>,
        learning_rate: f64,
        optimizer: &Optimizer,
        mode: &NNMode,
    ) -> NNResult<Array1<f64>>;
}
