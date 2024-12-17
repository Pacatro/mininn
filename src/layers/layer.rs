use dyn_clone::DynClone;
use ndarray::{ArrayD, ArrayViewD};
use std::{any::Any, fmt::Debug};

use crate::{
    core::{NNMode, NNResult},
    utils::{MSGPackFormat, Optimizer},
};

/// Trait defining the training behavior of a layer in a neural network.
///
/// The `TrainLayer` trait provides methods required to compute the forward and backward passes
/// during training and inference. It focuses on the computational transformations
/// performed by a layer and the calculation of gradients for backpropagation.
///
/// ## Key Responsibilities
/// - **Forward Pass**: Processes the input data to produce an output.
/// - **Backward Pass**: Calculates gradients with respect to the input and updates parameters.
///
pub trait TrainLayer {
    /// Computes the forward pass of the layer.
    ///
    /// The forward pass applies a transformation to the input data, producing the output.
    /// This is used both during training and inference.
    ///
    /// ## Arguments
    /// - `input`: A view of the input data as a multi-dimensional array.
    /// - `mode`: Specifies the mode of the neural network (e.g., training or inference).
    ///
    /// ## Returns
    /// - A result containing the transformed output data or an error if the computation fails.
    ///
    fn forward(&mut self, input: ArrayViewD<f32>, mode: &NNMode) -> NNResult<ArrayD<f32>>;

    /// Computes the backward pass of the layer.
    ///
    /// The backward pass calculates the gradient of the loss with respect to the input
    /// and updates the layer's parameters based on the provided gradients and optimizer.
    ///
    /// ## Arguments
    /// - `output_gradient`: The gradient of the loss with respect to the layer's output.
    /// - `learning_rate`: A scalar that determines the step size during parameter updates.
    /// - `optimizer`: The optimizer that defines how parameters are updated.
    /// - `mode`: Specifies the mode of the neural network (e.g., training or inference).
    ///
    /// ## Returns
    /// - A result containing the gradient of the loss with respect to the input or an error if the computation fails.
    ///
    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f32>,
        learning_rate: f32,
        optimizer: &Optimizer,
        mode: &NNMode,
    ) -> NNResult<ArrayD<f32>>;
}

/// Core trait defining the behavior of a layer in a neural network.
///
/// The `Layer` trait establishes a common interface for all layers, ensuring they can
/// be integrated into a neural network and participate in both inference and training phases.
///
/// ## Key Features
/// - **Type Identification**: Provides the layer's type as a string for debugging or serialization.
/// - **Runtime Polymorphism**: Enables dynamic layer management through `Any` and `DynClone`.
/// - **Extensibility**: Custom layers can implement this trait to seamlessly integrate into the framework.
///
/// ## Required Traits
/// Layers implementing `Layer` must also implement:
/// - `TrainLayer`: For forward and backward computations.
/// - `MSGPackFormat`: For serialization and deserialization in MessagePack format.
/// - `Any`: For runtime downcasting of the layer.
/// - `DynClone`: For cloning layer instances dynamically.
/// - `Debug`: For inspecting layer properties during debugging.
///
/// ## Example
/// ```rust
/// use mininn::prelude::*;
/// use mininn_derive::Layer;
/// use ndarray::{ArrayD, ArrayViewD};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Layer, Debug, Clone, Serialize, Deserialize)]
/// pub struct DenseLayer {
///     weights: ArrayD<f32>,
///     biases: ArrayD<f32>,
/// }
///
/// impl TrainLayer for DenseLayer {
///     fn forward(&mut self, input: ArrayViewD<f32>, mode: &NNMode) -> NNResult<ArrayD<f32>> {
///         // Perform forward pass computation
///         todo!()
///     }
///
///     fn backward(
///         &mut self,
///         output_gradient: ArrayViewD<f32>,
///         learning_rate: f32,
///         optimizer: &Optimizer,
///         mode: &NNMode,
///     ) -> NNResult<ArrayD<f32>> {
///         // Perform backward pass computation
///         todo!()
///     }
/// }
/// ```
///
pub trait Layer: TrainLayer + MSGPackFormat + Any + DynClone + Debug {
    /// Returns the type of the layer as a string.
    ///
    /// This method is useful for debugging, serialization, and distinguishing
    /// between different layer types at runtime.
    fn layer_type(&self) -> &str;

    /// Provides a reference to the layer as a trait object of type `Any`.
    ///
    /// This enables dynamic downcasting of the layer to its concrete type,
    /// which is useful for custom logic based on the specific layer type.
    fn as_any(&self) -> &dyn Any;
}

dyn_clone::clone_trait_object!(Layer);
