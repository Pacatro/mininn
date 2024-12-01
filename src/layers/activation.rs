use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use super::Layer;
use crate::{
    error::NNResult,
    nn::NNMode,
    utils::{ActivationFunc, ActivationFunction, Optimizer},
};

/// Represents an activation layer in a neural network.
///
/// The `Activation` layer applies a specific activation function to its input, modifying the data
/// based on the activation function used (e.g., `RELU`, `Sigmoid`, etc.). This layer is often used
/// in combination with other layers like `Dense` to introduce non-linearity into the model,
/// which is essential for learning complex patterns.
///
/// ## Attributes
///
/// * `input`: The input data to the activation layer. This is a 1D array of floating-point values
///   that represents the input from the previous layer in the network.
/// * `activation`: The activation function to apply to the input. It defines the non-linearity that
///   is applied to the input data (e.g., `RELU`, `Sigmoid`, `TANH`).
/// * `layer_type`: The type of the layer, which in this case is always `Activation`.
///   This helps identify the layer when saving or loading models, or when working with
///   dynamic layers in the network.
///
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Activation {
    input: ArrayD<f64>,
    activation: ActivationFunc, // TODO: CHANGE THIS TO USE A TRAIT
    layer_type: String,
}

impl Activation {
    /// Creates a new [`Activation`] layer
    ///
    /// ## Arguments
    ///
    /// - `activation`: The activation function of the layer
    ///
    /// ## Returns
    ///
    /// A new `Activation` layer with the specified activation function
    ///
    #[inline]
    pub fn new(activation: ActivationFunc) -> Self {
        Self {
            input: ArrayD::zeros(IxDyn(&[0])),
            activation,
            layer_type: "Activation".to_string(),
        }
    }

    /// Returns the activation function of this layer
    #[inline]
    pub fn activation(&self) -> ActivationFunc {
        self.activation
    }

    /// Sets the activation function of the layer
    ///
    /// ## Arguments
    ///
    /// - `activation`: The new activation function to be set for this layer
    ///
    #[inline]
    pub fn set_activation(&mut self, activation: ActivationFunc) {
        self.activation = activation;
    }
}

impl Layer for Activation {
    #[inline]
    fn layer_type(&self) -> String {
        self.layer_type.to_string()
    }

    #[inline]
    fn to_json(&self) -> NNResult<String> {
        Ok(serde_json::to_string(self)?)
    }

    #[inline]
    fn from_json(json_path: &str) -> NNResult<Box<dyn Layer>> {
        Ok(Box::new(serde_json::from_str::<Self>(json_path)?))
    }

    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &ArrayD<f64>, _mode: &NNMode) -> NNResult<ArrayD<f64>> {
        self.input = input.to_owned();
        Ok(self.activation.function(&self.input.view()))
    }

    #[inline]
    fn backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
        _learning_rate: f64,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f64>> {
        Ok(output_gradient.to_owned() * self.activation.derivate(&self.input.view()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_creation() {
        let activation = Activation::new(ActivationFunc::TANH);
        assert_eq!(activation.activation(), ActivationFunc::TANH);
    }

    #[test]
    fn test_forward_pass() {
        let mut activation = Activation::new(ActivationFunc::RELU);
        let input = vec![0.5, -0.3, 0.8];
        let input = ArrayD::from_shape_vec(IxDyn(&[input.len()]), input).unwrap();
        let output = activation.forward(&input, &NNMode::Test).unwrap();

        let expected_output = vec![0.5, 0.0, 0.8];
        let expected_output =
            ArrayD::from_shape_vec(IxDyn(&[expected_output.len()]), expected_output).unwrap();
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_backward_pass() {
        let mut activation = Activation::new(ActivationFunc::RELU);
        let input = vec![0.5, -0.3, 0.8];
        let input = ArrayD::from_shape_vec(IxDyn(&[input.len()]), input).unwrap();
        activation.forward(&input, &NNMode::Test).unwrap();

        let output_gradient = vec![1.0, 1.0, 1.0];
        let output_gradient =
            ArrayD::from_shape_vec(IxDyn(&[output_gradient.len()]), output_gradient).unwrap();
        let result = activation
            .backward(&output_gradient, 0.1, &Optimizer::GD, &NNMode::Test)
            .unwrap();

        let expected_result = vec![1.0, 0.0, 1.0];
        let expected_result =
            ArrayD::from_shape_vec(IxDyn(&[expected_result.len()]), expected_result).unwrap();
        assert_eq!(result, expected_result);
    }
}
