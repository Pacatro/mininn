use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_rand::{rand::distributions::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

use crate::{
    error::NNResult,
    nn::NNMode,
    utils::{ActivationFunc, Optimizer, OptimizerType},
    MininnError,
};

use super::Layer;

/// Represents a fully connected (dense) layer in a neural network.
///
/// A `Dense` layer is a core component of neural networks where every neuron in the layer is
/// connected to every neuron in the preceding layer. It performs the following operations:
///
/// 1. Computes the weighted sum of inputs.
/// 2. Adds a bias term to each weighted sum.
/// 3. Optionally applies an activation function to introduce non-linearity.
///
///
/// ## Attributes
///
/// - `weights`: A 2D array of weights [`Array2<f64>`] where each element represents the weight between
///   a neuron in this layer and a neuron in the previous layer.
/// - `biases`: A 1D array of biases [`Array1<f64>`] where each bias is applied to the corresponding neuron
///   in the layer.
/// - `input`: The input to the layer as a 1D array [`Array1<f64>`], which is the output from the previous layer.
/// - `activation`: An optional activation function [`ActivationFunc`] to be applied to the weighted sum
///   of the inputs. If `None`, no activation function is applied.
/// - `layer_type`: The type of the layer as a `String` which helps identify the layer in model operations
///   such as saving or loading.
///
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Dense {
    weights: Array2<f64>,
    biases: Array1<f64>,
    input: Array1<f64>,
    activation: Option<ActivationFunc>,
    layer_type: String,
}

impl Dense {
    ///  Creates a new [`Dense`] layer
    ///
    /// ## Arguments
    ///
    /// - `ninputs`: The number of inputs of the layer
    /// - `noutputs`: The number of outputs of the layer
    ///
    #[inline]
    pub fn new(ninputs: usize, noutputs: usize, activation: Option<ActivationFunc>) -> Self {
        Self {
            weights: Array2::random((noutputs, ninputs), Uniform::new(-1.0, 1.0)),
            biases: Array1::random(noutputs, Uniform::new(-1.0, 1.0)),
            input: Array1::zeros(ninputs),
            activation,
            layer_type: "Dense".to_string(),
        }
    }

    /// Returns the number of inputs for this layer
    #[inline]
    pub fn ninputs(&self) -> usize {
        self.weights.ncols()
    }

    /// Returns the number of outputs for this layer
    #[inline]
    pub fn noutputs(&self) -> usize {
        self.weights.nrows()
    }

    /// Returns a view of the weights matrix
    #[inline]
    pub fn weights(&self) -> ArrayView2<f64> {
        self.weights.view()
    }

    /// Returns a view of the biases vector
    #[inline]
    pub fn biases(&self) -> ArrayView1<f64> {
        self.biases.view()
    }

    /// Returns the activation function of this layer if any
    #[inline]
    pub fn activation(&self) -> Option<ActivationFunc> {
        self.activation
    }

    /// Sets the weights of the layer
    ///
    /// ## Arguments
    ///
    /// - `weights`: A reference to an [`Array2<f64>`] containing the new weights
    ///
    #[inline]
    pub fn set_weights(&mut self, weights: &Array2<f64>) {
        self.weights = weights.to_owned();
    }

    /// Sets the biases of the layer
    ///
    /// ## Arguments
    ///
    /// - `biases`: A reference to an [`Array1<f64>`] containing the new biases
    ///
    #[inline]
    pub fn set_biases(&mut self, biases: &Array1<f64>) {
        self.biases = biases.to_owned();
    }

    /// Sets the activation function of the layer
    ///
    /// ## Arguments
    ///
    /// - `activation`: An `Option<ActivationFunc>` representing the new activation function (or None)
    ///
    #[inline]
    pub fn set_activation(&mut self, activation: Option<ActivationFunc>) {
        self.activation = activation
    }
}

impl Layer for Dense {
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

    fn forward(&mut self, input: &Array1<f64>, _mode: &NNMode) -> NNResult<Array1<f64>> {
        self.input = input.to_owned();

        if self.input.is_empty() {
            return Err(MininnError::LayerError(
                "Input is empty, cannot forward pass".to_string(),
            ));
        }

        if self.weights.is_empty() {
            return Err(MininnError::LayerError(
                "Weights are empty, cannot forward pass".to_string(),
            ));
        }

        let sum = self.weights.dot(&self.input) + &self.biases;
        match self.activation {
            Some(act) => Ok(act.function(&sum.view())),
            None => Ok(sum),
        }
    }

    fn backward(
        &mut self,
        output_gradient: &Array1<f64>,
        learning_rate: f64,
        optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<Array1<f64>> {
        if self.input.is_empty() {
            return Err(MininnError::LayerError(
                "Input is empty, cannot backward pass".to_string(),
            ));
        }

        if self.weights.is_empty() {
            return Err(MininnError::LayerError(
                "Weights are empty, cannot backward pass".to_string(),
            ));
        }

        // Calculate gradients
        let weights_gradient = output_gradient
            .to_owned()
            .to_shape((output_gradient.len(), 1))?
            .dot(&self.input.view().to_shape((1, self.input.len()))?);

        let input_gradient = self.weights.t().dot(&output_gradient.view());

        let mut optimizer_type = match optimizer {
            Optimizer::GD => OptimizerType::GD,
            Optimizer::Momentum(momentum) => {
                OptimizerType::new_momentum(*momentum, self.weights.dim(), self.biases.len())
            } // Optimizer::Adam(beta1, beta2, epsilon) => OptimizerType::new_adam(
              //     self.weights.dim(),
              //     self.biases.len(),
              //     *beta1,
              //     *beta2,
              //     *epsilon,
              // ),
        };

        // Update weights and biases
        optimizer_type.optimize(
            &mut self.weights,
            &mut self.biases,
            &weights_gradient.view(),
            &output_gradient.view(),
            learning_rate,
        );

        match self.activation {
            Some(act) => Ok(input_gradient * act.derivate(&self.input.view())),
            None => Ok(input_gradient),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dense_creation() {
        let dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        assert_eq!(dense.ninputs(), 3);
        assert_eq!(dense.noutputs(), 2);
        assert_eq!(dense.activation(), Some(ActivationFunc::RELU));
    }

    #[test]
    fn test_forward_pass_without_activation() {
        let mut dense = Dense::new(3, 2, None);
        let input = array![0.5, -0.3, 0.8];
        let output = dense.forward(&input, &NNMode::Train).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_forward_pass_with_activation() {
        let mut dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        let input = array![0.5, -0.3, 0.8];
        let output = dense.forward(&input, &NNMode::Train).unwrap();

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_forward_pass_empty_input() {
        let mut dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        let input = array![];
        let result = dense.forward(&input, &NNMode::Train);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Layer Error: Input is empty, cannot forward pass."
        );
    }

    #[test]
    fn test_forward_pass_empty_weights() {
        let mut dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        dense.weights = array![[]];
        let input = array![0.5, -0.3, 0.8];
        let result = dense.forward(&input, &NNMode::Train);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Layer Error: Weights are empty, cannot forward pass."
        );
    }

    #[test]
    fn test_backward_pass() {
        let mut dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        let input = array![0.5, -0.3, 0.8];
        dense.forward(&input, &NNMode::Train).unwrap();
        let output_gradient = array![1.0, 1.0];
        let learning_rate = 0.01;
        let input_gradient = dense
            .backward(
                &output_gradient,
                learning_rate,
                &Optimizer::GD,
                &NNMode::Train,
            )
            .unwrap();
        assert_eq!(input_gradient.len(), 3);
    }

    #[test]
    fn test_backward_pass_empty_input() {
        let mut dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        dense.input = array![];
        let result = dense.backward(&array![], 0.1, &Optimizer::GD, &NNMode::Train);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Layer Error: Input is empty, cannot backward pass."
        );
    }

    #[test]
    fn test_backward_pass_empty_weights() {
        let mut dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        dense.weights = array![[]];
        let output_gradient = array![1.0, 1.0];
        let learning_rate = 0.01;
        let result = dense.backward(
            &output_gradient,
            learning_rate,
            &Optimizer::GD,
            &NNMode::Train,
        );
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Layer Error: Weights are empty, cannot backward pass."
        );
    }

    #[test]
    fn test_layer_type() {
        let dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        assert_eq!(dense.layer_type(), "Dense");
    }

    #[test]
    fn test_to_json() {
        let mut dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        dense.set_weights(&array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        dense.set_biases(&array![1.0, 2.0]);
        let json = dense.to_json().unwrap();
        assert_eq!(
            json,
            "{\"weights\":{\"v\":1,\"dim\":[3,2],\"data\":[1.0,2.0,3.0,4.0,5.0,6.0]},\"biases\":{\"v\":1,\"dim\":[2],\"data\":[1.0,2.0]},\"input\":{\"v\":1,\"dim\":[3],\"data\":[0.0,0.0,0.0]},\"activation\":\"RELU\",\"layer_type\":\"Dense\"}"
        );
    }
}
