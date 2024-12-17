use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD};
use ndarray_rand::{rand::distributions::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

use crate::{
    core::{MininnError, NNMode, NNResult},
    layers::{Layer, TrainLayer},
    utils::{ActivationFunction, MSGPackFormatting, Optimizer, OptimizerType},
};

use mininn_derive::Layer;

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
/// - `weights`: A 2D array of weights where each element represents the weight between
///   a neuron in this layer and a neuron in the previous layer.
/// - `biases`: A 1D array of biases where each bias is applied to the corresponding neuron
///   in the layer.
/// - `input`: The input to the layer as a 1D array which is the output from the previous layer.
/// - `activation`: An optional activation function to be applied to the weighted sum
///   of the inputs. If `None`, no activation function is applied.
/// - `layer_type`: The type of the layer as a `String` which helps identify the layer in model operations
///   such as saving or loading.
///
#[derive(Layer, Clone, Debug, Serialize, Deserialize)]
pub struct Dense {
    weights: Array2<f32>,
    biases: Array1<f32>,
    input: Array1<f32>,
    activation: Option<Box<dyn ActivationFunction>>,
}

impl Dense {
    ///  Creates a new [`Dense`] layer
    ///
    /// ## Arguments
    ///
    /// - `ninputs`: The number of inputs of the layer
    /// - `noutputs`: The number of outputs of the layer
    ///
    /// ## Returns
    ///
    /// A new `Dense` layer with the specified number of inputs and outputs
    ///
    /// ## Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// let dense = Dense::new(3, 2);
    /// assert_eq!(dense.ninputs(), 3);
    /// assert_eq!(dense.noutputs(), 2);
    /// ```
    ///
    #[inline]
    pub fn new(ninputs: usize, noutputs: usize) -> Self {
        Self {
            weights: Array2::random((noutputs, ninputs), Uniform::new(-1.0, 1.0)),
            biases: Array1::random(noutputs, Uniform::new(-1.0, 1.0)),
            input: Array1::zeros(ninputs),
            activation: None,
        }
    }

    /// Applies an activation function to the layer
    ///
    /// ## Arguments
    ///
    /// - `activation`: The activation function to be applied to the layer
    ///   (e.g., `Act::ReLU`)
    ///
    /// ## Returns
    ///
    /// A new `Dense` layer with the specified activation function
    ///
    /// ## Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    ///
    /// let dense = Dense::new(3, 2).apply(Act::ReLU);
    ///
    /// assert_eq!(dense.activation().unwrap(), "ReLU");
    /// ```
    ///
    pub fn apply(mut self, activation: impl ActivationFunction + 'static) -> Self {
        self.activation = Some(Box::new(activation));
        self
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
    pub fn weights(&self) -> ArrayView2<f32> {
        self.weights.view()
    }

    /// Returns a view of the biases vector
    #[inline]
    pub fn biases(&self) -> ArrayView1<f32> {
        self.biases.view()
    }

    /// Returns the activation function of this layer if any
    #[inline]
    pub fn activation(&self) -> Option<&str> {
        self.activation.as_ref().map(|a| a.as_ref().name())
    }

    /// Sets the weights of the layer
    ///
    /// ## Arguments
    ///
    /// - `weights`: A reference to an [`Array2<f32>`] containing the new weights
    ///
    #[inline]
    pub fn set_weights(&mut self, weights: &Array2<f32>) {
        self.weights = weights.to_owned();
    }

    /// Sets the biases of the layer
    ///
    /// ## Arguments
    ///
    /// - `biases`: A reference to an [`Array1<f32>`] containing the new biases
    ///
    #[inline]
    pub fn set_biases(&mut self, biases: &Array1<f32>) {
        self.biases = biases.to_owned();
    }
}

impl TrainLayer for Dense {
    fn forward(&mut self, input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        self.input = input.to_owned().into_dimensionality()?;

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
        match &self.activation {
            Some(act) => Ok(act.function(&sum.into_dimensionality()?.view())),
            None => Ok(sum.into_dimensionality()?),
        }
    }

    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f32>,
        learning_rate: f32,
        optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f32>> {
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

        let dim_output: Array1<f32> = output_gradient.to_owned().into_dimensionality()?;
        let dim_input: ArrayD<f32> = self.input.to_owned().into_dimensionality()?;

        let input_gradient = self.weights.t().dot(&dim_output);

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
            &dim_output.view(),
            learning_rate,
        );

        match &self.activation {
            Some(act) => Ok(input_gradient * act.derivate(&dim_input.view())),
            None => Ok(input_gradient.into_dyn()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::Act;

    use super::*;
    use ndarray::array;

    #[test]
    fn test_dense_creation() {
        let dense = Dense::new(3, 2).apply(Act::ReLU);
        assert_eq!(dense.ninputs(), 3);
        assert_eq!(dense.noutputs(), 2);
        assert!(dense.activation().is_some());
        assert_eq!(dense.activation().unwrap(), "ReLU");
    }

    #[test]
    fn test_forward_pass_without_name() {
        let mut dense = Dense::new(3, 2);
        let input = array![0.5, -0.3, 0.8].into_dyn();
        let output = dense.forward(input.view(), &NNMode::Train).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_forward_pass_with_name() {
        let mut dense = Dense::new(3, 2).apply(Act::ReLU);
        let input = array![0.5, -0.3, 0.8].into_dyn();
        let output = dense.forward(input.view(), &NNMode::Train).unwrap();

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_forward_pass_empty_input() {
        let mut dense = Dense::new(3, 2).apply(Act::ReLU);
        let input: Array1<f32> = array![];
        let result = dense.forward(input.into_dyn().view(), &NNMode::Train);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Layer Error: Input is empty, cannot forward pass."
        );
    }

    #[test]
    fn test_forward_pass_empty_weights() {
        let mut dense = Dense::new(3, 2).apply(Act::ReLU);
        dense.weights = array![[]];
        let input = array![0.5, -0.3, 0.8].into_dyn();
        let result = dense.forward(input.view(), &NNMode::Train);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Layer Error: Weights are empty, cannot forward pass."
        );
    }

    #[test]
    fn test_backward_pass() {
        let mut dense = Dense::new(3, 2).apply(Act::ReLU);
        let input = array![0.5, -0.3, 0.8].into_dyn();
        dense.forward(input.view(), &NNMode::Train).unwrap();
        let output_gradient = array![1.0, 1.0].into_dyn();
        let learning_rate = 0.01;
        let input_gradient = dense
            .backward(
                output_gradient.view(),
                learning_rate,
                &Optimizer::GD,
                &NNMode::Train,
            )
            .unwrap();
        assert_eq!(input_gradient.len(), 3);
    }

    #[test]
    fn test_backward_pass_empty_input() {
        let mut dense = Dense::new(3, 2).apply(Act::ReLU);
        let input: Array1<f32> = array![];
        dense.input = input.clone();

        let result = dense.backward(input.into_dyn().view(), 0.1, &Optimizer::GD, &NNMode::Train);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Layer Error: Input is empty, cannot backward pass."
        );
    }

    #[test]
    fn test_backward_pass_empty_weights() {
        let mut dense = Dense::new(3, 2).apply(Act::ReLU);
        dense.weights = array![[]];
        let output_gradient = array![1.0, 1.0].into_dyn();
        let learning_rate = 0.01;
        let result = dense.backward(
            output_gradient.view(),
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
        let dense = Dense::new(3, 2).apply(Act::ReLU);
        assert_eq!(dense.layer_type(), "Dense");
    }

    // #[test]
    // fn test_to_json() {
    //     let mut dense = Dense::new(3, 2).apply(Act::ReLU);
    //     dense.set_weights(&array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    //     dense.set_biases(&array![1.0, 2.0]);
    //     let json = dense.to_json().unwrap();
    //     assert_eq!(
    //         json,
    //         "{\"weights\":{\"v\":1,\"dim\":[3,2],\"data\":[1.0,2.0,3.0,4.0,5.0,6.0]},\"biases\":{\"v\":1,\"dim\":[2],\"data\":[1.0,2.0]},\"input\":{\"v\":1,\"dim\":[3],\"data\":[0.0,0.0,0.0]},\"activation\":\"ReLU\",\"layer_type\":\"Dense\"}"
    //     );
    // }

    // #[test]
    // fn test_to_msg_pack() {
    //     let mut dense = Dense::new(3, 2).apply(Act::ReLU);
    //     dense.set_weights(&array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    //     dense.set_biases(&array![1.0, 2.0]);
    //     let bytes = dense.to_msgpack().unwrap();
    //     assert!(!bytes.is_empty());
    //     let deserialized: Box<dyn Layer> = Dense::from_msgpack(&bytes).unwrap();
    //     assert_eq!(dense.layer_type(), deserialized.layer_type());
    // }
}
