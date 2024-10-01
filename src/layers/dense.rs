use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use serde::{Deserialize, Serialize};

use crate::{utils::ActivationFunc, error::NNResult};

use super::Layer;

/// Represents a fully connected (dense) layer in a neural network.
///
/// A `Dense` layer is a fundamental building block in neural networks where each neuron is connected 
/// to every neuron in the previous layer. It computes the weighted sum of the inputs, adds a bias, 
/// and then applies an optional activation function. 
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
    layer_type: String
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
            layer_type: "Dense".to_string()
        }
    }

    /// Returns the number of inputs for this layer
    ///
    /// ## Returns
    ///
    /// The number of columns in the weights matrix, which corresponds to the number of inputs
    ///
    #[inline]
    pub fn ninputs(&self) -> usize {
        self.weights.ncols()
    }

    /// Returns the number of outputs for this layer
    ///
    /// ## Returns
    ///
    /// The number of rows in the weights matrix, which corresponds to the number of outputs
    ///
    #[inline]
    pub fn noutputs(&self) -> usize {
        self.weights.nrows()
    }

    /// Returns a view of the weights matrix
    ///
    /// ## Returns
    ///
    /// An `ArrayView2<f64>` of the weights matrix
    ///
    #[inline]
    pub fn weights(&self) -> ArrayView2<f64> {
        self.weights.view()
    }

    /// Returns a view of the biases vector
    ///
    /// ## Returns
    ///
    /// An `ArrayView1<f64>` of the biases vector
    ///
    #[inline]
    pub fn biases(&self) -> ArrayView1<f64> {
        self.biases.view()
    }

    /// Returns the activation function of this layer
    ///
    /// ## Returns
    ///
    /// An `Option<ActivationFunc>` representing the activation function (if any)
    ///
    #[inline]
    pub fn activation(&self) -> Option<ActivationFunc> {
        self.activation
    }

    /// Sets the weights of the layer
    ///
    /// ## Arguments
    ///
    /// - `weights`: A reference to an `Array2<f64>` containing the new weights
    ///
    #[inline]
    pub fn set_weights(&mut self, weights: &Array2<f64>) {
        self.weights = weights.to_owned();
    }

    /// Sets the biases of the layer
    ///
    /// ## Arguments
    ///
    /// - `biases`: A reference to an `Array1<f64>` containing the new biases
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
        Ok(Box::new(serde_json::from_str::<Dense>(json_path)?))
    }

    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array1<f64>) -> NNResult<Array1<f64>> {
        self.input = input.clone();
        let sum = self.weights.dot(&self.input) + &self.biases;
        if let Some(act) = self.activation {
            Ok(act.function(&sum.view())?)
        } else {
            Ok(sum)
        }
    }

    fn backward(&mut self, output_gradient: ArrayView1<f64>, learning_rate: f64) -> NNResult<Array1<f64>> {
        // Calculate gradients
        let weights_gradient = output_gradient
            .to_owned()
            .to_shape((output_gradient.len(), 1))?
            .dot(&self.input.view().to_shape((1, self.input.len()))?);
        
        let input_gradient = self.weights.t().dot(&output_gradient);

        // Update weights and biases
        self.weights -= &(weights_gradient * learning_rate);
        self.biases -= &(output_gradient.to_owned() * learning_rate);

        if let Some(act) = self.activation {
            Ok(input_gradient * act.derivate(&self.input.view())?)
        } else {
            Ok(input_gradient)
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
        let output = dense.forward(&input).unwrap();

        // Verificamos que la salida tenga las dimensiones correctas (noutputs)
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_forward_pass_with_activation() {
        let mut dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        let input = array![0.5, -0.3, 0.8];
        let output = dense.forward(&input).unwrap();

        // Verificamos que la salida tenga las dimensiones correctas (noutputs) y que la activación sea correcta
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_backward_pass() {
        let mut dense = Dense::new(3, 2, Some(ActivationFunc::RELU));
        let input = array![0.5, -0.3, 0.8];
        dense.forward(&input).unwrap();

        // Simulamos un gradiente de salida
        let output_gradient = array![1.0, 1.0];
        let learning_rate = 0.01;
        let input_gradient = dense.backward(output_gradient.view(), learning_rate).unwrap();

        // Verificamos que el gradiente de entrada tenga las dimensiones correctas (ninputs)
        assert_eq!(input_gradient.len(), 3);
    }
}
