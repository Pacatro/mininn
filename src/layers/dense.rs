use std::error::Error;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use serde::{Deserialize, Serialize};

use crate::utils::ActivationFunc;

use super::{LayerType, Layer};

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
/// - `layer_type`: The type of the layer [`LayerType::Dense`], which helps identify the layer in model operations 
///   such as saving or loading.
///
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Dense {
    weights: Array2<f64>,
    biases: Array1<f64>,
    input: Array1<f64>,
    activation: Option<ActivationFunc>,
    layer_type: LayerType
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
            layer_type: LayerType::Dense
        }
    }

    // /// Returns the number of inputs of the layer
    // #[inline]
    // pub fn ninputs(&self) -> usize {
    //     self.weights.ncols()
    // }

    // /// Returns the number of outputs of the layer
    // #[inline]
    // pub fn noutputs(&self) -> usize {
    //     self.weights.nrows()
    // }
}

impl Layer for Dense {
    #[inline]
    fn ninputs(&self) -> usize {
        self.weights.ncols()
    }

    #[inline]
    fn noutputs(&self) -> usize {
        self.weights.nrows()
    }

    #[inline]
    fn weights(&self) -> ArrayView2<f64> {
        self.weights.view()
    }

    #[inline]
    fn biases(&self) -> ArrayView1<f64> {
        self.biases.view()
    }

    #[inline]
    fn activation(&self) -> Option<ActivationFunc> {
        self.activation
    }

    #[inline]
    fn set_activation(&mut self, activation: Option<ActivationFunc>) {
        self.activation = activation
    }

    #[inline]
    fn set_weights(&mut self, weights: &Array2<f64>) {
        self.weights = weights.to_owned();
    }

    #[inline]
    fn set_biases(&mut self, biases: &Array1<f64>) {
        self.biases = biases.to_owned();
    }

    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input = input.clone();
        let sum = self.weights.dot(&self.input) + &self.biases;
        if let Some(act) = self.activation {
            act.function(&sum.view())
        } else {
            sum
        }
    }

    fn backward(&mut self, output_gradient: ArrayView1<f64>, learning_rate: f64) -> Result<Array1<f64>, Box<dyn Error>> {
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
            Ok(input_gradient * act.derivate(&self.input.view()))
        } else {
            Ok(input_gradient)
        }

    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> LayerType {
        self.layer_type
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    
    fn from_json(json_path: &str) -> Box<dyn Layer> {
        Box::new(serde_json::from_str::<Dense>(json_path).unwrap())
    }
}