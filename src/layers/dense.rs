use std::error::Error;

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use crate::layers::Activation;

use super::Layer;

/// Represents a fully connected layer
/// 
/// ## Attributes
/// 
/// - `weights`: The weights of the layer as an [`Array2<f64>`]
/// - `biases`: The biases of the layer as an [`Array1<f64>`]
/// - `input`: The input of the layer as an [`Array1<f64>`]
/// - `activation`: The activation function of the layer as an [`Activation`]
/// 
#[derive(Debug, PartialEq, Clone)]
pub struct Dense {
    weights: Array2<f64>,
    biases: Array1<f64>,
    input: Array1<f64>,
    activation: Activation
}

impl Dense {
    ///  Creates a new [`Dense`] layer
    /// 
    /// ## Arguments
    /// 
    /// - `ninput`: The number of inputs of the layer
    /// - `noutput`: The number of outputs of the layer
    ///
    #[inline]
    pub fn new(ninput: usize, noutput: usize, activation: Activation) -> Self {
        Self {
            weights: Array2::random((noutput, ninput), Uniform::new(-1.0, 1.0)),
            biases: Array1::random(noutput, Uniform::new(-1.0, 1.0)),
            input: Array1::zeros(ninput),
            activation
        }
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.weights.ncols()
    }

    #[inline]
    pub fn output_size(&self) -> usize {
        self.weights.nrows()
    }

    /// Returns the weights of the layer
    #[inline]
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Returns the biases of the layer
    #[inline]
    pub fn biases(&self) -> &Array1<f64> {
        &self.biases
    }

    /// Returns the activation function of the layer
    #[inline]
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Sets a new activation function for the layer
    ///
    /// ## Arguments
    /// 
    /// - `activation`: The new activation fucntion of the layer
    /// 
    #[inline]
    pub fn set_activation(&mut self, activation: Activation) {
        self.activation = activation
    }

    /// Set the weights of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `weights`: The new weights of the layer
    /// 
    #[inline]
    pub fn set_weights(&mut self, weights: &Array2<f64>) {
        self.weights = weights.to_owned();
    }

    /// Set the biases of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `biases`: The new biases of the layer
    /// 
    #[inline]
    pub fn set_biases(&mut self, biases: &Array1<f64>) {
        self.biases = biases.to_owned();
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input = input.clone();
        let sum = self.weights.dot(&self.input) + &self.biases;
        self.activation.function(&sum.view())
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

        Ok(input_gradient * self.activation.derivate(&self.input.view()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
