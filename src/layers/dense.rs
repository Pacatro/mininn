use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use super::BaseLayer;

/// Represents a fully connected layer
/// 
/// ## Atributes
/// 
/// - `weights`: The weights of the layer as an [`Array2<f64>`]
/// - `biases`: The biases of the layer as an [`Array2<f64>`]
///
#[derive(Debug, PartialEq, Clone)]
pub struct Dense {
    weights: Array2<f64>,
    biases: Array1<f64>,
    input: Array1<f64>,
}

impl Dense {
    ///  Creates a new [`Dense`] layer
    /// 
    /// ## Arguments
    /// 
    /// - `ninput`: The number of inputs of the layer
    /// - `noutput`: The number of outputs of the layer
    /// 
    pub fn new(ninput: usize, noutput: usize) -> Self {
        Self {
            weights: Array2::random((noutput, ninput), Uniform::new(-1.0, 1.0)),
            biases: Array1::random(noutput, Uniform::new(-1.0, 1.0)),
            input: Array1::zeros(ninput),
        }
    }

    /// Returns the weights of the layer
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Returns the biases of the layer
    pub fn biases(&self) -> &Array1<f64> {
        &self.biases
    }

    /// Set the weights of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `weights`: The new weights of the layer
    /// 
    pub fn set_weights(&mut self, weights: &Array2<f64>) {
        self.weights = weights.to_owned();
    }

    /// Set the biases of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `biases`: The new biases of the layer
    /// 
    pub fn set_biases(&mut self, biases: &Array1<f64>) {
        self.biases = biases.to_owned();
    }
}

impl BaseLayer for Dense {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input = input.clone();
        self.weights.dot(&self.input) + &self.biases
    }

    fn backward(&mut self, output_gradient: ArrayView1<f64>, learning_rate: f64) -> Array1<f64> {
        // Calculate gradients
        let weights_gradient = output_gradient.to_owned().to_shape((output_gradient.len(), 1)).unwrap()
            .dot(&self.input.view().to_shape((1, self.input.len())).unwrap());
        let input_gradient = self.weights.t().dot(&output_gradient);

        // Update weights and biases
        self.weights -= &(weights_gradient * learning_rate);
        self.biases -= &(output_gradient.to_owned() * learning_rate);

        input_gradient
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
