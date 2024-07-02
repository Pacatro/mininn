use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use crate::activation::Activation;

/// Represents a layer of the neural network
/// 
/// ## Atributes
/// 
/// - `weights`: A 2D array that contains all the weights of the layer
/// - `biases`: A 1D array that contains all the biases of the layer
/// - `activation`: The activation function for the layer
/// 
#[derive(Debug, PartialEq, Clone)]
pub struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: Activation
}

impl Layer {
    /// Creates a new [`Layer`]
    /// 
    /// ## Arguments
    /// 
    /// - `num_neurons`: The number of neurons of the layer
    /// - `num_inputs`: The number of inputs that are connected to the layer
    /// - `activation`: The activation function for the layer
    /// 
    /// ## Returns
    /// 
    /// The [`Layer`] with the aport data
    /// 
    pub fn new(num_neurons: usize, num_inputs: usize, activation: Activation) -> Layer {
        Layer {
            weights: Array2::random((num_neurons, num_inputs), Uniform::new(-1.0, 1.0)),
            biases: Array1::random(num_neurons, Uniform::new(-1.0, 1.0)),
            activation
        }
    }

    /// Returns the weights of the layer
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Returns the weights of the layer
    pub fn biases(&self) -> &Array1<f64> {
        &self.biases
    }

    /// Returns the weights of the layer
    pub fn activation(&self) -> &Activation {
        &self.activation
    }

    /// Set the weights of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `weights`: The new weights of the layer
    /// 
    pub fn set_weights(&mut self, weights: &Array2<f64>) {
        self.weights = weights.to_owned()
    }

    /// Set the biases of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `biases`: The new biases of the layer
    /// 
    pub fn set_biases(&mut self, biases: &Array1<f64>) {
        self.biases = biases.to_owned()
    }

    /// Set the activation of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `activation`: The new activation of the layer
    /// 
    pub fn set_activation(&mut self, activation: &Activation) {
        self.activation = *activation
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new_layer() {
        let l = Layer::new(3, 2, Activation::SIGMOID);
        assert_eq!(l.weights().nrows(), 3);
        assert_eq!(l.weights().ncols(), 2);
        assert_eq!(l.biases().len(), 3);
        assert_eq!(l.activation(), &Activation::SIGMOID);
    }

    #[test]
    fn test_layer_setters() {
        let mut l = Layer::new(4, 3, Activation::SIGMOID);

        let old_weights = l.weights().clone();
        let old_biases = l.biases().clone();
        let old_activation = l.activation().clone();

        let new_weights = Array2::random((3, 4), Uniform::new(0.0, 1.0));
        let new_biases = Array1::random(4, Uniform::new(0.0, 1.0));
        let new_activation = Activation::RELU;

        l.set_weights(&new_weights);
        l.set_biases(&new_biases);
        l.set_activation(&new_activation);

        assert_ne!(l.weights(), &old_weights);
        assert_ne!(l.biases(), &old_biases);
        assert_ne!(l.activation(), &old_activation);
        assert_eq!(l.weights(), &new_weights);
        assert_eq!(l.biases(), &new_biases);
        assert_eq!(l.activation(), &new_activation);
    }
}