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
#[derive(Debug)]
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
            weights: Array2::random((num_inputs, num_neurons), Uniform::new(0.0, 1.0)),
            biases: Array1::random(num_neurons, Uniform::new(0.0, 1.0)),
            activation
        }
    }

    /// Returns the weights of the layer
    pub fn get_weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Returns the weights of the layer
    pub fn get_biases(&self) -> &Array1<f64> {
        &self.biases
    }

    /// Returns the weights of the layer
    pub fn get_activation(&self) -> &Activation {
        &self.activation
    }

    /// Set the weights of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `weights`: The new weights of the layer
    /// 
    pub fn set_weights(&mut self, weights: Array2<f64>) {
        self.weights = weights
    }

    /// Set the biases of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `biases`: The new biases of the layer
    /// 
    pub fn set_biases(&mut self, biases: Array1<f64>) {
        self.biases = biases
    }

    /// Set the activation of the layer
    /// 
    /// ## Arguments
    /// 
    /// - `activation`: The new activation of the layer
    /// 
    pub fn set_activation(&mut self, activation: Activation) {
        self.activation = activation
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new_layer() {
        let l = Layer::new(4, 3, Activation::SIGMOID);
        assert_eq!(l.get_weights().nrows(), 3);
        assert_eq!(l.get_weights().ncols(), 4);
        assert_eq!(l.get_biases().len(), 4);
        assert_eq!(l.get_activation(), &Activation::SIGMOID);
    }

    #[test]
    fn test_layer_setters() {
        let mut l = Layer::new(4, 3, Activation::SIGMOID);

        // Clonar los datos para evitar préstamos inmutables activos
        let old_weights = l.get_weights().clone();
        let old_biases = l.get_biases().clone();
        let old_activation = l.get_activation().clone();

        // Asignar nuevos valores
        let new_weights = Array2::random((3, 4), Uniform::new(0.0, 1.0));
        let new_biases = Array1::random(4, Uniform::new(0.0, 1.0));
        let new_activation = Activation::RELU;

        // Establecer nuevos valores
        l.set_weights(new_weights.clone());
        l.set_biases(new_biases.clone());
        l.set_activation(new_activation.clone());

        // Verificar que los valores antiguos sean diferentes de los nuevos
        assert_ne!(l.get_weights(), &old_weights);
        assert_ne!(l.get_biases(), &old_biases);
        assert_ne!(l.get_activation(), &old_activation);

        // Verificar que los valores nuevos se hayan establecido correctamente
        assert_eq!(l.get_weights(), &new_weights);
        assert_eq!(l.get_biases(), &new_biases);
        assert_eq!(l.get_activation(), &new_activation);
    }
}