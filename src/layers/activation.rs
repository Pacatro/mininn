use ndarray::Array2;

use crate::activation_type::ActivationType;

use super::BaseLayer;

/// Represents a fully connected layer
/// 
/// ## Atributes
/// 
/// - `activation`: The activation function of the layer
///
#[derive(Debug, PartialEq, Clone)]
pub struct Activation {
    input: Array2<f64>,
    activation: ActivationType
}

impl Activation {
    ///  Creates a new [`Activation`] layer
    /// 
    /// ## Arguments
    /// 
    /// - `activation`: The activation function of the layer
    /// 
    pub fn new(activation: ActivationType) -> Self {
        Self { input: Array2::zeros((1, 1)), activation }
    }

    /// Returns the activation function of the layer
    pub fn activation(&self) -> ActivationType {
        self.activation
    }

    /// Sets a new activation function for the layer
    ///
    /// ## Arguments
    /// 
    /// - `activation`: The new activation fucntion of the layer
    /// 
    pub fn set_activation(&mut self, activation: ActivationType) {
        self.activation = activation
    }
}

impl BaseLayer for Activation {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.input = input;
        self.activation.function(&self.input)
    }

    fn backward(&mut self, output_gradient: Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let _ = learning_rate;
        output_gradient * self.activation.derivate(&self.input)
    }
}