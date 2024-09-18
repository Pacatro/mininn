use ndarray::{Array1, ArrayView1};

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
    input: Array1<f64>,
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
        Self { input: Array1::zeros(1), activation }
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
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input = input.to_owned();
        self.activation.function(&self.input.view())
    }

    fn backward(&mut self, output_gradient: ArrayView1<f64>, _learning_rate: f64) -> Array1<f64> {
        output_gradient.to_owned() * self.activation.derivate(&self.input.view())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}