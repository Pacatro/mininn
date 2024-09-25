use std::error::Error;
use ndarray::{Array1, ArrayView1};

use super::Layer;
use crate::utils::ActivationFunc;

/// Represents a fully connected layer
/// 
/// ## Atributes
/// 
/// - `activation`: The activation function of the layer
///
#[derive(Debug, PartialEq, Clone)]
pub struct Activation {
    input: Array1<f64>,
    activation: ActivationFunc
}

impl Activation {
    ///  Creates a new [`Activation`] layer
    /// 
    /// ## Arguments
    /// 
    /// - `activation`: The activation function of the layer
    ///
    #[inline]
    pub fn new(activation: ActivationFunc) -> Self {
        Self {
            input: Array1::zeros(1),
            activation
        }
    }

    /// Returns the activation function of the layer
    #[inline]
    pub fn activation(&self) -> ActivationFunc {
        self.activation
    }

    /// Sets a new activation function for the layer
    ///
    /// ## Arguments
    /// 
    /// - `activation`: The new activation fucntion of the layer
    /// 
    #[inline]
    pub fn set_activation(&mut self, activation: ActivationFunc) {
        self.activation = activation
    }
}

impl Layer for Activation {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input = input.to_owned();
        self.activation.function(&self.input.view())
    }

    fn backward(&mut self, output_gradient: ArrayView1<f64>, _learning_rate: f64) -> Result<Array1<f64>, Box<dyn Error>> {
        Ok(output_gradient.to_owned() * self.activation.derivate(&self.input.view()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
