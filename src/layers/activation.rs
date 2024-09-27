use std::error::Error;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::{layer::LayerType, Layer};
use crate::utils::ActivationFunc;

/// Represents an activation layer in a neural network.
///
/// The `Activation` layer applies a specific activation function to its input, modifying the data 
/// based on the activation function used (e.g., `RELU`, `Sigmoid`, etc.). This layer is often used 
/// in combination with other layers like `Dense` to introduce non-linearities into the model, 
/// which is essential for learning complex patterns.
///
/// # Fields
///
/// * `input`: The input data to the activation layer. This is a 1D array of floating-point values 
///   that represents the input from the previous layer in the network.
/// * `activation`: The activation function to apply to the input. It defines the non-linearity that 
///   is applied to the input data (e.g., `RELU`, `Sigmoid`, `TANH`).
/// * `layer_type`: The type of the layer, which in this case is always `LayerType::Activation`. 
///   This helps identify the layer when saving or loading models, or when working with 
///   dynamic layers in the network.
///
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Activation {
    input: Array1<f64>,
    activation: ActivationFunc,
    layer_type: LayerType
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
            activation,
            layer_type: LayerType::Activation
        }
    }
}

impl Layer for Activation {
    fn ninputs(&self) -> usize {
        0
    }

    fn noutputs(&self) -> usize {
        0
    }

    fn weights(&self) -> ArrayView2<f64> {
        unimplemented!()
    }
    
    fn biases(&self) -> ArrayView1<f64> {
        unimplemented!()
    }
    
    fn activation(&self) -> Option<ActivationFunc> {
        Some(self.activation)
    }
    
    fn set_activation(&mut self, activation: Option<ActivationFunc>) {
        self.activation = activation.unwrap()
    }
    
    fn set_weights(&mut self, _weights: &Array2<f64>) {
        unimplemented!()
    }
    
    fn set_biases(&mut self, _biases: &Array1<f64>) {
        unimplemented!()
    }

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

    fn layer_type(&self) -> LayerType {
        self.layer_type
    }
 
    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    
    fn from_json(json_path: &str) -> Box<dyn Layer> {
        Box::new(serde_json::from_str::<Activation>(json_path).unwrap())
    }
}