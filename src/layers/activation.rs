use core::fmt;
use std::str::FromStr;

use ndarray::{Array1, ArrayView1};

/// Represents the diferents activations functions for the neural network
/// 
/// ## Types
/// 
/// - `STEP`
/// - `SIGMOID`
/// - `RELU`
/// - `TANH`
///
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Activation {
    STEP,
    SIGMOID,
    RELU,
    TANH
}

impl Activation {
    /// Returns the function of the diferents activations
    pub(crate) fn function(&self, z: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            Activation::STEP => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::SIGMOID => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::RELU => z.mapv(|x| if x > 0.0 { x } else { 0.0 }),
            Activation::TANH => z.mapv(|x| x.tanh()),
        }
    }
    
    /// Returns the derivate of the diferents activations
    pub(crate) fn derivate(&self, z: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            Activation::STEP => z.mapv(|_| 0.0),
            Activation::SIGMOID => self.function(z) * (1.0 - self.function(z)),
            Activation::RELU => Activation::STEP.function(z),
            Activation::TANH => 1.0 - self.function(z).mapv(|e| e.powi(2)),
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::RELU => write!(f, "RELU"),
            Activation::SIGMOID => write!(f, "SIGMOID"),
            Activation::TANH => write!(f, "TANH"),
            Activation::STEP => write!(f, "STEP"),
        }
    }
}

impl FromStr for Activation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "RELU" => Ok(Activation::RELU),
            "STEP" => Ok(Activation::STEP),
            "SIGMOID" => Ok(Activation::SIGMOID),
            "TANH" => Ok(Activation::TANH),
            _ => Err(format!("Unknown activation type: {}", s)),
        }
    }
}