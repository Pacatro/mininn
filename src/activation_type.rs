use core::fmt;
use std::str::FromStr;

use ndarray::Array2;

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
pub enum ActivationType {
    STEP,
    SIGMOID,
    RELU,
    TANH
}

impl ActivationType {
    /// Returns the function of the diferents activations
    pub(crate) fn function(&self, z: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationType::STEP => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationType::SIGMOID => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationType::RELU => z.mapv(|x| if x > 0.0 { x } else { 0.0 }),
            ActivationType::TANH => z.mapv(|x| x.tanh()),
        }
    }
    
    /// Returns the derivate of the diferents activations
    pub(crate) fn derivate(&self, z: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationType::STEP => z.mapv(|_| 0.0),
            ActivationType::SIGMOID => self.function(z) * (1.0 - self.function(z)),
            ActivationType::RELU => ActivationType::STEP.function(z),
            ActivationType::TANH => 1.0 - self.function(z).mapv(|e| e.powi(2)),
        }
    }
}

impl fmt::Display for ActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationType::RELU => write!(f, "RELU"),
            ActivationType::SIGMOID => write!(f, "SIGMOID"),
            ActivationType::TANH => write!(f, "TANH"),
            ActivationType::STEP => write!(f, "STEP"),
        }
    }
}

impl FromStr for ActivationType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "RELU" => Ok(ActivationType::RELU),
            "STEP" => Ok(ActivationType::STEP),
            "SIGMOID" => Ok(ActivationType::SIGMOID),
            "TANH" => Ok(ActivationType::TANH),
            _ => Err(()),
        }
    }
}