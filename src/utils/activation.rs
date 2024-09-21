use core::fmt;
use std::str::FromStr;
use ndarray::{Array1, ArrayView1};

/// Represents the different activation functions for the neural network
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Activation {
    /// `step(x) = 1 if x > 0 else 0`
    STEP,
    /// `sigmoid(x) = 1 / (1 + exp(-x))`
    SIGMOID,
    /// `ReLU(x) = x if x > 0 else 0`
    RELU,
    /// `tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))`
    TANH,
    /// `softmax(x) = exp(x) / sum(exp(x))`
    SOFTMAX,
}

impl Activation {
    /// Returns the function of the different activations
    pub fn function(&self, z: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            Activation::STEP => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::SIGMOID => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::RELU => z.mapv(|x| if x > 0.0 { x } else { 0.0 }),
            Activation::TANH => z.mapv(|x| x.tanh()),
            Activation::SOFTMAX => {
                let exp = z.mapv(|x| x.exp());
                let sum = exp.sum();
                exp.mapv(|x| x / sum)
            },
        }
    }

    /// Returns the derivative of the different activations
    pub fn derivate(&self, z: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            Activation::STEP => z.mapv(|_| 0.0),
            Activation::SIGMOID => self.function(z) * (1.0 - self.function(z)),
            Activation::RELU => Activation::STEP.function(z),
            Activation::TANH => 1.0 - self.function(z).mapv(|e| e.powi(2)),
            Activation::SOFTMAX => self.function(z) * (1.0 - self.function(z)),
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
            Activation::SOFTMAX => write!(f, "SOFTMAX"),
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
            "SOFTMAX" => Ok(Activation::SOFTMAX),
            _ => Err(format!("Unknown activation type: {}", s)),
        }
    }
}