use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

/// Represents the different activation functions for the neural network
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunc {
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

impl ActivationFunc {
    /// Returns the function of the different activations
    pub fn function(&self, z: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            ActivationFunc::STEP => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunc::SIGMOID => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunc::RELU => z.mapv(|x| if x > 0.0 { x } else { 0.0 }),
            ActivationFunc::TANH => z.mapv(|x| x.tanh()),
            ActivationFunc::SOFTMAX => {
                let exp = z.mapv(|x| x.exp());
                let sum = exp.sum();
                exp.mapv(|x| x / sum)
            },
        }
    }

    /// Returns the derivative of the different activations
    pub fn derivate(&self, z: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            ActivationFunc::STEP => z.mapv(|_| 0.0),
            ActivationFunc::SIGMOID => self.function(z) * (1.0 - self.function(z)),
            ActivationFunc::RELU => ActivationFunc::STEP.function(z),
            ActivationFunc::TANH => 1.0 - self.function(z).mapv(|e| e.powi(2)),
            ActivationFunc::SOFTMAX => self.function(z) * (1.0 - self.function(z)),
        }
    }
}