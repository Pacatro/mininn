use std::fmt::Debug;

use ndarray::{ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

pub trait ActivationFunction: Debug {
    /// Applies the activation function to the input array
    ///
    /// This method applies the chosen activation function element-wise to the input array.
    ///
    /// ## Arguments
    ///
    /// * `z`: The input values
    ///
    /// ## Returns
    ///
    /// The result of applying the activation function to the input
    ///
    fn function(&self, z: &ArrayViewD<f64>) -> ArrayD<f64>;

    /// Computes the derivative of the activation function
    ///
    /// This method calculates the derivative of the chosen activation function with respect to its input.
    /// For the SOFTMAX function, this returns the Jacobian matrix.
    ///
    /// ## Arguments
    ///
    /// * `z`: The input values
    ///
    /// ## Returns
    ///
    /// The derivatives of the activation function with respect to the input
    ///
    fn derivate(&self, z: &ArrayViewD<f64>) -> ArrayD<f64>;

    /// Returns the name of the activation function
    fn activation(&self) -> &str;
}

/// Represents the different activation functions for the neural network
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunc {
    /// `step(x) = 1 if x > 0 else 0`
    STEP,
    /// `sigmoid(x) = 1 / (1 + exp(-x))`
    SIGMOID,
    /// `RELU(x) = x if x > 0 else 0`
    RELU,
    /// `tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))`
    TANH,
    /// `softmax(x) = exp(x) / sum(exp(x))`
    SOFTMAX,
}

impl ActivationFunction for ActivationFunc {
    #[inline]
    fn function(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
        match self {
            ActivationFunc::STEP => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunc::SIGMOID => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunc::RELU => z.mapv(|x| if x > 0.0 { x } else { 0.0 }),
            ActivationFunc::TANH => z.mapv(|x| x.tanh()),
            ActivationFunc::SOFTMAX => {
                let exp = z.exp();
                let sum = exp.sum();
                exp.mapv(|x| x / sum)
            }
        }
    }

    #[inline]
    fn derivate(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
        match self {
            ActivationFunc::STEP => z.mapv(|_| 0.0),
            ActivationFunc::SIGMOID => self.function(z) * (1.0 - self.function(z)),
            ActivationFunc::RELU => ActivationFunc::STEP.function(z),
            ActivationFunc::TANH => 1.0 - self.function(z).mapv(|e| e.powi(2)),
            ActivationFunc::SOFTMAX => self.function(z) * (1.0 - self.function(z)),
        }
    }

    #[inline]
    fn activation(&self) -> &str {
        match self {
            ActivationFunc::STEP => "STEP",
            ActivationFunc::SIGMOID => "SIGMOID",
            ActivationFunc::RELU => "RELU",
            ActivationFunc::TANH => "TANH",
            ActivationFunc::SOFTMAX => "SOFTMAX",
        }
    }
}

impl PartialEq for Box<dyn ActivationFunction> {
    fn eq(&self, other: &Self) -> bool {
        self.activation() == other.activation()
    }
}

impl Eq for Box<dyn ActivationFunction> {}

impl Serialize for Box<dyn ActivationFunction> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.activation())
    }
}

impl<'de> Deserialize<'de> for Box<dyn ActivationFunction> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let activation: String = Deserialize::deserialize(deserializer)?;
        match activation.as_str() {
            "STEP" => Ok(Box::new(ActivationFunc::STEP)),
            "SIGMOID" => Ok(Box::new(ActivationFunc::SIGMOID)),
            "RELU" => Ok(Box::new(ActivationFunc::RELU)),
            "TANH" => Ok(Box::new(ActivationFunc::TANH)),
            "SOFTMAX" => Ok(Box::new(ActivationFunc::SOFTMAX)),
            _ => {
                // TODO: Implement a custom deserialization logic for your activation function
                todo!()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_step_function() {
        let input = array![1.0, -0.5, 0.0, 2.0];
        let activation = ActivationFunc::STEP;
        let output = activation.function(&input.into_dyn().view());
        let expected = array![1.0, 0.0, 0.0, 1.0];
        assert_eq!(output, expected.into_dyn());
    }

    #[test]
    fn test_sigmoid_function() {
        let input = array![0.0, 2.0, -2.0];
        let activation = ActivationFunc::SIGMOID;
        let output = activation.function(&input.into_dyn().view());
        assert!((output[0] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
        assert!((output[1] - 0.8808).abs() < 1e-4); // sigmoid(2)
        assert!((output[2] - 0.1192).abs() < 1e-4); // sigmoid(-2)
    }

    #[test]
    fn test_relu_function() {
        let input = array![-1.0, 0.0, 3.0];
        let activation = ActivationFunc::RELU;
        let output = activation.function(&input.into_dyn().view());
        let expected = array![0.0, 0.0, 3.0];
        assert_eq!(output, expected.into_dyn());
    }

    #[test]
    fn test_tanh_function() {
        let input = array![0.0, 1.0, -1.0];
        let activation = ActivationFunc::TANH;
        let output = activation.function(&input.into_dyn().view());
        assert!((output[0] - 0.0).abs() < 1e-6); // tanh(0) = 0
        assert!((output[1] - 0.7616).abs() < 1e-4); // tanh(1)
        assert!((output[2] + 0.7616).abs() < 1e-4); // tanh(-1)
    }

    #[test]
    fn test_softmax_function() {
        let input = array![1.0, 2.0, 3.0];
        let activation = ActivationFunc::SOFTMAX;
        let output = activation.function(&input.into_dyn().view());
        let sum: f64 = output.sum();
        assert!((sum - 1.0).abs() < 1e-6); // softmax outputs should sum to 1
    }

    #[test]
    fn test_sigmoid_derivate() {
        let input = array![0.0, 2.0, -2.0];
        let activation = ActivationFunc::SIGMOID;
        let derivative = activation.derivate(&input.into_dyn().view());
        assert!((derivative[0] - 0.25).abs() < 1e-6); // sigmoid'(0) = 0.25
        assert!((derivative[1] - 0.104993).abs() < 1e-4); // sigmoid'(2)
        assert!((derivative[2] - 0.104993).abs() < 1e-4); // sigmoid'(-2)
    }

    #[test]
    fn test_relu_derivate() {
        let input = array![1.0, -0.5, 0.0];
        let activation = ActivationFunc::RELU;
        let derivative = activation.derivate(&input.into_dyn().view());
        let expected = array![1.0, 0.0, 0.0];
        assert_eq!(derivative, expected.into_dyn());
    }

    #[test]
    fn test_tanh_derivate() {
        let input = array![0.0, 1.0, -1.0];
        let activation = ActivationFunc::TANH;
        let derivative = activation.derivate(&input.into_dyn().view());
        assert!((derivative[0] - 1.0).abs() < 1e-6); // tanh'(0) = 1
        assert!((derivative[1] - 0.419974).abs() < 1e-4); // tanh'(1)
        assert!((derivative[2] - 0.419974).abs() < 1e-4); // tanh'(-1)
    }

    #[test]
    fn test_softmax_derivate() {
        let input = array![1.0, 2.0, 3.0];
        let activation = ActivationFunc::SOFTMAX;
        let derivative = activation.derivate(&input.clone().into_dyn().view());
        let func = activation.function(&input.to_owned().into_dyn().view());
        let expected = &func * (1.0 - &func);
        assert_eq!(derivative, expected);
    }
}
