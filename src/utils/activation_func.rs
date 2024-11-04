use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

use crate::error::NNResult;

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

impl ActivationFunc {
    /// Applies the activation function to the input array
    ///
    /// This method applies the chosen activation function element-wise to the input array.
    ///
    /// ## Arguments
    ///
    /// * `z`: A reference to an `ArrayView1<f64>` representing the input values
    ///
    /// ## Returns
    ///
    /// An `Array1<f64>` containing the result of applying the activation function to the input
    ///
    #[inline]
    pub fn function(&self, z: &ArrayView1<f64>) -> NNResult<Array1<f64>> {
        match self {
            ActivationFunc::STEP => Ok(z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })),
            ActivationFunc::SIGMOID => Ok(z.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
            ActivationFunc::RELU => Ok(z.mapv(|x| if x > 0.0 { x } else { 0.0 })),
            ActivationFunc::TANH => Ok(z.mapv(|x| x.tanh())),
            ActivationFunc::SOFTMAX => {
                let exp = z.mapv(|x| x.exp());
                let sum = exp.sum();
                Ok(exp.mapv(|x| x / sum))
            }
        }
    }

    /// Computes the derivative of the activation function
    ///
    /// This method calculates the derivative of the chosen activation function with respect to its input.
    /// For the SOFTMAX function, this returns the Jacobian matrix.
    ///
    /// ## Arguments
    ///
    /// * `z`: A reference to an `ArrayView1<f64>` representing the input values
    ///
    /// ## Returns
    ///
    /// An `Array1<f64>` containing the derivatives of the activation function with respect to the input
    ///
    #[inline]
    pub fn derivate(&self, z: &ArrayView1<f64>) -> NNResult<Array1<f64>> {
        match self {
            ActivationFunc::STEP => Ok(z.mapv(|_| 0.0)),
            ActivationFunc::SIGMOID => Ok(self.function(z)? * (1.0 - self.function(z)?)),
            ActivationFunc::RELU => Ok(ActivationFunc::STEP.function(z)?),
            ActivationFunc::TANH => Ok(1.0 - self.function(z)?.mapv(|e| e.powi(2))),
            ActivationFunc::SOFTMAX => Ok(self.function(z)? * (1.0 - self.function(z)?)),
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
        let output = activation.function(&input.view()).unwrap();
        assert_eq!(output, array![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_function() {
        let input = array![0.0, 2.0, -2.0];
        let activation = ActivationFunc::SIGMOID;
        let output = activation.function(&input.view()).unwrap();
        assert!((output[0] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
        assert!((output[1] - 0.8808).abs() < 1e-4); // sigmoid(2)
        assert!((output[2] - 0.1192).abs() < 1e-4); // sigmoid(-2)
    }

    #[test]
    fn test_relu_function() {
        let input = array![-1.0, 0.0, 3.0];
        let activation = ActivationFunc::RELU;
        let output = activation.function(&input.view()).unwrap();
        assert_eq!(output, array![0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_tanh_function() {
        let input = array![0.0, 1.0, -1.0];
        let activation = ActivationFunc::TANH;
        let output = activation.function(&input.view()).unwrap();
        assert!((output[0] - 0.0).abs() < 1e-6); // tanh(0) = 0
        assert!((output[1] - 0.7616).abs() < 1e-4); // tanh(1)
        assert!((output[2] + 0.7616).abs() < 1e-4); // tanh(-1)
    }

    #[test]
    fn test_softmax_function() {
        let input = array![1.0, 2.0, 3.0];
        let activation = ActivationFunc::SOFTMAX;
        let output = activation.function(&input.view()).unwrap();
        let sum: f64 = output.sum();
        assert!((sum - 1.0).abs() < 1e-6); // softmax outputs should sum to 1
    }

    #[test]
    fn test_sigmoid_derivate() {
        let input = array![0.0, 2.0, -2.0];
        let activation = ActivationFunc::SIGMOID;
        let derivative = activation.derivate(&input.view()).unwrap();
        assert!((derivative[0] - 0.25).abs() < 1e-6); // sigmoid'(0) = 0.25
        assert!((derivative[1] - 0.104993).abs() < 1e-4); // sigmoid'(2)
        assert!((derivative[2] - 0.104993).abs() < 1e-4); // sigmoid'(-2)
    }

    #[test]
    fn test_relu_derivate() {
        let input = array![1.0, -0.5, 0.0];
        let activation = ActivationFunc::RELU;
        let derivative = activation.derivate(&input.view()).unwrap();
        assert_eq!(derivative, array![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tanh_derivate() {
        let input = array![0.0, 1.0, -1.0];
        let activation = ActivationFunc::TANH;
        let derivative = activation.derivate(&input.view()).unwrap();
        assert!((derivative[0] - 1.0).abs() < 1e-6); // tanh'(0) = 1
        assert!((derivative[1] - 0.419974).abs() < 1e-4); // tanh'(1)
        assert!((derivative[2] - 0.419974).abs() < 1e-4); // tanh'(-1)
    }

    #[test]
    fn test_softmax_derivate() {
        let input = array![1.0, 2.0, 3.0];
        let activation = ActivationFunc::SOFTMAX;
        let derivative = activation.derivate(&input.view()).unwrap();
        let expected = activation.function(&input.view()).unwrap()
            * (1.0 - activation.function(&input.view()).unwrap());
        assert_eq!(derivative, expected);
    }
}
