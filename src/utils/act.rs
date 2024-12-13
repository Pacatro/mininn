use std::fmt::Debug;

use dyn_clone::DynClone;
use ndarray::{ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

use crate::{
    core::{MininnError, NNResult},
    registers::ACT_REGISTER,
};

/// Allows users to define their own acrivation functions.
///
/// ## Methods
///
/// - `function`: Applies the activation function to the input array.
/// - `derivate`: Calculates the derivative of the activation function with respect to the input.
/// - `activation`: Returns the name of the activation function.
/// - `from_activation`: Creates a new instance of the activation function from a string.
///
pub trait ActivationFunction: Debug + DynClone {
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
    /// For the Softmax function, this returns the Jacobian matrix.
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

    /// Creates an activation function from a string
    ///
    /// ## Arguments
    ///
    /// * `activation`: The name of the activation function
    ///
    /// ## Returns
    ///
    /// A `Result` containing the activation function if successful, or an error if something goes wrong.
    ///
    fn from_activation(activation: &str) -> NNResult<Box<dyn ActivationFunction>>
    where
        Self: Sized;
}

dyn_clone::clone_trait_object!(ActivationFunction);

/// Some default implementations of Activation functions
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum Act {
    /// `step(x) = 1 if x > 0 else 0`
    Step,
    /// `sigmoid(x) = 1 / (1 + exp(-x))`
    Sigmoid,
    /// `ReLU(x) = x if x > 0 else 0`
    ReLU,
    /// `tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))`
    Tanh,
    /// `softmax(x) = exp(x) / sum(exp(x))`
    Softmax,
}

impl ActivationFunction for Act {
    #[inline]
    fn function(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
        match self {
            Act::Step => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Act::Sigmoid => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Act::ReLU => z.mapv(|x| if x > 0.0 { x } else { 0.0 }),
            Act::Tanh => z.mapv(|x| x.tanh()),
            Act::Softmax => {
                let exp = z.exp();
                let sum = exp.sum();
                exp.mapv(|x| x / sum)
            }
        }
    }

    #[inline]
    fn derivate(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
        match self {
            Act::Step => z.mapv(|_| 0.0),
            Act::Sigmoid => self.function(z) * (1.0 - self.function(z)),
            Act::ReLU => Act::Step.function(z),
            Act::Tanh => 1.0 - self.function(z).mapv(|e| e.powi(2)),
            Act::Softmax => self.function(z) * (1.0 - self.function(z)),
        }
    }

    #[inline]
    fn activation(&self) -> &str {
        match self {
            Act::Step => "Step",
            Act::Sigmoid => "Sigmoid",
            Act::ReLU => "ReLU",
            Act::Tanh => "Tanh",
            Act::Softmax => "Softmax",
        }
    }

    #[inline]
    fn from_activation(activation: &str) -> NNResult<Box<dyn ActivationFunction>>
    where
        Self: Sized,
    {
        match activation {
            "Step" => Ok(Box::new(Act::Step)),
            "Sigmoid" => Ok(Box::new(Act::Sigmoid)),
            "ReLU" => Ok(Box::new(Act::ReLU)),
            "Tanh" => Ok(Box::new(Act::Tanh)),
            "Softmax" => Ok(Box::new(Act::Softmax)),
            _ => Err(MininnError::ActivationError(
                "The activation function is not supported".to_string(),
            )),
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

        let act = ACT_REGISTER.with(|register| {
            register
                .borrow_mut()
                .create_activation(&activation)
                .map_err(|err| {
                    serde::de::Error::custom(format!(
                        "Failed to create activation function '{}': {}",
                        activation, err
                    ))
                })
        });

        act
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_step_function() {
        let input = array![1.0, -0.5, 0.0, 2.0];
        let activation = Act::Step;
        let output = activation.function(&input.into_dyn().view());
        let expected = array![1.0, 0.0, 0.0, 1.0];
        assert_eq!(output, expected.into_dyn());
    }

    #[test]
    fn test_sigmoid_function() {
        let input = array![0.0, 2.0, -2.0];
        let activation = Act::Sigmoid;
        let output = activation.function(&input.into_dyn().view());
        assert!((output[0] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
        assert!((output[1] - 0.8808).abs() < 1e-4); // sigmoid(2)
        assert!((output[2] - 0.1192).abs() < 1e-4); // sigmoid(-2)
    }

    #[test]
    fn test_relu_function() {
        let input = array![-1.0, 0.0, 3.0];
        let activation = Act::ReLU;
        let output = activation.function(&input.into_dyn().view());
        let expected = array![0.0, 0.0, 3.0];
        assert_eq!(output, expected.into_dyn());
    }

    #[test]
    fn test_tanh_function() {
        let input = array![0.0, 1.0, -1.0];
        let activation = Act::Tanh;
        let output = activation.function(&input.into_dyn().view());
        assert!((output[0] - 0.0).abs() < 1e-6); // tanh(0) = 0
        assert!((output[1] - 0.7616).abs() < 1e-4); // tanh(1)
        assert!((output[2] + 0.7616).abs() < 1e-4); // tanh(-1)
    }

    #[test]
    fn test_softmax_function() {
        let input = array![1.0, 2.0, 3.0];
        let activation = Act::Softmax;
        let output = activation.function(&input.into_dyn().view());
        let sum: f64 = output.sum();
        assert!((sum - 1.0).abs() < 1e-6); // softmax outputs should sum to 1
    }

    #[test]
    fn test_sigmoid_derivate() {
        let input = array![0.0, 2.0, -2.0];
        let activation = Act::Sigmoid;
        let derivative = activation.derivate(&input.into_dyn().view());
        assert!((derivative[0] - 0.25).abs() < 1e-6); // sigmoid'(0) = 0.25
        assert!((derivative[1] - 0.104993).abs() < 1e-4); // sigmoid'(2)
        assert!((derivative[2] - 0.104993).abs() < 1e-4); // sigmoid'(-2)
    }

    #[test]
    fn test_relu_derivate() {
        let input = array![1.0, -0.5, 0.0];
        let activation = Act::ReLU;
        let derivative = activation.derivate(&input.into_dyn().view());
        let expected = array![1.0, 0.0, 0.0];
        assert_eq!(derivative, expected.into_dyn());
    }

    #[test]
    fn test_tanh_derivate() {
        let input = array![0.0, 1.0, -1.0];
        let activation = Act::Tanh;
        let derivative = activation.derivate(&input.into_dyn().view());
        assert!((derivative[0] - 1.0).abs() < 1e-6); // tanh'(0) = 1
        assert!((derivative[1] - 0.419974).abs() < 1e-4); // tanh'(1)
        assert!((derivative[2] - 0.419974).abs() < 1e-4); // tanh'(-1)
    }

    #[test]
    fn test_softmax_derivate() {
        let input = array![1.0, 2.0, 3.0];
        let activation = Act::Softmax;
        let derivative = activation.derivate(&input.clone().into_dyn().view());
        let func = activation.function(&input.to_owned().into_dyn().view());
        let expected = &func * (1.0 - &func);
        assert_eq!(derivative, expected);
    }
}
