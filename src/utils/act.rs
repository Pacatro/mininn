use std::fmt::Debug;

use dyn_clone::DynClone;
use ndarray::{ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

use crate::{
    core::{MininnError, NNResult},
    registers::REGISTER,
};

use super::NNUtil;

/// Allows users to define custom activation functions with metadata and dynamic creation.
///
/// This trait extends `ActCore` by adding methods for obtaining the name of an activation
/// function and creating instances from a string representation.
///
/// ## Methods
///
/// - `activation`: Retrieves the name of the activation function.
/// - `from_activation`: Dynamically creates an activation function from its name.
pub trait ActivationFunction: NNUtil + ActCore + Debug + DynClone {}

dyn_clone::clone_trait_object!(ActivationFunction);

/// Core functionality for activation functions.
///
/// This trait defines the essential methods required to implement a custom activation function,
/// including the function itself and its derivative.
///
/// ## Methods
///
/// - `function`: Applies the activation function element-wise to the input array.
/// - `derivate`: Computes the derivative (or Jacobian, if applicable) of the activation function.
///
pub trait ActCore {
    /// Applies the activation function to the input array.
    ///
    /// This method evaluates the chosen activation function on each element of the input array.
    ///
    /// ## Arguments
    ///
    /// * `z`: The input values as an `ArrayViewD<f32>`.
    ///
    /// ## Returns
    ///
    /// An `ArrayD<f32>` containing the result of applying the activation function.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let relu = Act::ReLU;
    /// let input = array![[-1.0, 2.0], [0.0, 3.0]].into_dyn();
    /// let output = relu.function(&input.view());
    /// // Output: [[0.0, 2.0], [0.0, 3.0]]
    /// ```
    fn function(&self, z: &ArrayViewD<f32>) -> ArrayD<f32>;

    /// Computes the derivative of the activation function.
    ///
    /// This method calculates the derivative (or the Jacobian matrix, if applicable) of the activation
    /// function with respect to the input array.
    ///
    /// ## Arguments
    ///
    /// * `z`: The input values as an `ArrayViewD<f32>`.
    ///
    /// ## Returns
    ///
    /// An `ArrayD<f32>` containing the derivatives of the activation function with respect to the input.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let relu = Act::ReLU;
    /// let input = array![[-1.0, 2.0], [0.0, 3.0]].into_dyn();
    /// let output = relu.derivate(&input.view());
    /// // Output: [[0.0, 1.0], [0.0, 1.0]]
    /// ```
    fn derivate(&self, z: &ArrayViewD<f32>) -> ArrayD<f32>;
}

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

impl ActivationFunction for Act {}

impl NNUtil for Act {
    #[inline]
    fn name(&self) -> &str {
        match self {
            Act::Step => "Step",
            Act::Sigmoid => "Sigmoid",
            Act::ReLU => "ReLU",
            Act::Tanh => "Tanh",
            Act::Softmax => "Softmax",
        }
    }

    #[inline]
    fn from_name(name: &str) -> NNResult<Box<Self>>
    where
        Self: Sized,
    {
        match name {
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

impl ActCore for Act {
    #[inline]
    fn function(&self, z: &ArrayViewD<f32>) -> ArrayD<f32> {
        match self {
            Act::Step => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Act::Sigmoid => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Act::ReLU => z.mapv(|x| if x > 0.0 { x } else { 0.0 }),
            Act::Tanh => z.mapv(|x| x.tanh()),
            Act::Softmax => {
                let exp = z.exp();
                let sum = exp.sum();
                exp / sum
            }
        }
    }

    #[inline]
    fn derivate(&self, z: &ArrayViewD<f32>) -> ArrayD<f32> {
        match self {
            Act::Step => z.mapv(|_| 0.0),
            Act::Sigmoid => self.function(z) * (1.0 - self.function(z)),
            Act::ReLU => Act::Step.function(z),
            Act::Tanh => 1.0 - self.function(z).mapv(|e| e.powi(2)),
            Act::Softmax => self.function(z) * (1.0 - self.function(z)),
        }
    }
}

impl PartialEq for Box<dyn ActivationFunction> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for Box<dyn ActivationFunction> {}

impl Serialize for Box<dyn ActivationFunction> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for Box<dyn ActivationFunction> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let activation: String = Deserialize::deserialize(deserializer)?;

        let act = REGISTER.with_borrow(|register| {
            register.create_activation(&activation).map_err(|err| {
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
        let sum: f32 = output.sum();
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
