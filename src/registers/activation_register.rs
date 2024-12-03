use std::{cell::RefCell, collections::HashMap};

use crate::{
    core::{MininnError, NNResult},
    utils::{Act, ActivationFunction},
};

thread_local!(pub(crate) static ACT_REGISTER: RefCell<ActivationRegister> = RefCell::new(ActivationRegister::new()));

/// Registers a custom activation function in the register.
///
/// This function allows users to register their own custom activation function in the register.
/// The provided function should take a string argument and return a `Box<dyn ActivationFunction>`.
///
/// ## Arguments
///
/// * `activation`: The name of the activation function as a `&str` enum.
///
/// ## Returns
///
/// A `Result` containing the registration result if successful, or an error if something goes wrong.
///
/// ## Examples
///
/// ```rust
/// use mininn::prelude::*;
/// use mininn::utils::Act; // This should be your own activation function
///
/// register_activation::<Act>("ReLU").unwrap();
/// ```
///
#[inline]
pub fn register_activation<A: ActivationFunction>(activation: &str) -> NNResult<()> {
    Ok(ACT_REGISTER.with(|register| register.borrow_mut().register_activation::<A>(activation))?)
}

#[derive(Debug)]
pub(crate) struct ActivationRegister {
    registry: HashMap<String, fn(&str) -> NNResult<Box<dyn ActivationFunction>>>,
}

impl ActivationRegister {
    /// Creates a new `ActivationRegister` with default activations registered.
    ///
    /// By default, the register includes constructors for the following activation functions:
    /// - `Step`: Step function.
    /// - `Sigmoid`: Sigmoid function.
    /// - `ReLU`: Rectified Linear Unit (ReLU) function.
    /// - `Tanh`: Hyperbolic Tangent function.
    /// - `Softmax`: Softmax function.
    ///
    pub fn new() -> Self {
        let mut register = ActivationRegister {
            registry: HashMap::new(),
        };

        register
            .registry
            .insert("Step".to_string(), Act::from_activation);

        register
            .registry
            .insert("Sigmoid".to_string(), Act::from_activation);

        register
            .registry
            .insert("ReLU".to_string(), Act::from_activation);

        register
            .registry
            .insert("Tanh".to_string(), Act::from_activation);

        register
            .registry
            .insert("Softmax".to_string(), Act::from_activation);

        register
    }

    /// Registers a custom activation function in the register.
    ///
    /// This method allows users to register their own custom activation function in the register.
    /// The provided function should take a string argument and return a `Box<dyn ActivationFunction>`.
    ///
    /// ## Arguments
    ///
    /// * `activation`: The name of the activation function as a `&str` enum.
    /// * `constructor`: A function that takes a string and returns a `Box<dyn ActivationFunction>`.
    ///
    pub fn register_activation<A: ActivationFunction>(&mut self, activation: &str) -> NNResult<()> {
        if activation.is_empty() {
            return Err(MininnError::ActivationRegisterError(
                "Activation must be specified.".to_string(),
            ));
        }

        self.registry
            .insert(activation.to_string(), A::from_activation);

        Ok(())
    }

    /// Creates an activation function based on its name and activation function.
    ///
    /// This method retrieves the constructor associated with the given `Activation` name
    /// and creates an activation function by deserializing the provided JSON string.
    ///
    /// ## Arguments
    ///
    /// * `activation`: The name of the activation function to create.
    ///
    #[inline]
    pub fn create_activation(&self, activation: &str) -> NNResult<Box<dyn ActivationFunction>> {
        self.registry
            .get(activation)
            .map_or_else(
                || {
                    Err(MininnError::ActivationRegisterError(format!(
                        "Activation '{}' does not exist in the register. Please add it using the 'register_activation' method.",
                        activation
                    )))
                },
                |constructor| constructor(activation),
            )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, ArrayViewD};

    use super::*;

    #[test]
    fn test_default_activation_register() {
        let register = ActivationRegister::new();

        assert!(register.registry.contains_key("Step"));
        assert!(register.registry.contains_key("Sigmoid"));
        assert!(register.registry.contains_key("ReLU"));
        assert!(register.registry.contains_key("Tanh"));
        assert!(register.registry.contains_key("Softmax"));
    }

    #[test]
    fn test_create_activation() {
        let register = ActivationRegister::new();
        let activation = register.create_activation("Step").unwrap();
        assert_eq!(activation.activation(), "Step");
    }

    #[test]
    fn test_register_custom_activation() {
        #[derive(Debug)]
        struct CustomActivation;

        impl ActivationFunction for CustomActivation {
            fn function(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
                z.mapv(|x| x.powi(2))
            }

            fn derivate(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
                z.mapv(|x| 2. * x)
            }

            fn activation(&self) -> &str {
                "CUSTOM"
            }

            fn from_activation(_activation: &str) -> NNResult<Box<dyn ActivationFunction>>
            where
                Self: Sized,
            {
                Ok(Box::new(CustomActivation))
            }
        }

        let mut register = ActivationRegister::new();
        register
            .register_activation::<CustomActivation>("CUSTOM")
            .unwrap();
        let activation = register.create_activation("CUSTOM").unwrap();
        println!("{:?}", activation);
        assert_eq!(activation.activation(), "CUSTOM");
    }
}
