use std::{cell::RefCell, collections::HashMap};

use crate::{
    core::{MininnError, NNResult},
    utils::{Cost, CostFunction},
};

thread_local!(pub(crate) static COST_REGISTER: RefCell<CostRegister> = RefCell::new(CostRegister::new()));

/// Registers a custom cost function in the register.
///
/// This function allows users to register their own custom cost function in the register.
/// The provided function should take a string argument and return a `Box<dyn ActivationFunction>`.
///
/// ## Arguments
///
/// * `cost`: The name of the cost function as a `&str` enum.
///
/// ## Returns
///
/// A `Result` containing the registration result if successful, or an error if something goes wrong.
///
/// ## Examples
///
/// ```rust
/// use mininn::prelude::*;
/// use mininn::utils::Cost; // This should be your own cost function
///
/// register_cost::<Cost>("MSE").unwrap();
/// ```
///
#[inline]
pub fn register_cost<C: CostFunction>(cost: &str) -> NNResult<()> {
    Ok(COST_REGISTER.with(|register| register.borrow_mut().register_cost::<C>(cost))?)
}

#[derive(Debug)]
pub(crate) struct CostRegister {
    registry: HashMap<String, fn(&str) -> NNResult<Box<dyn CostFunction>>>,
}

impl CostRegister {
    /// Creates a new `CostRegister` with default costs registered.
    ///
    /// By default, the register includes constructors for the following cost functions:
    /// - `MSE`: Mean Squared Error.
    /// - `MAE`: Mean Absolute Error.
    /// - `BCE`: Binary Cross-Entropy.
    /// - `CCE`: Categorical Cross-Entropy.
    ///
    pub fn new() -> Self {
        let mut register = CostRegister {
            registry: HashMap::new(),
        };

        register.registry.insert("MSE".to_string(), Cost::from_cost);
        register.registry.insert("MAE".to_string(), Cost::from_cost);
        register.registry.insert("BCE".to_string(), Cost::from_cost);
        register.registry.insert("CCE".to_string(), Cost::from_cost);

        register
    }

    /// Registers a custom cost function in the register.
    ///
    /// This method allows users to register their own custom cost function in the register.
    /// The provided function should take a string argument and return a `Box<dyn ActivationFunction>`.
    ///
    /// ## Arguments
    ///
    /// * `cost`: The name of the cost function as a `&str` enum.
    /// * `constructor`: A function that takes a string and returns a `Box<dyn ActivationFunction>`.
    ///
    pub fn register_cost<C: CostFunction>(&mut self, cost: &str) -> NNResult<()> {
        if cost.is_empty() {
            return Err(MininnError::CostError(
                "Cost must be specified.".to_string(),
            ));
        }

        self.registry.insert(cost.to_string(), C::from_cost);

        Ok(())
    }

    /// Creates an cost function based on its name and cost function.
    ///
    /// This method retrieves the constructor associated with the given `Activation` name
    /// and creates an cost function by deserializing the provided JSON string.
    ///
    /// ## Arguments
    ///
    /// * `cost`: The name of the cost function to create.
    ///
    #[inline]
    pub fn create_cost(&self, cost: &str) -> NNResult<Box<dyn CostFunction>> {
        self.registry
            .get(cost)
            .map_or_else(
                || {
                    Err(MininnError::CostError(format!(
                        "Cost '{}' does not exist in the register. Please add it using the 'register_cost' method.",
                        cost
                    )))
                },
                |constructor| constructor(cost),
            )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, ArrayViewD};

    use super::*;

    #[test]
    fn test_default_cost_register() {
        let register = CostRegister::new();

        assert!(register.registry.contains_key("MSE"));
        assert!(register.registry.contains_key("MAE"));
        assert!(register.registry.contains_key("BCE"));
        assert!(register.registry.contains_key("CCE"));
    }

    #[test]
    fn test_create_cost() {
        let register = CostRegister::new();
        let cost = register.create_cost("MSE").unwrap();
        assert_eq!(cost.cost_name(), "MSE");
    }

    #[test]
    fn test_register_custom_cost() {
        #[derive(Debug)]
        struct CustomCost;

        impl CostFunction for CustomCost {
            fn function(&self, _y_p: &ArrayViewD<f64>, _y: &ArrayViewD<f64>) -> f64 {
                todo!()
            }

            fn derivate(&self, _y_p: &ArrayViewD<f64>, _y: &ArrayViewD<f64>) -> ArrayD<f64> {
                todo!()
            }

            fn cost_name(&self) -> &str {
                "CUSTOM"
            }

            fn from_cost(_cost: &str) -> NNResult<Box<dyn CostFunction>>
            where
                Self: Sized,
            {
                Ok(Box::new(CustomCost))
            }
        }

        let mut register = CostRegister::new();
        register.register_cost::<CustomCost>("CUSTOM").unwrap();
        let cost = register.create_cost("CUSTOM").unwrap();
        println!("{:?}", cost);
        assert_eq!(cost.cost_name(), "CUSTOM");
    }
}
