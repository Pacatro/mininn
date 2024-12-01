use std::collections::HashMap;

use crate::utils::ActivationFunction;

// TODO: Implement ActivationRegister
#[derive(Debug)]
pub struct ActivationRegister {
    _registry: HashMap<String, fn(&str) -> Box<dyn ActivationFunction>>,
}

// impl ActivationRegister {
//     pub fn new() -> Self {
//         let mut register = ActivationRegister {
//             registry: HashMap::new(),
//         };

//         register
//             .registry
//             .insert("STEP".to_string(), ActivationFunc::STEP.into());

//         register
//             .registry
//             .insert("SIGMOID".to_string(), ActivationFunc::SIGMOID.into());

//         register
//             .registry
//             .insert("RELU".to_string(), ActivationFunc::RELU.into());

//         register
//             .registry
//             .insert("TANH".to_string(), ActivationFunc::TANH.into());

//         register
//             .registry
//             .insert("SOFTMAX".to_string(), ActivationFunc::SOFTMAX.into());

//         register
//     }

//     /// Registers a custom activation function in the registry.
//     ///
//     /// This method allows users to register their own custom activation functions, enabling
//     /// the creation of new types of activation functions from JSON.
//     ///
//     /// ## Arguments
//     ///
//     /// - `activation_type`: The type of the activation function as a `&str` enum.
//     /// - `constructor`: A function that takes a JSON string and returns a `Box<dyn ActivationFunction>`.
//     ///
//     pub fn register_activation(
//         &mut self,
//         activation_type: &str,
//         constructor: fn(&str) -> Box<dyn ActivationFunction>,
//     ) -> NNResult<()> {
//         if activation_type.is_empty() {
//             return Err(MininnError::ActivationRegisterError(
//                 "Activation type must be specified.".to_string(),
//             ));
//         }

//         self.registry
//             .insert(activation_type.to_string(), constructor);

//         Ok(())
//     }

//     /// Creates an activation function based on its type and JSON representation.
//     ///
//     /// This method retrieves the constructor associated with the given `ActivationType`
//     /// and creates an activation function by deserializing the provided JSON string.
//     ///
//     /// ## Arguments
//     ///
//     /// - `activation_type`: The type of the activation function to create.
//     /// - `json`: The serialized representation of the activation function in JSON format.
//     ///
//     #[inline]
//     pub fn create_activation(
//         &self,
//         activation_type: &str,
//         json: &str,
//     ) -> NNResult<Box<dyn ActivationFunction>> {
//         self.registry
//             .get(activation_type)
//             .map_or_else(
//                 || {
//                     Err(MininnError::ActivationRegisterError(format!(
//                         "Activation '{}' does not exist in the register. Please add it using the 'register_activation' method.",
//                         activation_type
//                     )))
//                 },
//                 |constructor| constructor(json)
//             )
//     }
// }
