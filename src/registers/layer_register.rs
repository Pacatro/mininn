use std::{cell::RefCell, collections::HashMap};

use crate::{
    core::{MininnError, NNResult},
    layers::{Activation, Dense, Dropout, Layer},
};

thread_local!(pub(crate) static LAYER_REGISTER: RefCell<LayerRegister> = RefCell::new(LayerRegister::new()));

/// Registers a custom layer in the register.
///
/// This function allows users to register their own custom layer in the register.
/// The provided function should take a string argument and return a `Box<dyn Layer>`.
///
/// ## Arguments
///
/// * `layer_type`: The type of the layer as a `&str` enum.
///
/// ## Returns
///
/// A `Result` containing the registration result if successful, or an error if something goes wrong.
///
/// ## Examples
///
/// ```rust
/// use mininn::prelude::*;
/// use mininn::layers::Dense; // This should be your own layer
///
/// register_layer::<Dense>("Dense").unwrap();
/// ```
///
#[inline]
pub fn register_layer<L: Layer>(layer_type: &str) -> NNResult<()> {
    Ok(LAYER_REGISTER.with(|register| register.borrow_mut().register_layer::<L>(layer_type))?)
}

/// A registry for storing and creating neural network layers.
///
/// The `LayerRegister` struct manages a registry of layer constructors, allowing users to register
/// custom layers and create them dynamically from serialized JSON data. This is useful when you want to load a model from a file.
///
/// The registry maps the layer type to a function that can construct a layer from its JSON representation.
///
/// ## Fields
///
/// - `registry`: A `HashMap` where the key is a `String`, and the value is a function pointer that
///   creates a `Box<dyn Layer>` from a JSON string.
///
#[derive(Debug)]
pub(crate) struct LayerRegister {
    registry: HashMap<String, fn(&str) -> NNResult<Box<dyn Layer>>>,
}

impl LayerRegister {
    /// Creates a new `LayerRegister` with default layers registered.
    ///
    /// By default, the register includes constructors for the following layers:
    /// - `Dense`: Fully connected layers.
    /// - `Activation`: Activation layers (e.g., RELU, Sigmoid).
    ///
    pub fn new() -> Self {
        let mut register = LayerRegister {
            registry: HashMap::new(),
        };

        register
            .registry
            .insert("Dense".to_string(), Dense::from_json);

        register
            .registry
            .insert("Activation".to_string(), Activation::from_json);

        register
            .registry
            .insert("Dropout".to_string(), Dropout::from_json);

        register
    }

    /// Registers a custom layer in the registry.
    ///
    /// This method allows users to register their own custom layer constructors, enabling
    /// the creation of new types of layers from JSON.
    ///
    /// ## Arguments
    ///
    /// - `layer_type`: The type of the layer as a `&str` enum.
    /// - `constructor`: A function that takes a JSON string and returns a `Box<dyn Layer>`.
    ///
    pub fn register_layer<L: Layer>(&mut self, layer_type: &str) -> NNResult<()> {
        if layer_type.is_empty() {
            return Err(MininnError::LayerRegisterError(
                "Layer type must be specified.".to_string(),
            ));
        }

        self.registry.insert(layer_type.to_string(), L::from_json);

        Ok(())
    }

    /// Creates a layer based on its type and JSON representation.
    ///
    /// This method retrieves the constructor associated with the given `LayerType`
    /// and creates a layer by deserializing the provided JSON string.
    ///
    /// ## Arguments
    ///
    /// - `layer_type`: The type of the layer to create.
    /// - `json`: The serialized representation of the layer in JSON format.
    ///
    #[inline]
    pub fn create_layer(&self, layer_type: &str, json: &str) -> NNResult<Box<dyn Layer>> {
        self.registry
            .get(layer_type)
            .map_or_else(
                || {
                    Err(MininnError::LayerRegisterError(format!(
                        "Layer '{}' does not exist in the register. Please add it using the 'register_layer' method.",
                        layer_type
                    )))
                },
                |constructor| constructor(json)
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::NNMode,
        layers::{Activation, Dense},
        utils::Optimizer,
    };
    use ndarray::{array, Array1, Array2, ArrayD, ArrayViewD};
    use serde_json::json;

    /// Test default registration of layers in LayerRegister.
    #[test]
    fn test_default_layers_registered() {
        let register = LayerRegister::new();

        // Check if Dense and Activation layers are registered by default.
        assert!(register.registry.contains_key("Dense"));
        assert!(register.registry.contains_key("Activation"));
        assert!(register.registry.contains_key("Dropout"));
    }

    /// Test creating a Dense layer from JSON.
    #[test]
    fn test_create_dense_layer() {
        let register = LayerRegister::new();
        let weights: Array2<f64> = array![[]];
        let biases: Array1<f64> = array![];
        let input: Array1<f64> = array![];
        // JSON representation of a Dense layer.
        let dense_json = json!({
            "input_size": 3,
            "output_size": 2,
            "activation": "ReLU",
            "weights": weights,
            "biases": biases,
            "input": input,
            "layer_type": "Dense"
        })
        .to_string();

        let layer = register.create_layer("Dense", &dense_json).unwrap();
        assert!(layer.as_any().is::<Dense>(), "Expected a Dense layer");
    }

    /// Test creating an Activation layer from JSON.
    #[test]
    fn test_create_activation_layer() {
        let register = LayerRegister::new();
        let input: Array1<f64> = array![];
        // JSON representation of an Activation layer.
        let activation_json = json!({
            "activation": "Sigmoid",
            "input": input,
            "layer_type": "Activation"
        })
        .to_string();

        let layer = register
            .create_layer("Activation", &activation_json)
            .unwrap();
        assert!(
            layer.as_any().is::<Activation>(),
            "Expected an Activation layer"
        );
    }

    #[test]
    fn test_create_dropout_layer() {
        let register = LayerRegister::new();
        let input: Array1<f64> = array![];
        let mask: Array1<f64> = array![];
        let activation_json = json!({
            "input": input,
            "p": 0.5,
            "seed": 42,
            "mask": mask,
            "layer_type": "Dropout"
        })
        .to_string();

        let layer = register.create_layer("Dropout", &activation_json).unwrap();
        assert!(layer.as_any().is::<Dropout>(), "Expected an Dropout layer");
    }

    /// Test registering and creating a custom layer.
    #[test]
    fn test_register_custom_layer() {
        // A custom layer type.
        #[derive(Debug)]
        struct CustomLayer;

        impl Layer for CustomLayer {
            fn backward(
                &mut self,
                _output_gradient: ArrayViewD<f64>,
                _learning_rate: f64,
                _optimizer: &Optimizer,
                _mode: &NNMode,
            ) -> NNResult<ArrayD<f64>> {
                todo!()
            }
            fn forward(
                &mut self,
                _input: ArrayViewD<f64>,
                _mode: &NNMode,
            ) -> NNResult<ArrayD<f64>> {
                todo!()
            }
            fn from_json(_json: &str) -> NNResult<Box<dyn Layer>>
            where
                Self: Sized,
            {
                Ok(Box::new(CustomLayer))
            }
            fn layer_type(&self) -> String {
                "Custom".to_string()
            }
            fn to_json(&self) -> NNResult<String> {
                todo!()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        // Register the custom layer.
        let mut register = LayerRegister::new();
        register.register_layer::<CustomLayer>("Custom").unwrap();

        // JSON representation of the custom layer.
        let custom_json = "{}".to_string();

        let layer = register.create_layer("Custom", &custom_json).unwrap();
        assert!(layer.as_any().is::<CustomLayer>(), "Expected a CustomLayer");
    }
}
