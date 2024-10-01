use std::collections::HashMap;

use crate::{error::{MininnError, NNResult}, layers::{Activation, Conv, Dense, Layer}};

/// A registry for storing and creating neural network layers.
///
/// The `LayerRegister` struct manages a registry of layer constructors, allowing users to register 
/// custom layers and create them dynamically from serialized JSON data. This is useful for deserializing 
/// neural network models from files or other formats.
///
/// The registry maps a `String` enum to a function that can construct a layer from its JSON representation.
///
/// ## Fields
///
/// - `registry`: A `HashMap` where the key is a `String`, and the value is a function pointer that
///   creates a `Box<dyn Layer>` from a JSON string.
///
#[derive(Debug)]
pub struct LayerRegister {
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
        let mut register = LayerRegister { registry: HashMap::new() };

        register.registry.insert("Dense".to_string(), Dense::from_json);
        register.registry.insert("Activation".to_string(), Activation::from_json);
        register.registry.insert("Conv".to_string(), Conv::from_json);

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
    pub fn register_layer(&mut self, layer_type: &str, constructor: fn(&str) -> NNResult<Box<dyn Layer>>) -> NNResult<()> {
        if layer_type.is_empty() {
            return Err(MininnError::LayerRegisterError("Layer type must be specified.".to_string()));
        }

        self.registry.insert(layer_type.to_string(), constructor);

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
    use crate::layers::{Dense, Activation};
    use ndarray::{array, Array1, Array2, ArrayView1};
    use serde_json::json;

    /// Test default registration of layers in LayerRegister.
    #[test]
    fn test_default_layers_registered() {
        let register = LayerRegister::new();
        
        // Check if Dense and Activation layers are registered by default.
        assert!(register.registry.contains_key("Dense"));
        assert!(register.registry.contains_key("Activation"));
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
            "activation": "RELU",
            "weights": weights,
            "biases": biases,
            "input": input,
            "layer_type": "Dense"
        }).to_string();
        
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
            "activation": "SIGMOID",
            "input": input,
            "layer_type": "Activation"
        }).to_string();
        
        let layer = register.create_layer("Activation", &activation_json).unwrap();
        assert!(layer.as_any().is::<Activation>(), "Expected an Activation layer");
    }

    /// Test registering and creating a custom layer.
    #[test]
    fn test_register_custom_layer() {
        let mut register = LayerRegister::new();
        
        // A custom layer type.
        #[derive(Debug)]
        struct CustomLayer;
        
        impl Layer for CustomLayer {
            fn backward(&mut self, _output_gradient: ArrayView1<f64>, _learning_rate: f64) -> NNResult<Array1<f64>> {
                todo!()
            }
            fn forward(&mut self, _input: &Array1<f64>) -> NNResult<Array1<f64>> {
                todo!()
            }
            fn from_json(_json: &str) -> NNResult<Box<dyn Layer>> where Self: Sized {
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
        register.register_layer("Custom", CustomLayer::from_json).unwrap();
        
        // JSON representation of the custom layer.
        let custom_json = "{}".to_string();
        
        let layer = register.create_layer("Custom", &custom_json).unwrap();
        assert!(layer.as_any().is::<CustomLayer>(), "Expected a CustomLayer");
    }
}
