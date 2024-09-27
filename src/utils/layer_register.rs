use std::collections::HashMap;

use crate::layers::{Activation, Dense, Layer, LayerType};

/// A registry for storing and creating neural network layers.
///
/// The `LayerRegister` struct manages a registry of layer constructors, allowing users to register 
/// custom layers and create them dynamically from serialized JSON data. This is useful for deserializing 
/// neural network models from files or other formats.
///
/// The registry maps a `LayerType` enum to a function that can construct a layer from its JSON representation.
///
/// ## Fields
///
/// - `registry`: A `HashMap` where the key is a `LayerType`, and the value is a function pointer that
///   creates a `Box<dyn Layer>` from a JSON string.
///
pub(crate) struct LayerRegister {
    registry: HashMap<LayerType, fn(&str) -> Box<dyn Layer>>,
}

impl LayerRegister {
    /// Creates a new `LayerRegister` with default layers registered.
    ///
    /// By default, the register includes constructors for the following layers:
    /// - `Dense`: Fully connected layers.
    /// - `Activation`: Activation layers (e.g., ReLU, Sigmoid).
    ///
    pub fn new() -> Self {
        let mut register = LayerRegister { registry: HashMap::new() };

        register.register_layer(LayerType::Dense, Dense::from_json);
        register.register_layer(LayerType::Activation, Activation::from_json);

        register
    }

    /// Registers a custom layer in the registry.
    ///
    /// This method allows users to register their own custom layer constructors, enabling
    /// the creation of new types of layers from JSON.
    ///
    /// ## Arguments
    ///
    /// - `layer_type`: The type of the layer as a `LayerType` enum.
    /// - `constructor`: A function that takes a JSON string and returns a `Box<dyn Layer>`.
    ///
    pub fn register_layer(&mut self, layer_type: LayerType, constructor: fn(&str) -> Box<dyn Layer>) {
        self.registry.insert(layer_type, constructor);
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
    pub fn create_layer(&self, layer_type: &LayerType, json: &str) -> Box<dyn Layer> {
        let constructor = self.registry.get(layer_type).unwrap();
        constructor(json)
    }
}