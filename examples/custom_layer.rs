use mininn::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use ndarray::Array1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CustomLayer;

impl CustomLayer {
    fn new() -> Self { Self }
}

impl Layer for CustomLayer {
    fn layer_type(&self) -> String {
        "Custom".to_string()
    }

    fn to_json(&self) -> NNResult<String> {
        Ok(serde_json::to_string(self).unwrap())
    }

    fn from_json(json: &str) -> NNResult<Box<dyn Layer>> where Self: Sized {
        Ok(Box::new(serde_json::from_str::<CustomLayer>(json).unwrap()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn forward(&mut self, _input: &ndarray::Array1<f64>) -> NNResult<ndarray::Array1<f64>> {
        Ok(Array1::zeros(3))
    }

    fn backward(&mut self, _output_gradient: ndarray::ArrayView1<f64>, _learning_rate: f64) -> NNResult<ndarray::Array1<f64>> {
        Ok(Array1::zeros(3))
    }
}

fn main() {
    let nn = NN::new()
        .add(CustomLayer::new());

    let save = nn.save("load_models/custom_layer.h5");

    if save.is_ok() {
        // Imagine this is a different program (you need the implementation of the custom layer)
        let custom = CustomLayer::new();
        let mut register = LayerRegister::new();
        register.register_layer(&custom.layer_type(), CustomLayer::from_json);
        let load_nn = NN::load("load_models/custom_layer.h5", Some(register)).unwrap();
        assert!(!load_nn.is_empty());
        assert!(load_nn.extract_layers::<CustomLayer>().is_ok());
    }
}