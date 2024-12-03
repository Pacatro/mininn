use mininn::prelude::*;
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use serde_json;

// The implementation of the custom layer
#[derive(Debug, Serialize, Deserialize)]
struct CustomLayer;

impl CustomLayer {
    fn new() -> Self {
        Self
    }
}

// Implement the Layer trait for the custom layer
impl Layer for CustomLayer {
    fn layer_type(&self) -> String {
        "Custom".to_string()
    }

    fn to_json(&self) -> NNResult<String> {
        Ok(serde_json::to_string(self).unwrap())
    }

    fn from_json(json: &str) -> NNResult<Box<dyn Layer>>
    where
        Self: Sized,
    {
        Ok(Box::new(serde_json::from_str::<CustomLayer>(json).unwrap()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn forward(&mut self, _input: &ArrayD<f64>, _mode: &NNMode) -> NNResult<ArrayD<f64>> {
        Ok(ArrayD::zeros(IxDyn(&[3])))
    }

    fn backward(
        &mut self,
        _output_gradient: &ArrayD<f64>,
        _learning_rate: f64,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f64>> {
        Ok(ArrayD::zeros(IxDyn(&[3])))
    }
}

fn main() {
    let nn = NN::new().add(CustomLayer::new()).unwrap();

    if nn.save("custom_layer.h5").is_ok() {
        let register = LayerRegister::new()
            .register_layer("Custom", CustomLayer::from_json)
            .unwrap();

        let nn = NN::load("custom_layer.h5", Some(register)).unwrap();
        for layer in nn.extract_layers::<CustomLayer>().unwrap() {
            println!("{}", layer.layer_type())
        }
    }
}
