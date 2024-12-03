use mininn::prelude::*;
use ndarray::{ArrayD, ArrayViewD, IxDyn};
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

fn main() {
    let nn = NN::new()
        .add(CustomLayer::new())
        .unwrap()
        .add(Activation::new(CustomActivation))
        .unwrap();

    match nn.save("custom_layer.h5") {
        Ok(_) => println!("Model saved successfully!"),
        Err(err) => println!("Error: {}", err),
    }

    {
        register_layer::<CustomLayer>("Custom").unwrap();
        register_activation::<CustomActivation>("CUSTOM").unwrap();

        let nn = NN::load("custom_layer.h5").unwrap();
        for layer in nn.extract_layers::<CustomLayer>().unwrap() {
            println!("{}", layer.layer_type())
        }
    }

    std::fs::remove_file("custom_layer.h5").unwrap();
}
