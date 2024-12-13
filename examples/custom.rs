use mininn::prelude::*;
use ndarray::{ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

// The implementation of the custom layer
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    fn to_msg_pack(&self) -> NNResult<Vec<u8>> {
        Ok(rmp_serde::to_vec(self)?)
    }

    fn from_msg_pack(buff: &[u8]) -> NNResult<Box<dyn Layer>> {
        Ok(Box::new(rmp_serde::from_slice::<Self>(buff)?))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn forward(&mut self, _input: ArrayViewD<f64>, _mode: &NNMode) -> NNResult<ArrayD<f64>> {
        todo!()
    }

    fn backward(
        &mut self,
        _output_gradient: ArrayViewD<f64>,
        _learning_rate: f64,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f64>> {
        todo!()
    }
}

#[derive(Debug, Clone)]
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
