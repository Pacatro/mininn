use mininn::prelude::*;
use ndarray::{array, ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

// The implementation of the custom layer
#[derive(Layer, Debug, Clone, Serialize, Deserialize)]
struct CustomLayer;

impl CustomLayer {
    fn new() -> Self {
        Self
    }
}

impl TrainLayer for CustomLayer {
    fn forward(&mut self, input: ArrayViewD<f64>, _mode: &NNMode) -> NNResult<ArrayD<f64>> {
        Ok(input.mapv(|x| x.powi(2)))
    }

    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f64>,
        _learning_rate: f64,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f64>> {
        Ok(output_gradient.mapv(|x| 2. * x))
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
        "CustomActivation"
    }

    fn from_activation(_activation: &str) -> NNResult<Box<dyn ActivationFunction>>
    where
        Self: Sized,
    {
        Ok(Box::new(CustomActivation))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomCost;

impl CostFunction for CustomCost {
    fn function(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> f64 {
        (y - y_p).abs().mean().unwrap_or(0.)
    }

    fn derivate(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> ArrayD<f64> {
        (y_p - y).signum() / y.len() as f64
    }

    fn cost_name(&self) -> &str {
        "CustomCost"
    }

    fn from_cost(_cost: &str) -> NNResult<Box<dyn CostFunction>>
    where
        Self: Sized,
    {
        Ok(Box::new(CustomCost))
    }
}

fn main() {
    let mut nn = NN::new()
        .add(CustomLayer::new())
        .unwrap()
        .add(Activation::new(CustomActivation))
        .unwrap();

    let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let labels = array![[0.0], [1.0], [1.0], [0.0]];

    let train_config = TrainConfig::new()
        .epochs(1)
        .learning_rate(0.1)
        .cost(CustomCost);

    nn.train(train_data.view(), labels.view(), train_config)
        .unwrap();

    match nn.save("custom_layer.h5") {
        Ok(_) => println!("Model saved successfully!"),
        Err(err) => println!("Error: {}", err),
    }

    {
        Register::new()
            .with_layer::<CustomLayer>()
            .with_activation::<CustomActivation>()
            .with_cost::<CustomCost>()
            .register();

        // register!(layer: CustomLayer, act: CustomActivation, cost: CustomCost);

        let nn = NN::load("custom_layer.h5").unwrap();
        for layer in nn.extract_layers::<CustomLayer>().unwrap() {
            println!("{}", layer.layer_type())
        }
        println!("{}", nn.train_config().cost.cost_name());
    }

    std::fs::remove_file("custom_layer.h5").unwrap();
}
