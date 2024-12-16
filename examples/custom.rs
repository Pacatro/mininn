use mininn::{prelude::*, register};
use ndarray::{array, ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

// The implementation of the custom layer
#[derive(Layer, Debug, Clone, Serialize, Deserialize)]
struct CustomLayer;

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

#[derive(Layer, Debug, Clone, Serialize, Deserialize)]
pub struct CustomLayer1;

impl TrainLayer for CustomLayer1 {
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

#[derive(ActivationFunction, Debug, Clone)]
struct CustomActivation;

impl ActCore for CustomActivation {
    fn function(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
        z.mapv(|x| x.powi(2))
    }

    fn derivate(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
        z.mapv(|x| 2. * x)
    }
}

#[derive(CostFunction, Debug, Clone, Serialize, Deserialize)]
pub struct CustomCost;

impl CostCore for CustomCost {
    fn function(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> f64 {
        (y - y_p).abs().mean().unwrap_or(0.)
    }

    fn derivate(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> ArrayD<f64> {
        (y_p - y).signum() / y.len() as f64
    }
}

fn main() {
    let mut nn = NN::new()
        .add(CustomLayer)
        .unwrap()
        .add(CustomLayer1)
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
        // Or you can use the macro to register your own layers, activations and costs
        register!(
            layers: [CustomLayer, CustomLayer1],
            acts: [CustomActivation],
            costs: [CustomCost]
        );

        let nn = NN::load("custom_layer.h5").unwrap();
        for layer in nn.extract_layers::<CustomLayer>().unwrap() {
            println!("{}", layer.layer_type())
        }

        for layer in nn.extract_layers::<CustomLayer1>().unwrap() {
            println!("{}", layer.layer_type())
        }

        let activations = nn.extract_layers::<Activation>().unwrap();
        println!("Act: {}", activations[0].activation());
        println!("{}", nn.train_config().cost.cost_name());
    }

    std::fs::remove_file("custom_layer.h5").unwrap();
}
