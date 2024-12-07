# MiniNN

[![Crates.io](https://img.shields.io/crates/v/mininn.svg)](https://crates.io/crates/mininn)
[![Downloads](https://img.shields.io/crates/d/mininn.svg)](https://crates.io/crates/mininn)
[![Docs](https://docs.rs/mininn/badge.svg)](https://docs.rs/mininn)

A minimalist deep learnig crate for rust.

## ✏️ Usage

For this example we will resolve the classic XOR problem

```rust
use ndarray::{array, Array1};

use mininn::prelude::*;

fn main() -> NNResult<()> {
    let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],];

    let labels = array![[0.0], [1.0], [1.0], [0.0],];

    // Create the neural network
    let mut nn = NN::new()
        .add(Dense::new(2, 3, Some(ActivationFunc::TANH)))?
        .add(Dense::new(3, 1, Some(ActivationFunc::TANH)))?;

    // Train the neural network
    let loss = nn.train(
        &train_data,
        &labels,
        Cost::BCE,
        1000,
        0.1,
        2,
        Optimizer::GD,
        true,
    )?;

    println!("Predictions:\n");

    let predictions: Array1<f64> = train_data
        .rows()
        .into_iter()
        .map(|input| {
            let pred = nn.predict(&input.to_owned()).unwrap();
            let out = if pred[0] >= 0.9 { 1.0 } else { 0.0 };
            println!("{} --> {}", input, out);
            out
        })
        .collect();

    // Calc metrics using MetricsCalculator
    let metrics = MetricsCalculator::new(&labels, &predictions);

    println!("\nConfusion matrix:\n{}\n", metrics.confusion_matrix());

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\nLoss: {}",
        metrics.accuracy(),
        metrics.recall(),
        metrics.precision(),
        metrics.f1_score(),
        loss
    );

    // Save the model into a HDF5 file
    match nn.save("model.h5") {
        Ok(_) => println!("Model saved successfully!"),
        Err(e) => println!("Error saving model: {}", e),
    }

    Ok(())
}

```

### Output

```terminal
Epoch 1/1000 - Loss: 0.37767715592285533, Time: 0.000301444 sec
Epoch 2/1000 - Loss: 0.3209450799267143, Time: 0.000216753 sec
Epoch 3/1000 - Loss: 0.3180416337628711, Time: 0.00022032 sec
...
Epoch 998/1000 - Loss: 0.000011881245192030034, Time: 0.00021529 sec
Epoch 999/1000 - Loss: 0.000011090649737601982, Time: 0.000215882 sec
Epoch 1000/1000 - Loss: 0.000011604905569853055, Time: 0.000215721 sec

Training Completed!
Total Training Time: 0.22 sec
Predictions:

[0, 0] --> 0
[0, 1] --> 1
[1, 0] --> 1
[1, 1] --> 0

Confusion matrix:
[[2, 0],
 [0, 2]]

Accuracy: 1
Recall: 1
Precision: 1
F1: 1
Loss: 0.000011604905569853055
Model saved successfully!
```

### Metrics

You can also calculate metrics for your models using `MetricsCalculator`:

```rust
let metrics = MetricsCalculator::new(&labels, &predictions);

println!("\nConfusion matrix:\n{}\n", metrics.confusion_matrix());

println!(
    "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n",
    metrics.accuracy(), metrics.recall(), 
    metrics.precision(), metrics.f1_score()
);
```

This is the output of the `iris` example

```terminal
Confusion matrix:
[[26, 0, 0],
 [0, 28, 1],
 [0, 2, 18]]

Accuracy: 0.96
Recall: 0.9551724137931035
Precision: 0.960233918128655
F1: 0.9574098218166016
```

### Default Layers

For now, the crate only offers two types of layers:

| Layer          | Description                                                                                                      |
|----------------|------------------------------------------------------------------------------------------------------------------|
| `Dense`        | Fully connected layer where each neuron connects to every neuron in the previous layer. It computes the weighted sum of inputs, adds a bias term, and applies an optional activation function (e.g., ReLU, Sigmoid). This layer is fundamental for transforming input data in deep learning models. |
| `Activation`   | Applies a non-linear transformation (activation function) to its inputs. Common activation functions include ReLU, Sigmoid, Tanh, and Softmax. These functions introduce non-linearity to the model, allowing it to learn complex patterns. |
| `Dropout`      | Applies dropout, a regularization technique where randomly selected neurons are ignored during training. This helps prevent overfitting by reducing reliance on specific neurons and forces the network to learn more robust features. Dropout is typically used in the training phase and is deactivated during inference. |

> [!NOTE]
> More layers in the future.

### Save and load models

When you already have a trained model you can save it into a HDF5 file:

```rust
nn.save("model.h5").unwrap();
let mut nn = NN::load("model.h5", None).unwrap();
```

### Custom layers

All the layers that are in the network needs to implement the `Layer` trait, so is possible for users to create their own custom layers.

The only rule is that all the layers must implements the following traits (instead of the `Layer` trait):

- `Debug`: Standars traits.
- `Clone`: Standars traits.
- `Serialize` and `Deserialize`: From [`serde`](https://crates.io/crates/serde) crate.

Here is a little example about how to create custom layers:

```rust
use mininn::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use ndarray::Array1;

// The implementation of the custom layer
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CustomLayer;

impl CustomLayer {
    fn new() -> Self { Self }
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

    fn forward(&mut self, _input: &Array1<f64>, _mode: &NNMode) -> NNResult<Array1<f64>> {
        Ok(Array1::zeros(3))
    }

    fn backward(
        &mut self,
        _output_gradient: &Array1<f64>,
        _learning_rate: f64,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<Array1<f64>> {
        Ok(Array1::zeros(3))
    }
}

fn main() {
    let nn = NN::new()
        .add(CustomLayer::new()).unwrap();
    nn.save("custom_layer.h5").unwrap();
}
```

If you want to use a model with a custom layer, you need to add it into the `LayerRegister`, this is a data structure that stored all the types of layers that the `NN` struct is going to accept.

```rust
fn main() {
    // You need to have the implementation of the custom layer
    let custom = CustomLayer::new();
    // Create a new register.
    let mut register = LayerRegister::new();
    // Register the new layer
    register.register_layer(&custom.layer_type(), CustomLayer::from_json).unwrap();
    // Use the register as a parameter in the load method.
    let load_nn = NN::load("custom_layer.h5", Some(register)).unwrap();
    assert!(!load_nn.is_empty());
    assert!(load_nn.extract_layers::<CustomLayer>().is_ok());
}
```

## 🔧 Setup

You can add the crate with `cargo`

```terminal
cargo add mininn
```

Alternatively, you can manually add it to your project's Cargo.toml like this:

```toml
[dependencies]
mininn = "*" # Change the `*` to the current version
```

<!-- ## 💻 Contributing

If you want to help adding new features to this crate, you can contact with me to talk about it. -->

## 📋 Examples

There is a multitude of examples resolving classics ML problems, if you want to see the results just run these commands.

```bash
cargo run --example iris
cargo run --example xor [optional_path_to_model]     # If no path is provided, the model won't be saved
cargo run --example mnist [optional_path_to_model]   # If no path is provided, the model won't be saved
cargo run --example xor_load_nn <path_to_model>
cargo run --example mnist_load_nn <path_to_model>
```

## 📑 Libraries used

- [ndarray](https://docs.rs/ndarray/latest/ndarray/) - For manage N-Dimensional Arrays.
- [ndarray-rand](https://docs.rs/ndarray-rand/0.15.0/ndarray_rand/) - For manage Random N-Dimensional Arrays.
- [serde](https://docs.rs/serde/latest/serde/) - For serialization.
- [serde_json](https://docs.rs/serde_json/latest/serde_json/) - For JSON serialization.
- [hdf5](https://docs.rs/hdf5/latest/hdf5/) - For model storage.

## 🔑 License

[MIT](https://opensource.org/license/mit/) - Created by [**Paco Algar**](https://github.com/Pacatro).
