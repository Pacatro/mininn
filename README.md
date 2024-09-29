# MiniNN

A minimalist deep learnig crate for rust.

> [!WARNING]
> This crate is not complete. It will be updated and published on [crates.io](https://crates.io/) in the future.

## ✏️ Usage

For this example we will resolve the classic XOR problem

```rust
use ndarray::{array, Array1};

use mininn::prelude::*;

fn main() -> NNResult<()> {
    let train_data = array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];

    let labels = array![
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ];

    // Create the neural network
    let mut nn = NN::new()
        .add(Dense::new(2, 3, Some(ActivationFunc::TANH)))?
        .add(Dense::new(3, 1, Some(ActivationFunc::TANH)))?;

    // Train the neural network
    nn.train(Cost::MSE, &train_data, &labels, 1000, 0.1, true)?;

    let mut predictions = Vec::new();

    for input in train_data.rows() {
        // Use predict to see the resutl of the network
        let pred = nn.predict(&input.to_owned())?;
        let out = if pred[0] < 0.5 { 0 } else { 1 };
        predictions.push(out as f64);
        println!("{} --> {}", input, out)
    }

    // Calc metrics using MetricsCalculator
    let metrics = MetricsCalculator::new(&labels, &Array1::from_vec(predictions));

    println!("\n{}\n", metrics.confusion_matrix()?);

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n",
        metrics.accuracy()?, metrics.recall()?, metrics.precision()?,
        metrics.f1_score()?
    );

    // Save the model into a HDF5 file
    nn.save("load_models/xor.h5").unwrap_or_else(|err| eprintln!("{err}"));

    Ok(())
}
```

### Output

```terminal
Epoch 1/1000, error: 0.5241278261886898, time: 0.000293329 sec
Epoch 2/1000, error: 0.42558144595907677, time: 0.000232393 sec
Epoch 3/1000, error: 0.3776874679368199, time: 0.000237112 sec
...
Epoch 998/1000, error: 0.0017618690075552517, time: 0.000231691 sec
Epoch 999/1000, error: 0.001949346158027843, time: 0.00022584 sec
Epoch 1000/1000, error: 0.0022311549699578458, time: 0.000225159 sec
Training completed!
Training Error: 0.027032078040398873 , time: 0.23521075 sec
[0, 0] --> 0
[0, 1] --> 1
[1, 0] --> 1
[1, 1] --> 0

[[2, 0],
 [0, 2]]

Accuracy: 1
Recall: 1
Precision: 1
F1: 1
```

### Metrics

You can also calculate metrics for your models using `ClassMetrics`:

```rust
let metrics = MetricsCalculator::new(&labels, &Array1::from_vec(predictions));

println!("\nConfusion matrix:\n{}\n", metrics.confusion_matrix()?);

println!(
    "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n",
    metrics.accuracy()?, metrics.recall()?, metrics.precision()?,
    metrics.f1_score()?
);
```

```terminal
Confusion matrix:
[[2, 0],
 [0, 2]]

Accuracy: 1
Recall: 1
Precision: 1
F1: 1
```

### Default Layers

For now, the crate only offers two types of layers:

| Layer    | Description                         |
|----------|-------------------------------------|
| `Dense`         | Fully connected layer where each neuron connects to every neuron in the previous layer. It computes the weighted sum of inputs, adds a bias term, and applies an optional activation function (e.g., ReLU, Sigmoid). This layer is fundamental for transforming input data in deep learning models.       |
| `Activation`    | Applies a non-linear transformation (activation function) to its inputs. Common activation functions include ReLU, Sigmoid, Tanh, and Softmax. These functions introduce non-linearity to the model, allowing it to learn complex patterns.                       |

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
- `Serialize`: From [`serde`](https://crates.io/crates/serde) crate.
- `Deserialize` From [`serde`](https://crates.io/crates/serde) crate.

If you want to save your model with your new custom Layer, you need to add it into the `LayerRegister`, this is a data structure that stored all the types of layers that the `NN` struct is going to accept.

Here is a little example about how to create custom layers:

```rust
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

    let save = nn.save("custom_layer.h5");

    if save.is_ok() {
        // Imagine this is a different program (you need the implementation of the custom layer of course)
        let custom = CustomLayer::new();
        // Create a new register.
        let mut register = LayerRegister::new();
        // Register the new layer
        register.register_layer(&custom.layer_type(), CustomLayer::from_json);
        // Use the register as a parameter in the load method.
        let load_nn = NN::load("custom_layer.h5", Some(register)).unwrap();
        assert!(!load_nn.is_empty());
        assert!(load_nn.extract_layers::<CustomLayer>().is_ok());
    }
}
```

## 📖 Add the library to your project

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

## Examples

There is a multitude of examples if you want to learn how to use the library, just run these commands.

```terminal
cargo run --example xor
cargo run --example xor_load_nn
cargo run --example mnist
cargo run --example mnist_load_nn
cargo run --example custom_layer
```

## 📑 Libraries used

- [rand](https://docs.rs/rand/latest/rand/) - For Random stuffs.
- [ndarray](https://docs.rs/ndarray/latest/ndarray/) - For manage N-Dimensional Arrays.
- [ndarray-rand](https://docs.rs/ndarray-rand/0.15.0/ndarray_rand/) - For manage Random N-Dimensional Arrays.
- [serde](https://docs.rs/serde/latest/serde/) - For serialization.
- [serde_json](https://docs.rs/serde_json/latest/serde_json/) - For JSON serialization.
- [hdf5](https://docs.rs/hdf5/latest/hdf5/) - For model storage.

## TODOs 🏁

- [ ] Add Conv2D (try Conv3D) layer
- [ ] Add optimizers
<!-- - [ ] Create custom Cost and Activation functions -->

## 🔑 License

[MIT](https://opensource.org/license/mit/) - Created by [**Paco Algar**](https://github.com/Pacatro).
