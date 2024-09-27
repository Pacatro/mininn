# MiniNN

A minimalist deep learnig crate for rust using [ndarray](https://docs.rs/ndarray/latest/ndarray/).

> [!WARNING]
> This crate is not complete. It will be updated and published on [crates.io](https://crates.io/) in the future.

## ✏️ Usage

For this example we will resolve the classic XOR problem

```rust
use ndarray::array;

use mininn::prelude::*;

fn main() {
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
        .add(Dense::new(2, 3, Some(ActivationFunc::TANH)))
        .add(Dense::new(3, 1, Some(ActivationFunc::TANH)));

    nn.train(Cost::MSE, &train_data, &labels, 500, 0.1, true).unwrap();

    for input in train_data.rows() {
        let pred = nn.predict(&input.to_owned());
        let out = if pred[0] < 0.5 { 0 } else { 1 };
        println!("{} --> {}", input, out)
    }

    // Save the model into a HDF5 file
    nn.save("model.h5").unwrap();
}
```

### Output

```terminal
Epoch 1/300, error: 0.4616054910425124, time: 0.000347962 sec
Epoch 2/300, error: 0.3021019514321462, time: 0.000243915 sec
Epoch 3/300, error: 0.29083915749739214, time: 0.00024164 sec
...
Epoch 298/300, error: 0.0009148792200164942, time: 0.00025224 sec
Epoch 299/300, error: 0.0009105143390612294, time: 0.00026309 sec
Epoch 300/300, error: 0.0009061884741629226, time: 0.000249745 sec
[0, 0] --> 0
[0, 1] --> 1
[1, 0] --> 1
[1, 1] --> 0
```

### Metrics

You can also calculate metrics for your models using `ClassMetrics`:

```rust
let class_metrics = ClassMetrics::new(&test_labels, &predictions);

println!("\n{}\n", metrics.confusion_matrix());

println!(
    "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n",
    class_metrics.accuracy(), class_metrics.recall(), class_metrics.precision(),
    class_metrics.f1_score()
);
```

### Layers

The crate offers multiples types of layers:

| Layer    | Description                         |
|----------|-------------------------------------|
| `Dense`         | Each neuron is connected to every neuron in the previous layer. It computes the weighted sum of the inputs, adds a bias, and then applies an optional activation function.       |
| `Activation`    | Applies a specific activation function to its input                       |
<!---| `Conv2D`        | Cell 8                       |--->

### Save and load models

When you already have a trained model you can save it into a HDF5 file:

```rust
nn.save("model.h5").unwrap();
let mut nn = NN::load("model.h5").unwrap();
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

If you want to add new features to the libray, you need to follow this steps.

Clone this repository

```terminal
git clone https://github.com/Pacatro/mininn.git
cd mininn
```
``` -->

Run examples

```terminal
cargo run --example xor
cargo run --example xor_load_nn
cargo run --example mnist
cargo run --example mnist_load_nn
```

## TODOs 🏁

- [x] Try to solve XOR problem
- [x] Try to solve MNIST problem
- [x] Metrics for NN
- [x] Add Activation layer
- [x] Improve save and load system
- [ ] Add Conv2D (try Conv3D) layer
<!-- CAN BE PUBLISH -->
- [ ] Add optimizers
- [ ] Allow other files format for save the model

## 🔑 License

[MIT](https://opensource.org/license/mit/) - Created by [**Paco Algar**](https://github.com/Pacatro).
