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
        .add(Dense::new(2, 3, Activation::TANH))
        .add(Dense::new(3, 1, Activation::TANH));

    nn.train(Cost::MSE, &train_data, &labels, 500, 0.1, true).unwrap();

    for input in train_data.rows() {
        let pred = nn.predict(&input.to_owned());
        let out = if pred[0] < 0.5 { 0 } else { 1 };
        println!("{} --> {}", input, out)
    }

    // Save the model into a .toml file
    nn.save("model.toml").unwrap();
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

You can also calc classification metrics using `ClassMetrics`:

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
```

## TODOs 🏁

- [x] Try to solve XOR problem
- [x] Try to solve MNIST problem
- [x] Metrics for NN
- [ ] Add Activation layer
- [ ] Add Conv2D (try Conv3D) layer
<!-- CAN BE PUBLISH -->
- [ ] Add optimizers
- [ ] Allow other files format for save the model

## 🔑 License

[MIT](https://opensource.org/license/mit/) - Created by [**Paco Algar**](https://github.com/Pacatro).
