# RS-NN

A minimalist deep learnig crate for rust using [ndarray](https://docs.rs/ndarray/latest/ndarray/).

> [!WARNING]
> This crate is not complete. It will be updated and published on [crates.io](https://crates.io/) in the future.

## ✏️ Usage

For this example we will resolve the XOR problem

```rust
use ndarray::array;
use rs_nn::{
    NN,
    ActivationType,
    Cost,
    layers::{Activation, Dense}
};

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

    let mut nn = NN::new()
        .add(Dense::new(2, 3))
        .add(Activation::new(ActivationType::TANH))
        .add(Dense::new(3, 1))
        .add(Activation::new(ActivationType::TANH));

    nn.train(Cost::MSE, &train_data, &labels, 1000, 0.1, true);

    for input in train_data.rows() {
        let pred = nn.predict(&input.to_owned());
        let out = if pred[(0, 0)] < 0.5 { 0 } else { 1 };
        println!("{} --> {}", input, out)
    }
}
```

### Output

```terminal
Epoch 1/300, error: 0.3071834869826118, time: 0.000316989 seg
Epoch 2/300, error: 0.16930142566310555, time: 0.000220357 seg
Epoch 3/300, error: 0.1645315513850436, time: 0.000218866 seg
...
Epoch 298/300, error: 0.00535825686735959, time: 0.000227905 seg
Epoch 299/300, error: 0.005188265891455881, time: 0.000231953 seg
Epoch 300/300, error: 0.005027296687864061, time: 0.000232859 seg
[0, 0] --> 0
[0, 1] --> 1
[1, 0] --> 1
[1, 1] --> 0
```

<!-- ## 📖 Add the library to your project

You can add the crate with `cargo add`

```terminal
cargo add rs_nn
```

Alternatively, you can manually add it to your project's Cargo.toml like this:

```toml
[dependencies]
rs_nn = "*" # Change the `*` to the current version
``` -->

## 💻 Contributing

If you want to add new features to the libray, you need to follow this steps.

Clone this repository

```terminal
git clone https://github.com/Pacatro/rs-nn.git
cd rs-nn
```

Run example

```terminal
cargo run --example xor
```

## 🔑 License

[MIT](https://opensource.org/license/mit/) - Created by [**P4k0**](https://github.com/Pacatro).
