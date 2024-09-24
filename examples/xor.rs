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
    nn.save("load_models/xor.toml").unwrap();
}
