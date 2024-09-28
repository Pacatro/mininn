use ndarray::{array, Array1};

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

    // Train the neural network
    nn.train(Cost::MSE, &train_data, &labels, 1000, 0.1, true).unwrap();

    let mut predictions = Vec::new();

    for input in train_data.rows() {
        // Use predict to see the resutl of the network
        let pred = nn.predict(&input.to_owned());
        let out = if pred[0] < 0.5 { 0 } else { 1 };
        predictions.push(out as f64);
        println!("{} --> {}", input, out)
    }

    // Calc metrics using MetricsCalculator
    let metrics = MetricsCalculator::new(&labels, &Array1::from_vec(predictions));

    println!("\n{}\n", metrics.confusion_matrix());

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n",
        metrics.accuracy(), metrics.recall(), metrics.precision(),
        metrics.f1_score()
    );

    // Save the model into a HDF5 file
    nn.save("load_models/xor.h5").unwrap();
}
