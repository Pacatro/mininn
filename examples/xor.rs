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
    let loss = nn.train(Cost::MSE, &train_data, &labels, 1000, 0.1, true)?;

    println!("Predictions:\n");

    let predictions: Array1<f64> = train_data
        .rows()
        .into_iter()
        .map(|input| {
            let pred = nn.predict(&input.to_owned()).unwrap();
            let out = if pred[0] < 0.5 { 0.0 } else { 1.0 };
            println!("{} --> {}", input, out);
            out
        })
        .collect();

    // Calc metrics using MetricsCalculator
    let metrics = MetricsCalculator::new(&labels, &predictions);

    println!("\nConfusion matrix:\n{}\n", metrics.confusion_matrix()?);

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\nLoss: {}",
        metrics.accuracy()?, metrics.recall()?, metrics.precision()?,
        metrics.f1_score()?, loss
    );

    // Save the model into a HDF5 file
    if predictions.iter().zip(labels).all(|(pred, label)| *pred == label) {
        nn.save("load_models/xor.h5")?;
        println!("Model saved successfully!");
    }

    Ok(())
}
