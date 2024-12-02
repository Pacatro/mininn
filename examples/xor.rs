use ndarray::{array, Array1};

use mininn::prelude::*;

fn main() -> NNResult<()> {
    let args = std::env::args().collect::<Vec<String>>();

    let path = args.get(1);

    let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],];
    let labels = array![[0.0], [1.0], [1.0], [0.0],];

    // Create the neural network
    let mut nn = NN::new()
        .add(Dense::new(2, 3).with(Act::Tanh))?
        .add(Dense::new(3, 1).with(Act::Tanh))?;

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

    if let Some(p) = path {
        match nn.save(p) {
            Ok(_) => println!("Model saved successfully!"),
            Err(e) => println!("Error saving model: {}", e),
        }
    }

    Ok(())
}
