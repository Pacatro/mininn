use mininn::prelude::*;
use ndarray::{array, Array1};

fn main() -> NNResult<()> {
    let args = std::env::args().collect::<Vec<String>>();

    assert_eq!(
        args.len(),
        2,
        "Usage: cargo run --example xor <path_to_model>"
    );

    let path = args[1].clone();

    let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],];

    let labels = array![[0.0], [1.0], [1.0], [0.0],];

    let mut nn = NN::load(path)?;

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

    println!("\n{}\n", metrics.confusion_matrix());

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\nLoss: {}",
        metrics.accuracy(),
        metrics.recall(),
        metrics.precision(),
        metrics.f1_score(),
        nn.loss()
    );

    Ok(())
}
