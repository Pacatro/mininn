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
    
    let mut nn = NN::load("load_models/xor.h5").unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    let mut predictions = Vec::new();

    for input in train_data.rows() {
        let pred = nn.predict(&input.to_owned());
        let out = if pred[0] < 0.5 { 0 } else { 1 };
        predictions.push(out as f64);
        println!("{} --> {}", input, out)
    }

    let metrics = MetricsCalculator::new(&labels, &Array1::from_vec(predictions));

    println!("\n{}\n", metrics.confusion_matrix());

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n",
        metrics.accuracy(), metrics.recall(), metrics.precision(),
        metrics.f1_score()
    );
}
