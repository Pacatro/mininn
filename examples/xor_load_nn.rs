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
    
    let mut nn = NN::load("load_models/xor.h5", None).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

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

    println!("\n{}\n", metrics.confusion_matrix().unwrap());

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n",
        metrics.accuracy().unwrap(), metrics.recall().unwrap(), metrics.precision().unwrap(),
        metrics.f1_score().unwrap()
    );
}
