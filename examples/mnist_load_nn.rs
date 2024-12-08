use mininn::prelude::*;
use mnist::*; // Dataset
use ndarray::{Array1, Array2};

const MAX_TRAIN_LENGHT: u32 = 1000;
const MAX_TEST_LENGHT: u32 = 500;

fn load_mnist() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(MAX_TRAIN_LENGHT) // Max 50_000
        .validation_set_length(MAX_TEST_LENGHT) // Max 10_000
        .test_set_length(MAX_TEST_LENGHT) // Max 10_000
        .finalize();

    let train_data = Array2::from_shape_vec((MAX_TRAIN_LENGHT as usize, 28 * 28), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let train_labels = Array2::from_shape_vec((MAX_TRAIN_LENGHT as usize, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_data = Array2::from_shape_vec((MAX_TEST_LENGHT as usize, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.);

    let test_labels = Array2::from_shape_vec((MAX_TEST_LENGHT as usize, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    (train_data, train_labels, test_data, test_labels)
}

fn main() -> NNResult<()> {
    let args = std::env::args().collect::<Vec<String>>();

    assert_eq!(
        args.len(),
        2,
        "Usage: cargo run --example mnist_load_nn <path_to_model>"
    );

    let path = args[1].clone();

    let (_, _, test_data, test_labels) = load_mnist();
    let mut nn = NN::load(path).unwrap();

    let predictions = test_data
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let pred = nn.predict(row.view()).unwrap();

            let (pred_idx, _) = pred
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .expect("Can't get max value");

            println!(
                "Prediction: {} | Label: {}",
                pred_idx,
                test_labels.row(i)[0]
            );

            pred_idx as f64
        })
        .collect::<Array1<f64>>();

    let metrics = MetricsCalculator::new(test_labels.into_dyn().view(), predictions.view());

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
