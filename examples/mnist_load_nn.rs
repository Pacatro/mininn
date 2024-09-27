use mnist::*; // Dataset
use ndarray::{Array1, Array2};
use mininn::prelude::*;

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

    let train_data = Array2::from_shape_vec((MAX_TRAIN_LENGHT as usize, 28*28), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let train_labels = Array2::from_shape_vec((MAX_TRAIN_LENGHT as usize, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_data = Array2::from_shape_vec((MAX_TEST_LENGHT as usize, 28*28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.);

    let test_labels = Array2::from_shape_vec((MAX_TEST_LENGHT as usize, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    (train_data, train_labels, test_data, test_labels)
}

fn main() {
    let (_, _, test_data, test_labels) = load_mnist();
    
    let mut nn = NN::load("load_models/mnist_no_conv.h5").unwrap();

    let mut predictions = Vec::new();

    for i in 0..test_data.nrows() {
        let pred = nn.predict(&test_data.row(i).to_owned());
    
        let (pred_idx, _) = pred
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .expect("Can't get max value");
        
        println!("Prediction: {} | Label: {}", pred_idx, test_labels.row(i)[0]);

        predictions.push(pred_idx as f64);
    }

    let metrics = MetricsCalculator::new(&test_labels, &Array1::from_vec(predictions));

    println!("\n{}\n", metrics.confusion_matrix());

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n",
        metrics.accuracy(), metrics.recall(), metrics.precision(),
        metrics.f1_score()
    );
}