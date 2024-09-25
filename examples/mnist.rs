use mnist::*; // Dataset
use ndarray::Array2;
use mininn::prelude::*;

const MAX_TRAIN_LENGHT: u32 = 1000;
const MAX_TEST_LENGHT: u32 = 500;

fn one_hot_encode(labels: &Array2<f64>, num_classes: usize) -> Array2<f64> {
    let mut one_hot = Array2::zeros((labels.len(), num_classes));
    for (i, &label) in labels.iter().enumerate() {
        one_hot[(i, label as usize)] = 1.0;
    }
    one_hot
}

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
    let (train_data, train_labels, test_data, test_labels) = load_mnist();
    
    let train_labels_one_hot = one_hot_encode(&train_labels, 10);

    let mut nn = NN::new()
        .add(Dense::new(28*28, 40, Some(ActivationFunc::TANH)))
        .add(Dense::new(40, 10, Some(ActivationFunc::TANH)));

    nn.train(Cost::MSE, &train_data, &train_labels_one_hot, 100, 0.1, true)
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        });

    for i in 0..30 {
        let pred = nn.predict(&test_data.row(i).to_owned());
        let (pred_idx, _) = pred
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .expect("Can't get max value");
        
        println!("Prediction: {} | Label: {}", pred_idx, test_labels.row(i));
    }
}
