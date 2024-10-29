use mnist::*; // Dataset
use ndarray::Array2;
use mininn::prelude::*;

const MAX_TRAIN_LENGHT: usize = 1000;
const MAX_TEST_LENGHT: usize = 500;
const EPOCHS: u32 = 100;

fn load_mnist() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(MAX_TRAIN_LENGHT as u32) // Max 50_000
        .validation_set_length(MAX_TEST_LENGHT as u32) // Max 10_000
        .test_set_length(MAX_TEST_LENGHT as u32) // Max 10_000
        .label_format_one_hot()
        .finalize();

    let train_data = Array2::from_shape_vec((MAX_TRAIN_LENGHT, 28*28), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let train_labels = Array2::from_shape_vec((MAX_TRAIN_LENGHT, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_data = Array2::from_shape_vec((MAX_TEST_LENGHT, 28*28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.);

    let test_labels = Array2::from_shape_vec((MAX_TEST_LENGHT, 10), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    (train_data, train_labels, test_data, test_labels)
}

fn main() -> NNResult<()> {
    let (train_data, train_labels, _, _) = load_mnist();
    
    let mut nn = NN::new()
        .add(Dense::new(28*28, 40, Some(ActivationFunc::TANH)))?
        .add(Dense::new(40, 10, Some(ActivationFunc::TANH)))?;

    nn.train(Cost::MSE, &train_data, &train_labels, EPOCHS, 0.1, 32, true)?;

    nn.save("load_models/mnist_no_conv.h5")?;

    Ok(())
}
