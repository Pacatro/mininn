use mininn::constants::DEFAULT_DROPOUT_P;
use mininn::prelude::*;
use mnist::*; // Dataset
use ndarray::Array2;

const MAX_TRAIN_LENGHT: usize = 1000;
const MAX_TEST_LENGHT: usize = 500;

fn load_mnist() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
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

    let train_data = Array2::from_shape_vec((MAX_TRAIN_LENGHT, 28 * 28), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.0);

    let train_labels = Array2::from_shape_vec((MAX_TRAIN_LENGHT, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let test_data = Array2::from_shape_vec((MAX_TEST_LENGHT, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.);

    let test_labels = Array2::from_shape_vec((MAX_TEST_LENGHT, 10), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    (train_data, train_labels, test_data, test_labels)
}

fn main() -> NNResult<()> {
    let args = std::env::args().collect::<Vec<String>>();

    let path = args.get(1);

    let (train_data, train_labels, _, _) = load_mnist();

    let mut nn = NN::new()
        .add(Dropout::new(DEFAULT_DROPOUT_P))?
        .add(Dense::new(28 * 28, 40).apply(Act::Tanh))?
        .add(Dense::new(40, 10).apply(Act::Tanh))?;

    let train_config = TrainConfig::new()
        .with_cost(Cost::MSE)
        .with_epochs(10_000)
        .with_learning_rate(0.01)
        .with_batch_size(32)
        .with_optimizer(Optimizer::GD)
        .with_early_stopping(15, 0.01)
        .with_verbose(true);

    nn.train(train_data.view(), train_labels.view(), train_config)?;

    if let Some(p) = path {
        match nn.save(p) {
            Ok(_) => println!("Model saved successfully!"),
            Err(e) => println!("Error saving model: {}", e),
        }
    }

    Ok(())
}
