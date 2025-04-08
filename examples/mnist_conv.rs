use mininn::prelude::*;
use mnist::*;
use ndarray::{Array1, Array2, Array5};

const MAX_TRAIN_LENGHT: usize = 2000;
const MAX_TEST_LENGHT: usize = 400;

fn load_mnist() -> (
    Array5<f32>,
    Array2<f32>,
    Array1<f32>,
    Array5<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(MAX_TRAIN_LENGHT as u32)
        .validation_set_length(MAX_TEST_LENGHT as u32)
        .test_set_length(MAX_TEST_LENGHT as u32)
        .finalize();

    let Mnist {
        trn_lbl: trn_lbl_no_one_hot,
        tst_lbl: tst_lbl_no_one_hot,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(MAX_TRAIN_LENGHT as u32)
        .validation_set_length(MAX_TEST_LENGHT as u32)
        .test_set_length(MAX_TEST_LENGHT as u32)
        .finalize();

    let train_data = Array5::from_shape_vec((MAX_TRAIN_LENGHT, 1, 1, 28, 28), trn_img)
        .expect("Error converting images to Array5 struct")
        .map(|x| *x as f32 / 256.0);

    let train_labels = Array2::from_shape_vec((MAX_TRAIN_LENGHT, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let train_labels_no_one_hot = Array1::from_shape_vec(MAX_TRAIN_LENGHT, trn_lbl_no_one_hot)
        .expect("Error converting training labels to Array1 struct")
        .map(|x| *x as f32);

    let test_data = Array5::from_shape_vec((MAX_TEST_LENGHT, 1, 1, 28, 28), tst_img)
        .expect("Error converting images to Array5 struct")
        .map(|x| *x as f32 / 256.0);

    let test_labels = Array2::from_shape_vec((MAX_TEST_LENGHT, 10), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    let test_labels_no_one_hot = Array2::from_shape_vec((MAX_TEST_LENGHT, 1), tst_lbl_no_one_hot)
        .expect("Error converting testing labels to Array1 struct")
        .map(|x| *x as f32);

    (
        train_data,
        train_labels,
        train_labels_no_one_hot,
        test_data,
        test_labels,
        test_labels_no_one_hot,
    )
}

fn main() -> NNResult<()> {
    let args = std::env::args().collect::<Vec<String>>();
    let path = args.get(1);

    let (
        train_data,
        train_labels,
        _train_labels_no_one_hot,
        test_data,
        _test_labels,
        test_labels_no_one_hot,
    ) = load_mnist();

    let mut nn = NN::new()
        .add_layer(Conv::new(1, 32, (5, 5), 1, 2).apply(Act::ReLU))
        .add_layer(Conv::new(32, 64, (5, 5), 2, 2).apply(Act::ReLU))
        .add_layer(Flatten::new())
        .add_layer(Dense::new(64 * 14 * 14, 10).apply(Act::Tanh));

    let train_config = TrainConfig::new()
        .with_cost(Cost::CCE)
        .with_epochs(1000)
        .with_learning_rate(0.001)
        .with_batch_size(64)
        .with_optimizer(Optimizer::default_momentum())
        .with_early_stopping(5, 0.0001)
        .with_verbose();

    nn.train(train_data.view(), train_labels.view(), train_config)?;

    let predictions = test_data
        .outer_iter()
        .map(|row| {
            let pred = nn.predict(row.view()).unwrap();

            let (pred_idx, _) = pred
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .expect("Can't get max value");

            pred_idx as f32
        })
        .collect::<Array1<f32>>();

    let metrics = MetricsCalculator::new(test_labels_no_one_hot.view(), predictions.view());

    println!("\n{}\n", metrics.confusion_matrix());

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\nLoss: {}",
        metrics.accuracy(),
        metrics.recall(),
        metrics.precision(),
        metrics.f1_score(),
        nn.loss()
    );

    if let Some(p) = path {
        match nn.save(p) {
            Ok(_) => println!("Model saved successfully!"),
            Err(e) => println!("Error saving model: {}", e),
        }
    }

    Ok(())
}
