use mininn::prelude::*;
use mnist::*;
use ndarray::{Array2, Array4};

const MAX_TRAIN_LENGHT: usize = 7000;
const MAX_TEST_LENGHT: usize = 1000;

fn load_mnist() -> (Array4<f32>, Array2<f32>, Array4<f32>, Array2<f32>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(MAX_TRAIN_LENGHT as u32) // Máximo 50_000
        .validation_set_length(MAX_TEST_LENGHT as u32) // Máximo 10_000
        .test_set_length(MAX_TEST_LENGHT as u32) // Máximo 10_000
        .label_format_one_hot()
        .finalize();

    let train_data = Array4::from_shape_vec((MAX_TRAIN_LENGHT, 1, 28, 28), trn_img)
        .expect("Error converting images to Array4 struct")
        .map(|x| *x as f32 / 256.0);

    let train_labels = Array2::from_shape_vec((MAX_TRAIN_LENGHT, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let test_data = Array4::from_shape_vec((MAX_TEST_LENGHT, 1, 28, 28), tst_img)
        .expect("Error converting images to Array4 struct")
        .map(|x| *x as f32 / 256.0);

    let test_labels = Array2::from_shape_vec((MAX_TEST_LENGHT, 10), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    (train_data, train_labels, test_data, test_labels)
}

fn main() -> NNResult<()> {
    let args = std::env::args().collect::<Vec<String>>();
    let path = args.get(1);

    let (train_data, train_labels, _, _) = load_mnist();

    // Convertir las imágenes de forma vectorial (n, 784) a forma 4D (n, 1, 28, 28)
    // let train_data = train_data
    //     .to_shape((MAX_TRAIN_LENGHT, 1, 28, 28))
    //     .expect("Error reshaping training data");

    // Se define la red usando capas convolucionales.
    // La primera capa convolucional: entrada 1 canal, 32 kernels de tamaño 5x5, stride 1 y padding 2.
    // La segunda capa convolucional: entrada 32 canales, 64 kernels de tamaño 5x5, stride 2 y padding 2.
    // Luego se aplana y se conecta a una capa densa de salida de 10 neuronas.
    let mut nn = NN::new()
        .add_layer(Conv::new(1, 32, (5, 5), 1, 2).apply(Act::ReLU))
        .add_layer(Conv::new(32, 64, (5, 5), 2, 2).apply(Act::ReLU))
        .add_layer(Flatten::new())
        // La salida de la segunda convolución tiene forma (n, 64, 14, 14) por el stride 2,
        // es decir, 64*14*14 = 12544 entradas para la capa densa.
        .add_layer(Dense::new(64 * 14 * 14, 10).apply(Act::Tanh));

    let train_config = TrainConfig::new()
        .with_cost(Cost::CCE)
        .with_epochs(100)
        .with_learning_rate(0.001)
        .with_batch_size(64)
        .with_optimizer(Optimizer::default_momentum())
        .with_early_stopping(5, 0.0001)
        .with_verbose();

    nn.train(train_data.view(), train_labels.view(), train_config)?;

    if let Some(p) = path {
        match nn.save(p) {
            Ok(_) => println!("Model saved successfully!"),
            Err(e) => println!("Error saving model: {}", e),
        }
    }

    Ok(())
}
