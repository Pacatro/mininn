use mnist::*;
use ndarray::Array2;
use mininn::prelude::*;

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
        .training_set_length(1000) // Max 50_000
        .validation_set_length(500) // Max 10_000
        .test_set_length(1000)
        .finalize();

    let train_data = Array2::from_shape_vec((1000, 28*28), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let train_labels = Array2::from_shape_vec((1000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_data = Array2::from_shape_vec((1000, 28*28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.);

    let test_labels = Array2::from_shape_vec((1000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    (train_data, train_labels, test_data, test_labels)
}

fn main() {
    let (train_data, train_labels, _, _) = load_mnist();
    
    // Convertir etiquetas a formato one-hot
    let train_labels_one_hot = one_hot_encode(&train_labels, 10);

    // Crear la red neuronal
    let mut nn = NN::new()
        .add(Dense::new(28*28, 40, Activation::TANH))
        .add(Dense::new(40, 10, Activation::TANH));

    // Entrenar con CrossEntropy (ideal para clasificación)
    nn.train(Cost::MSE, &train_data, &train_labels_one_hot, 100, 0.1, true)
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        });

    nn.save("mnist_model.toml").unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    })
}
