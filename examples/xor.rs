use ndarray::array;
use rs_nn::{nn::NN, activation::Activation, cost::Cost};

fn main() {
    // Crear la red neuronal
    let mut nn = NN::new(&[2, 4, 1], &[Activation::TANH; 2]);

    // Preparar los datos de entrenamiento (problema XOR)
    let training_data = array![
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

    nn.train(1000, &training_data, &labels, Cost::MSE, 0.1);

    for input in training_data.rows() {
        let predictions = nn.predict(&input.to_owned());
        println!("Prediction: {predictions}")
    }
   
}