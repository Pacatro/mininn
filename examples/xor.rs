use std::{error::Error, io};

use ndarray::{array, Array1, Array2};
use rs_nn::{activation::Activation, cost::Cost, nn::NN};

fn transform_labels(data: &Array2<f64>, labels: &Array1<f64>) -> Array2<f64> {
    let mut inputs_labels = Array2::<f64>::zeros((data.nrows(), 2));
    
    for (i, &label) in labels.iter().enumerate() {
        inputs_labels[(i, label as usize)] = 1.0;
    }

    inputs_labels
}

fn get_predict(predictions: Array1<f64>) -> u8 {
    let mut num = 0;
    
    for (p, prediction) in predictions.iter().enumerate() {
        if *prediction > 0.5 {
            num = p as u8;
        }
    }

    num
}

fn main() -> Result<(), Box<dyn Error>>{
    let data = array![
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
    ];

    let labels = array![0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0.];

    let inputs_labels = transform_labels(&data, &labels);

    let mut nn = NN::new(&[2, 4, 2], &[Activation::SIGMOID, Activation::SIGMOID]);
    
    println!("Training...\n");
    nn.train(100, &data, &inputs_labels, Cost::MSE, 0.1);

    loop {
        let mut user_input1 = String::new();
        let mut user_input2 = String::new();
        
        println!("Inserte primera entrada (0 o 1): ");
        io::stdin().read_line(&mut user_input1)?;
        println!("Inserte segunda entrada (0 o 1): ");
        io::stdin().read_line(&mut user_input2)?;

        let user_input1: u8 = match user_input1.trim().parse() {
            Ok(num) if num == 0 || num == 1 => num,
            _ => {
                println!("Input incorrecto. Por favor, inserte 0 o 1.");
                continue;
            }
        };

        let user_input2: u8 = match user_input2.trim().parse() {
            Ok(num) if num == 0 || num == 1 => num,
            _ => {
                println!("Input incorrecto. Por favor, inserte 0 o 1.");
                continue;
            }
        };

        let input = array![user_input1 as f64, user_input2 as f64];
        let predictions = nn.predict(&input);
        let num = get_predict(predictions);

        println!("\nEl resultado es: {num}\n");
    }
}
