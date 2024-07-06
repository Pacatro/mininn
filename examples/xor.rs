use ndarray::Array2;
use rs_nn::{nn::NN, activation::Activation, cost::Cost};

fn main() {
    // Crear la red neuronal
    let mut nn = NN::new(&[2, 4, 16, 4, 1], &[Activation::SIGMOID; 4]);

    // Preparar los datos de entrenamiento (problema XOR)
    let training_data = Array2::from_shape_vec((4, 2), vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ]).unwrap();

    let labels = Array2::from_shape_vec((4, 1), vec![
        0.0,
        1.0,
        1.0,
        0.0,
    ]).unwrap();

    // Entrenar la red
    let epochs = 10000;
    let learning_rate = 2.0;
    println!("Entrenando la red neuronal...");
    println!("Épocas: {}, Tasa de aprendizaje: {}", epochs, learning_rate);

    for epoch in 0..=epochs {
        nn.train(1, &training_data, &labels, Cost::MSE, learning_rate);
        
        if epoch % 1000 == 0 {
            let mut total_cost = 0.0;
            for (input, label) in training_data.rows().into_iter().zip(labels.rows()) {
                let prediction = nn.predict(&input.to_owned());
                total_cost += nn.cost(&label.to_owned(), &prediction, Cost::MSE);
            }
            let avg_cost = total_cost / training_data.nrows() as f64;
            println!("Época {}: Costo promedio = {:.6}", epoch, avg_cost);
        }
    }

    // Probar la red con los datos de entrenamiento
    println!("\nResultados finales:");
    println!("|   Entrada    | Esperado | Predicción |");
    println!("|--------------|----------|------------|");

    for (input, label) in training_data.rows().into_iter().zip(labels.rows()) {
        let prediction = nn.predict(&input.to_owned());
        println!("| ({:.1}, {:.1})   | {:.1}      | {:.6}   |", 
                 input[0], input[1], label[0], prediction[0]);
    }

    // Calcular la precisión
    let mut correct = 0;
    for (input, label) in training_data.rows().into_iter().zip(labels.rows()) {
        let prediction = nn.predict(&input.to_owned());
        if (prediction[0].round() - label[0]).abs() < 1e-5 {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / training_data.nrows() as f64 * 100.0;
    println!("\nPrecisión: {:.2}%", accuracy);
}