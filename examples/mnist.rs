use std::{time::Instant, error::Error};

use ndarray::{Array1, Array2};
use polars::prelude::*;

use rs_nn::{activation::Activation, nn::NN, cost::Cost};

fn transform_labels(data: &Array2<f64>, labels: &Array1<f64>) -> Array2<f64> {
    let mut inputs_labels = Array2::<f64>::zeros((data.nrows(), 2));
    
    for (i, &label) in labels.iter().enumerate() {
        inputs_labels[(i, label as usize)] = 1.0;
    }

    inputs_labels
}

fn main() -> Result<(), Box<dyn Error>> {
    // Leer el archivo CSV
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("examples/data/train.csv".into()))?
        .finish()?;

    // let df_test = CsvReadOptions::default()
    //     .with_has_header(true)
    //     .try_into_reader_with_file_path(Some("examples/data/test.csv".into()))?
    //     .finish()?;

    let data = df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    let data_labels = data.column(0).mapv(|l| l as f64);

    let data = df.drop("label")?.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    let inputs_labels = transform_labels(&data, &data_labels);

    // let test = df_test.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    // let test_input = test.row(10).to_owned();

    let mut nn = NN::new(
        &[data.ncols(), 16, 16, 10],
        &[Activation::RELU, Activation::RELU, Activation::SIGMOID],
    );

    let input = data.row(0).to_owned();
    let label = inputs_labels.row(0).to_owned();

    let old_predictions = nn.predict(&input);
    
    let first_cost = nn.cost(&label, &old_predictions, Cost::MSE);
    println!("First cost: {first_cost}");

    println!("\nTraining...\n");
    
    let now = Instant::now();

    nn.train(100, &data, &inputs_labels, Cost::MSE, 0.5);

    let time = now.elapsed().as_secs_f32();

    println!("Training time: {time} seconds\n");

    let predictions = nn.predict(&input);
    for (p, prediction) in predictions.iter().enumerate() {
        println!("{p}: {}%", *prediction as f32 * 100f32);
    }

    let second_cost = nn.cost(&label, &predictions, Cost::MSE);
    println!("\nSecond cost: {second_cost}");

    let (num, _) = predictions
        .iter()
        .cloned() // para trabajar con valores en lugar de referencias
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("The number is: {num}");

    Ok(())
}