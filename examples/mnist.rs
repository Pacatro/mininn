use std::{time::Instant, error::Error};

use polars::prelude::*;

use rs_nn::{activation::Activation, nn::NN, cost::Cost};

fn main() -> Result<(), Box<dyn Error>> {
    // Leer el archivo CSV
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("examples/data/train.csv".into()))?
        .finish()?
        .drop("label")?;

    let df_test = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("examples/data/test.csv".into()))?
        .finish()?;

    let data = df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    let labels = data.column(0).mapv(|l| l as f64);

    let test = df_test.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    let test_input = test.row(10).to_owned();

    let mut nn = NN::new(
        &[df.get_columns().len(), 16, 16, 10],
        &[Activation::RELU, Activation::RELU, Activation::SIGMOID],
        0.5
    );

    let old_predictions = nn.predict(&test_input);
    
    let first_cost = nn.cost(&labels, &old_predictions, Cost::MSE);
    println!("First cost: {first_cost}");

    println!("\nTraining...\n");
    let now = Instant::now();
    nn.train(30, &data, &labels, Cost::MSE);

    let time = now.elapsed().as_secs_f32();

    println!("Training time: {time} seconds\n");

    let predictions = nn.predict(&test_input);
    for (p, prediction) in predictions.iter().enumerate() {
        println!("{p}: {}%", *prediction as f32 * 100f32);
    }

    let (num, _) = predictions
        .iter()
        .cloned() // para trabajar con valores en lugar de referencias
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("The number is: {num}");

    let second_cost = nn.cost(&labels, &predictions, Cost::MSE);
    println!("\nSecond cost: {second_cost}");

    Ok(())
}