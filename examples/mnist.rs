use std::error::Error;

use polars::prelude::*;
use ndarray::{Array1, Array2};

use rs_nn::{activation::Activation, nn::NN, cost::Cost};

fn main() -> Result<(), Box<dyn Error>> {
    // Leer el archivo CSV
    let mut df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("examples/data/train.csv".into()))?
        .finish()?;

    let labels: Array1<f64> = df.drop_in_place("label")?
        .i64()?
        .into_iter()
        .map(|opt| opt.unwrap() as f64)
        .collect::<Vec<f64>>()
        .into();

    let data: Array2<f64> = df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;

    let input: Array1<f64> = data.row(0).to_owned();

    let mut nn = NN::new(
        &[df.get_columns().len(), 16, 16, 10],
        &[Activation::RELU, Activation::RELU, Activation::SIGMOID],
        0.5
    );

    let old_prediction = nn.predict(&input);
    let old_cost = nn.cost(&labels, &old_prediction, Cost::MSE);

    for (l, layer) in nn.layers().iter().enumerate() {
        println!("Layer {} weights:\n{}\n", l, layer.weights())
    }

    nn.train(1, &data, &labels, Cost::MSE);

    let new_prediction = nn.predict(&input);
    let new_cost = nn.cost(&labels, &new_prediction, Cost::MSE);
    
    for (l, layer) in nn.layers().iter().enumerate() {
        println!("Layer {} weights:\n{}\n", l, layer.weights())
    }

    // println!("Old prediction: {old_prediction}");
    // println!("New prediction {new_prediction}");
    println!("Old cost: {old_cost}");
    println!("New cost {new_cost}");

    if new_cost < old_cost {
        println!("FURULA")
    } else {
        println!("NO FURULA")
    }

    Ok(())
}