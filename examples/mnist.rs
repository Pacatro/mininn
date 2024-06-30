use polars::prelude::*;
use ndarray::{Array1, Array2};

use rs_nn::{activation::Activation, nn::NN, cost::Cost};

fn main() {
    // Leer el archivo CSV
    let mut df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("examples/data/train.csv".into()))
        .unwrap()
        .finish()
        .unwrap();

    let labels: Array1<f64> = df.drop_in_place("label")
        .unwrap()
        .i64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap() as f64)
        .collect::<Vec<f64>>()
        .into();

    let data = df.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    let input = data.row(0).to_owned();

    let mut nn = NN::new(&[df.get_columns().len(), 16, 16, 10], &[Activation::SIGMOID; 3], 0.05);
    let result = nn.predict(&input);
    let old_cost = nn.cost(&labels, &result, Cost::MSE);

    nn.train(1, &data, &labels, Cost::MSE);

    // let new_cost = nn.cost(&labels, &result, Cost::MSE);

    println!("Old cost {old_cost}");
    // println!("New cost {new_cost}");

}