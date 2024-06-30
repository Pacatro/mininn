use polars::prelude::*;
use ndarray::Array1;

use rs_nn::{activation::Activation, nn::NN, cost::Cost};

fn main() {
    // Leer el archivo CSV
    let mut df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("examples/data/train.csv".into()))
        .unwrap()
        .finish()
        .unwrap();

    // Convertir la Series a Vec<f64> (o Vec<i32> o el tipo adecuado según tus datos)
    let labels_vec: Vec<f64> = df
        .drop_in_place("label")
        .unwrap()
        .i64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap() as f64)
        .collect();

    let input: Vec<f64> = df
        .get_row(0)
        .unwrap()
        .0
        .iter()
        .map(|elem| match elem {
            AnyValue::Int64(val) => *val as f64,
            _ => panic!("Expected Float64"),
        })
        .collect();

    let nn = NN::new(&[df.get_columns().len(), 16, 16, 10], &[Activation::SIGMOID; 3], 0.05);
    let result = nn.predict(&Array1::from(input));
    let cost = nn.cost(&Array1::from(labels_vec), &result, Cost::MSE);

    println!("Cost: {cost}")
}