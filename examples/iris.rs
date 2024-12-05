use mininn::prelude::*;
use ndarray::*;
use ndarray_rand::rand;

fn one_hot_encode(labels: &Array2<f64>) -> Array2<f64> {
    let num_classes = 3;
    let num_samples = labels.nrows();
    let mut one_hot = Array2::zeros((num_samples, num_classes));

    for (i, label) in labels.column(0).iter().enumerate() {
        let class = label.round() as usize;
        one_hot[[i, class]] = 1.0;
    }

    one_hot
}

fn load_data() -> NNResult<(ArrayD<f64>, ArrayD<f64>, ArrayD<f64>, ArrayD<f64>)> {
    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rand::thread_rng())
        .split_with_ratio(0.5);

    let train_data: Vec<Vec<f64>> = train
        .records()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let train_labels = train.targets().mapv(|x| x as f64).to_vec();

    let train_data = Array2::from_shape_vec(
        (train_data.len(), train_data[0].len()),
        train_data.into_iter().flatten().collect(),
    )?;

    let train_labels = Array2::from_shape_vec((train_labels.len(), 1), train_labels)?;

    let train_labels = one_hot_encode(&train_labels);

    let test_data: Vec<Vec<f64>> = test
        .records()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let test_labels = test.targets().mapv(|x| x as f64).to_vec();

    let test_data = Array2::from_shape_vec(
        (test_data.len(), test_data[0].len()),
        test_data.into_iter().flatten().collect(),
    )?;

    let test_labels = Array2::from_shape_vec((test_labels.len(), 1), test_labels)?;

    Ok((
        train_data.into_dyn(),
        train_labels.into_dyn(),
        test_data.into_dyn(),
        test_labels.into_dyn(),
    ))
}

fn main() -> NNResult<()> {
    let (train_data, train_labels, test_data, test_labels) = load_data()?;

    let mut nn = NN::new()
        .add(Dense::new(4, 16).with(Act::ReLU))?
        .add(Dense::new(16, 8).with(Act::ReLU))?
        .add(Dense::new(8, 3).with(Act::Softmax))?;

    let loss = nn.train(
        train_data.view(),
        train_labels.view(),
        Cost::CCE,
        500,
        0.001,
        32,
        Optimizer::GD,
        true,
    )?;

    let predictions = test_data
        .rows()
        .into_iter()
        .enumerate()
        .map(|(_i, row)| {
            let pred = nn.predict(&row.to_owned()).unwrap();

            let (pred_idx, _) = pred
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .expect("Can't get max value");

            // println!("Prediction: {} | Label: {}", pred_idx, test_labels.row(i)[0]);

            pred_idx as f64
        })
        .collect::<Array1<f64>>();

    let metrics = MetricsCalculator::new(&test_labels, &predictions);

    println!("\n{}\n", metrics.confusion_matrix());

    println!(
        "Accuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\nLoss: {}",
        metrics.accuracy(),
        metrics.recall(),
        metrics.precision(),
        metrics.f1_score(),
        loss
    );

    Ok(())
}
