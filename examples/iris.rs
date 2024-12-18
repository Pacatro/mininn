use mininn::constants::DEFAULT_MOMENTUM;
use mininn::prelude::*;
use ndarray::*;
use ndarray_rand::rand;

fn one_hot_encode(labels: &Array2<f32>) -> Array2<f32> {
    let num_classes = 3;
    let num_samples = labels.nrows();
    let mut one_hot = Array2::zeros((num_samples, num_classes));

    for (i, label) in labels.column(0).iter().enumerate() {
        let class = label.round() as usize;
        one_hot[[i, class]] = 1.0;
    }

    one_hot
}

fn load_data() -> NNResult<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)> {
    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rand::thread_rng())
        .split_with_ratio(0.5);

    let train_data: Vec<Vec<f32>> = train
        .records()
        .mapv(|x| x as f32)
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let train_labels = train.targets().mapv(|x| x as f32).to_vec();

    let train_data = Array2::from_shape_vec(
        (train_data.len(), train_data[0].len()),
        train_data.into_iter().flatten().collect(),
    )?;

    let train_labels = Array2::from_shape_vec((train_labels.len(), 1), train_labels)?;

    let train_labels = one_hot_encode(&train_labels);

    let test_data: Vec<Vec<f32>> = test
        .records()
        .mapv(|x| x as f32)
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let test_labels = test.targets().mapv(|x| x as f32).to_vec();

    let test_data = Array2::from_shape_vec(
        (test_data.len(), test_data[0].len()),
        test_data.into_iter().flatten().collect(),
    )?;

    let test_labels = Array2::from_shape_vec((test_labels.len(), 1), test_labels)?;

    Ok((train_data, train_labels, test_data, test_labels))
}

fn main() -> NNResult<()> {
    let (train_data, train_labels, test_data, test_labels) = load_data()?;

    let mut nn = NN::new()
        .add(Dense::new(4, 16).apply(Act::ReLU))
        .add(Dense::new(16, 8).apply(Act::ReLU))
        .add(Dense::new(8, 3).apply(Act::Softmax));

    let train_config = TrainConfig::new()
        .with_cost(Cost::CCE)
        .with_epochs(10000)
        .with_learning_rate(0.001)
        .with_batch_size(32)
        .with_optimizer(Optimizer::Momentum(DEFAULT_MOMENTUM))
        .with_early_stopping(10, 0.01)
        .with_verbose(true);

    let loss = nn.train(train_data.view(), train_labels.view(), train_config)?;

    let predictions = test_data
        .rows()
        .into_iter()
        .enumerate()
        .map(|(_i, row)| {
            let pred = nn.predict(row.view()).unwrap();

            let (pred_idx, _) = pred
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .expect("Can't get max value");

            // println!("Prediction: {} | Label: {}", pred_idx, test_labels.row(i)[0]);

            pred_idx as f32
        })
        .collect::<Array1<f32>>();

    let metrics = MetricsCalculator::new(test_labels.view(), predictions.view());

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
