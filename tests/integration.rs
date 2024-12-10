use mininn::prelude::*;
use ndarray::array;

#[test]
fn test_new() {
    let nn = NN::new();
    assert!(nn.is_empty());
    assert_eq!(nn.nlayers(), 0);
}

#[test]
fn test_add() {
    let nn = NN::new()
        .add(Dense::new(2, 3).apply(Act::ReLU))
        .unwrap()
        .add(Dense::new(3, 1).apply(Act::Sigmoid))
        .unwrap();
    assert_eq!(nn.nlayers(), 2);
    assert!(!nn.is_empty());
}

#[test]
fn test_dense_layers() {
    let nn = NN::new()
        .add(Dense::new(2, 3).apply(Act::ReLU))
        .unwrap()
        .add(Dense::new(3, 1).apply(Act::Sigmoid))
        .unwrap();
    let dense_layers = nn.extract_layers::<Dense>().unwrap();
    assert_eq!(dense_layers.len(), 2);
    assert_eq!(dense_layers[0].ninputs(), 2);
    assert_eq!(dense_layers[0].noutputs(), 3);
    assert_eq!(dense_layers[1].ninputs(), 3);
    assert_eq!(dense_layers[1].noutputs(), 1);
}

#[test]
fn test_activation_layers() {
    let nn = NN::new()
        .add(Activation::new(Act::ReLU))
        .unwrap()
        .add(Activation::new(Act::Sigmoid))
        .unwrap();
    let activation_layers = nn.extract_layers::<Activation>().unwrap();
    assert_eq!(activation_layers.len(), 2);
    assert_eq!(activation_layers[0].layer_type(), "Activation");
    assert_eq!(activation_layers[1].layer_type(), "Activation");
    assert_eq!(activation_layers[0].activation(), "ReLU");
    assert_eq!(activation_layers[1].activation(), "Sigmoid");
}

#[test]
fn test_extreact_layers_error() {
    let nn = NN::new()
        .add(Activation::new(Act::ReLU))
        .unwrap()
        .add(Activation::new(Act::Sigmoid))
        .unwrap();
    let activation_layers = nn.extract_layers::<Dense>();
    assert!(activation_layers.is_err());
    assert_eq!(
        activation_layers.unwrap_err().to_string(),
        "Neural Network Error: There is no layers of this type in the network.".to_string()
    );
}

#[test]
fn test_predict() {
    let mut nn = NN::new()
        .add(Dense::new(2, 3).apply(Act::ReLU))
        .unwrap()
        .add(Dense::new(3, 1).apply(Act::Sigmoid))
        .unwrap();
    let input = array![1.0, 2.0];
    let output = nn.predict(input.view()).unwrap();
    assert_eq!(output.len(), 1);
}

#[test]
fn test_train() {
    let mut nn = NN::new()
        .add(Dense::new(2, 3).apply(Act::Tanh))
        .unwrap()
        .add(Dense::new(3, 1).apply(Act::Tanh))
        .unwrap();

    let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn();
    let labels = array![[0.0], [1.0], [1.0], [0.0]].into_dyn();

    let prev_loss = nn.loss();

    assert_eq!(prev_loss, f64::MAX);
    assert!(
        nn.train(train_data.view(), labels.view(), TrainConfig::default())
            .is_ok(),
        "Training failed"
    );

    let new_loss = nn.loss();

    assert_ne!(prev_loss, new_loss);
    assert!(
        new_loss < prev_loss,
        "Expected new loss {} to be less than previous loss {}",
        new_loss,
        prev_loss
    );
}

#[test]
fn test_loss() {
    let mut nn = NN::new()
        .add(Dense::new(2, 3).apply(Act::ReLU))
        .unwrap()
        .add(Dense::new(3, 1).apply(Act::Sigmoid))
        .unwrap();

    let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn();
    let labels = array![[0.0], [1.0], [1.0], [0.0]].into_dyn();

    let loss = nn
        .train(train_data.view(), labels.view(), TrainConfig::default())
        .unwrap();

    assert!(loss < f64::MAX);
}

#[test]
fn test_save_and_load() {
    let mut nn = NN::new()
        .add(Dropout::new(DEFAULT_DROPOUT_P))
        .unwrap()
        .add(Dense::new(2, 3))
        .unwrap()
        .add(Activation::new(Act::ReLU))
        .unwrap()
        .add(Dense::new(3, 1).apply(Act::Sigmoid))
        .unwrap();

    let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn();
    let labels = array![[0.0], [1.0], [1.0], [0.0]].into_dyn();

    nn.train(train_data.view(), labels.view(), TrainConfig::default())
        .unwrap();

    assert_eq!(nn.mode(), NNMode::Test);

    // Save the model
    nn.save("test_model.h5").unwrap();

    // Load the model
    let loaded_nn = NN::load("test_model.h5").unwrap();

    assert_eq!(loaded_nn.mode(), NNMode::Test);
    assert_eq!(nn.nlayers(), loaded_nn.nlayers());

    let original_dense_layers = nn.extract_layers::<Dense>();
    let original_activation_layers = nn.extract_layers::<Activation>();
    let original_dropout_layers = nn.extract_layers::<Dropout>();
    let loaded_dense_layers = loaded_nn.extract_layers::<Dense>();
    let loaded_activation_layers = loaded_nn.extract_layers::<Activation>();
    let loaded_dropout_layers = loaded_nn.extract_layers::<Dropout>();

    assert!(original_dense_layers.is_ok());
    assert!(original_activation_layers.is_ok());
    assert!(original_dropout_layers.is_ok());
    assert!(loaded_dense_layers.is_ok());
    assert!(loaded_activation_layers.is_ok());
    assert!(loaded_dropout_layers.is_ok());

    assert_eq!(nn.loss(), loaded_nn.loss());

    std::fs::remove_file("test_model.h5").unwrap();
}
