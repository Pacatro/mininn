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
        .add(Dense::new(2, 3, Some(ActivationFunc::RELU))).unwrap()
        .add(Dense::new(3, 1, Some(ActivationFunc::SIGMOID))).unwrap();
    assert_eq!(nn.nlayers(), 2);
    assert!(!nn.is_empty());
}

#[test]
fn test_dense_layers() {
    let nn = NN::new()
        .add(Dense::new(2, 3, Some(ActivationFunc::RELU))).unwrap()
        .add(Dense::new(3, 1, Some(ActivationFunc::SIGMOID))).unwrap();
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
        .add(Activation::new(ActivationFunc::RELU)).unwrap()
        .add(Activation::new(ActivationFunc::SIGMOID)).unwrap();
    let activation_layers = nn.extract_layers::<Activation>().unwrap();
    assert_eq!(activation_layers.len(), 2);
    assert_eq!(activation_layers[0].layer_type(), "Activation");        
    assert_eq!(activation_layers[1].layer_type(), "Activation");
    assert_eq!(activation_layers[0].activation(), ActivationFunc::RELU);
    assert_eq!(activation_layers[1].activation(), ActivationFunc::SIGMOID);
}

#[test]
fn test_extreact_layers_error() {
    let nn = NN::new()
        .add(Activation::new(ActivationFunc::RELU)).unwrap()
        .add(Activation::new(ActivationFunc::SIGMOID)).unwrap();
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
        .add(Dense::new(2, 3, Some(ActivationFunc::RELU))).unwrap()
        .add(Dense::new(3, 1, Some(ActivationFunc::SIGMOID))).unwrap();
    let input = array![1.0, 2.0];
    let output = nn.predict(&input).unwrap();
    assert_eq!(output.len(), 1);
}

#[test]
fn test_train() {
    let mut nn = NN::new()
        .add(Dense::new(2, 3, Some(ActivationFunc::TANH))).unwrap()
        .add(Dense::new(3, 1, Some(ActivationFunc::TANH))).unwrap();

    let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let labels = array![[0.0], [1.0], [1.0], [0.0]];

    let prev_loss = nn.loss();

    assert_eq!(prev_loss, f64::MAX);
    assert!(
        nn.train(Cost::MSE, &train_data, &labels, 1, 0.1, 1, false).is_ok(),
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
        .add(Dense::new(2, 3, Some(ActivationFunc::RELU))).unwrap()
        .add(Dense::new(3, 1, Some(ActivationFunc::SIGMOID))).unwrap();

    let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let labels = array![[0.0], [1.0], [1.0], [0.0]];

    let loss = nn.train(Cost::MSE, &train_data, &labels, 100, 0.1, 1, false).unwrap();

    assert!(loss < f64::MAX);
}

#[test]
fn test_save_and_load() {
    let nn = NN::new()
        .add(Dense::new(2, 3, Some(ActivationFunc::RELU))).unwrap()
        .add(Dense::new(3, 1, Some(ActivationFunc::SIGMOID))).unwrap();

    // Save the model
    nn.save("load_models/test_model.h5").unwrap();

    // Load the model
    let loaded_nn = NN::load("load_models/test_model.h5", None).unwrap();

    assert_eq!(nn.nlayers(), loaded_nn.nlayers());

    let original_layers = nn.extract_layers::<Dense>();
    let loaded_layers = loaded_nn.extract_layers::<Dense>();

    assert!(original_layers.is_ok());
    assert!(loaded_layers.is_ok());
}