use hdf5::{types::VarLenUnicode, H5Type};
use ndarray::{s, Array1, Array2, ArrayD};
use std::{collections::VecDeque, path::Path, time::Instant};

use crate::{
    core::{MininnError, NNResult},
    layers::Layer,
    registers::LAYER_REGISTER,
    utils::{CostFunction, Optimizer},
};

/// Indicate if the neural network is in training or testing mode.
#[derive(Debug, PartialEq, Eq, H5Type, Clone, Copy)]
#[repr(u8)]
pub enum NNMode {
    Train = 0,
    Test = 1,
}

/// Represents a neural network.
///
/// This struct encapsulates the entire structure of a neural network, including all its layers.
/// It provides a foundation for building and managing complex neural network architectures.
///
/// # Attributes
///
/// * `layers` - A vector of boxed trait objects implementing the [`Layer`] trait.
///              Each element represents a layer in the neural network, allowing for
///              heterogeneous layer types within the same network.
/// * `register` - A register of the layers that the model have.
/// * `loss` - The loss of the model if training completes successfully.
///
/// ## Examples
///
/// ```
/// use mininn::prelude::*;
/// let mut nn = NN::new()
///     .add(Dense::new(784, 128).with(Act::ReLU)).unwrap()
///     .add(Dense::new(128, 10).with(Act::ReLU)).unwrap();
/// ```
///
#[derive(Debug)]
pub struct NN {
    layers: VecDeque<Box<dyn Layer>>,
    loss: f64,
    mode: NNMode,
}

impl NN {
    /// Creates a new empty neural network.
    ///
    /// ## Returns
    ///
    /// A new `NN` instance with no layers.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::core::NN;
    /// let nn = NN::new();
    /// assert!(nn.is_empty());
    /// ```
    ///
    #[inline]
    pub fn new() -> Self {
        Self {
            layers: VecDeque::new(),
            loss: f64::MAX,
            mode: NNMode::Train,
        }
    }

    /// Adds a new layer to the network.
    ///
    /// ## Arguments
    ///
    /// * `layer`: A struct that implements the [`Layer`](crate::layers::Layer) trait, e.g [`Dense`](crate::layers::Dense)
    ///
    /// ## Returns
    ///
    /// A mutable reference to `self`, allowing for method chaining.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// let nn = NN::new()
    ///     .add(Dense::new(784, 128).with(Act::ReLU)).unwrap()
    ///     .add(Dense::new(128, 10).with(Act::ReLU)).unwrap();
    /// ```
    ///
    pub fn add(mut self, layer: impl Layer + 'static) -> NNResult<Self> {
        // self.register
        //     .register_layer(&layer.layer_type(), L::from_json)?;
        self.layers.push_back(Box::new(layer));
        Ok(self)
    }

    /// Extracts layers of a specific type from the network.
    ///
    /// This generic method allows for flexible extraction of any layer type
    /// that implements the `Clone` trait and has a static lifetime. It uses
    /// dynamic typing to filter and extract layers of the specified type.
    ///
    /// ## Type Parameters
    ///
    /// * `T`: The type of layer to extract. Must implement `Clone`, [`Layer`](crate::layers::Layer) and have a `'static` lifetime.
    ///
    /// ### Returns
    ///
    /// A vector containing cloned instances of the specified layer type.
    ///
    /// ### Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// let nn = NN::new()
    ///     .add(Dense::new(784, 128).with(Act::ReLU)).unwrap()
    ///     .add(Activation::new(Act::ReLU)).unwrap()
    ///     .add(Dense::new(128, 10).with(Act::Sigmoid)).unwrap();
    ///
    /// let dense_layers = nn.extract_layers::<Dense>().unwrap();
    /// assert_eq!(dense_layers.len(), 2);
    ///
    /// let activation_layers = nn.extract_layers::<Activation>().unwrap();
    /// assert_eq!(activation_layers.len(), 1);
    /// ```
    ///
    /// ## Note
    ///
    /// This method uses dynamic casting, which may have a performance impact
    /// if called frequently or with a large number of layers. Consider caching
    /// the results if you need to access the extracted layers multiple times.
    ///
    #[inline]
    pub fn extract_layers<T: 'static + Layer>(&self) -> NNResult<Vec<&T>> {
        let layers: Vec<&T> = self
            .layers
            .iter()
            .filter_map(|l| l.as_any().downcast_ref::<T>())
            .collect();

        if layers.is_empty() {
            return Err(MininnError::NNError(
                "There is no layers of this type in the network".to_string(),
            ));
        }

        Ok(layers)
    }

    /// Returns the number of layers in the network.
    ///
    /// ## Returns
    ///
    /// The total number of layers in the network as a `usize`.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// let nn = NN::new()
    ///     .add(Dense::new(784, 128).with(Act::ReLU)).unwrap()
    ///     .add(Dense::new(128, 10).with(Act::ReLU)).unwrap();
    /// assert_eq!(nn.nlayers(), 2);
    /// ```
    ///
    #[inline]
    pub fn nlayers(&self) -> usize {
        self.layers.len()
    }

    /// Checks if the network has no layers.
    ///
    /// ## Returns
    ///
    /// `true` if the network has no layers, `false` otherwise.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// let nn = NN::new();
    /// assert!(nn.is_empty());
    ///
    /// let nn = nn.add(Dense::new(784, 128).with(Act::ReLU)).unwrap();
    /// assert!(!nn.is_empty());
    /// ```
    ///
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns the loss of the model if training completes successfully, or an error if something goes wrong.
    ///
    /// ## Returns
    ///
    /// The loss of the model as a `f64` value.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3).with(Act::ReLU)).unwrap()
    ///     .add(Dense::new(3, 1).with(Act::ReLU)).unwrap();
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
    /// let labels = array![[0.0], [1.0], [1.0]].into_dyn();
    /// let loss = nn.train(&train_data, &labels.into_dyn(), Cost::MSE, 100, 0.01, 1, Optimizer::GD, false).unwrap();
    /// assert!(loss < f64::MAX);
    /// ```
    ///
    #[inline]
    pub fn loss(&self) -> f64 {
        self.loss
    }

    /// Returns the mode of the neural network.
    ///
    /// ## Returns
    ///
    /// The mode of the neural network as a `NNMode` enum.
    ///
    /// ## Examples
    /// ```
    /// use mininn::prelude::*;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3).with(Act::ReLU)).unwrap()
    ///     .add(Dense::new(3, 1).with(Act::ReLU)).unwrap();
    /// assert_eq!(nn.mode(), NNMode::Train);
    /// ```
    ///
    #[inline]
    pub fn mode(&self) -> NNMode {
        self.mode
    }

    /// Performs a forward pass through the network to get a prediction.
    ///
    /// ## Arguments
    ///
    /// * `input` - The input to the network as an [`Array1<f64>`].
    ///
    /// ## Returns
    ///
    /// The output of the network as an [`Array1<f64>`].
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3).with(Act::ReLU)).unwrap()
    ///     .add(Dense::new(3, 1).with(Act::ReLU)).unwrap();
    /// let input = array![1.0, 2.0];
    /// let output = nn.predict(&input).unwrap();
    /// ```
    ///
    #[inline]
    pub fn predict(&mut self, input: &Array1<f64>) -> NNResult<ArrayD<f64>> {
        self.layers
            .iter_mut()
            .try_fold(input.to_owned().into_dimensionality()?, |output, layer| {
                layer.forward(&output, &self.mode)
            })
    }

    /// Trains the neural network using the provided data and parameters.
    ///
    /// ## Arguments
    ///
    /// * `train_data`: The training data.
    /// * `labels`: The labels corresponding to the training data.
    /// * `cost`: The cost function used to evaluate the error of the network.
    /// * `epochs`: The number of training epochs.
    /// * `learning_rate`: The learning rate for training.
    /// * `batch_size`: The size of each mini-batch.
    /// * `optimizer`: The optimizer used to update the weights and biases of the network.
    /// * `verbose`: Whether to print training progress.
    ///
    /// ## Returns
    ///
    /// The final loss of the model if training completes successfully, or an error if something goes wrong.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3).with(Act::ReLU)).unwrap()
    ///     .add(Dense::new(3, 1).with(Act::ReLU)).unwrap();
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
    /// let labels = array![[0.0], [1.0], [1.0]].into_dyn();
    /// let loss = nn.train(&train_data, &labels, Cost::MSE, 100, 0.01, 1, Optimizer::GD, false).unwrap();
    /// assert!(loss != f64::MAX);
    /// ```
    ///
    pub fn train(
        &mut self,
        train_data: &ArrayD<f64>,
        labels: &ArrayD<f64>,
        cost: impl CostFunction,
        epochs: u32,
        learning_rate: f64,
        batch_size: usize,
        optimizer: Optimizer,
        verbose: bool,
    ) -> NNResult<f64> {
        let train_data = train_data.to_owned().into_dimensionality()?;
        let labels: Array2<f64> = labels.to_owned().into_dimensionality()?;
        if epochs <= 0 {
            return Err(MininnError::NNError(
                "Number of epochs must be greater than 0".to_string(),
            ));
        }

        if learning_rate <= 0.0 {
            return Err(MininnError::NNError(
                "Learning rate must be greater than 0".to_string(),
            ));
        }

        if batch_size > train_data.nrows() {
            return Err(MininnError::NNError(
                "Batch size must be smaller than the number of training samples".to_string(),
            ));
        }

        self.mode = NNMode::Train;

        let total_start_time = Instant::now();

        for epoch in 1..=epochs {
            let epoch_start_time = Instant::now();
            let mut epoch_error = 0.0;

            for batch_start in (0..train_data.nrows()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(train_data.nrows());
                let batch_data = train_data.slice(s![batch_start..batch_end, ..]);
                let batch_labels = labels.slice(s![batch_start..batch_end, ..]);
                let mut batch_error = 0.0;

                for (input, label) in batch_data.rows().into_iter().zip(batch_labels.rows()) {
                    let output = self.predict(&input.to_owned())?;
                    let cost_value = cost.function(&output.view(), &label.into_dyn());
                    batch_error += cost_value;
                    let mut grad = cost.derivate(&output.view(), &label.into_dyn());

                    for layer in self.layers.iter_mut().rev() {
                        grad = layer.backward(&grad, learning_rate, &optimizer, &self.mode)?;
                    }
                }

                epoch_error += batch_error;
            }

            self.loss = epoch_error / train_data.nrows() as f64;

            if verbose {
                println!(
                    "Epoch {}/{} - Loss: {}, Time: {} sec",
                    epoch,
                    epochs,
                    self.loss,
                    epoch_start_time.elapsed().as_secs_f32()
                );
            }
        }

        if verbose {
            println!(
                "\nTraining Completed!\nTotal Training Time: {:.2} sec",
                total_start_time.elapsed().as_secs_f32()
            );
        }

        self.mode = NNMode::Test;

        Ok(self.loss)
    }

    /// Saves the neural network model into a HDF5 file.
    ///
    /// ## Arguments
    ///
    /// * `path`: The file path where the model will be saved.
    ///
    /// ## Returns
    ///
    /// `Ok(())` if the model is saved successfully, or an error if something goes wrong.
    ///
    pub fn save<P: AsRef<Path>>(&self, path: P) -> NNResult<()> {
        if self.is_empty() {
            return Err(MininnError::NNError("The model is empty".to_string()));
        }

        let path = path.as_ref();

        if path.extension().and_then(|s| s.to_str()) != Some("h5") {
            return Err(MininnError::IoError(
                "The file must be a .h5 file".to_string(),
            ));
        }

        let file = hdf5::File::create(path)?;

        file.new_attr::<f64>()
            .create("loss")?
            .write_scalar(&self.loss)?;

        file.new_attr::<NNMode>()
            .create("mode")?
            .write_scalar(&self.mode)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let group = file.create_group(&format!("model/layer_{}", i))?;

            group
                .new_attr::<VarLenUnicode>()
                .create("type")?
                .write_scalar(&layer.layer_type().parse::<VarLenUnicode>()?)?;

            group
                .new_attr::<VarLenUnicode>()
                .create("data")?
                .write_scalar(&layer.to_json()?.parse::<VarLenUnicode>()?)?;
        }

        file.close()?;

        Ok(())
    }

    /// Loads a neural network model from a TOML file.
    ///
    /// ## Arguments
    ///
    /// * `path`: The file path of the saved model.
    /// * `register`: A register of the layers that the model have
    ///
    /// ## Returns
    ///
    /// A `Result` containing the loaded `NN` if successful, or an error if something goes wrong.
    ///
    pub fn load<P: AsRef<Path>>(path: P) -> NNResult<NN> {
        let path = path.as_ref();

        if path.extension().and_then(|s| s.to_str()) != Some("h5") {
            return Err(MininnError::IoError(
                "The file must be a .h5 file".to_string(),
            ));
        }

        let mut nn = NN::new();

        let file = hdf5::File::open(path)?;
        let layer_count = file.groups()?[0].len();

        let loss = file.attr("loss")?.read_scalar::<f64>()?;
        let mode = file.attr("mode")?.read_scalar::<NNMode>()?;
        nn.loss = loss;
        nn.mode = mode;

        for i in 0..layer_count {
            let group = file.group(&format!("model/layer_{}", i))?;
            let layer_type = group.attr("type")?.read_scalar::<VarLenUnicode>()?;
            let json_data = group.attr("data")?.read_scalar::<VarLenUnicode>()?;
            let layer = LAYER_REGISTER.with(|register| {
                register
                    .borrow_mut()
                    .create_layer(&layer_type, json_data.as_str())
            })?;
            nn.layers.push_back(layer);
        }

        file.close()?;

        Ok(nn)
    }
}

impl Iterator for NN {
    type Item = Box<dyn Layer>;

    fn next(&mut self) -> Option<Self::Item> {
        self.layers.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, ArrayD, ArrayViewD, IxDyn};
    use serde::{Deserialize, Serialize};
    use serial_test::serial;

    use crate::{
        core::{NNMode, NNResult, NN},
        layers::{Activation, Dense, Dropout, Layer, DEFAULT_DROPOUT_P},
        registers::{register_activation, register_layer},
        utils::{Act, ActivationFunction, Cost, CostFunction, Optimizer},
    };

    #[test]
    fn test_new() {
        let nn = NN::new();
        assert!(nn.is_empty());
        assert_eq!(nn.nlayers(), 0);
    }

    #[test]
    fn test_add() {
        let nn = NN::new()
            .add(Dense::new(2, 3).with(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
            .unwrap();
        assert_eq!(nn.nlayers(), 2);
        assert!(!nn.is_empty());
    }

    #[test]
    fn test_dense_layers() {
        let nn = NN::new()
            .add(Dense::new(2, 3).with(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
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
    fn test_extract_layers_error() {
        let nn = NN::new()
            .add(Activation::new(Act::ReLU))
            .unwrap()
            .add(Activation::new(Act::Sigmoid))
            .unwrap();
        let activation_layers = nn.extract_layers::<Dense>();
        assert!(activation_layers.is_err());
        assert_eq!(
            activation_layers.unwrap_err().to_string(),
            "Neural Network Error: There is no layers of this type in the network."
        );
    }

    #[test]
    fn test_predict() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3).with(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
            .unwrap();
        let input = array![1.0, 2.0];
        let output = nn.predict(&input).unwrap();
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_train() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3).with(Act::Tanh))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Tanh))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let prev_loss = nn.loss();

        assert_eq!(prev_loss, f64::MAX);
        assert_eq!(nn.mode(), NNMode::Train);
        assert!(
            nn.train(
                &train_data.into_dyn(),
                &labels.into_dyn(),
                Cost::MSE,
                1,
                0.1,
                1,
                Optimizer::GD,
                false
            )
            .is_ok(),
            "Training failed"
        );
        assert_eq!(nn.mode(), NNMode::Test);

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
    fn test_train_bad_epochs() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3).with(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let result = nn.train(
            &train_data.into_dyn(),
            &labels.into_dyn(),
            Cost::MSE,
            0,
            0.1,
            1,
            Optimizer::GD,
            false,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Neural Network Error: Number of epochs must be greater than 0."
        );
    }

    #[test]
    fn test_train_bad_learning_rate() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3).with(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let result = nn.train(
            &train_data.into_dyn(),
            &labels.into_dyn(),
            Cost::MSE,
            1,
            0.0,
            1,
            Optimizer::GD,
            false,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Neural Network Error: Learning rate must be greater than 0."
        );
    }

    #[test]
    fn test_train_big_batch_size() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3).with(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let result = nn.train(
            &train_data.into_dyn(),
            &labels.into_dyn(),
            Cost::MSE,
            1,
            0.1,
            100,
            Optimizer::GD,
            false,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Neural Network Error: Batch size must be smaller than the number of training samples."
        );
    }

    #[test]
    fn test_loss() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3).with(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let loss = nn
            .train(
                &train_data.into_dyn(),
                &labels.into_dyn(),
                Cost::MSE,
                100,
                0.1,
                1,
                Optimizer::GD,
                false,
            )
            .unwrap();

        assert!(loss == nn.loss());
    }

    #[test]
    fn test_empty_nn_save() {
        let nn = NN::new();
        let result = nn.save("empty_model.h5");
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Neural Network Error: The model is empty."
        );
    }

    #[test]
    fn test_file_extension() {
        let nn = NN::new().add(Dense::new(2, 3).with(Act::ReLU)).unwrap();
        let result = nn.save("empty_model.json");
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "I/O Error: The file must be a .h5 file."
        );
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = NN::load("nonexistent_model.h5");
        assert!(result.is_err());
    }

    #[test]
    fn test_iter() {
        let nn = NN::new()
            .add(Dense::new(2, 3).with(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
            .unwrap();

        let mut iter = nn.into_iter();

        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_nn_extract_layers_error() {
        let nn = NN::new()
            .add(Activation::new(Act::ReLU))
            .unwrap()
            .add(Activation::new(Act::Sigmoid))
            .unwrap();
        let activation_layers = nn.extract_layers::<Dense>();
        assert!(activation_layers.is_err());
        assert_eq!(
            activation_layers.unwrap_err().to_string(),
            "Neural Network Error: There is no layers of this type in the network."
        );
    }

    #[test]
    fn test_train_custom_cost() {
        #[derive(Debug)]
        struct CustomCost;

        impl CostFunction for CustomCost {
            fn function(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> f64 {
                (y - y_p).abs().mean().unwrap_or(0.)
            }

            fn derivate(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> ArrayD<f64> {
                (y_p - y).signum() / y.len() as f64
            }

            fn cost_name(&self) -> &str {
                "Custom Cost"
            }
        }

        let mut nn = NN::new()
            .add(Dense::new(2, 3).with(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let prev_loss = nn.loss();

        assert_eq!(prev_loss, f64::MAX);
        assert_eq!(nn.mode(), NNMode::Train);
        assert!(
            nn.train(
                &train_data.into_dyn(),
                &labels.into_dyn(),
                CustomCost,
                100,
                0.1,
                1,
                Optimizer::GD,
                false
            )
            .is_ok(),
            "Training failed"
        );
        assert_eq!(nn.mode(), NNMode::Test);

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
    #[serial]
    fn test_save_and_load() {
        let mut nn = NN::new()
            .add(Dropout::new(DEFAULT_DROPOUT_P))
            .unwrap()
            .add(Dense::new(2, 3))
            .unwrap()
            .add(Activation::new(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).with(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        nn.train(
            &train_data.into_dyn(),
            &labels.into_dyn(),
            Cost::MSE,
            1,
            0.1,
            1,
            Optimizer::GD,
            false,
        )
        .unwrap();

        assert_eq!(nn.mode(), NNMode::Test);

        nn.save("test_model.h5").unwrap();

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

        for (original, loaded) in original_dense_layers
            .unwrap()
            .iter()
            .zip(loaded_dense_layers.unwrap().iter())
        {
            assert_eq!(original.ninputs(), loaded.ninputs());
            assert_eq!(original.noutputs(), loaded.noutputs());
        }

        for (original, loaded) in original_activation_layers
            .unwrap()
            .iter()
            .zip(loaded_activation_layers.unwrap().iter())
        {
            assert_eq!(original.activation(), loaded.activation());
        }

        for (original, loaded) in original_dropout_layers
            .unwrap()
            .iter()
            .zip(loaded_dropout_layers.unwrap().iter())
        {
            assert_eq!(original.p(), loaded.p());
            assert_eq!(original.seed(), loaded.seed());
        }

        assert_eq!(nn.loss(), loaded_nn.loss());

        std::fs::remove_file("test_model.h5").unwrap();
    }

    #[test]
    #[serial]
    fn test_save_and_load_custom_layer() {
        #[derive(Debug, Serialize, Deserialize)]
        struct CustomLayer;

        impl Layer for CustomLayer {
            fn layer_type(&self) -> String {
                "Custom".to_string()
            }

            fn to_json(&self) -> NNResult<String> {
                Ok(serde_json::to_string(self).unwrap())
            }

            fn from_json(json: &str) -> NNResult<Box<dyn Layer>>
            where
                Self: Sized,
            {
                Ok(Box::new(serde_json::from_str::<CustomLayer>(json).unwrap()))
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn forward(&mut self, _input: &ArrayD<f64>, _mode: &NNMode) -> NNResult<ArrayD<f64>> {
                Ok(ArrayD::zeros(IxDyn(&[3])))
            }

            fn backward(
                &mut self,
                _output_gradient: &ArrayD<f64>,
                _learning_rate: f64,
                _optimizer: &Optimizer,
                _mode: &NNMode,
            ) -> NNResult<ArrayD<f64>> {
                Ok(ArrayD::zeros(IxDyn(&[3])))
            }
        }

        let nn = NN::new()
            .add(CustomLayer)
            .unwrap()
            .add(Dense::new(3, 1).with(Act::ReLU))
            .unwrap();

        assert!(nn.save("custom_layer.h5").is_ok());

        register_layer::<CustomLayer>("Custom").unwrap();

        let nn = NN::load("custom_layer.h5").unwrap();

        let custom_layers = nn.extract_layers::<CustomLayer>().unwrap();
        let dense_layers = nn.extract_layers::<Dense>().unwrap();

        assert_eq!(dense_layers.len(), 1);
        assert_eq!(custom_layers.len(), 1);
        assert_eq!(custom_layers[0].layer_type(), "Custom");

        std::fs::remove_file("custom_layer.h5").unwrap();
    }

    #[test]
    #[serial]
    fn test_save_and_load_custom_activation() {
        #[derive(Debug)]
        struct CustomActivation;

        impl ActivationFunction for CustomActivation {
            fn function(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
                z.mapv(|x| x.powi(2))
            }

            fn derivate(&self, z: &ArrayViewD<f64>) -> ArrayD<f64> {
                z.mapv(|x| 2. * x)
            }

            fn activation(&self) -> &str {
                "CUSTOM"
            }

            fn from_activation(_activation: &str) -> NNResult<Box<dyn ActivationFunction>>
            where
                Self: Sized,
            {
                Ok(Box::new(CustomActivation))
            }
        }

        let nn = NN::new()
            .add(Dense::new(2, 3).with(CustomActivation))
            .unwrap()
            .add(Activation::new(Act::ReLU))
            .unwrap();

        // Save the model
        nn.save("test_model.h5").unwrap();

        register_activation::<CustomActivation>("CUSTOM").unwrap();

        // Load the model
        let loaded_nn = NN::load("test_model.h5").unwrap();

        assert_eq!(nn.nlayers(), loaded_nn.nlayers());

        let original_dense_layers = nn.extract_layers::<Dense>();
        let original_act_layer = nn.extract_layers::<Activation>();
        let loaded_dense_layers = loaded_nn.extract_layers::<Dense>();
        let loaded_act_layer = loaded_nn.extract_layers::<Activation>();

        assert!(original_dense_layers.is_ok());
        assert!(loaded_dense_layers.is_ok());

        assert_eq!(
            original_dense_layers.unwrap()[0]
                .activation()
                .unwrap()
                .activation(),
            loaded_dense_layers.unwrap()[0]
                .activation()
                .unwrap()
                .activation(),
        );

        assert_eq!(original_act_layer.unwrap()[0].activation(), "ReLU");
        assert_eq!(loaded_act_layer.unwrap()[0].activation(), "ReLU");

        assert_eq!(nn.loss(), loaded_nn.loss());

        std::fs::remove_file("test_model.h5").unwrap();
    }
}
