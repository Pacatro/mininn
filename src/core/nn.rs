use hdf5::{types::VarLenUnicode, H5Type};
use ndarray::{s, Array1, Array2, ArrayD, ArrayView1, ArrayView2};
use std::{collections::VecDeque, path::Path, time::Instant};

use crate::{
    core::{MininnError, NNResult},
    layers::{Dense, Layer},
    registers::REGISTER,
    utils::MSGPackFormatting,
};

use super::TrainConfig;

/// Indicate if the neural network is in training or testing mode.
#[derive(Debug, H5Type, PartialEq, Eq, Clone, Copy)]
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
///     .add(Dense::new(784, 128).apply(Act::ReLU))
///     .add(Dense::new(128, 10).apply(Act::ReLU));
/// ```
///
#[derive(Debug, Clone)]
pub struct NN {
    layers: VecDeque<Box<dyn Layer>>,
    train_config: TrainConfig,
    loss: f32,
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
            train_config: TrainConfig::default(),
            loss: f32::INFINITY,
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
    ///     .add(Dense::new(784, 128).apply(Act::ReLU))
    ///     .add(Dense::new(128, 10).apply(Act::ReLU));
    /// ```
    ///
    pub fn add(mut self, layer: impl Layer) -> Self {
        self.layers.push_back(Box::new(layer));
        self
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
    ///     .add(Dense::new(784, 128).apply(Act::ReLU))
    ///     .add(Activation::new(Act::ReLU))
    ///     .add(Dense::new(128, 10).apply(Act::Sigmoid));
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
    pub fn extract_layers<L: Layer>(&self) -> NNResult<Vec<&L>> {
        let layers: Vec<&L> = self
            .layers
            .iter()
            .filter_map(|l| l.as_any().downcast_ref::<L>())
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
    ///     .add(Dense::new(784, 128).apply(Act::ReLU))
    ///     .add(Dense::new(128, 10).apply(Act::ReLU));
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
    /// let nn = nn.add(Dense::new(784, 128).apply(Act::ReLU));
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
    /// The loss of the model as a `f32` value.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3).apply(Act::ReLU))
    ///     .add(Dense::new(3, 1).apply(Act::ReLU));
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let labels = array![[0.0], [1.0], [1.0]];
    /// let loss = nn.train(train_data.view(), labels.view(), TrainConfig::default()).unwrap();
    /// assert!(loss < f32::INFINITY);
    /// ```
    ///
    #[inline]
    pub fn loss(&self) -> f32 {
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
    ///     .add(Dense::new(2, 3).apply(Act::ReLU))
    ///     .add(Dense::new(3, 1).apply(Act::ReLU));
    /// assert_eq!(nn.mode(), NNMode::Train);
    /// ```
    ///
    #[inline]
    pub fn mode(&self) -> NNMode {
        self.mode
    }

    /// Returns the training configuration of the neural network.
    ///
    /// ## Examples
    /// ```
    /// use mininn::prelude::*;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3).apply(Act::ReLU))
    ///     .add(Dense::new(3, 1).apply(Act::ReLU));
    /// assert_eq!(nn.train_config().cost().name(), "MSE");
    /// assert_eq!(nn.train_config().epochs(), 100);
    /// assert_eq!(nn.train_config().learning_rate(), 0.1);
    /// assert_eq!(nn.train_config().batch_size(), 1);
    /// assert_eq!(nn.train_config().optimizer(), &Optimizer::GD);
    /// assert_eq!(nn.train_config().verbose(), true);
    /// ```
    ///
    #[inline]
    pub fn train_config(&self) -> &TrainConfig {
        &self.train_config
    }

    /// Performs a forward pass through the network to get a prediction.
    ///
    /// ## Arguments
    ///
    /// * `input` - The input to the network.
    ///
    /// ## Returns
    ///
    /// The output of the network.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3).apply(Act::ReLU))
    ///     .add(Dense::new(3, 1).apply(Act::ReLU));
    /// let input = array![1.0, 2.0];
    /// let output = nn.predict(input.view()).unwrap();
    /// ```
    ///
    #[inline]
    pub fn predict(&mut self, input: ArrayView1<f32>) -> NNResult<ArrayD<f32>> {
        self.layers
            .iter_mut()
            .try_fold(input.to_owned().into_dimensionality()?, |output, layer| {
                layer.forward(output.view(), &self.mode)
            })
    }

    /// Trains the neural network using the provided data and parameters.
    ///
    /// ## Arguments
    ///
    /// * `train_data`: The training data.
    /// * `labels`: The labels corresponding to the training data.
    /// * `train_config`: The training configuration to use, if none is provided, the default configuration will be used.
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
    ///     .add(Dense::new(2, 3).apply(Act::ReLU))
    ///     .add(Dense::new(3, 1).apply(Act::ReLU));
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let labels = array![[0.0], [1.0], [1.0]];
    /// let loss = nn.train(train_data.view(), labels.view(), TrainConfig::default()).unwrap();
    /// assert!(loss != f32::INFINITY);
    /// ```
    ///
    pub fn train(
        &mut self,
        train_data: ArrayView2<f32>,
        labels: ArrayView2<f32>,
        train_config: TrainConfig,
    ) -> NNResult<f32> {
        if train_config.epochs() == 0 {
            return Err(MininnError::NNError(
                "Number of epochs must be greater than 0".to_string(),
            ));
        }

        if train_config.learning_rate() <= 0.0 {
            return Err(MininnError::NNError(
                "Learning rate must be greater than 0".to_string(),
            ));
        }

        if train_config.batch_size() > train_data.nrows() {
            return Err(MininnError::NNError(
                "Batch size must be smaller than the number of training samples".to_string(),
            ));
        }

        if train_config.early_stopping() && train_config.patience() > train_config.epochs() {
            return Err(MininnError::NNError(format!(
                "Max epochs must be less than total epochs, got {} and {}",
                train_config.patience(),
                train_config.epochs()
            )));
        }

        if train_config.early_stopping() && train_config.verbose() {
            println!(
                "Early stopping enabled with patience = {} and tolerance = {}",
                train_config.patience(),
                train_config.tolerance()
            );
        }

        let mut best_loss = f32::INFINITY;
        let mut best_weights = Vec::new();
        let mut best_biases = Vec::new();
        let mut patience_counter = 0;

        self.train_config = train_config;
        self.mode = NNMode::Train;

        let total_start_time = Instant::now();

        for epoch in 1..=self.train_config.epochs() {
            let epoch_start_time = Instant::now();
            let mut epoch_error = 0.0;

            for batch_start in (0..train_data.nrows()).step_by(self.train_config.batch_size()) {
                let batch_end =
                    (batch_start + self.train_config.batch_size()).min(train_data.nrows());
                let batch_data = train_data.slice(s![batch_start..batch_end, ..]);
                let batch_labels = labels.slice(s![batch_start..batch_end, ..]);
                let mut batch_error = 0.0;

                for (input, label) in batch_data.rows().into_iter().zip(batch_labels.rows()) {
                    let output = self.predict(input)?;

                    let cost = self.train_config.cost().as_ref();

                    let cost_value = cost.function(&output.view(), &label.into_dyn());
                    let mut grad = cost.derivate(&output.view(), &label.into_dyn());

                    batch_error += cost_value;

                    for layer in self.layers.iter_mut().rev() {
                        grad = layer.backward(
                            grad.view(),
                            self.train_config.learning_rate(),
                            &self.train_config.optimizer(),
                            &self.mode,
                        )?;
                    }
                }

                epoch_error += batch_error;
            }

            self.loss = epoch_error / train_data.nrows() as f32;

            if self.train_config.verbose() {
                println!(
                    "Epoch {}/{} - Loss: {}, Time: {} sec",
                    epoch,
                    self.train_config.epochs(),
                    self.loss,
                    epoch_start_time.elapsed().as_secs_f32()
                );
            }

            if self.train_config.early_stopping() {
                let validation_loss = self.loss; // TODO: Implement validation loss

                if self.apply_early_stopping(
                    validation_loss,
                    &mut best_loss,
                    &mut patience_counter,
                    &mut best_weights,
                    &mut best_biases,
                ) {
                    if self.train_config.verbose() {
                        println!(
                            "Early stopping triggered at epoch {} - Best Loss: {}",
                            epoch, best_loss
                        );
                    }
                    break;
                }
            }
        }

        if self.train_config.verbose() {
            println!(
                "\nTraining Completed!\nTotal Training Time: {:.2} sec",
                total_start_time.elapsed().as_secs_f32()
            );
        }

        if self.train_config.early_stopping() && !best_weights.is_empty() && !best_biases.is_empty()
        {
            self.set_weights_biases(best_weights, best_biases);
        }

        self.mode = NNMode::Test;

        if self.train_config.early_stopping() {
            self.loss = best_loss;
        }

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
    pub fn save(&self, path: impl AsRef<Path>) -> NNResult<()> {
        if self.is_empty() {
            return Err(MininnError::NNError("The model is empty".to_string()));
        }

        let path = match path.as_ref().extension() {
            Some(ext) if ext == "h5" => path.as_ref().to_path_buf(),
            Some(ext) if ext != "h5" => {
                return Err(MininnError::IoError(
                    "The file must be a .h5 file".to_string(),
                ));
            }
            Some(_) => path.as_ref().with_extension("h5"),
            None => path.as_ref().with_extension("h5"),
        };

        let file = hdf5::File::create(path)?;

        file.new_attr::<f32>()
            .create("loss")?
            .write_scalar(&self.loss)?;

        file.new_attr::<NNMode>()
            .create("mode")?
            .write_scalar(&self.mode)?;

        let train_config_bytes = self.train_config.to_msgpack()?;

        file.new_dataset::<u8>()
            .shape(train_config_bytes.len())
            .create("train config")?
            .write(&train_config_bytes)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let group = file.create_group(&format!("model/layer_{}", i))?;

            group
                .new_attr::<VarLenUnicode>()
                .create("type")?
                .write_scalar(&layer.layer_type().parse::<VarLenUnicode>()?)?;

            let layer_bytes: Vec<u8> = layer.to_msgpack()?;

            group
                .new_dataset::<u8>()
                .shape(layer_bytes.len())
                .create("data")?
                .write(&layer_bytes)?;
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
    pub fn load(path: impl AsRef<Path>) -> NNResult<NN> {
        let path = match path.as_ref().extension() {
            Some(ext) if ext == "h5" => path.as_ref().to_path_buf(),
            Some(ext) if ext != "h5" => {
                return Err(MininnError::IoError(
                    "The file must be a .h5 file".to_string(),
                ));
            }
            Some(_) => path.as_ref().with_extension("h5"),
            None => path.as_ref().with_extension("h5"),
        };

        let mut nn = NN::new();

        let file = hdf5::File::open(path)?;
        let layer_count = file.groups()?[0].len();

        let loss = file.attr("loss")?.read_scalar::<f32>()?;
        let mode = file.attr("mode")?.read_scalar::<NNMode>()?;
        let train_config = file.dataset("train config")?.read()?.to_vec();

        nn.loss = loss;
        nn.mode = mode;
        nn.train_config = *TrainConfig::from_msgpack(&train_config)?;

        for i in 0..layer_count {
            let group = file.group(&format!("model/layer_{}", i))?;
            let layer_type = group.attr("type")?.read_scalar::<VarLenUnicode>()?;
            let data = group.dataset("data")?.read()?.to_vec();
            let layer =
                REGISTER.with_borrow(|register| register.create_layer(&layer_type, &data))?;
            nn.layers.push_back(layer);
        }

        file.close()?;

        Ok(nn)
    }

    /// Gets the weights and biases of the dense layers.
    ///
    /// ## Returns
    ///
    /// A tuple containing the weights and biases of the dense layers.
    ///
    fn get_weights_biases(&self) -> NNResult<(Vec<Array2<f32>>, Vec<Array1<f32>>)> {
        let denses = self.extract_layers::<Dense>()?;
        let weights = denses.iter().map(|d| d.weights().to_owned()).collect();
        let biases = denses.iter().map(|d| d.biases().to_owned()).collect();
        Ok((weights, biases))
    }

    /// Sets the weights and biases of the dense layers.
    ///
    /// ## Arguments
    ///
    /// * `weights`: The weights of the dense layers.
    /// * `biases`: The biases of the dense layers.
    ///
    fn set_weights_biases(&mut self, weights: Vec<Array2<f32>>, biases: Vec<Array1<f32>>) {
        let mut denses = self.extract_layers::<Dense>().unwrap();
        for (w, b) in weights.iter().zip(biases.iter()) {
            for dense in denses.iter_mut() {
                dense.to_owned().set_weights(w);
                dense.to_owned().set_biases(b);
            }
        }
    }

    /// Applies early stopping to the model.
    ///
    /// ## Arguments
    ///
    /// * `validation_loss`: The validation loss of the model.
    /// * `best_loss`: A mutable reference to the best loss of the model.
    /// * `patience_counter`: A mutable reference to the patience counter of the model.
    /// * `best_weights`: A mutable reference to the best weights of the model.
    /// * `best_biases`: A mutable reference to the best biases of the model.
    ///
    /// ## Returns
    ///
    /// A boolean indicating whether early stopping should be applied or not.
    ///
    fn apply_early_stopping(
        &mut self,
        validation_loss: f32,
        best_loss: &mut f32,
        patience_counter: &mut usize,
        best_weights: &mut Vec<Array2<f32>>,
        best_biases: &mut Vec<Array1<f32>>,
    ) -> bool {
        let absolute_improvement = *best_loss - validation_loss;
        let relative_improvement = absolute_improvement / best_loss.abs();

        if absolute_improvement > self.train_config.tolerance()
            || relative_improvement > self.train_config.tolerance()
        {
            *best_loss = validation_loss;
            (*best_weights, *best_biases) = self.get_weights_biases().unwrap();
            *patience_counter = 0;
        } else {
            *patience_counter += 1;
        }

        *patience_counter >= self.train_config.patience()
    }
}

impl Iterator for NN {
    type Item = Box<dyn Layer>;

    fn next(&mut self) -> Option<Self::Item> {
        self.layers.pop_front()
    }
}

#[macro_export]
macro_rules! nn {
    () => {
        NN::new()
    };
    ($( $layer:expr ),+ $(,)?) => {
        {
            let mut nn = NN::new();
            $(
                nn = nn.add($layer);
            )+
            nn
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::utils::NNUtil;
    use mininn_derive::{ActivationFunction, CostFunction, Layer};
    use ndarray::{array, ArrayD, ArrayViewD};
    use serde::{Deserialize, Serialize};
    use serial_test::serial;

    use crate::{
        core::{NNMode, NNResult, TrainConfig, NN},
        layers::DEFAULT_DROPOUT_P,
        layers::{Activation, Dense, Dropout, Layer, TrainLayer},
        prelude::Register,
        utils::{
            Act, ActCore, ActivationFunction, CostCore, CostFunction, MSGPackFormatting, Optimizer,
        },
    };

    #[derive(ActivationFunction, Debug, Clone)]
    struct CustomActivation;

    impl ActCore for CustomActivation {
        fn function(&self, z: &ArrayViewD<f32>) -> ArrayD<f32> {
            z.mapv(|x| x.powi(2))
        }

        fn derivate(&self, z: &ArrayViewD<f32>) -> ArrayD<f32> {
            z.mapv(|x| 2. * x)
        }
    }

    #[derive(CostFunction, Debug, Clone, Serialize, Deserialize)]
    struct CustomCost;

    impl CostCore for CustomCost {
        fn function(&self, y_p: &ArrayViewD<f32>, y: &ArrayViewD<f32>) -> f32 {
            (y - y_p).abs().mean().unwrap_or(0.)
        }

        fn derivate(&self, y_p: &ArrayViewD<f32>, y: &ArrayViewD<f32>) -> ArrayD<f32> {
            (y_p - y).signum() / y.len() as f32
        }
    }

    #[derive(Layer, Debug, Clone, Serialize, Deserialize)]
    struct CustomLayer;

    impl TrainLayer for CustomLayer {
        fn forward(&mut self, _input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
            todo!()
        }
        fn backward(
            &mut self,
            _output_gradient: ArrayViewD<f32>,
            _learning_rate: f32,
            _optimizer: &Optimizer,
            _mode: &NNMode,
        ) -> NNResult<ArrayD<f32>> {
            todo!()
        }
    }

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
            .add(Dense::new(3, 1).apply(Act::Sigmoid));
        assert!(!nn.is_empty());
        assert_eq!(nn.nlayers(), 2);
        assert_eq!(nn.loss(), f32::INFINITY);
        assert_eq!(nn.mode(), NNMode::Train);
    }

    #[test]
    fn test_dense_layers() {
        let nn = NN::new()
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .add(Dense::new(3, 1));
        assert!(!nn.is_empty());
        let dense_layers = nn.extract_layers::<Dense>().unwrap();
        assert!(!dense_layers.is_empty());
        assert_eq!(dense_layers.len(), 2);
        assert_eq!(dense_layers[0].ninputs(), 2);
        assert_eq!(dense_layers[0].noutputs(), 3);
        assert!(dense_layers[0].activation().is_some());
        assert_eq!(dense_layers[0].activation().unwrap(), "ReLU");
        assert_eq!(dense_layers[1].ninputs(), 3);
        assert_eq!(dense_layers[1].noutputs(), 1);
    }

    #[test]
    fn test_activation_layers() {
        let nn = NN::new()
            .add(Activation::new(Act::ReLU))
            .add(Activation::new(Act::Sigmoid));
        assert!(!nn.is_empty());
        let activation_layers = nn.extract_layers::<Activation>().unwrap();
        assert!(!activation_layers.is_empty());
        assert_eq!(activation_layers.len(), 2);
        assert_eq!(activation_layers[0].layer_type(), "Activation");
        assert_eq!(activation_layers[0].activation(), "ReLU");
        assert_eq!(activation_layers[1].layer_type(), "Activation");
        assert_eq!(activation_layers[1].activation(), "Sigmoid");
    }

    #[test]
    fn test_extract_layers_error() {
        let nn = NN::new()
            .add(Activation::new(Act::ReLU))
            .add(Activation::new(Act::Sigmoid));
        let activation_layers = nn.extract_layers::<Dense>();
        assert!(activation_layers.is_err());
        assert_eq!(
            activation_layers.unwrap_err().to_string(),
            "Neural Network Error: There is no layers of this type in the network."
        );
    }

    #[test]
    fn test_nn_macro() {
        let nn = nn!();
        assert!(nn.is_empty());
        assert_eq!(nn.nlayers(), 0);

        let nn = nn!(
            Dense::new(2, 3).apply(Act::ReLU),
            Dense::new(3, 1).apply(Act::Sigmoid)
        );
        assert!(!nn.is_empty());
        assert_eq!(nn.nlayers(), 2);
    }

    #[test]
    fn test_predict() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .add(Dense::new(3, 1).apply(Act::Sigmoid));
        let input = array![1.0, 2.0];
        let output = nn.predict(input.view()).unwrap();
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_train() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3).apply(Act::Tanh))
            .add(Dense::new(3, 1).apply(Act::Tanh));

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let prev_loss = nn.loss();

        assert_eq!(prev_loss, f32::INFINITY);
        assert_eq!(nn.mode(), NNMode::Train);

        let train_result = nn.train(train_data.view(), labels.view(), TrainConfig::default());

        assert!(train_result.is_ok(), "Training failed");
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
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .add(Dense::new(3, 1).apply(Act::Sigmoid));

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let result = nn.train(
            train_data.view(),
            labels.view(),
            TrainConfig::default().with_epochs(0).with_verbose(false),
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
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .add(Dense::new(3, 1).apply(Act::Sigmoid));

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let result = nn.train(
            train_data.view(),
            labels.view(),
            TrainConfig::default()
                .with_learning_rate(0.0)
                .with_verbose(false),
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
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .add(Dense::new(3, 1).apply(Act::Sigmoid));

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let result = nn.train(
            train_data.view(),
            labels.view(),
            TrainConfig::default()
                .with_batch_size(100)
                .with_verbose(false),
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
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .add(Dense::new(3, 1).apply(Act::Sigmoid));

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let loss = nn
            .train(
                train_data.view(),
                labels.view(),
                TrainConfig::default().with_verbose(false),
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
        let nn = NN::new().add(Dense::new(2, 3).apply(Act::ReLU));
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
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .add(Dense::new(3, 1).apply(Act::Sigmoid));

        let mut iter = nn.into_iter();

        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_nn_extract_layers_error() {
        let nn = NN::new()
            .add(Activation::new(Act::ReLU))
            .add(Activation::new(Act::Sigmoid));
        let activation_layers = nn.extract_layers::<Dense>();
        assert!(activation_layers.is_err());
        assert_eq!(
            activation_layers.unwrap_err().to_string(),
            "Neural Network Error: There is no layers of this type in the network."
        );
    }

    #[test]
    fn test_train_custom_cost() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .add(Dense::new(3, 1).apply(Act::Sigmoid));

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let prev_loss = nn.loss();

        let train_config = TrainConfig::default()
            .with_cost(CustomCost)
            .with_verbose(false);
        assert_eq!(prev_loss, f32::INFINITY);
        assert_eq!(nn.mode(), NNMode::Train);
        assert!(
            nn.train(train_data.view(), labels.view(), train_config)
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
            .add(Dense::new(2, 3))
            .add(Activation::new(Act::ReLU))
            .add(Dense::new(3, 1).apply(Act::Sigmoid));

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        nn.train(train_data.view(), labels.view(), TrainConfig::default())
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
        let nn = NN::new()
            .add(CustomLayer)
            .add(Dense::new(3, 1).apply(Act::ReLU));

        assert!(nn.save("custom_layer.h5").is_ok());

        Register::new().with_layer::<CustomLayer>().register();

        let nn = NN::load("custom_layer.h5").unwrap();

        let custom_layers = nn.extract_layers::<CustomLayer>().unwrap();
        let dense_layers = nn.extract_layers::<Dense>().unwrap();

        assert_eq!(dense_layers.len(), 1);
        assert_eq!(custom_layers.len(), 1);
        assert_eq!(custom_layers[0].layer_type(), "CustomLayer");

        std::fs::remove_file("custom_layer.h5").unwrap();
    }

    #[test]
    #[serial]
    fn test_save_and_load_custom_activation() {
        let nn = NN::new()
            .add(Dense::new(2, 3).apply(CustomActivation))
            .add(Activation::new(Act::ReLU));

        // Save the model
        nn.save("test_model.h5").unwrap();

        Register::new()
            .with_activation::<CustomActivation>()
            .register();

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
            original_dense_layers.unwrap()[0].activation().unwrap(),
            loaded_dense_layers.unwrap()[0].activation().unwrap()
        );

        assert_eq!(original_act_layer.unwrap()[0].activation(), "ReLU");
        assert_eq!(loaded_act_layer.unwrap()[0].activation(), "ReLU");

        assert_eq!(nn.loss(), loaded_nn.loss());

        std::fs::remove_file("test_model.h5").unwrap();
    }
}
