use hdf5::{types::VarLenUnicode, H5Type};
use ndarray::{s, Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, path::Path, time::Instant};

use crate::{
    core::{MininnError, NNResult},
    layers::{Dense, Layer},
    registers::LAYER_REGISTER,
    utils::{Cost, CostFunction, MSGPackFormat, Optimizer},
};

/// Training configuration for [`NN`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub cost: Box<dyn CostFunction>,
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub optimizer: Optimizer,
    pub early_stopping: bool,
    pub patience: usize,
    pub tolerance: f64,
    pub verbose: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            cost: Box::new(Cost::MSE),
            epochs: 100,
            learning_rate: 0.1,
            batch_size: 1,
            optimizer: Optimizer::GD,
            early_stopping: false,
            patience: 0,
            tolerance: 0.0,
            verbose: true,
        }
    }
}

impl TrainConfig {
    /// Creates a new empty [`TrainConfig`].
    ///
    /// ## Returns
    ///
    /// A new configuration instance with empty values.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// let train_config = TrainConfig::new();
    /// assert_eq!(train_config.cost.cost_name(), "MSE");
    /// assert_eq!(train_config.epochs, 0);
    /// assert_eq!(train_config.learning_rate, 0.0);
    /// assert_eq!(train_config.batch_size, 1);
    /// assert_eq!(train_config.optimizer, Optimizer::GD);
    /// assert_eq!(train_config.early_stopping, false);
    /// assert_eq!(train_config.patience, 0);
    /// assert_eq!(train_config.tolerance, 0.0);
    /// assert_eq!(train_config.verbose, false);
    /// ```
    ///
    pub fn new() -> Self {
        Self {
            cost: Box::new(Cost::MSE),
            epochs: 0,
            learning_rate: 0.0,
            batch_size: 1,
            optimizer: Optimizer::GD,
            early_stopping: false,
            patience: 0,
            tolerance: 0.0,
            verbose: false,
        }
    }

    /// Sets the number of epochs to train the network.
    ///
    /// The number of epochs determines the number of times the network will be trained on the
    /// entire training dataset.
    ///
    /// ## Arguments
    ///
    /// * `epochs` - The number of epochs to train the network.
    ///
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the cost function to be used during training.
    ///
    /// The cost function is responsible for calculating the loss of the network during training.
    /// It takes the predicted output and the actual output as input and returns a scalar value
    /// representing the loss.
    ///
    /// ## Arguments
    ///
    /// * `cost` - The cost function to be used during training.
    ///
    pub fn cost(mut self, cost: impl CostFunction + 'static) -> Self {
        self.cost = Box::new(cost);
        self
    }

    /// Sets the learning rate of the optimizer.
    ///
    /// The learning rate determines the step size of the optimization algorithm. A higher learning
    /// rate means that the optimizer will move faster, but may also lead to unstable training.
    ///
    /// ## Arguments
    ///
    /// * `learning_rate` - The learning rate of the optimizer.
    ///
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the batch size of the training dataset.
    ///
    /// The batch size determines the number of samples that are processed at a time during training.
    /// A larger batch size means that the network will be able to learn more quickly, but may also
    /// lead to unstable training.
    ///
    /// ## Arguments
    ///
    /// * `batch_size` - The batch size of the training dataset.
    ///
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the optimizer to be used during training.
    ///
    /// The optimizer is responsible for updating the weights of the network during training. It
    /// takes the current weights and the gradients of the loss function as input and returns the
    /// updated weights.
    ///
    /// ## Arguments
    ///
    /// * `optimizer` - The optimizer to be used during training.
    ///
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Sets whether the training process should stop early.
    ///
    /// If set to `true`, the training process will stop early if the validation loss does not
    /// improve for a certain number of epochs.
    ///
    /// ## Arguments
    ///
    /// * `early_stopping` - Whether to stop early or not.
    /// * `patience` - The limit of epochs without improvement before the training process stops.
    /// * `tolerance` - The minimum improvement required to continue training.
    ///
    pub fn early_stopping(mut self, patience: usize, tolerance: f64) -> Self {
        if patience > 0 && tolerance > 0.0 {
            self.early_stopping = true;
            self.patience = patience;
            self.tolerance = tolerance;
        }
        self
    }

    /// Sets whether the training process should be verbose.
    ///
    /// If set to `true`, the training process will print out information about the training
    /// process, such as the current loss and the current epoch.
    ///
    /// ## Arguments
    ///
    /// * `verbose` - Whether the training process should be verbose.
    ///
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl MSGPackFormat for TrainConfig {
    fn to_msgpack(&self) -> NNResult<Vec<u8>> {
        Ok(rmp_serde::to_vec(self)?)
    }

    fn from_msgpack(buff: &[u8]) -> NNResult<Box<Self>>
    where
        Self: Sized,
    {
        Ok(Box::new(rmp_serde::from_slice::<Self>(buff)?))
    }
}

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
///     .add(Dense::new(784, 128).apply(Act::ReLU)).unwrap()
///     .add(Dense::new(128, 10).apply(Act::ReLU)).unwrap();
/// ```
///
#[derive(Debug, Clone)]
pub struct NN {
    layers: VecDeque<Box<dyn Layer>>,
    train_config: TrainConfig,
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
            train_config: TrainConfig::default(),
            loss: f64::INFINITY,
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
    ///     .add(Dense::new(784, 128).apply(Act::ReLU)).unwrap()
    ///     .add(Dense::new(128, 10).apply(Act::ReLU)).unwrap();
    /// ```
    ///
    pub fn add(mut self, layer: impl Layer + 'static) -> NNResult<Self> {
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
    ///     .add(Dense::new(784, 128).apply(Act::ReLU)).unwrap()
    ///     .add(Activation::new(Act::ReLU)).unwrap()
    ///     .add(Dense::new(128, 10).apply(Act::Sigmoid)).unwrap();
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
    ///     .add(Dense::new(784, 128).apply(Act::ReLU)).unwrap()
    ///     .add(Dense::new(128, 10).apply(Act::ReLU)).unwrap();
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
    /// let nn = nn.add(Dense::new(784, 128).apply(Act::ReLU)).unwrap();
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
    ///     .add(Dense::new(2, 3).apply(Act::ReLU)).unwrap()
    ///     .add(Dense::new(3, 1).apply(Act::ReLU)).unwrap();
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let labels = array![[0.0], [1.0], [1.0]];
    /// let loss = nn.train(train_data.view(), labels.view(), TrainConfig::default()).unwrap();
    /// assert!(loss < f64::INFINITY);
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
    ///     .add(Dense::new(2, 3).apply(Act::ReLU)).unwrap()
    ///     .add(Dense::new(3, 1).apply(Act::ReLU)).unwrap();
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
    ///     .add(Dense::new(2, 3).apply(Act::ReLU)).unwrap()
    ///     .add(Dense::new(3, 1).apply(Act::ReLU)).unwrap();
    /// assert_eq!(nn.train_config().cost.cost_name(), "MSE");
    /// assert_eq!(nn.train_config().epochs, 100);
    /// assert_eq!(nn.train_config().learning_rate, 0.1);
    /// assert_eq!(nn.train_config().batch_size, 1);
    /// assert_eq!(nn.train_config().optimizer, Optimizer::GD);
    /// assert_eq!(nn.train_config().verbose, true);
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
    ///     .add(Dense::new(2, 3).apply(Act::ReLU)).unwrap()
    ///     .add(Dense::new(3, 1).apply(Act::ReLU)).unwrap();
    /// let input = array![1.0, 2.0];
    /// let output = nn.predict(input.view()).unwrap();
    /// ```
    ///
    #[inline]
    pub fn predict(&mut self, input: ArrayView1<f64>) -> NNResult<ArrayD<f64>> {
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
    ///     .add(Dense::new(2, 3).apply(Act::ReLU)).unwrap()
    ///     .add(Dense::new(3, 1).apply(Act::ReLU)).unwrap();
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let labels = array![[0.0], [1.0], [1.0]];
    /// let loss = nn.train(train_data.view(), labels.view(), TrainConfig::default()).unwrap();
    /// assert!(loss != f64::INFINITY);
    /// ```
    ///
    pub fn train(
        &mut self,
        train_data: ArrayView2<f64>,
        labels: ArrayView2<f64>,
        train_config: TrainConfig,
    ) -> NNResult<f64> {
        if train_config.epochs == 0 {
            return Err(MininnError::NNError(
                "Number of epochs must be greater than 0".to_string(),
            ));
        }

        if train_config.learning_rate <= 0.0 {
            return Err(MininnError::NNError(
                "Learning rate must be greater than 0".to_string(),
            ));
        }

        if train_config.batch_size > train_data.nrows() {
            return Err(MininnError::NNError(
                "Batch size must be smaller than the number of training samples".to_string(),
            ));
        }

        if train_config.early_stopping && train_config.patience > train_config.epochs {
            return Err(MininnError::NNError(format!(
                "Max epochs must be less than total epochs, got {} and {}",
                train_config.patience, train_config.epochs
            )));
        }

        if train_config.early_stopping && train_config.verbose {
            println!(
                "Early stopping enabled with patience = {} and tolerance = {}",
                train_config.patience, train_config.tolerance
            );
        }

        let mut best_loss = f64::INFINITY;
        let mut best_weights = Vec::new();
        let mut best_biases = Vec::new();
        let mut patience_counter = 0;

        self.train_config = train_config;
        self.mode = NNMode::Train;

        let total_start_time = Instant::now();

        for epoch in 1..=self.train_config.epochs {
            let epoch_start_time = Instant::now();
            let mut epoch_error = 0.0;

            for batch_start in (0..train_data.nrows()).step_by(self.train_config.batch_size) {
                let batch_end =
                    (batch_start + self.train_config.batch_size).min(train_data.nrows());
                let batch_data = train_data.slice(s![batch_start..batch_end, ..]);
                let batch_labels = labels.slice(s![batch_start..batch_end, ..]);
                let mut batch_error = 0.0;

                for (input, label) in batch_data.rows().into_iter().zip(batch_labels.rows()) {
                    let output = self.predict(input)?;

                    let (cost_value, mut grad) =
                        self.calc_gradient(output.view(), label.into_dyn());

                    batch_error += cost_value;

                    for layer in self.layers.iter_mut().rev() {
                        grad = layer.backward(
                            grad.view(),
                            self.train_config.learning_rate,
                            &self.train_config.optimizer,
                            &self.mode,
                        )?;
                    }
                }

                epoch_error += batch_error;
            }

            self.loss = epoch_error / train_data.nrows() as f64;

            if self.train_config.verbose {
                println!(
                    "Epoch {}/{} - Loss: {}, Time: {} sec",
                    epoch,
                    self.train_config.epochs,
                    self.loss,
                    epoch_start_time.elapsed().as_secs_f32()
                );
            }

            if self.train_config.early_stopping {
                let validation_loss = self.loss; // TODO: Implement validation loss

                if self.apply_early_stopping(
                    validation_loss,
                    &mut best_loss,
                    &mut patience_counter,
                    &mut best_weights,
                    &mut best_biases,
                ) {
                    if self.train_config.verbose {
                        println!(
                            "Early stopping triggered at epoch {} - Best Loss: {}",
                            epoch, best_loss
                        );
                    }
                    break;
                }
            }
        }

        if self.train_config.verbose {
            println!(
                "\nTraining Completed!\nTotal Training Time: {:.2} sec",
                total_start_time.elapsed().as_secs_f32()
            );
        }

        if self.train_config.early_stopping && !best_weights.is_empty() && !best_biases.is_empty() {
            self.set_weights_biases(best_weights, best_biases);
        }

        self.mode = NNMode::Test;

        if self.train_config.early_stopping {
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
    pub fn save<P: AsRef<Path>>(&self, path: P) -> NNResult<()> {
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

        file.new_attr::<f64>()
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
    pub fn load<P: AsRef<Path>>(path: P) -> NNResult<NN> {
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

        let loss = file.attr("loss")?.read_scalar::<f64>()?;
        let mode = file.attr("mode")?.read_scalar::<NNMode>()?;
        let train_config = file.dataset("train config")?.read()?.to_vec();

        nn.loss = loss;
        nn.mode = mode;
        nn.train_config = *TrainConfig::from_msgpack(&train_config)?;

        for i in 0..layer_count {
            let group = file.group(&format!("model/layer_{}", i))?;
            let layer_type = group.attr("type")?.read_scalar::<VarLenUnicode>()?;
            let data = group.dataset("data")?.read()?.to_vec();
            let layer = LAYER_REGISTER
                .with(|register| register.borrow_mut().create_layer(&layer_type, &data))?;
            nn.layers.push_back(layer);
        }

        file.close()?;

        Ok(nn)
    }

    ///Calculates the loss and the gradient of the loss function.
    ///
    /// ## Arguments
    ///
    /// * `output`: The output of the network.
    /// * `label`: The label of the output.
    ///
    /// ## Returns
    ///
    /// The loss and the gradient of the loss function.
    ///
    #[inline]
    fn calc_gradient(&self, output: ArrayViewD<f64>, label: ArrayViewD<f64>) -> (f64, ArrayD<f64>) {
        (
            self.train_config.cost.function(&output, &label),
            self.train_config.cost.derivate(&output, &label),
        )
    }

    /// Gets the weights and biases of the dense layers.
    ///
    /// ## Returns
    ///
    /// A tuple containing the weights and biases of the dense layers.
    ///
    fn get_weights_biases(&self) -> NNResult<(Vec<Array2<f64>>, Vec<Array1<f64>>)> {
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
    fn set_weights_biases(&mut self, weights: Vec<Array2<f64>>, biases: Vec<Array1<f64>>) {
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
        validation_loss: f64,
        best_loss: &mut f64,
        patience_counter: &mut usize,
        best_weights: &mut Vec<Array2<f64>>,
        best_biases: &mut Vec<Array1<f64>>,
    ) -> bool {
        let absolute_improvement = *best_loss - validation_loss;
        let relative_improvement = absolute_improvement / best_loss.abs();

        if absolute_improvement > self.train_config.tolerance
            || relative_improvement > self.train_config.tolerance
        {
            *best_loss = validation_loss;
            (*best_weights, *best_biases) = self.get_weights_biases().unwrap();
            *patience_counter = 0;
        } else {
            *patience_counter += 1;
        }

        *patience_counter >= self.train_config.patience
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
    use ndarray::{array, ArrayD, ArrayViewD};
    use serde::{Deserialize, Serialize};
    use serial_test::serial;

    use crate::{
        core::{NNMode, NNResult, TrainConfig, NN},
        layers::{Activation, Dense, Dropout, Layer, DEFAULT_DROPOUT_P},
        registers::{register_activation, register_layer},
        utils::{Act, ActivationFunction, Cost, CostFunction, MSGPackFormat, Optimizer},
    };

    #[derive(Debug, Clone)]
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

    #[derive(Debug, Clone)]
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

        fn from_cost(_cost: &str) -> NNResult<Box<dyn CostFunction>>
        where
            Self: Sized,
        {
            todo!()
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct CustomLayer;

    impl Layer for CustomLayer {
        fn layer_type(&self) -> String {
            "Custom".to_string()
        }
        // fn to_msgpack(&self) -> NNResult<Vec<u8>> {
        //     Ok(rmp_serde::to_vec(self)?)
        // }
        // fn from_msgpack(buff: &[u8]) -> NNResult<Box<Self>> {
        //     Ok(Box::new(rmp_serde::from_slice::<Self>(buff)?))
        // }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn forward(&mut self, _input: ArrayViewD<f64>, _mode: &NNMode) -> NNResult<ArrayD<f64>> {
            todo!()
        }
        fn backward(
            &mut self,
            _output_gradient: ArrayViewD<f64>,
            _learning_rate: f64,
            _optimizer: &Optimizer,
            _mode: &NNMode,
        ) -> NNResult<ArrayD<f64>> {
            todo!()
        }
    }

    impl MSGPackFormat for CustomLayer {
        fn to_msgpack(&self) -> NNResult<Vec<u8>> {
            Ok(rmp_serde::to_vec(self)?)
        }

        fn from_msgpack(buff: &[u8]) -> NNResult<Box<Self>>
        where
            Self: Sized,
        {
            Ok(Box::new(rmp_serde::from_slice::<Self>(buff)?))
        }
    }

    #[test]
    fn test_new() {
        let nn = NN::new();
        assert!(nn.is_empty());
        assert_eq!(nn.nlayers(), 0);
    }

    #[test]
    fn test_train_config_new() {
        let train_config = TrainConfig::new();
        assert_eq!(train_config.cost.cost_name(), "MSE");
        assert_eq!(train_config.epochs, 0);
        assert_eq!(train_config.learning_rate, 0.0);
        assert_eq!(train_config.batch_size, 1);
        assert_eq!(train_config.optimizer, Optimizer::GD);
        assert_eq!(train_config.early_stopping, false);
        assert_eq!(train_config.patience, 0);
        assert_eq!(train_config.tolerance, 0.0);
        assert_eq!(train_config.verbose, false);
    }

    #[test]
    fn test_train_config_default() {
        let train_config = TrainConfig::default();
        assert_eq!(train_config.cost.cost_name(), "MSE");
        assert_eq!(train_config.epochs, 100);
        assert_eq!(train_config.learning_rate, 0.1);
        assert_eq!(train_config.batch_size, 1);
        assert_eq!(train_config.optimizer, Optimizer::GD);
        assert_eq!(train_config.verbose, true);
    }

    #[test]
    fn test_custom_train_config() {
        let train_config = TrainConfig::new()
            .epochs(1000)
            .cost(Cost::CCE)
            .learning_rate(0.01)
            .batch_size(32)
            .optimizer(Optimizer::default_momentum())
            .verbose(true);

        assert_eq!(train_config.cost.cost_name(), "CCE");
        assert_eq!(train_config.epochs, 1000);
        assert_eq!(train_config.learning_rate, 0.01);
        assert_eq!(train_config.batch_size, 32);
        assert_eq!(train_config.optimizer, Optimizer::default_momentum());
        assert_eq!(train_config.verbose, true);
    }

    #[test]
    fn test_add() {
        let nn = NN::new()
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).apply(Act::Sigmoid))
            .unwrap();
        assert!(!nn.is_empty());
        assert_eq!(nn.nlayers(), 2);
        assert_eq!(nn.loss(), f64::INFINITY);
        assert_eq!(nn.mode(), NNMode::Train);
    }

    #[test]
    fn test_dense_layers() {
        let nn = NN::new()
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1))
            .unwrap();
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
            .unwrap()
            .add(Activation::new(Act::Sigmoid))
            .unwrap();
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

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let prev_loss = nn.loss();

        assert_eq!(prev_loss, f64::INFINITY);
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
            .unwrap()
            .add(Dense::new(3, 1).apply(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let result = nn.train(
            train_data.view(),
            labels.view(),
            TrainConfig::default().epochs(0).verbose(false),
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
            .unwrap()
            .add(Dense::new(3, 1).apply(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let result = nn.train(
            train_data.view(),
            labels.view(),
            TrainConfig::default().learning_rate(0.0).verbose(false),
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
            .unwrap()
            .add(Dense::new(3, 1).apply(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let result = nn.train(
            train_data.view(),
            labels.view(),
            TrainConfig::default().batch_size(100).verbose(false),
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
            .unwrap()
            .add(Dense::new(3, 1).apply(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let loss = nn
            .train(
                train_data.view(),
                labels.view(),
                TrainConfig::default().verbose(false),
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
        let nn = NN::new().add(Dense::new(2, 3).apply(Act::ReLU)).unwrap();
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
            .unwrap()
            .add(Dense::new(3, 1).apply(Act::Sigmoid))
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
        let mut nn = NN::new()
            .add(Dense::new(2, 3).apply(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).apply(Act::Sigmoid))
            .unwrap();

        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];

        let prev_loss = nn.loss();

        let train_config = TrainConfig::default().cost(CustomCost);
        assert_eq!(prev_loss, f64::INFINITY);
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
            .unwrap()
            .add(Dense::new(2, 3))
            .unwrap()
            .add(Activation::new(Act::ReLU))
            .unwrap()
            .add(Dense::new(3, 1).apply(Act::Sigmoid))
            .unwrap();

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
            .unwrap()
            .add(Dense::new(3, 1).apply(Act::ReLU))
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
        let nn = NN::new()
            .add(Dense::new(2, 3).apply(CustomActivation))
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
            original_dense_layers.unwrap()[0].activation().unwrap(),
            loaded_dense_layers.unwrap()[0].activation().unwrap()
        );

        assert_eq!(original_act_layer.unwrap()[0].activation(), "ReLU");
        assert_eq!(loaded_act_layer.unwrap()[0].activation(), "ReLU");

        assert_eq!(nn.loss(), loaded_nn.loss());

        std::fs::remove_file("test_model.h5").unwrap();
    }
}
