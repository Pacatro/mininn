use std::{path::Path, time::Instant};
use ndarray::{Array1, Array2};
use hdf5::{types::VarLenUnicode, File};

use crate::{
    layers::Layer,
    utils::{Cost, LayerRegister},
    error::{MininnError, NNResult},
};

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
/// # Examples
///
/// ```
/// use mininn::prelude::*;
/// let mut nn = NN::new()
///     .add(Dense::new(784, 128, Some(ActivationFunc::RELU))).unwrap()
///     .add(Dense::new(128, 10, Some(ActivationFunc::RELU))).unwrap();
/// ```
///
/// # Notes
///
/// - The order of layers in the `layers` vector corresponds to the forward pass order.
/// - This structure supports various types of layers, as long as they implement the `Layer` trait.
/// - Memory management for layers is handled automatically through the use of `Box<dyn Layer>`.
/// 
#[derive(Debug)]
pub struct NN {
    layers: Vec<Box<dyn Layer>>,
    register: LayerRegister,
    loss: f64,
}

impl NN {
    /// Creates a new empty neural network.
    ///
    /// # Returns
    ///
    /// A new `NN` instance with no layers.
    ///
    /// # Examples
    ///
    /// ```
    /// use mininn::NN;
    /// let nn = NN::new();
    /// assert!(nn.is_empty());
    /// ```
    /// 
    #[inline]
    pub fn new() -> Self {
        Self { layers: vec![], register: LayerRegister::default(), loss: f64::MAX }
    }

    /// Adds a new layer to the network.
    ///
    /// # Arguments
    ///
    /// * `layer`: A struct that implements the `Layer` trait, e.g [`Dense`](crate::layers::Dense)
    ///
    /// # Returns
    ///
    /// A mutable reference to `self`, allowing for method chaining.
    ///
    /// # Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// let nn = NN::new()
    ///     .add(Dense::new(784, 128, Some(ActivationFunc::RELU))).unwrap()
    ///     .add(Dense::new(128, 10, Some(ActivationFunc::RELU))).unwrap();
    /// ```
    /// 
    pub fn add<L: Layer + 'static>(mut self, layer: L) -> NNResult<Self> {
        self.register.register_layer(&layer.layer_type(), L::from_json)?;
        self.layers.push(Box::new(layer));
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
    /// * `T`: The type of layer to extract. Must implement `Clone`, `Layer` and have a `'static` lifetime.
    ///
    /// ## Returns
    ///
    /// A vector containing cloned instances of the specified layer type.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// let nn = NN::new()
    ///     .add(Dense::new(784, 128, Some(ActivationFunc::RELU))).unwrap()
    ///     .add(Activation::new(ActivationFunc::RELU)).unwrap()
    ///     .add(Dense::new(128, 10, Some(ActivationFunc::SIGMOID))).unwrap();
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
    pub fn extract_layers<T: 'static + Clone + Layer>(&self) -> NNResult<Vec<T>> {
        let layers: Vec<T> = self.layers
            .iter()
            .filter_map(|l| l.as_any().downcast_ref::<T>())
            .cloned()
            .collect();

        if layers.is_empty() {
            return Err(MininnError::NNError("There is no layers of this type in the network".to_string()));
        }

        Ok(layers)
    }

    /// Returns the number of layers in the network.
    ///
    /// # Returns
    ///
    /// The total number of layers in the network as a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// let nn = NN::new()
    ///     .add(Dense::new(784, 128, Some(ActivationFunc::RELU))).unwrap()
    ///     .add(Dense::new(128, 10, Some(ActivationFunc::RELU))).unwrap();
    /// assert_eq!(nn.nlayers(), 2);
    /// ```
    /// 
    #[inline]
    pub fn nlayers(&self) -> usize {
        self.layers.len()
    }

    /// Checks if the network has no layers.
    ///
    /// # Returns
    ///
    /// `true` if the network has no layers, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// let nn = NN::new();
    /// assert!(nn.is_empty());
    ///
    /// let nn = nn.add(Dense::new(784, 128, Some(ActivationFunc::RELU))).unwrap();
    /// assert!(!nn.is_empty());
    /// ```
    /// 
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns the loss of the model if training completes successfully, or an error if something goes wrong.
    ///
    /// # Returns
    ///
    /// The loss of the model as a `f64` value.
    ///
    /// # Examples
    /// 
    /// ```
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3, Some(ActivationFunc::RELU))).unwrap()
    ///     .add(Dense::new(3, 1, Some(ActivationFunc::RELU))).unwrap();
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let labels = array![[0.0], [1.0], [1.0]];
    /// let loss = nn.train(Cost::MSE, &train_data, &labels, 100, 0.01, false).unwrap();
    /// assert!(loss < f64::MAX);
    /// ```
    ///
    #[inline]
    pub fn loss(&self) -> f64 {
        self.loss
    }

    /// Performs a forward pass through the network to get a prediction.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to the network as an [`Array1<f64>`].
    ///
    /// # Returns
    ///
    /// The output of the network as an [`Array1<f64>`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3, Some(ActivationFunc::RELU))).unwrap()
    ///     .add(Dense::new(3, 1, Some(ActivationFunc::RELU))).unwrap();
    /// let input = array![1.0, 2.0];
    /// let output = nn.predict(&input).unwrap();
    /// ```
    /// 
    #[inline]
    pub fn predict(&mut self, input: &Array1<f64>) -> NNResult<Array1<f64>> {
        self.layers.iter_mut().try_fold(input.clone(), |output, layer| {
            layer.forward(&output)
        })
    }

    /// Trains the neural network using the provided data and parameters.
    ///
    /// # Arguments
    ///
    /// * `cost`: The cost function used to evaluate the error of the network.
    /// * `train_data`: The training data as an [`Array2<f64>`].
    /// * `labels`: The labels corresponding to the training data as an [`Array2<f64>`].
    /// * `epochs`: The number of training epochs.
    /// * `learning_rate`: The learning rate for training.
    /// * `verbose`: Whether to print training progress.
    ///
    /// # Returns
    ///
    /// The final loss of the mdoel if training completes successfully, or an error if something goes wrong.
    ///
    /// # Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3, Some(ActivationFunc::RELU))).unwrap()
    ///     .add(Dense::new(3, 1, Some(ActivationFunc::RELU))).unwrap();
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let labels = array![[0.0], [1.0], [1.0]];
    /// let loss = nn.train(Cost::MSE, &train_data, &labels, 100, 0.01, false).unwrap();
    /// assert!(loss != f64::NAN);
    /// ```
    /// 
    pub fn train(
        &mut self,
        cost: Cost,
        train_data: &Array2<f64>,
        labels: &Array2<f64>,
        epochs: u32,
        learning_rate: f64,
        verbose: bool,
    ) -> NNResult<f64> {
        let total_start_time = Instant::now();
    
        for epoch in 1..=epochs {
            let epoch_start_time = Instant::now();
            let mut epoch_error = 0.0;

            for (input, label) in train_data.rows().into_iter().zip(labels.rows()) {
                let output = self.predict(&input.to_owned())?;
                let cost_value = cost.function(&output.view(), &label);
                epoch_error += cost_value;
    
                let mut grad = cost.derivate(&output.view(), &label);
    
                for layer in self.layers.iter_mut().rev() {
                    grad = layer.backward(grad.view(), learning_rate)?;
                }
            }

            self.loss = epoch_error / train_data.nrows() as f64;
    
            if verbose {
                println!(
                    "Epoch {}/{} - Loss: {}, Time: {} sec",
                    epoch, epochs, self.loss, epoch_start_time.elapsed().as_secs_f32()
                );
            }
        }
    
        if verbose {
            println!(
                "\nTraining Completed!\nTotal Training Time: {:.2} sec",
                total_start_time.elapsed().as_secs_f32()
            );
        }
    
        Ok(self.loss)
    }

    /// Saves the neural network model into a HDF5 file.
    ///
    /// # Arguments
    ///
    /// * `path`: The file path where the model will be saved.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the model is saved successfully, or an error if something goes wrong.
    ///
    pub fn save<P: AsRef<Path>>(&self, path: P) -> NNResult<()> {
        if self.is_empty() {
            return Err(MininnError::NNError("The model is empty".to_string()));
        }

        let path = path.as_ref();

        if path.extension().and_then(|s| s.to_str()) != Some("h5") {
            return Err(MininnError::NNError("The file must be a .h5 file".to_string()));
        }

        let file = File::create(path)?;
        
        for (i, layer) in self.layers.iter().enumerate() {
            let group = file.create_group(&format!("model/layer_{}", i))?;

            group.new_attr::<VarLenUnicode>()
                .create("type")?
                .write_scalar(&layer.layer_type().parse::<VarLenUnicode>()?)?;

            group.new_attr::<VarLenUnicode>()
                .create("data")?
                .write_scalar(&layer.to_json()?.parse::<VarLenUnicode>()?)?;
        }

        Ok(())
    }

    /// Loads a neural network model from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path`: The file path of the saved model.
    /// * `register`: A register of the layers that the model have
    ///
    /// # Returns
    ///
    /// A `Result` containing the loaded `NN` if successful, or an error if something goes wrong.
    ///
    pub fn load<P: AsRef<Path>>(path: P, register: Option<LayerRegister>) -> NNResult<NN> {
        let path = path.as_ref();

        if path.extension().and_then(|s| s.to_str()) != Some("h5") {
            return Err(MininnError::NNError("The file must be a .h5 file".to_string()));
        }
        
        let mut nn = NN::new();
        
        nn.register = register.unwrap_or_else(LayerRegister::new);

        let file = File::open(path)?;
        let layer_count = file.groups()?[0].len();

        for i in 0..layer_count {
            let group = file.group(&format!("model/layer_{}", i))?;
            let layer_type = group.attr("type")?.read_scalar::<VarLenUnicode>()?;
            let json_data = group.attr("data")?.read_scalar::<VarLenUnicode>()?;
            let layer = nn.register.create_layer(&layer_type, json_data.as_str())?;
            nn.layers.push(layer);
        }

        Ok(nn)
    }
}

impl Default for NN {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::prelude::*;

    use super::*;

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
            nn.train(Cost::MSE, &train_data, &labels, 1, 0.1, false).is_ok(),
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

        let loss = nn.train(Cost::MSE, &train_data, &labels, 100, 0.1, false).unwrap();

        assert!(loss == nn.loss());
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

        for (original, loaded) in original_layers.unwrap().iter().zip(loaded_layers.unwrap().iter()) {
            assert_eq!(original.ninputs(), loaded.ninputs());
            assert_eq!(original.noutputs(), loaded.noutputs());
        }

        std::fs::remove_file("load_models/test_model.h5").unwrap();
    }

    #[test]
    fn test_empty_nn_save() {
        let nn = NN::new();
        let result = nn.save("load_models/empty_model.h5");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "Neural Network Error: The model is empty.".to_string());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = NN::load("load_models/nonexistent_model.h5", None);
        assert!(result.is_err());
    }
}
