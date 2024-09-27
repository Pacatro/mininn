use std::time::Instant;

use ndarray::{Array1, Array2};
use hdf5::{types::VarLenUnicode, File};

use crate::{
    layers::Layer,
    utils::{layer_register::LayerRegister, Cost}, 
    NNResult
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
///
/// # Examples
///
/// ```
/// use mininn::prelude::*;
/// let mut nn = NN::new()
///     .add(Dense::new(784, 128, Some(ActivationFunc::RELU)))
///     .add(Dense::new(128, 10, Some(ActivationFunc::RELU)));
/// ```
///
/// # Notes
///
/// - The order of layers in the `layers` vector corresponds to the forward pass order.
/// - This structure supports various types of layers, as long as they implement the `Layer` trait.
/// - Memory management for layers is handled automatically through the use of `Box<dyn Layer>`.
/// 
pub struct NN {
    layers: Vec<Box<dyn Layer>>,
    register: LayerRegister
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
        Self { layers: vec![], register: LayerRegister::new() }
    }

    /// Adds a new layer to the network.
    ///
    /// # Arguments
    ///
    /// * `layer` - A struct that implements the `Layer` trait, e.g [`Dense`](crate::layers::Dense)
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
    ///     .add(Dense::new(784, 128, Some(ActivationFunc::RELU)))
    ///     .add(Dense::new(128, 10, Some(ActivationFunc::RELU)));
    /// ```
    /// 
    pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.register.register_layer(&layer.layer_type(), L::from_json);
        self.layers.push(Box::new(layer));
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
    ///     .add(Dense::new(784, 128, Some(ActivationFunc::RELU)))
    ///     .add(Activation::new(ActivationFunc::RELU))
    ///     .add(Dense::new(128, 10, Some(ActivationFunc::SIGMOID)));
    ///
    /// let dense_layers = nn.extract_layers::<Dense>();
    /// assert_eq!(dense_layers.len(), 2);
    ///
    /// let activation_layers = nn.extract_layers::<Activation>();
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
    pub fn extract_layers<T: 'static + Clone + Layer>(&self) -> Vec<T> {
        self.layers
            .iter()
            .filter_map(|l| l.as_any().downcast_ref::<T>())
            .cloned()
            .collect()
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
    ///     .add(Dense::new(784, 128, Some(ActivationFunc::RELU)))
    ///     .add(Dense::new(128, 10, Some(ActivationFunc::RELU)));
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
    /// let nn = nn.add(Dense::new(784, 128, Some(ActivationFunc::RELU)));
    /// assert!(!nn.is_empty());
    /// ```
    /// 
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
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
    ///     .add(Dense::new(2, 3, Some(ActivationFunc::RELU)))
    ///     .add(Dense::new(3, 1, Some(ActivationFunc::RELU)));
    /// let input = array![1.0, 2.0];
    /// let output = nn.predict(&input);
    /// ```
    /// 
    #[inline]
    pub fn predict(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.layers
            .iter_mut()
            .fold(
                input.to_owned(),
                |output, layer| layer.forward(&output)
            )
    }

    /// Trains the neural network using the provided data and parameters.
    ///
    /// # Arguments
    ///
    /// * `cost` - The cost function used to evaluate the error of the network.
    /// * `train_data` - The training data as an [`Array2<f64>`].
    /// * `labels` - The labels corresponding to the training data as an [`Array2<f64>`].
    /// * `epochs` - The number of training epochs.
    /// * `learning_rate` - The learning rate for training.
    /// * `verbose` - Whether to print training progress.
    ///
    /// # Returns
    ///
    /// `Ok(())` if training completes successfully, or an error if something goes wrong.
    ///
    /// # Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// use ndarray::array;
    /// let mut nn = NN::new()
    ///     .add(Dense::new(2, 3, Some(ActivationFunc::RELU)))
    ///     .add(Dense::new(3, 1, Some(ActivationFunc::RELU)));
    /// let train_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let labels = array![[0.0], [1.0], [1.0]];
    /// nn.train(Cost::MSE, &train_data, &labels, 1000, 0.01, true).unwrap();
    /// ```
    /// 
    pub fn train(
        &mut self,
        cost: Cost,
        train_data: &Array2<f64>,
        labels: &Array2<f64>,
        epochs: u32,
        learning_rate: f64,
        verbose: bool
    ) -> NNResult<()> {
        for epoch in 1..=epochs {
            let now = Instant::now();
            let mut total_error = 0.0;

            for (x, y) in train_data.rows().into_iter().zip(labels.rows()) {
                let output = self.predict(&x.to_owned());
                total_error += cost.function(&output.view(), &y);
                let mut grad = cost.derivate(&output.view(), &y);

                for layer in self.layers.iter_mut().rev() {
                    grad = layer.backward(grad.view(), learning_rate)?;
                }
            }

            let avg_error = total_error / train_data.nrows() as f64;

            if verbose {
                println!(
                    "Epoch {}/{}, error: {}, time: {} sec",
                    epoch, epochs, avg_error, now.elapsed().as_secs_f32()
                );
            }
        }

        Ok(())
    }

    /// Saves the neural network model into a HDF5 file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the model will be saved.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the model is saved successfully, or an error if something goes wrong.
    ///
    pub fn save(&self, path: &str) -> NNResult<()> {
        if self.is_empty() {
            return Err("Can not save the model because it is empty.".into());
        }

        if !path.contains(".h5") {
            return Err("The file must be a .h5 file".into());
        }

        let file = File::create(path)?;
        
        for (i, layer) in self.layers.iter().enumerate() {
            let group = file.create_group(&format!("model/layer_{}", i))?;

            group
                .new_attr::<VarLenUnicode>()
                .create("type")?
                .write_scalar(&layer.layer_type().parse::<VarLenUnicode>()?)?;

            group
                .new_attr::<VarLenUnicode>()
                .create("data")?
                .write_scalar(&layer.to_json().parse::<VarLenUnicode>()?)?;
        }

        Ok(())
    }
    /// Loads a neural network model from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path of the saved model.
    ///
    /// # Returns
    ///
    /// A `Result` containing the loaded `NN` if successful, or an error if something goes wrong.
    ///
    pub fn load(path: &str) -> NNResult<NN> {
        let file = File::open(path)?;
        let mut nn = NN::new();

        let layer_count = file.groups().unwrap()[0].len();

        for i in 0..layer_count {
            let group = file.group(&format!("model/layer_{}", i))?;
            let layer_type = group.attr("type")?.read_scalar::<VarLenUnicode>()?;
            let json_data = group.attr("data")?.read_scalar::<VarLenUnicode>()?;
            let layer = nn.register.create_layer(&layer_type, json_data.as_str());
            nn.layers.push(layer);
        }

        Ok(nn)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
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
            .add(Dense::new(2, 3, Some(ActivationFunc::RELU)))
            .add(Dense::new(3, 1, Some(ActivationFunc::SIGMOID)));
        assert_eq!(nn.nlayers(), 2);
        assert!(!nn.is_empty());
    }

    #[test]
    fn test_dense_layers() {
        let nn = NN::new()
            .add(Dense::new(2, 3, Some(ActivationFunc::RELU)))
            .add(Dense::new(3, 1, Some(ActivationFunc::SIGMOID)));
        let dense_layers = nn.extract_layers::<Dense>();
        assert_eq!(dense_layers.len(), 2);
        assert_eq!(dense_layers[0].ninputs(), 2);
        assert_eq!(dense_layers[0].noutputs(), 3);
        assert_eq!(dense_layers[1].ninputs(), 3);
        assert_eq!(dense_layers[1].noutputs(), 1);
    }

    #[test]
    fn test_predict() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3, Some(ActivationFunc::RELU)))
            .add(Dense::new(3, 1, Some(ActivationFunc::SIGMOID)));
        let input = array![1.0, 2.0];
        let output = nn.predict(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_train() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3, Some(ActivationFunc::TANH)))
            .add(Dense::new(3, 1, Some(ActivationFunc::STEP)));
        let train_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let labels = array![[0.0], [1.0], [1.0], [0.0]];
        let result = nn.train(Cost::MSE, &train_data, &labels, 1000, 0.1, false);
        assert!(result.is_ok());

        // Test predictions after training
        let predictions: Vec<f64> = train_data.rows()
            .into_iter()
            .map(|row| {
                let pred = nn.predict(&row.to_owned())[0];
                if pred < 0.5 { 0.0 } else { 1.0 }
            })
            .collect();
        
        assert_relative_eq!(predictions[0], 0.0, epsilon = 0.1);
        assert_relative_eq!(predictions[1], 1.0, epsilon = 0.1);
        assert_relative_eq!(predictions[2], 1.0, epsilon = 0.1);
        assert_relative_eq!(predictions[3], 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_save_and_load() {
        let nn = NN::new()
            .add(Dense::new(2, 3, Some(ActivationFunc::RELU)))
            .add(Dense::new(3, 1, Some(ActivationFunc::SIGMOID)));
        
        // Save the model
        nn.save("load_models/test_model.h5").unwrap();

        // Load the model
        let loaded_nn = NN::load("load_models/test_model.h5").unwrap();

        assert_eq!(nn.nlayers(), loaded_nn.nlayers());
        
        let original_layers = nn.extract_layers::<Dense>();
        let loaded_layers = loaded_nn.extract_layers::<Dense>();

        for (original, loaded) in original_layers.iter().zip(loaded_layers.iter()) {
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
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = NN::load("load_models/nonexistent_model.h5");
        assert!(result.is_err());
    }
}