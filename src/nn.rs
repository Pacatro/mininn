use std::{
    error::Error,
    fs::{self, File},
    io::Write, 
    str::FromStr, 
    time::Instant
};

use ndarray::{Array1, Array2};

use crate::{
    utils::Cost,
    layers::{Layer, Dense, Activation},
    save_config::SaveConfig,
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
///     .add(Dense::new(784, 128, Activation::RELU))
///     .add(Dense::new(128, 10, Activation::RELU));
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
        Self { layers: vec![] }
    }

    /// Adds a new layer to the network.
    ///
    /// # Arguments
    ///
    /// * `layer` - A struct that implements the `Layer` trait, e.g [`Dense`]
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
    ///     .add(Dense::new(784, 128, Activation::RELU))
    ///     .add(Dense::new(128, 10, Activation::RELU));
    /// ```
    /// 
    pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Returns only the dense layers of the network.
    ///
    /// # Returns
    ///
    /// A vector containing cloned [`Dense`] layers from the network.
    ///
    /// # Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// let nn = NN::new()
    ///     .add(Dense::new(784, 128, Activation::RELU))
    ///     .add(Dense::new(128, 10, Activation::RELU));
    /// 
    /// let dense_layers = nn.dense_layers();
    /// assert_eq!(dense_layers.len(), 2);
    /// ```
    /// 
    #[inline]
    pub fn dense_layers(&self) -> Vec<Dense> {
        self.layers
            .iter()
            .filter_map(|l| {
                l.as_any().downcast_ref::<Dense>()
            })
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
    ///     .add(Dense::new(784, 128, Activation::RELU))
    ///     .add(Dense::new(128, 10, Activation::RELU));
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
    /// let nn = nn.add(Dense::new(784, 128, Activation::RELU));
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
    ///     .add(Dense::new(2, 3, Activation::RELU))
    ///     .add(Dense::new(3, 1, Activation::RELU));
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
    ///     .add(Dense::new(2, 3, Activation::RELU))
    ///     .add(Dense::new(3, 1, Activation::RELU));
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
    ) -> Result<(), Box<dyn Error>> {
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

    /// Saves the neural network model into a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the model will be saved.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the model is saved successfully, or an error if something goes wrong.
    ///
    /// # Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    /// let nn = NN::new()
    ///     .add(Dense::new(784, 128, Activation::RELU))
    ///     .add(Dense::new(128, 10, Activation::RELU));
    /// nn.save("model.toml").unwrap();
    /// ```
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        if self.is_empty() {
            return Err("Can not save the model because it is empty.".into());
        }
        
        let save_config = SaveConfig::new(self);
        let toml_string = toml::to_string(&save_config)?;
        let mut file = File::create(path)?;
        file.write_all(toml_string.as_bytes())?;

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
    /// # Examples
    ///
    /// ```
    /// use mininn::NN;
    /// let nn = NN::load("model.toml").unwrap();
    /// ```
    /// 
    pub fn load(path: &str) -> Result<NN, Box<dyn Error>> {
        let content = fs::read_to_string(path)?;

        if content.is_empty() {
            return Err(format!("'{path}' is empty").into());
        }

        let save_config: SaveConfig = toml::from_str(&content)?;

        if save_config.nn_weights().is_empty() ||
           save_config.nn_biases().is_empty() || 
           save_config.nn_layers_activation().is_empty() {
            return Err(format!("The path '{path}' does not contains any model").into());
        }

        let mut nn = NN::new();
        
        for ((w, b), a) in save_config.nn_weights().iter()
            .zip(save_config.nn_biases())
            .zip(save_config.nn_layers_activation()) {
            
            let weights = Array2::from_shape_vec(
                (w.len(), w[0].len()),
                w.iter().flatten().cloned().collect()
            )?;

            let biases = Array1::from_shape_vec(
                b.len(),
                b.to_vec()
            )?;
            
            let mut dense = Dense::new(weights.shape()[0], weights.shape()[1], Activation::STEP);
            dense.set_weights(&weights);
            dense.set_biases(&biases);
            dense.set_activation(Activation::from_str(a)?);
            
            nn.layers.push(Box::new(dense));
        }
        
        Ok(nn)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

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
            .add(Dense::new(2, 3, Activation::RELU))
            .add(Dense::new(3, 1, Activation::SIGMOID));
        assert_eq!(nn.nlayers(), 2);
        assert!(!nn.is_empty());
    }

    #[test]
    fn test_dense_layers() {
        let nn = NN::new()
            .add(Dense::new(2, 3, Activation::RELU))
            .add(Dense::new(3, 1, Activation::SIGMOID));
        let dense_layers = nn.dense_layers();
        assert_eq!(dense_layers.len(), 2);
        assert_eq!(dense_layers[0].input_size(), 2);
        assert_eq!(dense_layers[0].output_size(), 3);
        assert_eq!(dense_layers[1].input_size(), 3);
        assert_eq!(dense_layers[1].output_size(), 1);
    }

    #[test]
    fn test_predict() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3, Activation::RELU))
            .add(Dense::new(3, 1, Activation::SIGMOID));
        let input = array![1.0, 2.0];
        let output = nn.predict(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_train() {
        let mut nn = NN::new()
            .add(Dense::new(2, 3, Activation::TANH))
            .add(Dense::new(3, 1, Activation::STEP));
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
            .add(Dense::new(2, 3, Activation::RELU))
            .add(Dense::new(3, 1, Activation::SIGMOID));
        
        // Save the model
        nn.save("test_model.toml").unwrap();

        // Load the model
        let loaded_nn = NN::load("test_model.toml").unwrap();

        assert_eq!(nn.nlayers(), loaded_nn.nlayers());
        
        let original_layers = nn.dense_layers();
        let loaded_layers = loaded_nn.dense_layers();

        for (original, loaded) in original_layers.iter().zip(loaded_layers.iter()) {
            assert_eq!(original.input_size(), loaded.input_size());
            assert_eq!(original.output_size(), loaded.output_size());
            // You might want to add more detailed comparisons here,
            // such as checking weights and biases
        }

        // Clean up
        std::fs::remove_file("test_model.toml").unwrap();
    }

    #[test]
    fn test_empty_nn_save() {
        let nn = NN::new();
        let result = nn.save("empty_model.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = NN::load("nonexistent_model.toml");
        assert!(result.is_err());
    }
}