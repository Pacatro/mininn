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

/// Represents a neural network
/// 
/// ## Atributes
/// 
/// - `layers`: The layers of the neural network
/// 
pub struct NN {
    layers: Vec<Box<dyn Layer>>,
}

impl NN {
    /// Creates a new [`NN`] without layers
    #[inline]
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    /// Adds a new layer to the network
    /// 
    /// ## Atributes
    /// 
    /// - `layer`: A struct that implement the [`Layer`] trait like [`Dense`](crate::layers::Dense), [`Activation`](crate::layers::Activation), etc
    /// 
    /// ## Returns
    /// 
    /// A mutable [`NN`] with the new layer
    /// 
    pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Returns only the dense layers of the network
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

    /// Returns the number of layers in the network
    #[inline]
    pub fn nlayers(&self) -> usize {
        self.layers.len()
    }

    /// Returns true if the network has no layers
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get the prediction of the network
    /// 
    /// ## Arguments
    /// 
    /// - `input`: The reference to input of the network as an [`Array1<f64>`](ndarray::Array1)
    /// 
    /// ## Returns
    /// 
    /// The prediction of the network as an [`Array1<f64>`](ndarray::Array1)
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

    /// Trains the neural network
    ///
    /// ## Arguments
    ///
    /// - `cost`: The cost function used to evaluate the error of the network
    /// - `train_data`: The reference to the training data as an [`Array2<f64>`](ndarray::Array2)
    /// - `labels`: The reference to the labels corresponding to the training data as an [`Array2<f64>`](ndarray::Array2)
    /// - `epochs`: The number of epochs for training as a `u32`
    /// - `learning_rate`: The learning rate for training as an `f64`
    /// - `verbose`: A boolean indicating whether to print training progress
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

    /// Stored all the important information into a `.toml` file
    /// 
    /// ## Arguments
    /// 
    /// - `path`: The path where save the data
    /// 
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

    /// Load a model from a `.toml` file.
    /// 
    /// ## Atributes
    /// 
    /// - `path`: The path of the model file.
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
mod test {
    use crate::layers::{Dense, Activation};

    use super::*;

    #[test]
    fn test_new_nn() {
        let nn = NN::new()
            .add(Dense::new(1, 2, Activation::SIGMOID))
            .add(Dense::new(2, 2, Activation::SIGMOID));

        assert_eq!(nn.nlayers(), 2);
    }
}