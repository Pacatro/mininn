use std::{error::Error, time::Instant, fs::File, io::Write};
use ndarray::{Array1, Array2};

use crate::{cost::Cost, layers::{Activation, BaseLayer, Dense}};

use crate::save_config::SaveConfig;

/// Represents a neural network
/// 
/// ## Atributes
/// 
/// - `layers`: The layers of the neural network
/// 
pub struct NN {
    layers: Vec<Box<dyn BaseLayer>>,
}

impl NN {
    /// Creates a new [`NN`] without layers
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    /// Adds a new layer to the network
    /// 
    /// ## Atributes
    /// 
    /// - `layer`: A struct that implement the [`BaseLayer`] trait like [`Dense`](crate::layers::Dense), [`Activation`](crate::layers::Activation), etc
    /// 
    /// ## Returns
    /// 
    /// A mutable [`NN`] with the new layer
    /// 
    pub fn add<L: BaseLayer + 'static>(mut self, layer: L ) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Returns only the dense layers of the network
    pub fn dense_layers(&self) -> Vec<Dense> {
        self.layers
            .iter()
            .filter_map(|l| {
                l.as_any().downcast_ref::<Dense>()
            })
            .cloned()
            .collect()
    }

    /// Returns only the activation layers of the network
    pub fn activation_layers(&self) -> Vec<Activation> {
        self.layers
            .iter()
            .filter_map(|l| {
                l.as_any().downcast_ref::<Activation>()
            })
            .cloned()
            .collect()
    }

    /// Returns the number of layers in the network
    pub fn nlayers(&self) -> usize {
        self.layers.len()
    }

    /// Get the prediction of the network
    /// 
    /// ## Arguments
    /// 
    /// - `input`: The input of the network as an [`Array1<f64>`](ndarray::Array1)
    /// 
    /// ## Returns
    /// 
    /// The prediction of the network as an [`Array2<f64>`](ndarray::Array2)
    /// 
    pub fn predict(&mut self, input: &Array1<f64>) -> Array2<f64> {
        let mut output = Array2::from_shape_vec((input.len(), 1), input.to_vec()).unwrap();

        for layer in self.layers.iter_mut() {
            output = layer.forward(output);
        }

        output
    }

    /// Trains the neural network
    ///
    /// ## Arguments
    ///
    /// - `cost`: The cost function used to evaluate the error of the network
    /// - `train_data`: The training data as an [`Array2<f64>`](ndarray::Array2)
    /// - `labels`: The labels corresponding to the training data as an [`Array2<f64>`](ndarray::Array2)
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
    ) {
        for e in 0..epochs {
            let now = Instant::now();
            let mut error = 0.0;

            for (x, y) in train_data.rows().into_iter().zip(labels.rows()) {
                let y = Array2::from_shape_vec((1, y.len()), y.to_vec()).unwrap();
                let output = self.predict(&x.to_owned());

                error += cost.function(&output.to_owned(), &y);

                let mut grad = cost.derivate(&output, &y);
                for layer in self.layers.iter_mut().rev() {
                    grad = layer.backward(grad, learning_rate);
                }
            }

            error /= train_data.len() as f64;

            if verbose {
                println!("Epoch {}/{}, error: {}, time: {} seg", e+1, epochs, error, now.elapsed().as_secs_f32());
            }
        }
    }

    /// Stored all the important information into a `.toml` file
    /// 
    /// ## Arguments
    /// 
    /// - `path`: The path where save the data
    /// 
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let save_config = SaveConfig::new(self);

        let toml_string = toml::to_string(&save_config)?;
        let mut file = File::create(path)?;
        file.write_all(toml_string.as_bytes())?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{activation_type::ActivationType, layers::{Dense, Activation}};

    use super::*;

    #[test]
    fn test_new_nn() {
        let nn = NN::new()
            .add(Dense::new(1, 2))
            .add(Activation::new(ActivationType::SIGMOID))
            .add(Dense::new(2, 2));

        assert_eq!(nn.nlayers(), 3);
    }
}