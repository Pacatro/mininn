use serde::{Deserialize, Serialize};

use crate::NN;

/// Represents all the network information that must be stored
/// 
/// ## Atributes
/// 
/// - `nn_weights`: All the weights of the network
/// - `nn_biases`: All the biases of the network
/// 
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct SaveConfig {
    nn_weights: Vec<Vec<Vec<f64>>>,
    nn_biases: Vec<Vec<f64>>,
    nn_layers_activation: Vec<String>
}

impl SaveConfig {
    /// Creates a new save configuration
    pub fn new(nn: &NN) -> Self {
        let nn_weights = nn
            .dense_layers()
            .iter()
            .map(|d| {
                d.weights()
                    .outer_iter()
                    .map(|row| row.to_vec())
                    .collect()
            })
            .collect();

        let nn_biases = nn
            .dense_layers()
            .iter()
            .map(|d| d.biases().to_vec())
            .collect();

        let nn_layers_activation = nn
            .activation_layers()
            .iter()
            .map(|l| l.activation().to_string())
            .collect();

        Self { nn_weights, nn_biases, nn_layers_activation }
    }

    pub fn nn_weights(&self) -> &Vec<Vec<Vec<f64>>> {
        &self.nn_weights
    }

    pub fn nn_biases(&self) -> &Vec<Vec<f64>> {
        &self.nn_biases
    }

    pub fn nn_layers_activation(&self) -> &Vec<String> {
        &self.nn_layers_activation
    }
}