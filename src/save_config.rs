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
pub struct SaveConfig {
    nn_weights: Vec<Vec<Vec<f64>>>,
    nn_biases: Vec<Vec<f64>>,
}

impl SaveConfig {
    /// Creates a new save configuration
    pub fn new(nn: &NN) -> Self {
        let nn_weights: Vec<Vec<Vec<f64>>> = nn
            .dense_layers()
            .iter()
            .map(|d| {
                d.weights()
                    .outer_iter()
                    .map(|row| row.to_vec())
                    .collect()
            })
            .collect();

        let nn_biases: Vec<Vec<f64>> = nn
            .dense_layers()
            .iter()
            .map(|d| d.biases().flatten().to_vec())
            .collect();

        Self { nn_weights, nn_biases }
    }
}