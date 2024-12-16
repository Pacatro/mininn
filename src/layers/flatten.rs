use mininn_derive::Layer;
use ndarray::{Array1, ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

use crate::{
    core::{NNMode, NNResult},
    utils::{MSGPackFormat, Optimizer},
};

use super::{layer::TrainLayer, Layer};

/// Flattens the input into a 1D array.
///
/// This layer is useful when the input is a 2D array, but you want to treat it as a 1D array.
///
/// ## Attributes
///
/// - `input`: The input array.
/// - `original_shape`: The original shape of the input array.
///
/// ## Example
///
/// ```
/// use ndarray::array;
/// use mininn::prelude::*;
///
/// let mut layer = Flatten::new();
/// let input = array![[1., 2., 3.], [4., 5., 6.]].into_dyn();
/// let output = layer.forward(input.view(), &NNMode::Train).unwrap();
/// assert_eq!(output.shape(), &[6]);
/// ```
#[derive(Layer, Debug, Clone, Serialize, Deserialize)]
pub struct Flatten {
    input: Array1<f64>,
    original_shape: Vec<usize>,
}

impl Flatten {
    /// Creates a new instance of the Flatten layer.
    pub fn new() -> Self {
        Self {
            input: Array1::zeros(0),
            original_shape: Vec::new(),
        }
    }
}

impl TrainLayer for Flatten {
    fn forward(&mut self, input: ArrayViewD<f64>, _mode: &NNMode) -> NNResult<ArrayD<f64>> {
        self.original_shape = input.shape().to_vec();
        self.input = input.flatten().to_owned();
        Ok(self.input.clone().into_dyn())
    }

    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f64>,
        _learning_rate: f64,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f64>> {
        let reshaped_gradient = output_gradient
            .to_shape(self.original_shape.clone())?
            .to_owned();
        Ok(reshaped_gradient.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_flatten_forward() {
        let mut layer = Flatten::new();
        let input = array![[1., 2., 3.], [4., 5., 6.]].into_dyn();
        let output = layer.forward(input.view(), &NNMode::Train).unwrap();
        assert_eq!(output.shape(), &[6]);
    }

    #[test]
    fn test_flatten_backward() {
        let mut layer = Flatten::new();
        let input = array![[1., 2., 3.], [4., 5., 6.]].into_dyn();
        let output = layer.forward(input.view(), &NNMode::Train).unwrap();

        assert_eq!(output.shape(), &[6]);

        let output_gradient = layer
            .backward(output.view(), 0.1, &Optimizer::GD, &NNMode::Train)
            .unwrap();

        assert_eq!(output_gradient.shape(), &[2, 3]);
    }
}
