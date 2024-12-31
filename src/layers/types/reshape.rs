use ndarray::{ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

use crate::core::{NNMode, NNResult};
use crate::layers::{Layer, TrainLayer};
use crate::utils::{MSGPackFormatting, Optimizer};
use mininn_derive::Layer;

#[derive(Debug, Clone, Serialize, Deserialize, Layer)]
pub struct Reshape {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl Reshape {
    pub fn new(input_shape: impl Into<Vec<usize>>, output_shape: impl Into<Vec<usize>>) -> Self {
        Self {
            input_shape: input_shape.into(),
            output_shape: output_shape.into(),
        }
    }
}

impl TrainLayer for Reshape {
    fn forward(&mut self, input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        Ok(input
            .to_shape(self.output_shape.to_vec())?
            .to_owned()
            .into_dyn())
    }

    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f32>,
        _learning_rate: f32,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f32>> {
        Ok(output_gradient
            .to_owned()
            .to_shape(self.input_shape.to_vec())?
            .to_owned()
            .into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::{
        core::NNMode,
        layers::{types::reshape::Reshape, TrainLayer},
    };

    #[test]
    fn test_reshape() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let mut reshape = Reshape::new([2, 3], vec![3, 2]);

        let result = reshape
            .forward(input.view().into_dyn(), &NNMode::Train)
            .unwrap()
            .into_dimensionality()
            .unwrap();

        let expected = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        assert_eq!(result.dim(), expected.dim());
        assert_eq!(result, expected);
    }
}
