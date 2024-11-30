use std::any::Any;

use ndarray::{Array1, ArrayD};
use ndarray_rand::{rand, rand::distributions::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

use crate::{layers::Layer, nn::NNMode, utils::Optimizer, NNResult};

/// Default probability of keeping neurons on the layer.
pub const DEFAULT_DROPOUT_P: f64 = 0.5;

/// Applies dropout regularization to a neural network layer.
///
/// Dropout is a regularization technique used to mitigate overfitting in neural networks.
/// It randomly deactivates a fraction of neurons in a layer during training, forcing the network
/// to learn redundant representations and improving its ability to generalize to unseen data.
/// Dropout is typically used in conjunction with other regularization methods, such as L2 Regularization.
///
/// ## Training Phase
/// During training, a specified fraction of neurons in the layer are randomly deactivated (set to zero).
/// This encourages the layer to distribute learning across different subsets of neurons,
/// reducing reliance on specific activations to represent patterns or features.
///
/// ## Prediction Phase
/// During inference (prediction), dropout is deactivated.
///
/// ## Considerations
/// - Using dropout may increase the training loss slightly because the network is constrained during training.
/// - It is recommended to carefully choose the dropout probability (`p`) as high values can lead to underfitting,
///   while low values might not provide sufficient regularization.
///
/// ## Attributes
///
/// - `input`: A 1D array of floating-point values representing the input data from the previous layer.
/// - `p`: The probability of retaining each neuron in the layer during training.
///         Values typically range between 0.5 and 0.8 for hidden layers.
/// - `seed`: A seed value used for generating the random dropout mask, ensuring reproducibility.
/// - `layer_type`: The type identifier for this layer, always set to `Dropout`.
///
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Dropout {
    input: Array1<f64>,
    p: f64,
    seed: u64,
    mask: Array1<f64>,
    layer_type: String,
}

impl Dropout {
    /// Creates a new [`Dropout`] layer
    ///
    /// ## Arguments
    ///
    /// - `p`: The probability of keeping neurons on the layer
    /// - `seed`: The seed used to generate the random mask
    ///
    #[inline]
    pub fn new(p: f64, seed: Option<u64>) -> Self {
        Self {
            input: Array1::zeros(0),
            p,
            seed: seed.unwrap_or_else(rand::random),
            mask: Array1::zeros(0),
            layer_type: "Dropout".to_string(),
        }
    }

    /// Returns the probability of keeping neurons on the layer
    #[inline]
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Returns the seed used to generate the random mask
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Sets the probability of keeping neurons on the layer
    ///
    /// ## Arguments
    ///
    /// - `p`: The new probability of keeping neurons on the layer
    ///
    #[inline]
    pub fn set_p(&mut self, p: f64) {
        self.p = p;
    }

    /// Sets the seed used to generate the random mask
    ///
    /// ## Arguments
    ///
    /// - `seed`: The new seed used to generate the random mask
    ///
    #[inline]
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }
}

impl Layer for Dropout {
    #[inline]
    fn layer_type(&self) -> String {
        self.layer_type.to_string()
    }

    #[inline]
    fn to_json(&self) -> NNResult<String> {
        Ok(serde_json::to_string(self)?)
    }

    #[inline]
    fn from_json(json_path: &str) -> NNResult<Box<dyn Layer>> {
        Ok(Box::new(serde_json::from_str::<Self>(json_path)?))
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward(&mut self, input: &ArrayD<f64>, mode: &NNMode) -> NNResult<ArrayD<f64>> {
        self.input = input.to_owned().into_dimensionality()?;
        match mode {
            NNMode::Train => {
                self.mask = Array1::random(input.len(), Uniform::new(0.0, 1.0)).mapv(|v| {
                    if v < self.p {
                        1.
                    } else {
                        0.
                    }
                }) / self.p;
                Ok((&self.input * &self.mask).into_dyn())
            }
            NNMode::Test => Ok((self.input.to_owned()).into_dyn()),
        }
    }

    #[inline]
    fn backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
        _learning_rate: f64,
        _optimizer: &Optimizer,
        mode: &NNMode,
    ) -> NNResult<ArrayD<f64>> {
        match mode {
            NNMode::Train => Ok(output_gradient * &self.mask),
            NNMode::Test => Ok(output_gradient.to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{layers::Layer, utils::Optimizer};
    use ndarray::array;

    #[test]
    fn test_new_dropout() {
        let dropout = Dropout::new(DEFAULT_DROPOUT_P, Some(42));
        assert_eq!(dropout.p(), DEFAULT_DROPOUT_P);
        assert_eq!(dropout.seed(), 42);
        assert_eq!(dropout.input.len(), 0);
        assert_eq!(dropout.mask.len(), 0);
    }

    #[test]
    fn test_dropout_forward_pass_train() {
        let mut dropout = Dropout::new(0.5, Some(42));
        let input = array![1.0, 2.0, 3.0, 4.0];
        let output = dropout
            .forward(&input.clone().into_dyn(), &NNMode::Train)
            .unwrap();

        // Validar que la máscara tenga la misma longitud que el input
        assert_eq!(dropout.mask.len(), input.len());
        // Validar que los valores de salida sean escalados correctamente
        for (o, (i, m)) in output.iter().zip(input.iter().zip(dropout.mask.iter())) {
            assert_eq!(*o, *i * *m);
        }
    }

    #[test]
    fn test_dropout_backward_pass_train() {
        let mut dropout = Dropout::new(0.5, Some(42));
        let input = array![1.0, 2.0, 3.0, 4.0];
        let output_gradient = array![0.1, 0.2, 0.3, 0.4];
        dropout
            .forward(&input.clone().into_dyn(), &NNMode::Train)
            .unwrap();
        let backprop_output = dropout
            .backward(
                &output_gradient.clone().into_dyn(),
                0.01,
                &Optimizer::default(),
                &NNMode::Train,
            )
            .unwrap();

        // Validar que el gradiente esté escalado correctamente
        for (bp, (og, m)) in backprop_output
            .iter()
            .zip(output_gradient.iter().zip(dropout.mask.iter()))
        {
            assert_eq!(*bp, *og * *m);
        }
    }

    #[test]
    fn test_dropout_forward_pass_test() {
        let mut dropout = Dropout::new(0.5, Some(42));
        let input = array![1.0, 2.0, 3.0, 4.0];
        let output = dropout
            .forward(&input.clone().into_dyn(), &NNMode::Test)
            .unwrap();

        assert_eq!(output, input.into_dyn());
    }

    #[test]
    fn test_dropout_backward_pass_test() {
        let mut dropout = Dropout::new(0.5, Some(42));
        let input = array![1.0, 2.0, 3.0, 4.0];
        let output_gradient = array![0.1, 0.2, 0.3, 0.4];
        dropout
            .forward(&input.clone().into_dyn(), &NNMode::Test)
            .unwrap();
        let backprop_output = dropout
            .backward(
                &output_gradient.clone().into_dyn(),
                0.01,
                &Optimizer::default(),
                &NNMode::Test,
            )
            .unwrap();

        assert_eq!(backprop_output, output_gradient.into_dyn());
    }

    #[test]
    fn test_dropout_serialization() {
        let dropout = Dropout::new(DEFAULT_DROPOUT_P, Some(42));
        let json = dropout.to_json().unwrap();
        let deserialized: Box<dyn Layer> = Dropout::from_json(&json).unwrap();
        assert_eq!(dropout.layer_type(), deserialized.layer_type());
    }
}
