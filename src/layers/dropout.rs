use mininn_derive::Layer;
use ndarray::{Array1, ArrayD, ArrayViewD};
use ndarray_rand::{rand, rand::distributions::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

use crate::{
    core::{NNMode, NNResult},
    layers::Layer,
    utils::{MSGPackFormat, Optimizer},
};

use super::layer::TrainLayer;

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
#[derive(Layer, Debug, Clone, Serialize, Deserialize)]
pub struct Dropout {
    input: Array1<f64>,
    p: f64,
    seed: u64,
    mask: Array1<f64>,
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
    pub fn new(p: f64) -> Self {
        Self {
            input: Array1::zeros(0),
            p,
            seed: rand::random(),
            mask: Array1::zeros(0),
        }
    }

    /// Sets the seed used to generate the random mask
    ///
    /// ## Arguments
    ///
    /// - `seed`: The new seed used to generate the random mask
    ///
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
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

impl Default for Dropout {
    /// Creates a new [`Dropout`] layer with `p = DEFAULT_DROPOUT_P`
    #[inline]
    fn default() -> Self {
        Self::new(DEFAULT_DROPOUT_P)
    }
}

impl TrainLayer for Dropout {
    fn forward(&mut self, input: ArrayViewD<f64>, mode: &NNMode) -> NNResult<ArrayD<f64>> {
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
            NNMode::Test => Ok(self.input.to_owned().into_dyn()),
        }
    }

    #[inline]
    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f64>,
        _learning_rate: f64,
        _optimizer: &Optimizer,
        mode: &NNMode,
    ) -> NNResult<ArrayD<f64>> {
        match mode {
            NNMode::Train => Ok(output_gradient.to_owned() * &self.mask),
            NNMode::Test => Ok(output_gradient.to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Optimizer;
    use ndarray::array;

    #[test]
    fn test_new_dropout() {
        let dropout = Dropout::new(DEFAULT_DROPOUT_P).with_seed(42);
        assert_eq!(dropout.p(), DEFAULT_DROPOUT_P);
        assert_eq!(dropout.seed(), 42);
        assert_eq!(dropout.input.len(), 0);
        assert_eq!(dropout.mask.len(), 0);
    }

    #[test]
    fn test_new_dropout_default() {
        let dropout = Dropout::default();
        assert_eq!(dropout.p(), DEFAULT_DROPOUT_P);
    }

    #[test]
    fn test_dropout_forward_pass_train() {
        let mut dropout = Dropout::new(0.5).with_seed(42);
        let input = array![1.0, 2.0, 3.0, 4.0].into_dyn();
        let output = dropout.forward(input.view(), &NNMode::Train).unwrap();

        // Validar que la máscara tenga la misma longitud que el input
        assert_eq!(dropout.mask.len(), input.len());
        // Validar que los valores de salida sean escalados correctamente
        for (o, (i, m)) in output.iter().zip(input.iter().zip(dropout.mask.iter())) {
            assert_eq!(*o, *i * *m);
        }
    }

    #[test]
    fn test_dropout_backward_pass_train() {
        let mut dropout = Dropout::new(0.5).with_seed(42);
        let input = array![1.0, 2.0, 3.0, 4.0].into_dyn();
        let output_gradient = array![0.1, 0.2, 0.3, 0.4].into_dyn();
        dropout.forward(input.view(), &NNMode::Train).unwrap();
        let backprop_output = dropout
            .backward(
                output_gradient.view(),
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
        let mut dropout = Dropout::new(0.5).with_seed(42);
        let input = array![1.0, 2.0, 3.0, 4.0].into_dyn();
        let output = dropout.forward(input.view(), &NNMode::Test).unwrap();

        assert_eq!(output, input);
    }

    #[test]
    fn test_dropout_backward_pass_test() {
        let mut dropout = Dropout::new(0.5).with_seed(42);
        let input = array![1.0, 2.0, 3.0, 4.0].into_dyn();
        let output_gradient = array![0.1, 0.2, 0.3, 0.4].into_dyn();
        dropout.forward(input.view(), &NNMode::Test).unwrap();
        let backprop_output = dropout
            .backward(
                output_gradient.view(),
                0.01,
                &Optimizer::default(),
                &NNMode::Test,
            )
            .unwrap();

        assert_eq!(backprop_output, output_gradient.into_dyn());
    }

    // #[test]
    // fn test_dropout_serialization() {
    //     let dropout = Dropout::new(DEFAULT_DROPOUT_P).with_seed(42);
    //     let json = dropout.to_json().unwrap();
    //     let deserialized: Box<dyn Layer> = Dropout::from_json(&json).unwrap();
    //     assert_eq!(dropout.layer_type(), deserialized.layer_type());
    // }

    // #[test]
    // fn test_dropout_msg_pack() {
    //     let dropout = Dropout::new(DEFAULT_DROPOUT_P).with_seed(42);
    //     let bytes = dropout.to_msgpack().unwrap();
    //     assert!(!bytes.is_empty());
    //     let deserialized: Box<dyn Layer> = Dropout::from_msgpack(&bytes).unwrap();
    //     assert_eq!(dropout.layer_type(), deserialized.layer_type());
    // }
}
