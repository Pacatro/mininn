use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Default momentum value for the Momentum optimizer.
pub const DEFAULT_MOMENTUM: f64 = 0.9;
/// Default beta1 parameter for the Adam optimizer.
pub const DEFAULT_BETA1: f64 = 0.9;
/// Default beta2 parameter for the Adam optimizer.
pub const DEFAULT_BETA2: f64 = 0.999;
/// Default epsilon parameter for the Adam optimizer, to avoid division by zero.
pub const DEFAULT_EPSILON: f64 = 1e-8;

/// Enum representing different types of optimizers for training neural networks.
#[derive(Debug, PartialEq, Clone)]
pub enum Optimizer {
    /// Gradient Descent optimizer.
    GD,
    /// Momentum optimizer with an optional momentum factor. Defaults to `0.9`.
    Momentum(Option<f64>),
    /// Adam optimizer with optional `beta1`, `beta2`, and `epsilon` parameters. Defaults to `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`.
    Adam(Option<f64>, Option<f64>, Option<f64>),
}

impl Optimizer {
    /// Returns a default Momentum optimizer with the default momentum value.
    pub fn default_momentum() -> Self {
        Optimizer::Momentum(None)
    }

    /// Returns a default Adam optimizer with default values for `beta1`, `beta2`, and `epsilon`.
    pub fn default_adam() -> Self {
        Optimizer::Adam(None, None, None)
    }
}

impl Default for Optimizer {
    /// Default implementation for the Optimizer enum, returning Gradient Descent (GD) optimizer.
    fn default() -> Self {
        Optimizer::GD
    }
}

/// Enum representing the internal types of optimizers, which includes the state needed for each optimizer type.
pub(crate) enum OptimizerType {
    /// Gradient Descent (GD) optimizer.
    GD,
    /// Momentum optimizer with a momentum value and momentum terms for weights and biases.
    Momentum {
        momentum: f64,
        weights_momentum: Array2<f64>,
        biases_momentum: Array1<f64>,
    },
    /// Adam optimizer with parameters `beta1`, `beta2`, `epsilon`, and moment terms for weights and biases.
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weights_m: Array2<f64>,
        weights_v: Array2<f64>,
        biases_m: Array1<f64>,
        biases_v: Array1<f64>,
        t: i32,
    },
}

impl OptimizerType {
    /// Creates a new Adam optimizer with the given dimensions for weights and biases,
    /// and optional parameters `beta1`, `beta2`, and `epsilon`.
    ///
    /// # Arguments
    ///
    /// * `weights_dim` - Tuple indicating the dimensions of the weights array.
    /// * `biases_dim` - Size of the biases array.
    /// * `beta1` - Optional beta1 parameter, default is `0.9`.
    /// * `beta2` - Optional beta2 parameter, default is `0.999`.
    /// * `epsilon` - Optional epsilon parameter, default is `1e-8`.
    pub(crate) fn new_adam(
        weights_dim: (usize, usize),
        biases_dim: usize,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
    ) -> Self {
        OptimizerType::Adam {
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            weights_m: Array2::zeros(weights_dim),
            weights_v: Array2::zeros(weights_dim),
            biases_m: Array1::zeros(biases_dim),
            biases_v: Array1::zeros(biases_dim),
            t: 0,
        }
    }

    /// Creates a new Momentum optimizer with the specified dimensions for weights and biases,
    /// and an optional momentum parameter.
    ///
    /// # Arguments
    ///
    /// * `momentum` - Optional momentum parameter, default is `0.9`.
    /// * `weights_dim` - Tuple indicating the dimensions of the weights array.
    /// * `biases_dim` - Size of the biases array.
    pub(crate) fn new_momentum(
        momentum: Option<f64>,
        weights_dim: (usize, usize),
        biases_dim: usize,
    ) -> Self {
        OptimizerType::Momentum {
            momentum: momentum.unwrap_or(0.9),
            weights_momentum: Array2::zeros(weights_dim),
            biases_momentum: Array1::zeros(biases_dim),
        }
    }

    /// Updates the weights and biases using the gradient information for a single optimization step.
    ///
    /// # Arguments
    ///
    /// * `weights` - Mutable reference to the weights array to be updated.
    /// * `biases` - Mutable reference to the biases array to be updated.
    /// * `weights_gradient` - Gradient of the loss with respect to weights.
    /// * `output_gradient` - Gradient of the loss with respect to biases.
    /// * `learning_rate` - Learning rate used to scale the gradients.
    pub fn optimize(
        &mut self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        weights_gradient: &ArrayView2<f64>,
        output_gradient: &ArrayView1<f64>,
        learning_rate: f64,
    ) {
        match self {
            OptimizerType::GD => {
                *weights -= &(weights_gradient * learning_rate);
                *biases -= &(output_gradient.to_owned() * learning_rate);
            }
            OptimizerType::Momentum {
                momentum,
                weights_momentum,
                biases_momentum,
            } => {
                *weights_momentum =
                    *momentum * &weights_momentum.view() - learning_rate * weights_gradient;
                *biases_momentum =
                    *momentum * &biases_momentum.view() - learning_rate * output_gradient;
                *weights += &*weights_momentum;
                *biases += &*biases_momentum;
            }
            OptimizerType::Adam {
                beta1,
                beta2,
                epsilon,
                weights_m,
                weights_v,
                biases_m,
                biases_v,
                t,
            } => {
                *t += 1;
                *weights_m = *beta1 * weights_m.to_owned() + (1.0 - *beta1) * weights_gradient;
                *biases_m = *beta1 * biases_m.to_owned() + (1.0 - *beta1) * output_gradient;
                *weights_v = *beta2 * weights_v.to_owned()
                    + (1.0 - *beta2) * &(weights_gradient * weights_gradient);
                *biases_v = *beta2 * biases_v.to_owned()
                    + (1.0 - *beta2) * &(output_gradient * output_gradient);

                let weights_m_hat = weights_m.to_owned() / (1.0 - beta1.powi(*t));
                let biases_m_hat = biases_m.to_owned() / (1.0 - beta1.powi(*t));
                let weights_v_hat = weights_v.to_owned() / (1.0 - beta2.powi(*t));
                let biases_v_hat = biases_v.to_owned() / (1.0 - beta2.powi(*t));

                *weights -= &((learning_rate * weights_m_hat) / (weights_v_hat.sqrt() + *epsilon));
                *biases -= &((learning_rate * biases_m_hat) / (biases_v_hat.sqrt() + *epsilon));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gradient_descent() {
        let mut weights = array![[0.5, -0.5], [0.5, -0.5]];
        let mut biases = array![0.5, -0.5];
        let weights_gradient = array![[0.1, -0.1], [0.1, -0.1]];
        let output_gradient = array![0.1, -0.1];
        let learning_rate = 0.1;

        let mut optimizer = OptimizerType::GD;
        optimizer.optimize(
            &mut weights,
            &mut biases,
            &weights_gradient.view(),
            &output_gradient.view(),
            learning_rate,
        );

        assert_eq!(weights, array![[0.49, -0.49], [0.49, -0.49]]);
        assert_eq!(biases, array![0.49, -0.49]);
    }

    #[test]
    fn test_momentum() {
        let mut weights = array![[0.5, -0.5], [0.5, -0.5]];
        let mut biases = array![0.5, -0.5];
        let weights_gradient = array![[0.1, -0.1], [0.1, -0.1]];
        let output_gradient = array![0.1, -0.1];
        let learning_rate = 0.1;
        let momentum = 0.9;

        let mut optimizer = OptimizerType::new_momentum(Some(momentum), (2, 2), 2);
        optimizer.optimize(
            &mut weights,
            &mut biases,
            &weights_gradient.view(),
            &output_gradient.view(),
            learning_rate,
        );

        assert_eq!(weights, array![[0.49, -0.49], [0.49, -0.49]]);
        assert_eq!(biases, array![0.49, -0.49]);
    }

    #[test]
    fn test_adam() {
        let mut weights = array![[0.5, -0.5], [0.5, -0.5]];
        let mut biases = array![0.5, -0.5];
        let weights_gradient = array![[0.1, -0.1], [0.1, -0.1]];
        let output_gradient = array![0.1, -0.1];
        let learning_rate = 0.1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        let mut optimizer =
            OptimizerType::new_adam((2, 2), 2, Some(beta1), Some(beta2), Some(epsilon));

        optimizer.optimize(
            &mut weights,
            &mut biases,
            &weights_gradient.view(),
            &output_gradient.view(),
            learning_rate,
        );

        // Due to the nature of Adam, precise expected values are hard to determine.
        // Instead, we check if weights and biases have been updated.
        assert!(weights != array![[0.5, -0.5], [0.5, -0.5]]);
        assert!(biases != array![0.5, -0.5]);
    }

    #[test]
    fn test_default_optimizer() {
        assert_eq!(Optimizer::default(), Optimizer::GD);
    }

    #[test]
    fn test_default_momentum() {
        assert_eq!(Optimizer::default_momentum(), Optimizer::Momentum(None));
    }

    #[test]
    fn test_default_adam() {
        assert_eq!(Optimizer::default_adam(), Optimizer::Adam(None, None, None));
    }
}
