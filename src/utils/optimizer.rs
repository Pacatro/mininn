use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

const DEFAULT_MOMENTUM: f64 = 0.9;
const DEFAULT_BETA1: f64 = 0.9;
const DEFAULT_BETA2: f64 = 0.999;
const DEFAULT_EPSILON: f64 = 1e-8;

pub enum Optimizer {
    /// Gradient Descent optimizer
    GD,
    /// Momentum optimizer, with optional momentum (default is 0.9)
    Momentum(Option<f64>),
    /// Adam optimizer, with optional beta1 (default is 0.9), beta2 (default is 0.999) and epsilon (default is 1e-8)
    Adam(Option<f64>, Option<f64>, Option<f64>),
}

impl Optimizer {
    pub fn default_momentum() -> Self {
        Optimizer::Momentum(None)
    }

    pub fn default_adam() -> Self {
        Optimizer::Adam(None, None, None)
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::GD
    }
}

pub(crate) enum OptimizerType {
    GD,
    Momentum {
        momentum: f64,
        weights_momentum: Array2<f64>,
        biases_momentum: Array1<f64>,
    },
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
    pub(crate) fn new_adam(
        weights_dim: (usize, usize),
        biases_dim: usize,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
    ) -> Self {
        OptimizerType::Adam {
            beta1: beta1.unwrap_or(DEFAULT_BETA1),
            beta2: beta2.unwrap_or(DEFAULT_BETA2),
            epsilon: epsilon.unwrap_or(DEFAULT_EPSILON),
            weights_m: Array2::zeros(weights_dim),
            weights_v: Array2::zeros(weights_dim),
            biases_m: Array1::zeros(biases_dim),
            biases_v: Array1::zeros(biases_dim),
            t: 0,
        }
    }

    pub(crate) fn new_momentum(
        momentum: Option<f64>,
        weights_dim: (usize, usize),
        biases_dim: usize,
    ) -> Self {
        OptimizerType::Momentum {
            momentum: momentum.unwrap_or(DEFAULT_MOMENTUM),
            weights_momentum: Array2::zeros(weights_dim),
            biases_momentum: Array1::zeros(biases_dim),
        }
    }

    pub(crate) fn optimize(
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
                // TODO: Check why this is not working

                unimplemented!("Adam optimizer is not implemented yet");

                // *t += 1;

                // *weights_m = *beta1 * weights_m.to_owned() + (1.0 - *beta1) * weights_gradient;
                // *biases_m = *beta1 * biases_m.to_owned() + (1.0 - *beta1) * output_gradient;

                // *weights_v = *beta2 * weights_v.to_owned() + (1.0 - *beta2) * &(weights_gradient * weights_gradient);
                // *biases_v = *beta2 * biases_v.to_owned() + (1.0 - *beta2) * &(output_gradient * output_gradient);

                // let weights_m_hat = weights_m.to_owned() / (1.0 - beta1.powi(*t));
                // let biases_m_hat = biases_m.to_owned() / (1.0 - beta1.powi(*t));
                // let weights_v_hat = weights_v.to_owned() / (1.0 - beta2.powi(*t));
                // let biases_v_hat = biases_v.to_owned() / (1.0 - beta2.powi(*t));

                // *weights -= &((learning_rate * weights_m_hat) / (weights_v_hat.sqrt() + *epsilon));
                // *biases -= &((learning_rate * biases_m_hat) / (biases_v_hat.sqrt() + *epsilon));
            }
        }
    }
}
