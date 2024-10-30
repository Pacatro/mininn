use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

const DEFAULT_MOMENTUM: f64 = 0.9;
const DEFAULT_BETA1: f64 = 0.9;
const DEFAULT_BETA2: f64 = 0.999;
const DEFAULT_EPSILON: f64 = 1e-8;

pub enum Optimizer {
    /// Gradient Descent
    GD,
    /// Momentum
    Momentum(Option<f64>),
    /// Adam
    Adam,
}

pub(crate) enum OptimizerType {
    GD,
    Momentum {
        momentum: f64,
        weights_momentum: Array2<f64>,
        biases_momentum: Array1<f64>
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
    pub(crate) fn new_adam(weights_dim: (usize, usize), biases_dim: usize) -> Self {
        OptimizerType::Adam {
            beta1: DEFAULT_BETA1,
            beta2: DEFAULT_BETA2,
            epsilon: DEFAULT_EPSILON,
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
            biases_momentum: Array1::zeros(biases_dim)
        }
    }

    pub(crate) fn optimize(
        &mut self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        weights_gradient: &ArrayView2<f64>,
        output_gradient: &ArrayView1<f64>,
        learning_rate: f64
    ) {
        match self {
            OptimizerType::GD => {
                *weights -= &(weights_gradient * learning_rate);
                *biases -= &(output_gradient.to_owned() * learning_rate);
            }
            OptimizerType::Momentum { momentum, weights_momentum, biases_momentum } => {
                *weights_momentum = *momentum * &weights_momentum.view() - learning_rate * weights_gradient;
                *biases_momentum = *momentum * &biases_momentum.view() - learning_rate * output_gradient;
                
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
                *t += 1;

                *weights_m = *beta1 * &weights_m.view() + (1.0 - *beta1) * weights_gradient;
                *biases_m = *beta1 * &biases_m.view() + (1.0 - *beta1) * output_gradient;

                *weights_v = *beta2 * &weights_v.view() + (1.0 - *beta2) * weights_gradient.powi(2);
                *biases_v = *beta2 * &biases_v.view() + (1.0 - *beta2) * output_gradient.powi(2);

                let weights_m_hat = &weights_m.view() / (1.0 - (*beta1).powi(*t));
                let biases_m_hat = &biases_m.view() / (1.0 - (*beta1).powi(*t));
                let weights_v_hat = &weights_v.view() / (1.0 - (*beta2).powi(*t));
                let biases_v_hat = &biases_v.view() / (1.0 - (*beta2).powi(*t));

                *weights -= &(((weights_m_hat / weights_v_hat.sqrt()) + *epsilon) * learning_rate);  
                *biases -= &(((biases_m_hat / biases_v_hat.sqrt()) + *epsilon) * learning_rate);
            }
        }
    }
}