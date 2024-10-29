use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

const DEFAULT_MOMENTUM: f64 = 0.9;

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
    Adam,
}

impl OptimizerType {
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
            OptimizerType::Adam => {
                todo!()
            }
        }
    }
}