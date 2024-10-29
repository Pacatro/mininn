use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

const DEFAULT_MOMENTUM: f64 = 0.9;

pub enum Optimizer {
    /// Gradient Descent
    GD,
    /// Momentum
    Momentum {
        factor: f64,
        weights_momentum: Option<Array2<f64>>,
        biases_momentum: Option<Array1<f64>>,
    },
    /// Adam
    Adam,
}

impl Optimizer {
    pub fn new_momentum(
        factor: Option<f64>,
        weights_momentum: Option<Array2<f64>>,
        biases_momentum: Option<Array1<f64>>
    ) -> Self {
        Optimizer::Momentum {
            factor: factor.unwrap_or(DEFAULT_MOMENTUM),
            weights_momentum: weights_momentum,
            biases_momentum: biases_momentum,
        }
    }

    pub fn optimize(
        &self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        weights_gradient: &ArrayView2<f64>,
        output_gradient: &ArrayView1<f64>,
        learning_rate: f64
    ) {
        match self {
            Optimizer::GD => {
                *weights -= &(weights_gradient * learning_rate);
                *biases -= &(output_gradient.to_owned() * learning_rate);
            }
            Optimizer::Momentum {
                factor,
                weights_momentum,
                biases_momentum,
            } => {
                // Get or initialize momentum
                todo!()
            }
            Optimizer::Adam => {
                todo!()
            }
        }
    }
}