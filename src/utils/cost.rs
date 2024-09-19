use ndarray::{Array1, ArrayView1};

/// Represents the different cost functions for the neural network
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Cost {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,    
    /// Binary Cross-Entropy
    BCE,
    /// Categorical Cross-Entropy
    CCE,
}

impl Cost {
    /// Returns the cost function
    pub fn function(&self, y_p: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        match self {
            Cost::MSE => (y - y_p).map(|x| x.powi(2)).mean().unwrap(),
            Cost::MAE => (y - y_p).map(|x| x.abs()).mean().unwrap(),
            Cost::BCE => {
                -(y *   y_p.map(|x| x.ln()) + (1.0 - y) * y_p.map(|x| (1.0 - x).ln())).mean().unwrap()
            }
            Cost::CCE => {
                -y.iter()
                    .zip(y_p.iter())
                    .map(|(y_i, y_p_i)| y_i * y_p_i.ln())
                    .sum::<f64>()
                    / y.len() as f64
            }
        }
    }

    /// Returns the cost derivative
    pub fn derivate(&self, y_p: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            Cost::MSE => 2.0 * (y_p - y) / y.len() as f64,
            Cost::MAE => (y_p - y).map(|x| x.signum()) / y.len() as f64,
            Cost::BCE => (y_p - y) / (y_p * (1.0 - y_p)),
            Cost::CCE => -y / y_p,
        }
    }
}