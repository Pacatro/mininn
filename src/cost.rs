use ndarray::Array2;

/// Represents the diferents cost functions for the neural network
/// 
/// ## Types
/// 
/// - `MSE`
///
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Cost {
    MSE,
    // TODO: ADD MORE
}

impl Cost {
    /// Returns the cost function
    pub(crate) fn function(&self, y_p: &Array2<f64>, y: &Array2<f64>) -> f64 {
        match self {
            Cost::MSE => (y - y_p).map(|x| x.powi(2)).mean().unwrap(),
        }
    }
    
    /// Returns the cost derivate
    pub(crate) fn derivate(&self, y_p: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        match self {
            Cost::MSE => 2.0*(y_p - y) / y.len() as f64,
        }
    }
}