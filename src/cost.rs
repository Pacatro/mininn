use ndarray::Array1;

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
    pub(crate) fn function(&self) -> fn(prediction: &Array1<f64>, label: &Array1<f64>) -> f64 {
        match self {
            Cost::MSE => |prediction: &Array1<f64>, label: &Array1<f64>| (label - prediction).map(|x| x.powi(2)).mean().unwrap(),
        }
    }
    
    /// Returns the cost derivate
    pub(crate) fn derivate(&self) -> fn(prediction: &Array1<f64>, label: &Array1<f64>) -> Array1<f64> {
        match self {
            Cost::MSE => |prediction: &Array1<f64>, label: &Array1<f64>| 2.0*(prediction - label),
        }
    }
}
