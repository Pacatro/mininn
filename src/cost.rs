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
    pub fn function(&self) -> fn(labels: Array1<f64>, predictions: Array1<f64>) -> f64 {
        match self {
            Cost::MSE => |labels: Array1<f64>, predictions: Array1<f64>| (labels - predictions).map(|elem| elem.powi(2)).mean().unwrap(),
        }
    }
    
    /// Returns the cost derivate
    pub fn derivate(&self) -> fn(labels: Array1<f64>, predictions: Array1<f64>) -> f64 {
        match self {
            Cost::MSE => |labels: Array1<f64>, predictions: Array1<f64>| (2.0*(labels - predictions)).mean().unwrap(),
        }
    }
}
