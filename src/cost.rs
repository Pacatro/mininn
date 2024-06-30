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
    pub fn function(&self) -> fn(prediction: &f64, label: &f64) -> f64 {
        match self {
            Cost::MSE => |prediction: &f64, label: &f64| (prediction - label).powi(2),
        }
    }
    
    /// Returns the cost derivate
    pub fn derivate(&self) -> fn(prediction: &f64, label: &f64) -> f64 {
        match self {
            Cost::MSE => |prediction: &f64, label: &f64| 2.0*(prediction - label),
        }
    }
}
