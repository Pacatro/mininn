/// Represents the diferents activations functions for the neural network
/// 
/// ## Types
/// 
/// - `STEP`
/// - `SIGMOID`
/// - `RELU`
/// - `TANH`
/// 
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Activation {
    STEP,
    SIGMOID,
    RELU,
    TANH
}

impl Activation {
    /// Returns the function of the diferents activations
    pub fn function(&self) -> fn(x: &f64) -> f64 {
        match self {
            Activation::STEP => |x| if *x > 0.0 { 1.0 } else { 0.0 },
            Activation::SIGMOID => |x| 1.0 / (1.0 + (-x).exp()),
            Activation::RELU => |x| if *x > 0.0 { *x } else { 0.0 },
            Activation::TANH => |x| x.tanh(),
        }
    }
    
    /// Returns the derivate of the diferents activations
    pub fn derivate(&self) -> fn(x: &f64) -> f64 {
        match self {
            Activation::STEP => |_| 0.0,
            Activation::SIGMOID => |x| x * (1.0 - x),
            Activation::RELU => |x| if *x > 0.0 { 1.0 } else { 0.0 },
            Activation::TANH => |x| 1.0 - x * x,
        }
    }
}

