use ndarray::Array1;

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
    pub(crate) fn function(&self) -> fn(z: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::STEP => |z| z.map(|x| if *x > 0.0 { 1.0 } else { 0.0 }),
            Activation::SIGMOID => |z| z.map(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::RELU => |z| z.map(|x| if *x > 0.0 { *x } else { 0.0 }),
            Activation::TANH => |z| z.map(|x| x.tanh()),
        }
    }
    
    /// Returns the derivate of the diferents activations
    pub(crate) fn derivate(&self) -> fn(z: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::STEP => |z| z.map(|_| 0.0),
            Activation::SIGMOID => |z| z.map(|x| x * (1.0 - x)),
            Activation::RELU => |z| z.map(|x| if *x > 0.0 { 1.0 } else { 0.0 }),
            Activation::TANH => |z| z.map(|x| 1.0 - x * x),
        }
    }
}

