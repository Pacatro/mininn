use core::fmt;
use std::{error::Error, io};

/// Type alias for Minnin Results
pub type NNResult<T> = Result<T, MininnError>;

/// Enum representing all possible errors that can occur in the `mininn` crate.
#[derive(Debug)]
pub enum MininnError {
    /// Error that occurs during a forward or backward pass in a network layer.
    LayerError(String),
    
    /// Error related to an activation function (e.g., invalid operation or unsupported function).
    ActivationFuncError(String),
    
    /// Error that occurs while calculating the cost/loss function during training.
    CostError(String),
    
    /// Error related to the registration of a custom layer in the neural network.
    LayerRegisterError(String),
    
    /// Error that occurs while calculating performance metrics (e.g., accuracy, precision).
    MetricsError(String),
    
    /// General error related to the neural network's internal operations.
    NNError(String),
    
    /// Error related to input/output operations, typically raised during file handling or other I/O tasks.
    IoError(io::Error),
    
    /// Error that occurs during serialization or deserialization of data (e.g., JSON parsing issues).
    SerdeError(serde_json::Error),
    
    /// Error related to the shape or dimensions of a data array, often caused by mismatches between expected and actual data shapes.
    ShapeError(ndarray::ShapeError),
    
    /// Error that occurs while interacting with HDF5 files or datasets.
    HDF5Error(hdf5::Error),
    
    /// Error related to HDF5 string operations, typically involving string encoding or decoding in HDF5 files.
    StringError(hdf5::types::StringError),
}

impl Error for MininnError {}

impl fmt::Display for MininnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MininnError::LayerError(msg) => write!(f, "Layer Error: {msg}."),
            MininnError::ActivationFuncError(msg) => write!(f, "Activation Function Error: {msg}."),
            MininnError::CostError(msg) => write!(f, "Cost Function Error: {msg}."),
            MininnError::LayerRegisterError(msg) => write!(f, "Layer Registration Error: {msg}."),
            MininnError::MetricsError(msg) => write!(f, "Metrics Calculation Error: {msg}."),
            MininnError::NNError(msg) => write!(f, "Neural Network Error: {msg}."),
            MininnError::IoError(err) => write!(f, "I/O Error: {}", err),
            MininnError::SerdeError(err) => write!(f, "Serialization/Deserialization Error: {}", err),
            MininnError::ShapeError(err) => write!(f, "Shape Error: {}", err),
            MininnError::HDF5Error(err) => write!(f, "HDF5 Error: {}", err),
            MininnError::StringError(err) => write!(f, "HDF5 String Error: {}", err),
        }
    }
}

impl From<io::Error> for MininnError {
    fn from(err: io::Error) -> MininnError {
        MininnError::IoError(err)
    }
}

impl From<serde_json::Error> for MininnError {
    fn from(err: serde_json::Error) -> MininnError {
        MininnError::SerdeError(err)
    }
}

impl From<ndarray::ShapeError> for MininnError {
    fn from(err: ndarray::ShapeError) -> MininnError {
        MininnError::ShapeError(err)
    }
}

impl From<hdf5::Error> for MininnError {
    fn from(err: hdf5::Error) -> MininnError {
        MininnError::HDF5Error(err)
    }
}

impl From<hdf5::types::StringError> for MininnError {
    fn from(err: hdf5::types::StringError) -> MininnError {
        MininnError::StringError(err)
    }
}