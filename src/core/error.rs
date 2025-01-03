use core::fmt;
use std::error::Error;

/// Type alias for Minnin Results
pub type NNResult<T> = Result<T, MininnError>;

/// Enum representing all possible errors that can occur in the `mininn` crate.
#[derive(Debug)]
pub enum MininnError {
    /// Error that occurs during a forward or backward pass in a network layer.
    LayerError(String),

    /// Error related to the cost function used in a neural network layer.
    CostError(String),

    /// Error related to the activation function used in a neural network layer.
    ActivationError(String),

    /// Error related to the registration of a custom layer in the neural network.
    LayerRegisterError(String),

    /// Error related to the registration of a custom activation function in the neural network.
    ActivationRegisterError(String),

    /// Error related to the neural network's training configuration
    TrainConfigError(String),

    /// General error related to the neural network's internal operations.
    NNError(String),

    /// Error related to input/output operations, typically raised during file handling or other I/O tasks.
    IoError(String),

    /// Error that occurs during serialization of MessagePack data.
    SerializeMsgPackError(rmp_serde::encode::Error),

    /// Error that occurs during deserialization of MessagePack data.
    DeserializeMsgPackError(rmp_serde::decode::Error),

    /// Error that occurs during serialization or deserialization of data (e.g., JSON parsing issues).
    SerdeError(serde::de::value::Error),

    /// Error related to the shape or dimensions of a data array, often caused by mismatches between expected and actual data shapes.
    ShapeError(ndarray::ShapeError),

    /// Error that occurs while interacting with HDF5 files or datasets.
    HDF5Error(hdf5::Error),

    /// Error related to HDF5 string operations, typically involving string encoding or decoding in HDF5 files.
    HDF5StringError(hdf5::types::StringError),
}

impl Error for MininnError {}

impl fmt::Display for MininnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MininnError::LayerError(msg) => write!(f, "Layer Error: {msg}."),
            MininnError::ActivationError(msg) => write!(f, "Activation Error: {msg}."),
            MininnError::CostError(msg) => write!(f, "Cost Error: {msg}."),
            MininnError::LayerRegisterError(msg) => write!(f, "Layer Registration Error: {msg}."),
            MininnError::ActivationRegisterError(msg) => {
                write!(f, "Activation Registration Error: {msg}.")
            }
            MininnError::TrainConfigError(msg) => write!(f, "Train Config Error: {msg}."),
            MininnError::NNError(msg) => write!(f, "Neural Network Error: {msg}."),
            MininnError::IoError(msg) => write!(f, "I/O Error: {}.", msg),
            MininnError::SerializeMsgPackError(err) => {
                write!(f, "MessagePack Serialization Error: {}.", err)
            }
            MininnError::DeserializeMsgPackError(err) => {
                write!(f, "MessagePack Deserialization Error: {}.", err)
            }
            MininnError::SerdeError(err) => {
                write!(f, "Serialization/Deserialization Error: {}.", err)
            }
            MininnError::ShapeError(err) => write!(f, "Shape Error: {}.", err),
            MininnError::HDF5Error(err) => write!(f, "HDF5 Error: {}.", err),
            MininnError::HDF5StringError(err) => write!(f, "HDF5 String Error: {}.", err),
        }
    }
}

impl From<rmp_serde::encode::Error> for MininnError {
    fn from(err: rmp_serde::encode::Error) -> MininnError {
        MininnError::SerializeMsgPackError(err)
    }
}

impl From<rmp_serde::decode::Error> for MininnError {
    fn from(err: rmp_serde::decode::Error) -> MininnError {
        MininnError::DeserializeMsgPackError(err)
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
        MininnError::HDF5StringError(err)
    }
}

impl From<serde::de::value::Error> for MininnError {
    fn from(err: serde::de::value::Error) -> MininnError {
        MininnError::SerdeError(err)
    }
}
