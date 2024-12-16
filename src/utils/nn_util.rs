use dyn_clone::DynClone;

use crate::core::NNResult;

/// Represents a utility function that can be used in neural networks.
///
/// This trait defines the essential methods required to implement a custom utility function.
/// It provides a way to create a new instance of the utility function from a string.
///
/// ## Methods
///
/// - `name`: Returns the name of the utility function.
/// - `from_name`: Creates a new instance of the utility function from a string.
///
pub trait NNUtil: DynClone {
    /// Returns the name of the util
    fn name(&self) -> &str;

    /// Creates an util from a string
    ///
    /// ## Arguments
    ///
    /// * `name`: The name of the util
    ///
    /// ## Returns
    ///
    /// A `Result` containing the util if successful, or an error if something goes wrong.
    ///
    fn from_name(name: &str) -> NNResult<Box<Self>>
    where
        Self: Sized;
}

dyn_clone::clone_trait_object!(NNUtil);
