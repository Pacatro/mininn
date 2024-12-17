use crate::core::NNResult;

/// Defines the methods required for serialization and deserialization for savings and loading models.
pub trait MSGPackFormatting {
    /// Serializes the object to a MesgPack bytes representation.
    fn to_msgpack(&self) -> NNResult<Vec<u8>>;

    /// Deserializes bytes into a new instance of the object.
    fn from_msgpack(buff: &[u8]) -> NNResult<Box<Self>>
    where
        Self: Sized;
}
