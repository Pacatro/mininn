//! This module contains the recorder for activations, costs and layers.

mod global_recorder;
mod recorder;

pub(crate) use global_recorder::RECORDER;

pub use recorder::Recorder;
