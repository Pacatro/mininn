//! Utility functions and types for the library.
//!
//! This module contains various utility functions and types that are used throughout the library.
//! These include activation functions, cost functions, metrics calculators, and optimizers.
//!
mod formatting;
mod metrics;
mod nn_util;

pub use formatting::MSGPackFormatting;
pub use metrics::MetricsCalculator;
pub use nn_util::NNUtil;
