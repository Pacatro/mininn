use crate::{
    core::NNResult,
    layers::Layer,
    utils::{ActivationFunction, CostFunction},
};

use super::global_register::{GlobalRegister, RegisterItems, REGISTER};

/// Represents a registry for layers, cost functions, and activation functions.
///
/// The `Register` struct allows you to dynamically register layers, cost functions,
/// and activation functions for use in your neural network models.
/// Each component is stored as a vector of optional tuples, where each tuple
/// contains the name of the component and a function pointer to construct it.
#[derive(Debug)]
pub struct Register {
    layers: Vec<Option<(String, fn(&[u8]) -> NNResult<Box<dyn Layer>>)>>,
    costs: Vec<Option<(String, fn(&str) -> NNResult<Box<dyn CostFunction>>)>>,
    activations: Vec<Option<(String, fn(&str) -> NNResult<Box<dyn ActivationFunction>>)>>,
}

impl Register {
    /// Creates a new, empty `Register`.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            activations: Vec::new(),
            costs: Vec::new(),
        }
    }

    /// Registers a new layer type with the `Register`.
    ///
    /// # Type Parameters
    /// - `L`: A type that implements the `Layer` trait.
    ///
    /// # Returns
    /// - The updated `Register` instance.
    ///
    pub fn with_layer<L: Layer>(mut self) -> Self {
        let layer_type = std::any::type_name::<L>()
            .split("::")
            .last()
            .expect("The layer type is empty")
            .to_string();
        self.layers.push(Some((
            layer_type,
            GlobalRegister::from_msgpack_adapter::<L>,
        )));
        self
    }

    /// Registers a new activation function with the `Register`.
    ///
    /// # Type Parameters
    /// - `A`: A type that implements the `ActivationFunction` trait.
    ///
    /// # Returns
    /// - The updated `Register` instance.
    ///
    pub fn with_activation<A: ActivationFunction + 'static>(mut self) -> Self {
        let activation_type = std::any::type_name::<A>()
            .split("::")
            .last()
            .expect("The activation type is empty")
            .to_string();
        self.activations.push(Some((
            activation_type,
            GlobalRegister::from_act_adapter::<A>,
        )));
        self
    }

    /// Registers a new cost function with the `Register`.
    ///
    /// # Type Parameters
    /// - `C`: A type that implements the `CostFunction` trait.
    ///
    /// # Returns
    /// - The updated `Register` instance.
    ///
    pub fn with_cost<C: CostFunction + 'static>(mut self) -> Self {
        let cost_type = std::any::type_name::<C>()
            .split("::")
            .last()
            .expect("The cost type is empty")
            .to_string();
        self.costs
            .push(Some((cost_type, GlobalRegister::from_cost_adapter::<C>)));
        self
    }

    /// Finalizes the registration process by adding all registered components to the global registry.
    ///
    /// This method iterates over the stored layers, activations, and cost functions, and adds
    /// them to the global `REGISTER`.
    ///
    pub fn register(self) {
        for layer in self.layers {
            if let Some((name, constructor)) = layer {
                REGISTER.with_borrow_mut(|register| {
                    register
                        .records
                        .insert(name.to_string(), RegisterItems::Layer(constructor));
                });
            }
        }

        for activation in self.activations {
            if let Some((name, constructor)) = activation {
                REGISTER.with_borrow_mut(|register| {
                    register
                        .records
                        .insert(name.to_string(), RegisterItems::Activation(constructor));
                });
            }
        }

        for cost in self.costs {
            if let Some((name, constructor)) = cost {
                REGISTER.with_borrow_mut(|register| {
                    register
                        .records
                        .insert(name.to_string(), RegisterItems::Cost(constructor));
                });
            }
        }
    }
}

/// Macro to register your own layers, activations and costs
#[macro_export]
macro_rules! register {
    (
        $(layers: [$( $layer_type:ty ),* ])?$(,)?
        $(acts: [$( $activation_type:ty ),* ])?$(,)?
        $(costs: [$( $cost_type:ty ),* ])?$(,)?
    ) => {{
        let mut register = Register::new();

        $(
            $(
                register = register.with_layer::<$layer_type>();
            )*
        )?

        $(
            $(
                register = register.with_activation::<$activation_type>();
            )*
        )?

        $(
            $(
                register = register.with_cost::<$cost_type>();
            )*
        )?

        register.register();
    }};
}

#[cfg(test)]
mod tests {
    use crate::utils::MSGPackFormatting;
    use mininn_derive::Layer;
    use ndarray::{ArrayD, ArrayViewD};
    use serde::{Deserialize, Serialize};

    use crate::{core::NNMode, layers::TrainLayer, utils::Optimizer};

    use super::*;

    #[derive(Layer, Debug, Clone, Serialize, Deserialize)]
    struct CustomLayer;

    impl TrainLayer for CustomLayer {
        fn forward(&mut self, _input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
            todo!()
        }
        fn backward(
            &mut self,
            _output_gradient: ArrayViewD<f32>,
            _learning_rate: f32,
            _optimizer: &Optimizer,
            _mode: &NNMode,
        ) -> NNResult<ArrayD<f32>> {
            todo!()
        }
    }

    #[test]
    fn test_register() {
        let register = Register::new().with_layer::<CustomLayer>();
        assert!(!register.layers.is_empty());
        let layer = register.layers.first().unwrap();
        assert!(layer.is_some());
        assert_eq!(layer.as_ref().unwrap().0, "CustomLayer");
    }
}
