use crate::{
    core::NNResult,
    layers::Layer,
    utils::{ActivationFunction, CostFunction, MSGPackFormat},
};

use super::global_register::{GlobalRegister, RegisterItems, REGISTER};

pub struct Register {
    layer: Option<(String, fn(&[u8]) -> NNResult<Box<dyn Layer>>)>,
    cost: Option<(String, fn(&str) -> NNResult<Box<dyn CostFunction>>)>,
    activation: Option<(String, fn(&str) -> NNResult<Box<dyn ActivationFunction>>)>,
}

impl Register {
    pub fn new() -> Self {
        Self {
            layer: None,
            activation: None,
            cost: None,
        }
    }

    pub fn with_layer<L: Layer + MSGPackFormat>(mut self) -> Self {
        let layer_type = std::any::type_name::<L>()
            .split("::")
            .last()
            .expect("The layer type is empty")
            .to_string();
        self.layer = Some((layer_type, GlobalRegister::from_msgpack_adapter::<L>));
        self
    }

    pub fn with_activation<A: ActivationFunction>(mut self) -> Self {
        let activation_type = std::any::type_name::<A>()
            .split("::")
            .last()
            .expect("The activation type is empty")
            .to_string();
        self.activation = Some((activation_type, A::from_activation));
        self
    }

    pub fn with_cost<C: CostFunction>(mut self) -> Self {
        let cost_type = std::any::type_name::<C>()
            .split("::")
            .last()
            .expect("The cost type is empty")
            .to_string();
        self.cost = Some((cost_type, C::from_cost));
        self
    }

    pub fn register(self) {
        if let Some((name, constructor)) = self.layer {
            REGISTER.with_borrow_mut(|register| {
                register
                    .records
                    .insert(name.to_string(), RegisterItems::Layer(constructor));
            });
        }

        if let Some((name, constructor)) = self.activation {
            REGISTER.with_borrow_mut(|register| {
                register
                    .records
                    .insert(name.to_string(), RegisterItems::Activation(constructor));
            });
        }

        if let Some((name, constructor)) = self.cost {
            REGISTER.with_borrow_mut(|register| {
                register
                    .records
                    .insert(name.to_string(), RegisterItems::Cost(constructor));
            });
        }
    }
}

#[macro_export]
macro_rules! register {
    (layer: $layer_type:ty, act: $activation_type:ty, cost: $cost_type:ty) => {
        {
            let mut register = Register::new()
                .with_layer::<$layer_type>()
                .with_activation::<$activation_type>()
                .with_cost::<$cost_type>();
            register.register();
        }
    };
    (layer: $layer_type:ty, act: $activation_type:ty) => {
        {
            let mut register = Register::new()
                .with_layer::<$layer_type>()
                .with_activation::<$activation_type>();
            register.register();
        }
    };
    (layer: $layer_type:ty, cost: $cost_type:ty) => {
        {
            let mut register = Register::new()
                .with_layer::<$layer_type>()
                .with_cost::<$cost_type>();
            register.register();
        }
    };
    (layer: $layer_type:ty) => {
        {
            let mut register = Register::new().with_layer::<$layer_type>();
            register.register();
        }
    };
    (layers: $( $layer_type:ty ),* ) => {
        {
            let mut register = Register::new();
            $(
                register = register.with_layer::<$layer_type>();
            )*
            register.register();
        }
    };
    (activations: $( $activation_type:ty ),* ) => {
        {
            let mut register = Register::new();
            $(
                register = register.with_activation::<$activation_type>();
            )*
            register.register();
        }
    };
    (costs: $( $cost_type:ty ),* ) => {
        {
            let mut register = Register::new();
            $(
                register = register.with_cost::<$cost_type>();
            )*
            register.register();
        }
    };
    (layers: $( $layer_type:ty ),*, acts: $( $activation_type:ty ),*, costs: $( $cost_type:ty ),*) => {
        {
            let mut register = Register::new();
            $(
                register = register.with_layer::<$layer_type>();
            )*
            $(
                register = register.with_activation::<$activation_type>();
            )*
            $(
                register = register.with_cost::<$cost_type>();
            )*
            register.register();
        }
    };
    (layers: $( $layer_type:ty ),*, acts: $( $activation_type:ty ),*) => {
        {
            let mut register = Register::new();
            $(
                register = register.with_layer::<$layer_type>();
            )*
            $(
                register = register.with_activation::<$activation_type>();
            )*
            register.register();
        }
    };
    (layers: $( $layer_type:ty ),*, costs: $( $cost_type:ty ),*) => {
        {
            let mut register = Register::new();
            $(
                register = register.with_layer::<$layer_type>();
            )*
            $(
                register = register.with_cost::<$cost_type>();
            )*
            register.register();
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct CustomLayer;

    impl Layer for CustomLayer {
        fn layer_type(&self) -> String {
            todo!()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            todo!()
        }

        fn forward(
            &mut self,
            _input: ndarray::ArrayViewD<f64>,
            _mode: &crate::prelude::NNMode,
        ) -> NNResult<ndarray::ArrayD<f64>> {
            todo!()
        }

        fn backward(
            &mut self,
            _output_gradient: ndarray::ArrayViewD<f64>,
            _learning_rate: f64,
            _optimizer: &crate::prelude::Optimizer,
            _mode: &crate::prelude::NNMode,
        ) -> NNResult<ndarray::ArrayD<f64>> {
            todo!()
        }
    }

    impl MSGPackFormat for CustomLayer {
        fn to_msgpack(&self) -> NNResult<Vec<u8>> {
            todo!()
        }

        fn from_msgpack(_buff: &[u8]) -> NNResult<Box<Self>>
        where
            Self: Sized,
        {
            todo!()
        }
    }

    #[test]
    fn test_register() {
        let register = Register::new().with_layer::<CustomLayer>();
        assert!(register.layer.is_some());
        assert_eq!(register.layer.unwrap().0, "CustomLayer");
    }
}
