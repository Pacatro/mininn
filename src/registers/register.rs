use std::{cell::RefCell, collections::HashMap};

use crate::{
    core::{MininnError, NNResult},
    layers::{Activation, Dense, Dropout, Flatten, Layer},
    utils::{Act, ActivationFunction, Cost, CostFunction, MSGPackFormat},
};

pub(crate) enum RegisterItems {
    Layer(fn(&[u8]) -> NNResult<Box<dyn Layer>>),
    Activation(fn(&str) -> NNResult<Box<dyn ActivationFunction>>),
    Cost(fn(&str) -> NNResult<Box<dyn CostFunction>>),
}

pub(crate) struct GlobalRegister {
    pub(crate) records: HashMap<String, RegisterItems>,
}

impl GlobalRegister {
    pub fn new() -> Self {
        // TODO: Try to improve this --> Register all layers, costs and activation meanwhile the nn is created
        let mut records = HashMap::new();

        // Insert default layers
        records.insert(
            "Dense".to_string(),
            RegisterItems::Layer(GlobalRegister::from_msgpack_adapter::<Dense>),
        );
        records.insert(
            "Activation".to_string(),
            RegisterItems::Layer(GlobalRegister::from_msgpack_adapter::<Activation>),
        );
        records.insert(
            "Dropout".to_string(),
            RegisterItems::Layer(GlobalRegister::from_msgpack_adapter::<Dropout>),
        );
        records.insert(
            "Flatten".to_string(),
            RegisterItems::Layer(GlobalRegister::from_msgpack_adapter::<Flatten>),
        );

        // Insert default costs
        records.insert("MSE".to_string(), RegisterItems::Cost(Cost::from_cost));
        records.insert("MAE".to_string(), RegisterItems::Cost(Cost::from_cost));
        records.insert("BCE".to_string(), RegisterItems::Cost(Cost::from_cost));
        records.insert("CCE".to_string(), RegisterItems::Cost(Cost::from_cost));

        // Insert default activations
        records.insert(
            "Step".to_string(),
            RegisterItems::Activation(Act::from_activation),
        );
        records.insert(
            "Sigmoid".to_string(),
            RegisterItems::Activation(Act::from_activation),
        );
        records.insert(
            "ReLU".to_string(),
            RegisterItems::Activation(Act::from_activation),
        );
        records.insert(
            "Tanh".to_string(),
            RegisterItems::Activation(Act::from_activation),
        );
        records.insert(
            "Softmax".to_string(),
            RegisterItems::Activation(Act::from_activation),
        );

        Self { records }
    }

    pub fn create_layer(&self, name: &str, buff: &[u8]) -> NNResult<Box<dyn Layer>> {
        match self.records.get(name) {
            Some(RegisterItems::Layer(constructor)) => constructor(buff),
            _ => Err(MininnError::LayerRegisterError(format!(
                "Layer '{}' does not exist",
                name
            ))),
        }
    }

    pub fn create_activation(&self, name: &str) -> NNResult<Box<dyn ActivationFunction>> {
        match self.records.get(name) {
            Some(RegisterItems::Activation(constructor)) => constructor(name),
            _ => Err(MininnError::LayerRegisterError(format!(
                "Activation '{}' does not exist",
                name
            ))),
        }
    }

    pub fn create_cost(&self, name: &str) -> NNResult<Box<dyn CostFunction>> {
        match self.records.get(name) {
            Some(RegisterItems::Cost(constructor)) => constructor(name),
            _ => Err(MininnError::LayerRegisterError(format!(
                "Cost '{}' does not exist",
                name
            ))),
        }
    }

    fn from_msgpack_adapter<T>(buff: &[u8]) -> NNResult<Box<dyn Layer>>
    where
        T: Layer + MSGPackFormat + 'static,
    {
        T::from_msgpack(buff).map(|layer| layer as Box<dyn Layer>)
    }
}

thread_local!(pub(crate) static REGISTER: RefCell<GlobalRegister> = RefCell::new(GlobalRegister::new()));

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

    pub fn with_layer<L: Layer + MSGPackFormat>(mut self, name: &str) -> Self {
        self.layer = Some((name.to_string(), GlobalRegister::from_msgpack_adapter::<L>));
        self
    }

    pub fn with_activation<A: ActivationFunction>(mut self, name: &str) -> Self {
        self.activation = Some((name.to_string(), A::from_activation));
        self
    }

    pub fn with_cost<C: CostFunction>(mut self, name: &str) -> Self {
        self.cost = Some((name.to_string(), C::from_cost));
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
