use std::{cell::RefCell, collections::HashMap};

use crate::{
    core::{MininnError, NNResult},
    layers::{Activation, Dense, Dropout, Flatten, Layer},
    utils::{Act, ActivationFunction, Cost, CostFunction, MSGPackFormat},
};

thread_local!(pub(crate) static REGISTER: RefCell<GlobalRegister> = RefCell::new(GlobalRegister::new()));

pub(crate) enum RegisterItems {
    Layer(fn(&[u8]) -> NNResult<Box<dyn Layer>>),
    Activation(fn(&str) -> NNResult<Box<dyn ActivationFunction>>),
    Cost(fn(&str) -> NNResult<Box<dyn CostFunction>>),
}

pub(crate) struct GlobalRegister {
    pub(crate) records: HashMap<String, RegisterItems>,
}

impl GlobalRegister {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn create_layer(&self, name: &str, buff: &[u8]) -> NNResult<Box<dyn Layer>> {
        match self.records.get(name) {
            Some(RegisterItems::Layer(constructor)) => constructor(buff),
            _ => Err(MininnError::LayerRegisterError(format!(
                "Layer '{}' does not exist",
                name
            ))),
        }
    }

    pub(crate) fn create_activation(&self, name: &str) -> NNResult<Box<dyn ActivationFunction>> {
        match self.records.get(name) {
            Some(RegisterItems::Activation(constructor)) => constructor(name),
            _ => Err(MininnError::LayerRegisterError(format!(
                "Activation '{}' does not exist",
                name
            ))),
        }
    }

    pub(crate) fn create_cost(&self, name: &str) -> NNResult<Box<dyn CostFunction>> {
        match self.records.get(name) {
            Some(RegisterItems::Cost(constructor)) => constructor(name),
            _ => Err(MininnError::LayerRegisterError(format!(
                "Cost '{}' does not exist",
                name
            ))),
        }
    }

    pub(crate) fn from_msgpack_adapter<T>(buff: &[u8]) -> NNResult<Box<dyn Layer>>
    where
        T: Layer + MSGPackFormat + 'static,
    {
        T::from_msgpack(buff).map(|layer| layer as Box<dyn Layer>)
    }
}
