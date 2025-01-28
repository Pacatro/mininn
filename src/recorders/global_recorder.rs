use std::{cell::RefCell, collections::HashMap};

use crate::{
    core::{MininnError, NNResult},
    layers::{Activation, Dense, Dropout, Flatten, Layer},
    utils::{Act, ActivationFunction, Cost, CostFunction},
};

thread_local!(pub(crate) static RECORDER: RefCell<GlobalRecorder> = RefCell::new(GlobalRecorder::new()));

pub(crate) enum RecorderItems {
    Layer(fn(&[u8]) -> NNResult<Box<dyn Layer>>),
    Activation(fn(&str) -> NNResult<Box<dyn ActivationFunction>>),
    Cost(fn(&str) -> NNResult<Box<dyn CostFunction>>),
}

pub(crate) struct GlobalRecorder {
    pub(crate) records: HashMap<String, RecorderItems>,
}

impl GlobalRecorder {
    pub(crate) fn new() -> Self {
        let mut records = HashMap::new();

        // Insert default layers
        records.insert(
            "Dense".to_string(),
            RecorderItems::Layer(GlobalRecorder::from_msgpack_adapter::<Dense>),
        );
        records.insert(
            "Activation".to_string(),
            RecorderItems::Layer(GlobalRecorder::from_msgpack_adapter::<Activation>),
        );
        records.insert(
            "Dropout".to_string(),
            RecorderItems::Layer(GlobalRecorder::from_msgpack_adapter::<Dropout>),
        );
        records.insert(
            "Flatten".to_string(),
            RecorderItems::Layer(GlobalRecorder::from_msgpack_adapter::<Flatten>),
        );

        // Insert default costs
        records.insert(
            "MSE".to_string(),
            RecorderItems::Cost(GlobalRecorder::from_cost_adapter::<Cost>),
        );
        records.insert(
            "MAE".to_string(),
            RecorderItems::Cost(GlobalRecorder::from_cost_adapter::<Cost>),
        );
        records.insert(
            "BCE".to_string(),
            RecorderItems::Cost(GlobalRecorder::from_cost_adapter::<Cost>),
        );
        records.insert(
            "CCE".to_string(),
            RecorderItems::Cost(GlobalRecorder::from_cost_adapter::<Cost>),
        );

        // Insert default activations
        records.insert(
            "Step".to_string(),
            RecorderItems::Activation(GlobalRecorder::from_act_adapter::<Act>),
        );
        records.insert(
            "Sigmoid".to_string(),
            RecorderItems::Activation(GlobalRecorder::from_act_adapter::<Act>),
        );
        records.insert(
            "ReLU".to_string(),
            RecorderItems::Activation(GlobalRecorder::from_act_adapter::<Act>),
        );
        records.insert(
            "Tanh".to_string(),
            RecorderItems::Activation(GlobalRecorder::from_act_adapter::<Act>),
        );
        records.insert(
            "Softmax".to_string(),
            RecorderItems::Activation(GlobalRecorder::from_act_adapter::<Act>),
        );

        Self { records }
    }

    pub(crate) fn create_layer(&self, name: &str, buff: &[u8]) -> NNResult<Box<dyn Layer>> {
        match self.records.get(name) {
            Some(RecorderItems::Layer(constructor)) => constructor(buff),
            _ => Err(MininnError::LayerRecorderError(format!(
                "Layer '{}' does not exist",
                name
            ))),
        }
    }

    pub(crate) fn create_activation(&self, name: &str) -> NNResult<Box<dyn ActivationFunction>> {
        match self.records.get(name) {
            Some(RecorderItems::Activation(constructor)) => constructor(name),
            _ => Err(MininnError::ActivationRecorderError(format!(
                "Activation '{}' does not exist",
                name
            ))),
        }
    }

    pub(crate) fn create_cost(&self, name: &str) -> NNResult<Box<dyn CostFunction>> {
        match self.records.get(name) {
            Some(RecorderItems::Cost(constructor)) => constructor(name),
            _ => Err(MininnError::CostRecorderError(format!(
                "Cost '{}' does not exist",
                name
            ))),
        }
    }

    pub(crate) fn from_msgpack_adapter<T: Layer>(buff: &[u8]) -> NNResult<Box<dyn Layer>> {
        T::from_msgpack(buff).map(|layer| layer as Box<dyn Layer>)
    }

    pub(crate) fn from_cost_adapter<T: CostFunction + 'static>(
        name: &str,
    ) -> NNResult<Box<dyn CostFunction>> {
        T::from_name(name).map(|cost| cost as Box<dyn CostFunction>)
    }

    pub(crate) fn from_act_adapter<T: ActivationFunction + 'static>(
        name: &str,
    ) -> NNResult<Box<dyn ActivationFunction>> {
        T::from_name(name).map(|act| act as Box<dyn ActivationFunction>)
    }
}
