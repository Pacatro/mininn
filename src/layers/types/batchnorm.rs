use mininn_derive::Layer;
use ndarray::{Array1, ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

use crate::core::{NNMode, NNResult};
use crate::layers::{Layer, TrainLayer};
use crate::utils::MSGPackFormatting;

// use crate::{layers::Layer, NNMode, NNResult};

#[derive(Layer, Debug, Serialize, Deserialize, Clone)]
pub(crate) struct BatchNorm {
    input: Array1<f32>,
    gamma: Array1<f32>,
    beta: Array1<f32>,
    epsilon: f32,
    momentum: f32,
    running_mean: Array1<f32>,
    running_var: Array1<f32>,
    mu: f32,
    xmu: Array1<f32>,
    carre: Array1<f32>,
    var: f32,
    sqrtvar: f32,
    invvar: f32,
    va2: Array1<f32>,
    va3: Array1<f32>,
    xbar: Array1<f32>,
}

impl BatchNorm {
    #[inline]
    pub fn _new(
        epsilon: f32,
        momentum: f32,
        running_mean: Option<Array1<f32>>,
        running_var: Option<Array1<f32>>,
    ) -> Self {
        Self {
            input: Array1::zeros(0),
            gamma: Array1::ones(0),
            beta: Array1::zeros(0),
            epsilon,
            momentum,
            running_mean: running_mean.unwrap_or(Array1::zeros(0)),
            running_var: running_var.unwrap_or(Array1::zeros(0)),
            mu: 0.,
            xmu: Array1::zeros(0),
            carre: Array1::zeros(0),
            var: 0.,
            sqrtvar: 0.,
            invvar: 0.,
            va2: Array1::zeros(0),
            va3: Array1::zeros(0),
            xbar: Array1::zeros(0),
        }
    }

    #[inline]
    pub fn _gamma(&self) -> Array1<f32> {
        self.gamma.to_owned()
    }

    #[inline]
    pub fn _beta(&self) -> Array1<f32> {
        self.beta.to_owned()
    }

    #[inline]
    pub fn _epsilon(&self) -> f32 {
        self.epsilon
    }

    #[inline]
    pub fn _momentum(&self) -> f32 {
        self.momentum
    }

    #[inline]
    pub fn _running_mean(&self) -> Array1<f32> {
        self.running_mean.to_owned()
    }

    #[inline]
    pub fn _running_var(&self) -> Array1<f32> {
        self.running_var.to_owned()
    }
}

impl TrainLayer for BatchNorm {
    fn forward(&mut self, _input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        todo!()
    }

    fn backward(
        &mut self,
        _output_gradient: ArrayViewD<f32>,
        _learning_rate: f32,
        _optimizer: &crate::prelude::Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f32>> {
        todo!()
    }
}
