use mininn_derive::Layer;
use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis};
use serde::{Deserialize, Serialize};

use crate::core::{MininnError, NNMode, NNResult};
use crate::layers::{Layer, TrainLayer};
use crate::utils::{MSGPackFormatting, Optimizer};

#[derive(Layer, Debug, Serialize, Deserialize, Clone)]
pub struct BatchNorm {
    input: Array2<f32>,
    gamma: Array1<f32>,
    beta: Array1<f32>,
    epsilon: f32,
    momentum: f32,
    running_mean: Array1<f32>,
    running_var: Array1<f32>,
    mu: Array1<f32>,
    xmu: Array2<f32>,
    carre: Array2<f32>,
    var: Array1<f32>,
    sqrtvar: Array1<f32>,
    invvar: Array1<f32>,
    va2: Array2<f32>,
    va3: Array2<f32>,
    xbar: Array2<f32>,
}

impl Default for BatchNorm {
    fn default() -> Self {
        Self::new(1e-5, 0.9)
    }
}

impl BatchNorm {
    #[inline]
    pub fn new(epsilon: f32, momentum: f32) -> Self {
        Self {
            input: Array2::zeros((0, 0)),
            gamma: Array1::ones(0),
            beta: Array1::zeros(0),
            epsilon,
            momentum,
            running_mean: Array1::zeros(0),
            running_var: Array1::zeros(0),
            mu: Array1::zeros(0),
            xmu: Array2::zeros((0, 0)),
            carre: Array2::zeros((0, 0)),
            var: Array1::zeros(0),
            sqrtvar: Array1::zeros(0),
            invvar: Array1::zeros(0),
            va2: Array2::zeros((0, 0)),
            va3: Array2::zeros((0, 0)),
            xbar: Array2::zeros((0, 0)),
        }
    }

    #[inline]
    pub fn with_running_mean(self, running_mean: Array1<f32>) -> Self {
        Self {
            running_mean,
            ..self
        }
    }

    #[inline]
    pub fn with_running_var(self, running_var: Array1<f32>) -> Self {
        Self {
            running_var,
            ..self
        }
    }

    #[inline]
    pub fn gamma(&self) -> Array1<f32> {
        self.gamma.to_owned()
    }

    #[inline]
    pub fn beta(&self) -> Array1<f32> {
        self.beta.to_owned()
    }

    #[inline]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    #[inline]
    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    #[inline]
    pub fn running_mean(&self) -> Array1<f32> {
        self.running_mean.to_owned()
    }

    #[inline]
    pub fn running_var(&self) -> Array1<f32> {
        self.running_var.to_owned()
    }
}

impl TrainLayer for BatchNorm {
    fn forward(&mut self, input: ArrayViewD<f32>, mode: &NNMode) -> NNResult<ArrayD<f32>> {
        self.input = input.to_owned().into_dimensionality()?;

        if self.input.is_empty() {
            return Err(MininnError::LayerError(
                "Input is empty, cannot forward pass".to_string(),
            ));
        }

        if self.gamma.is_empty() {
            return Err(MininnError::LayerError(
                "Gamma is empty, cannot forward pass".to_string(),
            ));
        }

        if self.beta.is_empty() {
            return Err(MininnError::LayerError(
                "Beta is empty, cannot forward pass".to_string(),
            ));
        }

        match mode {
            NNMode::Train => {
                self.mu = self.input.mean_axis(Axis(0)).unwrap();
                self.xmu = &self.input - &self.mu;
                self.carre = self.xmu.powi(2);
                self.var = self.carre.mean_axis(Axis(0)).unwrap();
                self.sqrtvar = (&self.var + self.epsilon).sqrt();
                self.invvar = 1.0 / &self.sqrtvar;
                self.va2 = &self.xmu * &self.invvar;
                self.va3 = &self.va2 * &self.gamma;
                let out = &self.va3 + &self.beta;
                self.running_mean =
                    self.momentum * &self.running_mean + (1.0 - self.momentum) * &self.mu;
                self.running_var =
                    self.momentum * &self.running_var + (1.0 - self.momentum) * &self.var;

                Ok(out.into_dyn())
            }
            NNMode::Test => {
                let xbar =
                    (&self.input - &self.running_mean) / (&self.running_var + self.epsilon).sqrt();

                let out = &xbar * &self.gamma + &self.beta;
                Ok(out.into_dyn())
            }
        }
    }

    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f32>,
        _learning_rate: f32,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f32>> {
        let dout: Array2<f32> = output_gradient.to_owned().into_dimensionality()?;
        let (n, _): (usize, usize) = dout.dim();

        // let dbeta = dout.sum_axis(Axis(0));
        let dva3 = dout.to_owned();

        // let dgamma = (&self.va2 * &dva3).sum_axis(Axis(0));
        let dva2 = &self.gamma * &dva3;

        let dxmu = &self.invvar * &dva2;
        let dinvvar = (&self.xmu * &dva2).sum_axis(Axis(0));

        let dsqrtvar = -1.0 / (self.sqrtvar.pow2()) * dinvvar;

        let dvar = 0.5 * (&self.var + self.epsilon).powf(-0.5) * dsqrtvar;

        let dcarre = (1.0 / n as f32) * Array2::ones(self.carre.dim()) * &dvar;
        let dxmu = dxmu + 2.0 * &self.xmu * &dcarre;

        let dx = dxmu.to_owned();
        let dmu = -dxmu.sum_axis(Axis(0));

        let dx = dx + (1.0 / n as f32) * Array2::ones(dxmu.dim()) * &dmu;

        Ok(dx.into_dyn())
    }
}
