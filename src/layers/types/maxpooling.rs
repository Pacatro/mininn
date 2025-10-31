use ndarray::{s, Array4, ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

use crate::{
    core::{NNMode, NNResult, Optimizer},
    layers::{Layer, Trainable},
    utils::MSGPackFormatting,
};
use mininn_derive::Layer;

#[derive(Layer, Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct MaxPooling {
    input: Array4<f32>,
    pool_size: (usize, usize),
    stride: usize,
}

impl MaxPooling {
    pub fn new(pool_size: (usize, usize), stride: usize) -> Self {
        Self {
            input: Array4::zeros((0, 0, 0, 0)),
            pool_size,
            stride,
        }
    }

    pub fn pool_size(&self) -> (usize, usize) {
        self.pool_size
    }

    pub fn stride(&self) -> usize {
        self.stride
    }
}

impl Trainable for MaxPooling {
    fn forward(&mut self, input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        self.input = input.to_owned().into_dimensionality()?;

        let (n, c, h, w) = self.input.dim();
        let (h_p, w_p) = self.pool_size;
        let out_h = 1 + (h - h_p) / self.stride;
        let out_w = 1 + (w - w_p) / self.stride;

        let mut out = Array4::<f32>::zeros((n, c, out_h, out_w));
        for i in 0..n {
            for j in 0..c {
                for r in 0..out_h {
                    let rs = r * self.stride;
                    let re = rs + h_p;
                    for col in 0..out_w {
                        let cs = col * self.stride;
                        let ce = cs + w_p;

                        let window = self.input.slice(s![i, j, rs..re, cs..ce]);

                        let max_val = window.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));

                        out[[i, j, r, col]] = max_val;
                    }
                }
            }
        }

        Ok(out.into_dyn())
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
