use ndarray::{s, Array1, Array2, Array3, Array4, ArrayD, ArrayView2, ArrayViewD, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

use crate::core::{NNMode, NNResult};
use crate::layers::{Layer, TrainLayer};
use crate::utils::{ActivationFunction, MSGPackFormatting, Optimizer};
use mininn_derive::Layer;

// TODO: REMAKE THIS WHOLE MODULE
// Links:
// https://poloclub.github.io/cnn-explainer/
// https://www.youtube.com/watch?v=pj9-rr1wDhM
// https://www.youtube.com/watch?v=KuXjwB4LzSA

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Padding {
    Valid,
    Same,
}

pub fn conv2d(input: ArrayView2<f32>, kernel: ArrayView2<f32>, padding: Padding) -> Array2<f32> {
    match padding {
        Padding::Valid => {
            let (h, w) = input.dim();
            let (kh, kw) = kernel.dim();
            let output_h = h - kh + 1;
            let output_w = w - kw + 1;
            let mut output = Array2::zeros((output_h, output_w));
            for i in 0..output_h {
                for j in 0..output_w {
                    let mut sum = 0.0;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            sum += input[[i + ki, j + kj]] * kernel[[ki, kj]];
                        }
                    }
                    output[[i, j]] = sum;
                }
            }
            output
        }
        Padding::Same => todo!(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Layer)]
pub struct Conv2D {
    input: Array3<f32>,
    weights: Array4<f32>,
    biases: Array1<f32>,
    kernel_size: (usize, usize),
    stride: usize,
    padding: Padding,
    activation: Option<Box<dyn ActivationFunction>>,
}

impl Conv2D {
    pub fn new(
        nkernels: usize,
        kernel_size: (usize, usize), // (H, W)
        stride: usize,
        padding: Padding,
        channels: usize,
    ) -> Self {
        // TODO: Xaviers initialization1
        Self {
            input: Array3::zeros((0, 0, 0)),
            weights: Array4::random(
                (nkernels, channels, kernel_size.0, kernel_size.1),
                Uniform::new(-1.0, 1.0),
            ),
            biases: Array1::random(nkernels, Uniform::new(-1.0, 1.0)),
            kernel_size,
            stride,
            padding,
            activation: None,
        }
    }

    pub fn apply(mut self, activation: impl ActivationFunction + 'static) -> Self {
        self.activation = Some(Box::new(activation));
        self
    }

    #[inline]
    pub fn activation(&self) -> Option<&dyn ActivationFunction> {
        self.activation.as_deref()
    }

    #[inline]
    pub fn padding(&self) -> Padding {
        self.padding
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }
}

impl TrainLayer for Conv2D {
    fn forward(&mut self, input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        todo!()
    }

    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f32>,
        learning_rate: f32,
        optimizer: &Optimizer,
        mode: &NNMode,
    ) -> NNResult<ArrayD<f32>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::layers::types::conv2d::{conv2d, Padding};

    #[test]
    fn test_conv2d_valid() {
        let input = array![
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
        ];

        let kernel = array![[1., 0., -1.], [1., 0., -1.], [1., 0., -1.],];

        let expected = array![
            [0., 30., 30., 0.],
            [0., 30., 30., 0.],
            [0., 30., 30., 0.],
            [0., 30., 30., 0.],
        ];

        let result = conv2d(input.view(), kernel.view(), Padding::Valid);

        assert_eq!(result, expected);
    }
}
