use ndarray::{Array1, Array2, Array3, Array4, ArrayD, ArrayView2, ArrayViewD, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

use crate::core::{MininnError, NNMode, NNResult};
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

pub fn conv2d(
    input: ArrayView2<f32>,
    kernel: ArrayView2<f32>,
    padding: Padding,
    stride: usize,
) -> NNResult<Array2<f32>> {
    if stride == 0 {
        return Err(MininnError::LayerError(
            "Stride must be greater than 0".to_string(),
        ));
    }

    let (h, w) = input.dim();
    let (kh, kw) = kernel.dim();

    match padding {
        Padding::Valid => {
            if h < kh || w < kw {
                return Err(MininnError::LayerError(
                    "Kernel size larger than input size".to_string(),
                ));
            }

            let output_h = (h - kh) / stride + 1;
            let output_w = (w - kw) / stride + 1;
            let mut output = Array2::zeros((output_h, output_w));
            for i in 0..output_h {
                for j in 0..output_w {
                    let mut sum = 0.0;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            sum += input[[i * stride + ki, j * stride + kj]] * kernel[[ki, kj]];
                        }
                    }
                    output[[i, j]] = sum;
                }
            }
            Ok(output)
        }
        Padding::Same => todo!(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Layer)]
pub struct Conv2D {
    input: Array3<f32>,   // FORMAT: (C, H, W)
    weights: Array4<f32>, // FORMAT: (N, C, H, W)
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
        padding: Padding,
        channels: usize,
    ) -> Self {
        let scale = (2.0 / (kernel_size.0 * kernel_size.1 * channels) as f32).sqrt();

        Self {
            input: Array3::zeros((0, 0, 0)),
            weights: Array4::random(
                (nkernels, channels, kernel_size.0, kernel_size.1),
                Uniform::new(-scale, scale),
            ),
            biases: Array1::random(nkernels, Uniform::new(-scale, scale)),
            kernel_size,
            stride: 1,
            padding,
            activation: None,
        }
    }

    pub fn apply(mut self, activation: impl ActivationFunction + 'static) -> Self {
        self.activation = Some(Box::new(activation));
        self
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn set_weights(&mut self, weights: &Array4<f32>) {
        self.weights = weights.to_owned();
    }

    pub fn set_biases(&mut self, biases: &Array1<f32>) {
        self.biases = biases.to_owned();
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
        self.input = input.into_owned().into_dimensionality()?;

        let (n_kernels, _channels, kh, kw) = self.weights.dim();
        let (_, h, w) = self.input.dim();

        let padding = match self.padding {
            Padding::Valid => 0,
            Padding::Same => (kh - 1) / 2,
        };

        let h_out = (h + 2 * padding - kh) / self.stride + 1;
        let w_out = (w + 2 * padding - kw) / self.stride + 1;

        let mut output = Array3::<f32>::zeros((n_kernels, h_out, w_out));

        for (k, kernel) in self.weights.axis_iter(Axis(0)).enumerate() {
            let mut kernel_output = Array2::<f32>::zeros((h_out, w_out));

            for (c, img_channel) in self.input.axis_iter(Axis(0)).enumerate() {
                let kernel_channel = kernel.index_axis(Axis(0), c);
                kernel_output = kernel_output
                    + conv2d(
                        img_channel.view(),
                        kernel_channel.view(),
                        self.padding,
                        self.stride,
                    )?;
            }

            kernel_output += self.biases[k];
            output.index_axis_mut(Axis(0), k).assign(&kernel_output);
        }

        let output = output.into_dyn();

        match &self.activation {
            Some(activation) => Ok(activation.function(&output.view())),
            None => Ok(output),
        }
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

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::{
        core::NNMode,
        layers::{
            types::conv2d::{conv2d, Padding},
            Conv2D, TrainLayer,
        },
        utils::Act,
    };

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

        let result = conv2d(input.view(), kernel.view(), Padding::Valid, 1).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_conv2d_with_stride() {
        let input = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];

        let kernel = array![[1.0, 0.0], [0.0, -1.0]];

        let stride = 2;

        let result = conv2d(input.view(), kernel.view(), Padding::Valid, stride).unwrap();

        let expected = array![[1.0 - 6.0, 3.0 - 8.0], [9.0 - 14.0, 11.0 - 16.0]];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_conv2d_forward() {
        let input = array![[
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
        ]];

        let weights = array![[[[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]]]];
        let biases = array![0.0, 0.0];

        let mut conv = Conv2D::new(1, (3, 3), Padding::Valid, 1);

        conv.set_weights(&weights);
        conv.set_biases(&biases);

        let result = conv
            .forward(input.into_dyn().view(), &NNMode::Train)
            .unwrap()
            .into_dimensionality()
            .unwrap();

        let expected = array![[
            [0., 30., 30., 0.],
            [0., 30., 30., 0.],
            [0., 30., 30., 0.],
            [0., 30., 30., 0.],
        ]];

        assert_eq!(result, expected);
    }
}
