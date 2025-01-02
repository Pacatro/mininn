use ndarray::{s, Array3, Array4, ArrayD, ArrayView3, ArrayView4, ArrayViewD, Axis, Ix3};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

use crate::core::{MininnError, NNMode, NNResult};
use crate::layers::{Layer, TrainLayer};
use crate::utils::{
    cross_correlation2d, ActivationFunction, MSGPackFormatting, Optimizer, Padding,
};
use mininn_derive::Layer;

// TODO: REMAKE THIS WHOLE MODULE
// Links:
// https://www.youtube.com/watch?v=z9hJzduHToc&t=94s
// https://poloclub.github.io/cnn-explainer/
// https://www.youtube.com/watch?v=pj9-rr1wDhM
// https://www.youtube.com/watch?v=KuXjwB4LzSA

#[derive(Debug, Clone, Serialize, Deserialize, Layer)]
pub struct Conv2D {
    input: Array3<f32>,   // FORMAT: (C, H, W)
    kernels: Array4<f32>, // FORMAT: (N, C, H, W)
    biases: Array3<f32>,
    nkernels: usize,                     // Number of output channels/filters
    input_depth: usize,                  // Number of input channels
    input_shape: (usize, usize, usize),  // (C, H, W)
    output_shape: (usize, usize, usize), // (N, H, W)
    kernel_size: usize,                  // Assuming square kernel for simplicity
    stride: usize,
    activation: Option<Box<dyn ActivationFunction>>,
}

impl Conv2D {
    pub fn new(nkernels: usize, kernel_size: usize, input_shape: (usize, usize, usize)) -> Self {
        let (input_depth, input_height, input_width) = input_shape;
        let output_height = input_height - kernel_size + 1;
        let output_width = input_width - kernel_size + 1;
        let output_shape = (nkernels, output_height, output_width);
        let kernels_shape = (nkernels, input_depth, kernel_size, kernel_size);

        // Initialize using Xavier/Glorot initialization
        let fan_in = input_depth * kernel_size * kernel_size;
        let scale = (2.0 / fan_in as f32).sqrt();

        Self {
            input: Array3::zeros((0, 0, 0)),
            kernels: Array4::random(kernels_shape, Uniform::new(-scale, scale)),
            biases: Array3::zeros(output_shape),
            nkernels,
            input_depth,
            input_shape,
            output_shape,
            kernel_size,
            stride: 1,
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

    pub fn output_shape(&self) -> (usize, usize, usize) {
        self.output_shape
    }

    pub fn kernels_shape(&self) -> (usize, usize, usize, usize) {
        (
            self.nkernels,
            self.input_depth,
            self.kernel_size,
            self.kernel_size,
        )
    }

    pub fn nkernels(&self) -> usize {
        self.nkernels
    }

    pub fn set_biases(&mut self, biases: ArrayView3<f32>) {
        self.biases = biases.to_owned();
    }

    pub fn set_kernels(&mut self, kernels: ArrayView4<f32>) {
        self.kernels = kernels.to_owned();
    }
}

impl TrainLayer for Conv2D {
    fn forward(&mut self, input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        let input = input.to_slice().unwrap().to_vec();
        self.input = Array3::from_shape_vec(self.input_shape, input)?;

        // Validate input shape
        if self.input.shape() != [self.input_depth, self.input_shape.1, self.input_shape.2] {
            return Err(MininnError::LayerError(format!(
                "Expected input shape {:?}, got {:?}",
                self.input_shape,
                self.input.shape()
            )));
        }

        let mut output = Array3::zeros(self.output_shape);

        for i in 0..self.nkernels {
            for j in 0..self.input_depth {
                let corr = cross_correlation2d(
                    self.input.slice(s![j, .., ..]),
                    self.kernels.slice(s![i, j, .., ..]),
                    self.stride,
                    Padding::Valid,
                )?;

                output.index_axis_mut(Axis(0), i).assign(&corr);
            }
        }

        output += &self.biases;

        Ok(output.into_dyn())
    }

    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f32>,
        learning_rate: f32,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f32>> {
        let output_gradient = output_gradient.into_owned().into_dimensionality::<Ix3>()?;
        let mut kernels_gradient = Array4::zeros(self.kernels_shape());
        let mut input_gradient = Array3::zeros(self.input_shape);

        for i in 0..self.nkernels {
            for j in 0..self.input_depth {
                let kernel_corr = cross_correlation2d(
                    self.input.slice(s![j, .., ..]),
                    output_gradient.slice(s![i, .., ..]),
                    self.stride,
                    Padding::Valid,
                )?;

                let input_corr = cross_correlation2d(
                    output_gradient.slice(s![i, .., ..]),
                    self.kernels.slice(s![i, j, .., ..]),
                    self.stride,
                    Padding::Full,
                )?;

                kernels_gradient
                    .slice_mut(s![i, j, .., ..])
                    .assign(&kernel_corr);

                input_gradient
                    .slice_mut(s![j, .., ..])
                    .zip_mut_with(&input_corr, |x, &y| *x += y);
            }
        }

        // Update parameters
        self.kernels -= &(&kernels_gradient * learning_rate);
        self.biases -= &(&output_gradient * learning_rate);

        Ok(input_gradient.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array3};

    use crate::{
        core::NNMode,
        layers::{Conv2D, TrainLayer},
    };

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

        let kernels = array![[[[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]]]];
        let biases = Array3::<f32>::zeros((1, 4, 4));

        let mut conv = Conv2D::new(1, 3, input.dim());

        conv.set_kernels(kernels.view());
        conv.set_biases(biases.view());

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
