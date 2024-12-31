use ndarray::{s, Array1, Array2, Array3, Array4, ArrayD, ArrayView2, ArrayViewD, Axis, Ix3};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

use crate::core::{MininnError, NNMode, NNResult};
use crate::layers::{Layer, TrainLayer};
use crate::utils::{ActivationFunction, MSGPackFormatting, Optimizer};
use mininn_derive::Layer;

// TODO: REMAKE THIS WHOLE MODULE
// Links:
// https://www.youtube.com/watch?v=z9hJzduHToc&t=94s
// https://poloclub.github.io/cnn-explainer/
// https://www.youtube.com/watch?v=pj9-rr1wDhM
// https://www.youtube.com/watch?v=KuXjwB4LzSA

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Padding {
    Valid,
    Full,
}

pub fn cross_correlation2d(
    input: ArrayView2<f32>,
    kernel: ArrayView2<f32>,
    stride: usize,
    padding: Padding,
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
        Padding::Full => todo!(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Layer)]
pub struct Conv2D {
    input: Array3<f32>,   // FORMAT: (C, H, W)
    kernels: Array4<f32>, // FORMAT: (N, C, H, W)
    biases: Array3<f32>,
    kernel_size: (usize, usize),
    stride: usize,
    padding: Padding,
    activation: Option<Box<dyn ActivationFunction>>,
}

impl Conv2D {
    pub fn new(
        nkernels: usize,
        kernel_size: (usize, usize),       // (H, W)
        input_size: (usize, usize, usize), // (C, H, W)
    ) -> Self {
        let (kh, kw) = kernel_size;
        let (c, h, w) = input_size;

        let output_size = (nkernels, h - kh + 1, w - kw + 1);
        let kernels_size = (nkernels, c, kh, kw);

        let scale = (2.0 / (kh * kw * c) as f32).sqrt();

        Self {
            input: Array3::zeros((0, 0, 0)),
            kernels: Array4::random(kernels_size, Uniform::new(-scale, scale)),
            biases: Array3::random(output_size, Uniform::new(-scale, scale)),
            kernel_size,
            stride: 1,
            padding: Padding::Valid,
            activation: None,
        }
    }

    pub fn apply(mut self, activation: impl ActivationFunction + 'static) -> Self {
        self.activation = Some(Box::new(activation));
        self
    }

    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn set_kernels(&mut self, kernels: &Array4<f32>) {
        self.kernels = kernels.to_owned();
    }

    pub fn set_biases(&mut self, biases: &Array3<f32>) {
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

        let mut output = self.biases.clone();

        // Obtener dimensiones
        let (c, _, _) = self.input.dim(); // Dimensiones del input: (canales, altura, ancho)
        let (_, c_kernel, _, _) = self.kernels.dim(); // Dimensiones de los kernels

        // Validar que los canales coincidan
        if c != c_kernel {
            return Err(MininnError::LayerError(
                "Number of input channels does not match kernel channels".to_string(),
            ));
        }

        // Realizar la correlación 2D para cada kernel
        for (i, mut output_channel) in output.axis_iter_mut(Axis(0)).enumerate() {
            for (j, input_channel) in self.input.axis_iter(Axis(0)).enumerate() {
                let kernel = self.kernels.slice(s![i, j, .., ..]);
                let result = cross_correlation2d(
                    input_channel.view(),
                    kernel.view(),
                    self.stride,
                    self.padding,
                )?;
                output_channel += &result;
            }
        }

        let output = output.into_dyn();

        match &self.activation {
            Some(activation) => Ok(activation.function(&output.view())),
            None => Ok(output),
        }
    }

    fn backward(
        &mut self,
        output_gradient: ArrayViewD<f32>,
        learning_rate: f32,
        optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ArrayD<f32>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array3};

    use crate::{
        core::NNMode,
        layers::{
            types::conv2d::{cross_correlation2d, Padding},
            Conv2D, TrainLayer,
        },
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

        let result = cross_correlation2d(input.view(), kernel.view(), 1, Padding::Valid).unwrap();

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

        let result =
            cross_correlation2d(input.view(), kernel.view(), stride, Padding::Valid).unwrap();

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

        let kernels = array![[[[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]]]];
        let biases = Array3::<f32>::zeros((1, 4, 4));

        let mut conv = Conv2D::new(1, (3, 3), input.dim());

        conv.set_kernels(&kernels);
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
