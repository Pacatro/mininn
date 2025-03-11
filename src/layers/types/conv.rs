use ndarray::{
    s, Array, Array1, Array2, Array3, Array4, ArrayD, ArrayView1, ArrayView2, ArrayView3,
    ArrayView4, ArrayViewD, Axis, Slice,
};
use ndarray_rand::{rand::distributions::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

use crate::{
    core::{NNMode, NNResult},
    layers::{Layer, Trainable},
    utils::{ActivationFunction, MSGPackFormatting, Optimizer},
};

use mininn_derive::Layer;

/// Represents a convolutional layer in a neural network.
///
/// A `Conv` layer is a core component of neural networks where it applies a set of filters to an input image.
/// that can be trained and used for various tasks, like image classification, object detection, and image segmentation.
///
/// ## Note:
///
/// - This implementation use the `im2col` method to perform the convolution operation, which makes it more faster but consumes more memory.
/// - The format used for the input and output arrays is `(batch, channels, height, width)`.
///
/// ## Attributes
///
/// - `input`: The input to the layer as a 4D array (batch, channels, height, width).
/// - `weights`: The weights/kernels/filters of the layer as a 4D array (output channels, input channels, kernel height, kernel width).
/// - `biases`: The biases of the layer as a 1D array (output channels).
/// - `activation`: The activation function to be applied to the layer (e.g., ReLU).
/// - `stride`: The stride of the convolution operation.
/// - `padding`: The amount of padding to be applied to the input.
///
#[derive(Layer, Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Conv {
    input: Array4<f32>,
    weights: Array4<f32>,
    biases: Array1<f32>,
    activation: Option<Box<dyn ActivationFunction>>,
    stride: usize,
    padding: usize,
}

impl Conv {
    /// Creates a new [`Conv`] layer
    ///
    /// ## Arguments
    ///
    /// - `n_channels`: The number of input channels
    /// - `n_kernels`: The number of kernels in the layer
    /// - `kernel_size`: The size of the kernel (height and width)
    /// - `stride`: The stride of the convolution operation
    /// - `padding`: The amount of padding to be applied to the input
    ///
    pub fn new(
        n_channels: usize,
        n_kernels: usize,
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
    ) -> Self {
        let (kernel_h, kernel_w) = kernel_size;

        let fan_in = n_channels * kernel_h * kernel_w;
        let fan_out = n_kernels * kernel_h * kernel_w;
        let xavier = 6f32.sqrt() / ((fan_in + fan_out) as f32).sqrt();

        Self {
            weights: Array4::random(
                (n_kernels, n_channels, kernel_h, kernel_w),
                Uniform::new(-xavier, xavier),
            ),
            biases: Array1::random(n_kernels, Uniform::new(-xavier, xavier)),
            input: Array4::zeros((0, n_channels, 0, 0)),
            activation: None,
            stride,
            padding,
        }
    }

    /// Applies an activation function to the layer
    ///
    /// ## Arguments
    ///
    /// - `activation`: The activation function to be applied to the layer
    ///   (e.g., `Act::ReLU`)
    ///
    /// ## Returns
    ///
    /// A new `Conv` layer with the specified activation function
    ///
    /// ## Examples
    ///
    /// ```
    /// use mininn::prelude::*;
    ///
    /// let conv = Conv::new(3, 3, (3, 3), 1, 1).apply(Act::ReLU);
    ///
    /// assert_eq!(conv.activation().unwrap().name(), "ReLU");
    /// ```
    ///
    pub fn apply(mut self, activation: impl ActivationFunction + 'static) -> Self {
        self.activation = Some(Box::new(activation));
        self
    }

    /// Returns the number of kernels of the layer
    #[inline]
    pub fn n_kernels(&self) -> usize {
        self.weights.dim().0
    }

    /// Returns the kernel size of the layer
    pub fn kernel_size(&self) -> (usize, usize) {
        let (_, _, kernel_h, kernel_w) = self.weights.dim();
        (kernel_h, kernel_w)
    }

    /// Returns the stride of the layer
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the padding of the layer
    #[inline]
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// Returns a view of the weights of the layer
    #[inline]
    pub fn weights(&self) -> ArrayView4<f32> {
        self.weights.view()
    }

    /// Returns a view of the biases of the layer
    #[inline]
    pub fn biases(&self) -> ArrayView1<f32> {
        self.biases.view()
    }

    /// Returns the activation function of the layer if any
    #[inline]
    pub fn activation(&self) -> Option<&dyn ActivationFunction> {
        self.activation.as_deref()
    }
}

impl Trainable for Conv {
    fn forward(&mut self, input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        self.input = input.to_owned().into_dimensionality()?;

        let (n, _, in_h, in_w) = self.input.dim();
        let (f, c, k_h, k_w) = self.weights.dim();

        let h_prime = (in_h + 2 * self.padding - k_h) / self.stride + 1;
        let w_prime = (in_w + 2 * self.padding - k_w) / self.stride + 1;

        let mut out = Array4::<f32>::zeros((n, f, h_prime, w_prime));

        for im_num in 0..n {
            let im = input.slice(s![im_num, .., .., ..]);
            let pad_config = vec![
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
            ];
            let im_pad = pad(im.to_owned(), pad_config, 0.0);
            let im_col = im2col(im_pad.view(), k_h, k_w, self.stride);
            let filter_col = self.weights.to_shape((f, c * k_h * k_w))?;
            let mul = im_col.dot(&filter_col.t()) + self.biases.view();
            let col_im = col2im_2d(mul.view(), h_prime, w_prime)?;

            out.slice_mut(s![im_num, .., .., ..]).assign(&col_im);
        }

        let output = out.into_dyn();

        match &self.activation {
            Some(activation) => Ok(activation.function(&output.view())),
            None => Ok(output),
        }
    }

    fn backward(
        &mut self,
        output_gradient: ndarray::ArrayViewD<f32>,
        learning_rate: f32,
        optimizer: &Optimizer,
        mode: &NNMode,
    ) -> NNResult<ndarray::ArrayD<f32>> {
        todo!()
    }
}

fn im2col(img: ArrayView3<f32>, filter_h: usize, filter_w: usize, stride: usize) -> Array2<f32> {
    let (c, h, w) = img.dim();
    let new_h = (h - filter_h) / stride + 1;
    let new_w = (w - filter_w) / stride + 1;

    let mut col: Array2<f32> = Array2::zeros((new_h * new_w, c * filter_h * filter_w));

    for i in 0..new_h {
        for j in 0..new_w {
            let patch = img.slice(s![
                ..,
                i * stride..i * stride + filter_h,
                j * stride..j * stride + filter_w,
            ]);
            // The flatten() function use extra memory, maybe should be a better option
            let flatten_patch = patch.flatten();
            col.slice_mut(s![i * new_w + j, ..]).assign(&flatten_patch);
        }
    }

    col
}

fn col2im_2d(mul: ArrayView2<f32>, h_prime: usize, w_prime: usize) -> NNResult<Array3<f32>> {
    let f = mul.shape()[1];
    let mut out = Array3::zeros((f, h_prime, w_prime));

    for i in 0..f {
        let col = mul.slice(s![.., i]);
        let new_out = col.to_shape((h_prime, w_prime))?;
        out.slice_mut(s![i, .., ..]).assign(&new_out);
    }

    Ok(out)
}

fn col2im_3d(
    mul: ArrayView2<f32>,
    h_prime: usize,
    w_prime: usize,
    c: usize,
) -> NNResult<Array4<f32>> {
    let f = mul.shape()[1];
    let mut out = Array4::zeros((f, c, h_prime, w_prime));

    for i in 0..f {
        let col = mul.slice(s![.., i]);
        let reshaped_col = col.to_shape((c, h_prime, w_prime))?;
        out.slice_mut(s![i, .., .., ..]).assign(&reshaped_col);
    }

    Ok(out)
}

/// Pads an image with a specified value
///
/// ## Parameters
///
/// - `img`: The input image to be padded
/// - `pad_with`: A vector of tuples representing the padding configuration
/// - `value`: The value to be used for padding
///
/// ## Returns
///
/// The padded image
///
fn pad(img: Array3<f32>, pad_with: Vec<[usize; 2]>, value: f32) -> Array3<f32> {
    assert_eq!(
        img.ndim(),
        pad_with.len(),
        "Array ndim must match length of `pad_with`."
    );

    let mut padded_shape = img.raw_dim();

    for (ax, (&ax_len, &[pad_lo, pad_hi])) in img.shape().iter().zip(&pad_with).enumerate() {
        padded_shape[ax] = ax_len + pad_lo + pad_hi;
    }

    let mut padded = Array::from_elem(padded_shape, value);
    let padded_dim = padded.raw_dim();
    {
        let mut orig_portion = padded.view_mut();
        for (ax, &[pad_lo, pad_hi]) in pad_with.iter().enumerate() {
            orig_portion.slice_axis_inplace(
                Axis(ax),
                Slice::from(pad_lo as isize..padded_dim[ax] as isize - (pad_hi as isize)),
            );
        }
        orig_portion.assign(&img);
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3, Array4};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    #[test]
    fn test_im2col() {
        let img = Array3::random((3, 10, 10), Uniform::new(-1., 1.));
        let col = im2col(img.view(), 3, 3, 1);
        assert_eq!(col.dim(), (64, 27));
    }

    #[test]
    fn test_col2im_2d() {
        let mul = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0]
        ];
        let expected = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
        ];
        let result = col2im_2d(mul.view(), 2, 3).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_col2im_3d() {
        let mul = array![
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
            [3.0, 13.0, 23.0],
            [4.0, 14.0, 24.0],
            [5.0, 15.0, 25.0],
            [6.0, 16.0, 26.0],
            [7.0, 17.0, 27.0],
            [8.0, 18.0, 28.0]
        ];

        let expected = Array4::from_shape_vec(
            (3, 2, 2, 2),
            vec![
                // f = 0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // f = 1
                11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, // f = 2
                21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
            ],
        )
        .unwrap();

        let result = col2im_3d(mul.view(), 2, 2, 2).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pad() {
        let img: Array3<f32> = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];

        let pad_with = vec![[1, 1], [1, 2], [0, 0]];
        let pad_value = 0.0;

        let padded = pad(img.clone(), pad_with.clone(), pad_value);

        let expected_shape = [4, 5, 2];
        assert_eq!(padded.shape(), &expected_shape);

        let inner = padded.slice(s![1..3, 1..3, ..]);
        assert_eq!(inner, img.view());

        assert_eq!(padded[[0, 1, 1]], pad_value);
    }

    #[test]
    fn test_conv_forward() {
        let mut conv = Conv::new(1, 1, (3, 3), 1, 0);

        conv.weights.fill(1.0);
        conv.biases.fill(1.0);

        let input = Array::from_elem((1, 1, 4, 4), 1.0).into_dyn();
        let output = conv.forward(input.view(), &NNMode::Train).unwrap();
        let expected = Array::from_elem((1, 1, 2, 2), 10.0).into_dyn();

        assert_eq!(output, expected);
    }
}
