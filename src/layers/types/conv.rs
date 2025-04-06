use ndarray::{
    s, Array, Array1, Array2, Array3, Array4, ArrayD, ArrayView1, ArrayView2, ArrayView3,
    ArrayView4, ArrayViewD, Axis, Ix4, Slice,
};
use ndarray_rand::{rand::distributions::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

use crate::{
    core::{MininnError, NNMode, NNResult},
    layers::{Layer, Trainable},
    utils::{ActivationFunction, MSGPackFormatting, Optimizer},
};

use mininn_derive::Layer;

/// Represents a convolutional layer in a neural network.
///
/// A `Conv` layer is a core component of neural networks that applies a set of filters to an input image,
/// which can be trained and used for various tasks such as image classification, object detection, and image segmentation.
///
/// ## Note:
///
/// - This implementation uses the `im2col` method to perform the convolution operation, which makes it faster but consumes more memory.
/// - The format used for the input and output arrays is `(batch, channels, height, width)`.
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
    /// - `n_channels`: The number of input channels.
    /// - `n_kernels`: The number of kernels in the layer.
    /// - `kernel_size`: The size of the kernel (height and width).
    /// - `stride`: The stride of the convolution operation.
    /// - `padding`: The amount of padding to be applied to the input.
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
        let xavier = (6.0f32).sqrt() / ((fan_in + fan_out) as f32).sqrt();

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

    /// Applies an activation function to the layer.
    ///
    /// ## Arguments
    ///
    /// - `activation`: The activation function to be applied to the layer (e.g., `Act::ReLU`).
    ///
    /// ## Returns
    ///
    /// A new `Conv` layer with the specified activation function.
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

    /// Returns the number of kernels of the layer.
    #[inline]
    pub fn n_kernels(&self) -> usize {
        self.weights.dim().0
    }

    /// Returns the kernel size of the layer.
    pub fn kernel_size(&self) -> (usize, usize) {
        let (_, _, kernel_h, kernel_w) = self.weights.dim();
        (kernel_h, kernel_w)
    }

    /// Returns the stride of the layer.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the padding of the layer.
    #[inline]
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// Returns a view of the weights of the layer.
    #[inline]
    pub fn weights(&self) -> ArrayView4<f32> {
        self.weights.view()
    }

    /// Returns a view of the biases of the layer.
    #[inline]
    pub fn biases(&self) -> ArrayView1<f32> {
        self.biases.view()
    }

    /// Returns the activation function of the layer if any.
    #[inline]
    pub fn activation(&self) -> Option<&dyn ActivationFunction> {
        self.activation.as_deref()
    }
}

impl Trainable for Conv {
    fn forward(&mut self, input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        self.input = input.to_owned().into_dimensionality::<Ix4>()?;

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
            // Use 1 instead of `c` since each filter produces a single map (ignoring the input channel).
            let col_im = col2im(mul.view(), h_prime, w_prime, 1)?;
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
        output_gradient: ArrayViewD<f32>,
        _learning_rate: f32,
        _optimizer: &Optimizer,
        _mode: &NNMode,
    ) -> NNResult<ndarray::ArrayD<f32>> {
        let output_gradient: Array4<f32> = output_gradient.to_owned().into_dimensionality()?;

        let (n, _, h, w) = self.input.dim();
        let (f, c, k_h, k_w) = self.weights.dim();
        let h_prime = (h + 2 * self.padding - k_h) / self.stride + 1;
        let w_prime = (w + 2 * self.padding - k_w) / self.stride + 1;

        let mut dw = Array4::<f32>::zeros(self.weights.dim());
        let mut dx = Array4::<f32>::zeros(self.input.dim());
        let mut db = Array1::<f32>::zeros(self.biases.dim());

        for i in 0..n {
            let im = self.input.slice(s![i, .., .., ..]);
            let pad_config = vec![
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
            ];
            let im_pad = pad(im.to_owned(), pad_config, 0.0);
            let im_col = im2col(im_pad.view(), k_h, k_w, self.stride);
            let filter_col = self.weights.to_shape((f, c * k_h * k_w))?;
            let filter_col = filter_col.t();

            let dout_i = output_gradient.slice(s![i, .., .., ..]);
            let dbias_sum = dout_i.to_shape((f, h_prime * w_prime))?;
            let dbias_sum = dbias_sum.t();

            db.scaled_add(1.0, &dbias_sum.sum_axis(Axis(0)));
            let dmul = dbias_sum;

            let dfilter_col = im_col.t().dot(&dmul);
            let dim_col = dmul.dot(&filter_col.t());

            let dx_padded = col2im_back(dim_col, h_prime, w_prime, self.stride, k_h, k_w, c)?;
            dx.slice_mut(s![i, .., .., ..]).assign(&dx_padded.slice(s![
                ..,
                self.padding..h + self.padding,
                self.padding..w + self.padding
            ]));
            let dfilter_col = dfilter_col.t();
            dw.scaled_add(1.0, &dfilter_col.to_shape((f, c, k_h, k_w))?);
        }

        Ok(dx.into_dyn())
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
            // The flatten() function creates extra memory; perhaps this could be optimized in the future.
            let flatten_patch = patch.flatten();
            col.slice_mut(s![i * new_w + j, ..]).assign(&flatten_patch);
        }
    }

    col
}

/// Reconstructs the convolution output from the im2col result.
///
/// # Parameters
/// - `mul`: The result of the matrix multiplication between the input patches and the filters, with shape `(h_prime*w_prime, n_kernels)`.
/// - `h_prime`: The height of the convolution output.
/// - `w_prime`: The width of the convolution output.
/// - `c`: This parameter must be 1; otherwise, an error is returned.
///
fn col2im(mul: ArrayView2<f32>, h_prime: usize, w_prime: usize, c: usize) -> NNResult<ArrayD<f32>> {
    if c != 1 {
        return Err(MininnError::LayerError(
            "col2im only supports c == 1".to_string(),
        ));
    }
    let f = mul.shape()[1];

    let mut out = Array3::<f32>::zeros((f, h_prime, w_prime));

    for i in 0..f {
        let col = mul.slice(s![.., i]);
        let reshaped = col.to_shape((h_prime, w_prime))?;
        out.slice_mut(s![i, .., ..]).assign(&reshaped);
    }

    Ok(out.into_dyn())
}

fn col2im_back(
    dim_col: Array2<f32>,
    h_prime: usize,
    w_prime: usize,
    stride: usize,
    filter_h: usize,
    filter_w: usize,
    filter_c: usize,
) -> NNResult<Array3<f32>> {
    let h = (h_prime - 1) * stride + filter_h;
    let w = (w_prime - 1) * stride + filter_w;
    let mut dx = Array3::<f32>::zeros((filter_c, h, w));

    for i in 0..(h_prime * w_prime) {
        let row = dim_col.slice(s![i, ..]);
        let h_start = (i / w_prime) * stride;
        let w_start = (i % w_prime) * stride;

        dx.slice_mut(s![
            ..,
            h_start..h_start + filter_h,
            w_start..w_start + filter_w
        ])
        .scaled_add(1.0, &row.to_shape((filter_c, filter_h, filter_w))?);
    }

    Ok(dx)
}

/// Pads an image with a specified value.
///
/// ## Parameters
///
/// - `img`: The input image to be padded.
/// - `pad_with`: A vector of arrays representing the padding configuration for each dimension.
/// - `value`: The value to be used for padding.
///
/// ## Returns
///
/// The image with the applied padding.
///
fn pad(img: Array3<f32>, pad_with: Vec<[usize; 2]>, value: f32) -> Array3<f32> {
    assert_eq!(
        img.ndim(),
        pad_with.len(),
        "The number of dimensions of the array must match the length of `pad_with`."
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
    use ndarray::{array, Array};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    #[test]
    fn test_im2col() {
        let img = Array3::random((3, 10, 10), Uniform::new(-1., 1.));
        let col = im2col(img.view(), 3, 3, 1);
        assert_eq!(col.dim(), (64, 27));
    }

    #[test]
    fn test_col2im() {
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
        ]
        .into_dyn();
        // Passing c = 1, which should work correctly.
        let result = col2im(mul.view(), 2, 3, 1).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_col2im_invalid() {
        // An error is expected when passing c different from 1.
        let mul = Array2::<f32>::zeros((6, 2));
        let result = col2im(mul.view(), 2, 3, 3);
        assert!(result.is_err());
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

    #[test]
    fn test_col2im_back() {
        let dim_col = Array::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();
        let expected = Array::from_shape_vec(
            (1, 3, 3),
            vec![1.0, 7.0, 6.0, 12.0, 34.0, 22.0, 11.0, 27.0, 16.0],
        )
        .unwrap();
        let result = col2im_back(dim_col, 2, 2, 1, 2, 2, 1).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_conv_backward() {
        let mut conv = Conv::new(1, 1, (3, 3), 1, 0);
        conv.weights.fill(1.0);
        conv.biases.fill(1.0);
        let input = Array::from_elem((1, 1, 4, 4), 1.0).into_dyn();
        let output = conv.forward(input.view(), &NNMode::Train).unwrap();
        let grad = Array::ones(output.dim());
        let dummy_optimizer = Optimizer::SGD;
        let dx = conv
            .backward(grad.view(), 0.1, &dummy_optimizer, &NNMode::Train)
            .unwrap();
        assert_eq!(dx.shape(), input.shape());
    }
}
