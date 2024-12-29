use ndarray::{
    s, Array1, Array2, Array3, Array4, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayView4,
    ArrayViewD,
};
use serde::{Deserialize, Serialize};

use crate::core::{MininnError, NNMode, NNResult};
use crate::layers::{Layer, TrainLayer};
use crate::utils::{ActivationFunction, MSGPackFormatting, Optimizer};
use mininn_derive::Layer;

pub fn im2col(
    input: ArrayView3<f32>,
    kernel_size: (usize, usize),
    stride: usize,
) -> NNResult<Array2<f32>> {
    let (input_h, input_w, input_c) = input.dim();
    let (kernel_h, kernel_w) = kernel_size;

    if kernel_h > input_h || kernel_w > input_w || stride == 0 {
        return Err(MininnError::LayerError(
            "Invalid kernel size or stride".to_string(),
        ));
    }

    let new_h = (input_h - kernel_h) / stride + 1;
    let new_w = (input_w - kernel_w) / stride + 1;

    let mut col = Array2::<f32>::zeros((new_h * new_w, input_c * kernel_h * kernel_w));

    let patch_size = kernel_h * kernel_w * input_c;

    for i in 0..new_h {
        for j in 0..new_w {
            let patch = input.slice(s![
                i * stride..i * stride + kernel_h,
                j * stride..j * stride + kernel_w,
                ..
            ]);
            col.slice_mut(s![i * new_w + j, ..])
                .assign(&patch.to_shape((patch_size,))?);
        }
    }

    Ok(col)
}

// FIXME
pub fn col2im(
    mul: ArrayView2<f32>,
    kernel_size: (usize, usize),
    c: usize,
) -> NNResult<ArrayD<f32>> {
    let f = mul.shape()[0];
    let (kernel_h, kernel_w) = kernel_size;

    let out = if c == 1 {
        let mut out = Array3::<f32>::zeros((kernel_h, kernel_w, f));
        for i in 0..f {
            let col = mul.slice(s![i, ..]);
            out.slice_mut(s![i, .., ..])
                .assign(&col.to_shape((kernel_h, kernel_w))?);
        }
        out.into_dyn()
    } else {
        let mut out = Array4::<f32>::zeros((kernel_h, kernel_w, f, c));
        for i in 0..f {
            let col = mul.slice(s![i, ..]);
            out.slice_mut(s![i, .., .., ..])
                .assign(&col.to_shape((kernel_h, kernel_w, c))?);
        }
        out.into_dyn()
    };

    Ok(out)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Padding {
    Valid,
    Same,
}

#[derive(Debug, Clone, Serialize, Deserialize, Layer)]
pub struct Conv {
    input: Array3<f32>,
    weights: Array4<f32>,
    biases: Array1<f32>,
    batch_size: usize,
    nfilters: usize,
    kernel_width: usize,
    kernel_height: usize,
    img_width: usize,
    img_height: usize,
    conv_img_width: usize,
    conv_img_height: usize,
    stride: usize,
    pad: usize,
    padding: Padding,
    depth: usize,
    output_depth: usize,
    activation: Option<Box<dyn ActivationFunction>>,
}

impl Conv {
    #[inline]
    pub fn new(nfilters: usize, kernel_size: (usize, usize)) -> Self {
        Self {
            input: Array3::zeros((0, 0, 0)),
            weights: Array4::zeros((0, 0, 0, 0)),
            biases: Array1::zeros(0),
            batch_size: 0,
            nfilters,
            kernel_width: kernel_size.0,
            kernel_height: kernel_size.1,
            img_width: 0,
            img_height: 0,
            conv_img_width: 0,
            conv_img_height: 0,
            stride: 1,
            pad: 1,
            padding: Padding::Valid,
            depth: 0,
            output_depth: 0,
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

    #[inline]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[inline]
    pub fn nfilters(&self) -> usize {
        self.nfilters
    }

    #[inline]
    pub fn kernel_size(&self) -> (usize, usize) {
        (self.kernel_width, self.kernel_height)
    }

    #[inline]
    pub fn img_size(&self) -> (usize, usize) {
        (self.img_width, self.img_height)
    }

    #[inline]
    pub fn conv_img_size(&self) -> (usize, usize) {
        (self.conv_img_width, self.conv_img_height)
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    pub fn pad(&self) -> usize {
        self.pad
    }

    #[inline]
    pub fn padding(&self) -> Padding {
        self.padding
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    #[inline]
    pub fn output_depth(&self) -> usize {
        self.output_depth
    }

    #[inline]
    pub fn activation(&self) -> Option<&str> {
        self.activation.as_ref().map(|a| a.as_ref().name())
    }

    #[inline]
    pub fn weights(&self) -> ArrayView4<f32> {
        self.weights.view()
    }

    #[inline]
    pub fn biases(&self) -> ArrayView1<f32> {
        self.biases.view()
    }
}

impl Default for Conv {
    fn default() -> Self {
        Self {
            input: Array3::zeros((0, 0, 0)),
            weights: Array4::zeros((0, 0, 0, 0)),
            biases: Array1::zeros(0),
            batch_size: 0,
            nfilters: 0,
            kernel_width: 0,
            kernel_height: 0,
            img_width: 0,
            img_height: 0,
            conv_img_width: 0,
            conv_img_height: 0,
            stride: 1,
            pad: 1,
            padding: Padding::Valid,
            depth: 0,
            output_depth: 0,
            activation: None,
        }
    }
}

impl TrainLayer for Conv {
    fn forward(&mut self, input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        // self.input = input.to_owned().into_dimensionality()?;
        // let (n, c, h, w) = self.input.dim();
        // let (f, c, hh, ww) = self.weights.dim();

        // let output_height = 1 + (h + 2 * self.pad - self.kernel_height) / self.stride;
        // let output_width = 1 + (w + 2 * self.pad - self.kernel_width) / self.stride;
        // let output = Array4::<f32>::zeros((n, f, output_height, output_width));

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
    use ndarray::{array, Array3};

    use crate::layers::types::conv::{col2im, im2col, Conv, Padding};

    #[test]
    fn test_conv_creation() {
        let conv = Conv::new(3, (3, 3));
        assert_eq!(conv.nfilters(), 3);
        assert_eq!(conv.kernel_size(), (3, 3));
    }

    #[test]
    fn test_conv_creation_with_stride() {
        let conv = Conv::new(3, (3, 3)).with_stride(2);
        assert_eq!(conv.stride(), 2);
    }

    #[test]
    fn test_conv_creation_with_padding() {
        let conv = Conv::new(3, (3, 3)).with_padding(Padding::Same);
        assert_eq!(conv.padding(), Padding::Same);
    }

    #[test]
    fn test_conv_creation_with_activation() {
        let conv = Conv::new(3, (3, 3)).apply(crate::utils::Act::ReLU);
        assert_eq!(conv.activation().unwrap(), "ReLU");
    }

    #[test]
    fn test_conv_im2col() {
        let input = Array3::from_shape_vec(
            (5, 5, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
        )
        .unwrap();

        let kernel_size = (3, 3);
        let stride = 1;

        let col = im2col(input.view(), kernel_size, stride).unwrap();

        let expected = array![
            [1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0],
            [2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0],
            [3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 16.0, 17.0, 18.0],
            [7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0],
            [8.0, 9.0, 10.0, 13.0, 14.0, 15.0, 18.0, 19.0, 20.0],
            [11.0, 12.0, 13.0, 16.0, 17.0, 18.0, 21.0, 22.0, 23.0],
            [12.0, 13.0, 14.0, 17.0, 18.0, 19.0, 22.0, 23.0, 24.0],
            [13.0, 14.0, 15.0, 18.0, 19.0, 20.0, 23.0, 24.0, 25.0]
        ];

        assert_eq!(col, expected);
    }

    #[test]
    fn test_col2im() {
        let input = Array3::from_shape_vec(
            (5, 5, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
        )
        .unwrap();

        let kernel_size = (3, 3);

        let expected = array![
            [1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0],
            [2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0],
            [3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 16.0, 17.0, 18.0],
            [7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0],
            [8.0, 9.0, 10.0, 13.0, 14.0, 15.0, 18.0, 19.0, 20.0],
            [11.0, 12.0, 13.0, 16.0, 17.0, 18.0, 21.0, 22.0, 23.0],
            [12.0, 13.0, 14.0, 17.0, 18.0, 19.0, 22.0, 23.0, 24.0],
            [13.0, 14.0, 15.0, 18.0, 19.0, 20.0, 23.0, 24.0, 25.0]
        ];

        let col = col2im(expected.view(), kernel_size, 1).unwrap();

        assert_eq!(col.into_dimensionality().unwrap(), input);
    }
}
