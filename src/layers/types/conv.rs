use ndarray::{
    s, Array1, Array2, Array3, Array4, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayView4,
    ArrayViewD,
};
use serde::{Deserialize, Serialize};

use crate::core::{MininnError, NNMode, NNResult};
use crate::layers::{Layer, TrainLayer};
use crate::utils::{ActivationFunction, MSGPackFormatting, Optimizer};
use mininn_derive::Layer;

// TODO: REMAKE THIS WHOLE MODULE
// Links:
// https://www.youtube.com/watch?v=pj9-rr1wDhM
// https://www.youtube.com/watch?v=KuXjwB4LzSA

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

pub fn col2im(
    col: ArrayView2<f32>,
    input_shape: (usize, usize, usize),
    kernel_size: (usize, usize),
    stride: usize,
) -> NNResult<Array3<f32>> {
    let (input_h, input_w, input_c) = input_shape;
    let (kernel_h, kernel_w) = kernel_size;

    if kernel_h > input_h || kernel_w > input_w || stride == 0 {
        return Err(MininnError::LayerError(
            "Invalid kernel size or stride".to_string(),
        ));
    }

    let new_h = (input_h - kernel_h) / stride + 1;
    let new_w = (input_w - kernel_w) / stride + 1;

    let mut output = Array3::<f32>::zeros((input_h, input_w, input_c));

    for i in 0..new_h {
        for j in 0..new_w {
            let row_idx = i * new_w + j;
            let col_row = col.row(row_idx);
            let patch = col_row.to_shape((kernel_h, kernel_w, input_c))?;
            output
                .slice_mut(s![
                    i * stride..i * stride + kernel_h,
                    j * stride..j * stride + kernel_w,
                    ..
                ])
                .zip_mut_with(&patch, |out, &val| *out += val);
        }
    }

    Ok(output)
}

pub(crate) fn col2im_backward(
    col_gradient: ArrayView2<f32>,
    feature_h: usize,
    feature_w: usize,
    stride: usize,
    kernel_size: (usize, usize, usize),
) -> NNResult<Array3<f32>> {
    let (kernel_h, kernel_w, kernel_c) = kernel_size;
    let h = (feature_h - 1) * stride + kernel_h;
    let w = (feature_w - 1) * stride + kernel_w;

    // Inicializar el tensor de gradiente de entrada con ceros
    let mut dx = Array3::<f32>::zeros((h, w, kernel_c));

    // Iterar sobre las posiciones de las características
    for i in 0..(feature_h * feature_w) {
        let col_row = col_gradient.row(i);
        let row_col_reshaped = col_row.to_shape((kernel_h, kernel_w, kernel_c))?;

        let h_start = (i / feature_w) * stride;
        let w_start = (i % feature_w) * stride;

        // Actualizar la vista mutable directamente
        dx.slice_mut(s![
            h_start..h_start + kernel_h,
            w_start..w_start + kernel_w,
            ..
        ])
        .zip_mut_with(&row_col_reshaped, |a, &b| *a += b);
    }

    Ok(dx)
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
        self.input = input.to_owned().into_dimensionality()?;

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

    use crate::layers::types::conv::{col2im, im2col};

    #[test]
    fn test_im2col_basic() {
        let input = array![
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
            [[7.0], [8.0], [9.0]]
        ];

        let kernel_size = (2, 2);
        let stride = 1;

        let expected = array![
            [1.0, 2.0, 4.0, 5.0],
            [2.0, 3.0, 5.0, 6.0],
            [4.0, 5.0, 7.0, 8.0],
            [5.0, 6.0, 8.0, 9.0]
        ];

        let result = im2col(input.view(), kernel_size, stride).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_col2im_basic() {
        let col = array![
            [1.0, 2.0, 4.0, 5.0],
            [2.0, 3.0, 5.0, 6.0],
            [4.0, 5.0, 7.0, 8.0],
            [5.0, 6.0, 8.0, 9.0]
        ];

        let input_shape = (3, 3, 1);
        let kernel_size = (2, 2);
        let stride = 1;

        let expected = array![
            [[1.0], [4.0], [3.0]],
            [[8.0], [20.0], [12.0]],
            [[7.0], [16.0], [9.0]]
        ];

        let result = col2im(col.view(), input_shape, kernel_size, stride).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_im2col_col2im_roundtrip() {
        let input = array![
            [[1.0], [2.0], [3.0], [4.0]],
            [[5.0], [6.0], [7.0], [8.0]],
            [[9.0], [10.0], [11.0], [12.0]],
            [[13.0], [14.0], [15.0], [16.0]]
        ];

        let kernel_size = (2, 2);
        let stride = 2;

        let col = im2col(input.view(), kernel_size, stride).unwrap();
        let reconstructed = col2im(col.view(), (4, 4, 1), kernel_size, stride).unwrap();

        assert_eq!(input, reconstructed);
    }

    #[test]
    fn test_im2col_multichannel() {
        let input = array![
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            [[7.0, 70.0], [8.0, 80.0], [9.0, 90.0]]
        ];

        let kernel_size = (2, 2);
        let stride = 1;

        let result = im2col(input.view(), kernel_size, stride).unwrap();

        assert_eq!(result.dim(), (4, 8)); // 4 patches, each of size 2x2x2.
    }

    #[test]
    fn test_col2im_multichannel() {
        let col = array![
            [1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 40.0, 50.0],
            [2.0, 3.0, 5.0, 6.0, 20.0, 30.0, 50.0, 60.0],
            [4.0, 5.0, 7.0, 8.0, 40.0, 50.0, 70.0, 80.0],
            [5.0, 6.0, 8.0, 9.0, 50.0, 60.0, 80.0, 90.0]
        ];

        let input_shape = (3, 3, 2);
        let kernel_size = (2, 2);
        let stride = 1;

        let result = col2im(col.view(), input_shape, kernel_size, stride).unwrap();

        assert_eq!(result.dim(), input_shape);
    }

    #[test]
    fn test_invalid_kernel_or_stride() {
        let input = array![[[1.0], [2.0]], [[3.0], [4.0]]];

        // Invalid kernel size
        assert!(im2col(input.view(), (3, 3), 1).is_err());

        // Invalid stride
        assert!(im2col(input.view(), (2, 2), 0).is_err());
    }
}
