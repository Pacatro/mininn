use ndarray::{s, Array2, Array3, Array4, ArrayD, ArrayView2, ArrayView3, ArrayViewD};
use serde::{Deserialize, Serialize};

use crate::{
    core::{NNMode, NNResult},
    layers::{Layer, Trainable},
    utils::{ActivationFunction, MSGPackFormatting, Optimizer, OptimizerType},
};

use mininn_derive::Layer;

// img FORMAT --> (C, H, W)

#[derive(Layer, Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Conv {}

impl Trainable for Conv {
    fn forward(&mut self, input: ArrayViewD<f32>, _mode: &NNMode) -> NNResult<ArrayD<f32>> {
        todo!()
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
}
