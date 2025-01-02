use ndarray::{s, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::core::{MininnError, NNResult};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) enum Padding {
    Valid,
    Full,
}

pub(crate) fn cross_correlation2d(
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
            compute_correlation(input, kernel, stride, None)
        }
        Padding::Full => {
            let pad_h = kh - 1;
            let pad_w = kw - 1;
            let padded = pad_input(&input, pad_h, pad_w);
            compute_correlation(padded.view(), kernel, stride, None)
        }
    }
}

fn pad_input(input: &ArrayView2<f32>, pad_h: usize, pad_w: usize) -> Array2<f32> {
    let (h, w) = input.dim();
    let new_h = h + 2 * pad_h;
    let new_w = w + 2 * pad_w;
    let mut padded = Array2::zeros((new_h, new_w));

    padded
        .slice_mut(s![pad_h..pad_h + h, pad_w..pad_w + w])
        .assign(input);
    padded
}

fn compute_correlation(
    input: ArrayView2<f32>,
    kernel: ArrayView2<f32>,
    stride: usize,
    output_size: Option<(usize, usize)>,
) -> NNResult<Array2<f32>> {
    let (h, w) = input.dim();
    let (kh, kw) = kernel.dim();

    let (output_h, output_w) = output_size.unwrap_or_else(|| {
        let out_h = (h - kh) / stride + 1;
        let out_w = (w - kw) / stride + 1;
        (out_h, out_w)
    });

    let mut output = Array2::zeros((output_h, output_w));

    for i in 0..output_h {
        for j in 0..output_w {
            output[[i, j]] = compute_window_sum(
                input.slice(s![i * stride..i * stride + kh, j * stride..j * stride + kw]),
                kernel,
            );
        }
    }
    Ok(output)
}

fn compute_window_sum(window: ArrayView2<f32>, kernel: ArrayView2<f32>) -> f32 {
    (&window * &kernel).sum()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::utils::{correlation::pad_input, cross_correlation2d, Padding};

    #[test]
    fn test_pad() {
        let input = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.],];

        let pad_h = 1;
        let pad_w = 1;

        let expected = array![
            [0., 0., 0., 0., 0.],
            [0., 1., 2., 3., 0.],
            [0., 4., 5., 6., 0.],
            [0., 7., 8., 9., 0.],
            [0., 0., 0., 0., 0.],
        ];

        let result = pad_input(&input.view(), pad_h, pad_w);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_correlation_valid() {
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
    fn test_correlation_full() {
        let input = array![
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
            [10., 10., 10., 0., 0., 0.],
        ];

        let kernel = array![[1., 0., -1.], [1., 0., -1.], [1., 0., -1.],];

        let stride = 1;
        let result =
            cross_correlation2d(input.view(), kernel.view(), stride, Padding::Full).unwrap();

        let expected = array![
            [-10., -10., 0., 10., 10., 0., 0., 0.],
            [-20., -20., 0., 20., 20., 0., 0., 0.],
            [-30., -30., 0., 30., 30., 0., 0., 0.],
            [-30., -30., 0., 30., 30., 0., 0., 0.],
            [-30., -30., 0., 30., 30., 0., 0., 0.],
            [-30., -30., 0., 30., 30., 0., 0., 0.],
            [-20., -20., 0., 20., 20., 0., 0., 0.],
            [-10., -10., 0., 10., 10., 0., 0., 0.],
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_correlation_with_stride() {
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
}
