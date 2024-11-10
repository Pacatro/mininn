use ndarray::{Array1, Array2, Array3, Array4, ArrayView2};
use ndarray_rand::{rand::distributions::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

use super::Layer;
use crate::{error::NNResult, utils::Optimizer};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Padding {
    Valid,
    Full,
}

/// Computes the valid cross-correlation between an input array and a kernel.
///
/// # Arguments
/// * `input` - The input 2D array
/// * `kernel` - The kernel to convolve with the input
///
/// # Returns
/// * `Result<Array2<f64>, CrossCorrelationError>` - The resulting cross-correlation array or an error
///
/// # Examples
/// ```
/// use ndarray::array;
/// let input = array![[1.0, 2.0, 3.0],
///                    [4.0, 5.0, 6.0],
///                    [7.0, 8.0, 9.0]];
/// let kernel = array![[1.0, 0.0],
///                     [0.0, 1.0]];
/// let result = cross_correlation_valid(&input, &kernel).unwrap();
/// ```
///
pub(crate) fn cross_correlation_valid(input: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let mut sums = Vec::new();

    let (input_rows, input_cols) = input.dim();
    let (kernel_rows, kernel_cols) = kernel.dim();

    let valid_rows = input_rows - kernel_rows + 1;
    let valid_cols = input_cols - kernel_cols + 1;

    for i in 0..valid_rows {
        for j in 0..valid_cols {
            let mut sum = 0.0;

            for ki in 0..kernel_rows {
                for kj in 0..kernel_cols {
                    sum += input[[i + ki, j + kj]] * kernel[[ki, kj]];
                }
            }

            sums.push(sum);
        }
    }

    Array2::from_shape_vec((valid_rows, valid_cols), sums).unwrap()
}

// /// Computes the full cross-correlation between an input array and a kernel.
// /// The output size will be (input_size + kernel_size - 1) in each dimension.
// ///
// /// # Arguments
// /// * `input` - The input 2D array
// /// * `kernel` - The kernel to convolve with the input
// ///
// /// # Returns
// /// * `Result<Array2<f64>, CrossCorrelationError>` - The resulting cross-correlation array or an error
// ///
// /// # Examples
// /// ```
// /// use ndarray::array;
// /// let input = array![[1.0, 2.0],
// ///                    [3.0, 4.0]];
// /// let kernel = array![[1.0, 0.5],
// ///                     [0.5, 0.25]];
// /// let result = cross_correlation_full(&input, &kernel).unwrap();
// /// // Result will be 3x3 array with full padding
// /// ```
// pub(crate) fn cross_correlation_full(input: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
//     let (input_rows, input_cols) = input.dim();
//     let (kernel_rows, kernel_cols) = kernel.dim();

//     let output_rows = input_rows + kernel_rows - 1;
//     let output_cols = input_cols + kernel_cols - 1;

//     let mut output = Array2::zeros((output_rows, output_cols));

//     let pad_top = kernel_rows - 1;
//     let pad_left = kernel_cols - 1;

//     let input_view = input.view();
//     let kernel_view = kernel.view();

//     for i in 0..output_rows {
//         for j in 0..output_cols {
//             let sum = compute_padded_window_sum(input_view, kernel_view, i, j, pad_top, pad_left);
//             output[[i, j]] = sum;
//         }
//     }

//     output
// }

// /// Computes the sum of element-wise multiplication between the kernel and the corresponding
// /// section of the input array, taking padding into account
// #[inline]
// fn compute_padded_window_sum(
//     input: ArrayView2<f64>,
//     kernel: ArrayView2<f64>,
//     out_row: usize,
//     out_col: usize,
//     pad_top: usize,
//     pad_left: usize,
// ) -> f64 {
//     let (kernel_rows, kernel_cols) = kernel.dim();
//     let (input_rows, input_cols) = input.dim();
//     let mut sum = 0.0;

//     for ki in 0..kernel_rows {
//         for kj in 0..kernel_cols {
//             // Calcular la posición correspondiente en el input
//             let input_row = out_row.saturating_sub(pad_top) + ki;
//             let input_col = out_col.saturating_sub(pad_left) + kj;

//             // Verificar si la posición está dentro del input
//             if input_row < input_rows && input_col < input_cols {
//                 sum += input[[input_row, input_col]]
//                     * kernel[[kernel_rows - 1 - ki, kernel_cols - 1 - kj]];
//             }
//         }
//     }

//     sum
// }

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Conv {
    // Shape = [depth, height, width]
    input_shape: [usize; 3],
    output_shape: [usize; 3],
    // Kernel shape = [nkernerls, input_depth, kernel_size, kernel_size]
    kernel_shape: [usize; 4],
    kernels: Array4<f64>,
    biases: Array3<f64>,
    padding: Padding,
}

impl Conv {
    pub fn new(
        input_shape: [usize; 3],
        kernel_size: usize,
        nkernels: usize,
        padding: Padding,
    ) -> Self {
        let input_depth = input_shape[0];
        let input_height = input_shape[1];
        let input_width = input_shape[2];
        let kernel_shape = [nkernels, input_depth, kernel_size, kernel_size];
        let output_shape = [
            nkernels,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        ];

        let kernels = Array4::random(kernel_shape, Uniform::new(-1.0, 1.0));
        let biases = Array3::random(output_shape, Uniform::new(-1.0, 1.0));

        println!("kernels:\n{}", kernels);

        Self {
            input_shape,
            output_shape,
            kernel_shape,
            kernels,
            biases,
            padding,
        }
    }
}

impl Default for Conv {
    fn default() -> Self {
        Self::new([1, 2, 3], 2, 4, Padding::Valid)
    }
}

impl Layer for Conv {
    fn layer_type(&self) -> String {
        "Conv".to_string()
    }

    fn to_json(&self) -> NNResult<String> {
        Ok(serde_json::to_string(self)?)
    }

    fn from_json(json: &str) -> NNResult<Box<dyn Layer>> {
        Ok(Box::new(serde_json::from_str::<Conv>(json)?))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array1<f64>) -> NNResult<Array1<f64>> {
        todo!()
    }

    fn backward(
        &mut self,
        output_gradient: ndarray::ArrayView1<f64>,
        learning_rate: f64,
        optimizer: &Optimizer,
    ) -> NNResult<Array1<f64>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_conv_creation() {
        let conv = Conv::new([1, 2, 3], 2, 4, Padding::Valid);
        assert_eq!(conv.input_shape, [1, 2, 3]);
        assert_eq!(conv.output_shape, [4, 1, 2]);
        assert_eq!(conv.kernel_shape, [4, 1, 2, 2]);
        assert_eq!(conv.kernels.shape(), [4, 1, 2, 2]);
        assert_eq!(conv.biases.shape(), [4, 1, 2]);
    }

    #[test]
    fn test_conv_default() {
        let conv_new = Conv::new([1, 2, 3], 2, 4, Padding::Valid);
        let conv_default = Conv::default();
        assert_eq!(conv_new.input_shape, conv_default.input_shape);
        assert_eq!(conv_new.output_shape, conv_default.output_shape);
        assert_eq!(conv_new.kernel_shape, conv_default.kernel_shape);
    }

    #[test]
    fn test_cross_correlation_valid() {
        let input = array![[1., 6., 2.], [5., 3., 1.], [7., 0., 4.],];
        let kernel = array![[1., 2.], [-1., 0.]];
        let output = cross_correlation_valid(&input, &kernel);
        assert_eq!(output, array![[8., 7.], [4., 5.]]);
    }

    // #[test]
    // fn test_cross_correlation_full() {
    //     let input = array![[1., 6., 2.], [5., 3., 1.], [7., 0., 4.],];
    //     let kernel = array![[1., 2.], [-1., 0.]];
    //     let real_output = array![
    //         [0., -1., -6., -2.],
    //         [2., 8., 7., 1.],
    //         [10., 4., 5., -3.],
    //         [14., 7., 8., 4.],
    //     ];
    //     let output = cross_correlation_full(&input, &kernel);
    //     assert_eq!(output, real_output);
    // }
}
