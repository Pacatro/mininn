use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_rand::{rand::distributions::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

use super::Layer;
use crate::{error::NNResult, nn::NNMode, utils::Optimizer};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub(crate) enum Padding {
    Valid,
    Full,
}

// SEE THIS FOR DOC: https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub(crate) struct Conv {
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
        let (input_depth, input_height, input_width) =
            (input_shape[0], input_shape[1], input_shape[2]);

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

    // TODO: IMPROVE THIS
    fn _cross_correlation(
        &self,
        input: &Array2<f64>,
        kernel: &Array2<f64>,
    ) -> NNResult<Array2<f64>> {
        let mut sums = Vec::new();

        match self.padding {
            Padding::Valid => {
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

                Ok(Array2::from_shape_vec((valid_rows, valid_cols), sums)?)
            }
            Padding::Full => Ok(Array2::zeros((0, 0))),
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

    fn forward(&mut self, _input: &Array1<f64>, _mode: &NNMode) -> NNResult<Array1<f64>> {
        todo!()
    }

    fn backward(
        &mut self,
        _output_gradient: &Array1<f64>,
        _learning_rate: f64,
        _optimizer: &Optimizer,
        _mode: &NNMode,
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
        let conv = Conv::new([1, 2, 3], 2, 4, Padding::Valid);
        let output = conv._cross_correlation(&input, &kernel);
        assert!(output.is_ok());
        assert_eq!(output.unwrap(), array![[8., 7.], [4., 5.]]);
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
