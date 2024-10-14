use ndarray::{Array1, ArrayView1};

/// Represents the different cost functions for the neural network
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Cost {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
}

impl Cost {
    /// Computes the cost between predicted and actual values
    ///
    /// This method calculates the cost (loss) between the predicted values and the actual values
    /// using the specified cost function.
    ///
    /// ## Arguments
    ///
    /// * `y_p`: A reference to an `ArrayView1<f64>` representing the predicted values
    /// * `y`: A reference to an `ArrayView1<f64>` representing the actual values
    ///
    /// ## Returns
    ///
    /// A `f64` value representing the computed cost
    ///
    #[inline]
    pub fn function(&self, y_p: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        match self {
            Cost::MSE => (y - y_p).map(|x| x.powi(2)).mean().unwrap_or(0.),
            Cost::MAE => (y - y_p).map(|x| x.abs()).mean().unwrap_or(0.),
        }
    }

    /// Computes the derivative of the cost function
    ///
    /// This method calculates the derivative of the cost function with respect to the predicted values.
    /// It is used in the backpropagation process for updating the network weights.
    ///
    /// ## Arguments
    ///
    /// * `y_p`: A reference to an `ArrayView1<f64>` representing the predicted values
    /// * `y`: A reference to an `ArrayView1<f64>` representing the actual values
    ///
    /// ## Returns
    ///
    /// An `Array1<f64>` containing the computed derivatives
    ///
    #[inline]
    pub fn derivate(&self, y_p: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        match self {
            Cost::MSE => 2.0 * (y_p - y) / y.len() as f64,
            Cost::MAE => (y_p - y).map(|x| x.signum()) / y.len() as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mse_function() {
        let y_p = array![0.1, 0.4, 0.6];
        let y = array![0.0, 0.5, 1.0];
        let cost = Cost::MSE;
        let result = cost.function(&y_p.view(), &y.view());
        assert_eq!(result as f32, 0.06);
    }

    #[test]
    fn test_mae_function() {
        let y_p = array![0.1, 0.4, 0.6];
        let y = array![0.0, 0.5, 1.0];
        let cost = Cost::MAE;
        let result = cost.function(&y_p.view(), &y.view());
        assert_eq!(result as f32, 0.2); // Expected MAE
    }

    #[test]
    fn test_mse_derivate() {
        let y_p = array![0.1, 0.4, 0.6];
        let y = array![0.0, 0.5, 1.0];
        let cost = Cost::MSE;
        let result = cost.derivate(&y_p.view(), &y.view());
        let expected = array![0.066666667, -0.066666667, -0.266666668]; // Expected MSE derivative
        assert_eq!(result.mapv(|v| v as f32), expected);
    }

    #[test]
    fn test_mae_derivate() {
        let y_p = array![0.1, 0.4, 0.6];
        let y = array![0.0, 0.5, 1.0];
        let cost = Cost::MAE;
        let result = cost.derivate(&y_p.view(), &y.view());
        let expected = array![0.33333334, -0.33333334, -0.33333334]; // Expected MAE derivative
        assert_eq!(result.mapv(|v| v as f32), expected);
    }
}
