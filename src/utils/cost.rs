use std::fmt::Debug;

use ndarray::{ArrayD, ArrayViewD};

/// Allows users to define their own cost functions
///
/// ## Methods
/// - `function`: Calculates the cost between the predicted and actual values
/// - `derivate`: Calculates the derivative of the cost function
/// - `get_name`: Returns the name of the cost function
///
pub trait CostFunction: Debug {
    /// Computes the cost between predicted and actual values
    ///
    /// This method calculates the cost (loss) between the predicted values and the actual values
    /// using the specified cost function.
    ///
    /// ## Arguments
    ///
    /// * `y_p`: The predicted values
    /// * `y`: The actual values
    ///
    /// ## Returns
    ///
    /// A `f64` value representing the computed cost
    ///
    fn function(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> f64;

    /// Computes the derivative of the cost function
    ///
    /// This method calculates the derivative of the cost function with respect to the predicted values.
    /// It is used in the backpropagation process for updating the network weights.
    ///
    /// ## Arguments
    ///
    /// * `y_p`: The predicted values
    /// * `y`: The actual values
    ///
    /// ## Returns
    ///
    /// An `ArrayD<f64>` containing the computed derivatives
    ///
    fn derivate(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> ArrayD<f64>;

    /// Returns the name of the cost function
    fn cost_name(&self) -> &str;
}

/// Represents the different cost functions for the neural network
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Cost {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Binary Cross-Entropy
    BCE,
    /// Categorical Cross-Entropy
    CCE,
}

impl CostFunction for Cost {
    #[inline]
    fn function(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> f64 {
        match self {
            Cost::MSE => (y - y_p).pow2().mean().unwrap_or(0.),
            Cost::MAE => (y - y_p).abs().mean().unwrap_or(0.),
            Cost::BCE => -((y * y_p.ln() + (1. - y) * (1. - y_p).ln()).sum()),
            Cost::CCE => -(y * y_p.ln()).sum(),
        }
    }

    #[inline]
    fn derivate(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> ArrayD<f64> {
        match self {
            Cost::MSE => 2.0 * (y_p - y) / y.len() as f64,
            Cost::MAE => (y_p - y).signum() / y.len() as f64,
            Cost::BCE => y_p - y,
            Cost::CCE => y_p - y,
        }
    }

    #[inline]
    fn cost_name(&self) -> &str {
        match self {
            Cost::MSE => "MSE",
            Cost::MAE => "MAE",
            Cost::BCE => "BCE",
            Cost::CCE => "CCE",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cost_name() {
        let cost = Cost::MSE;
        assert_eq!(cost.cost_name(), "MSE");
    }

    #[test]
    fn test_mse_function() {
        let y_p = array![0.1, 0.4, 0.6].into_dyn();
        let y = array![0.0, 0.5, 1.0].into_dyn();
        let cost = Cost::MSE;
        let result = cost.function(&y_p.view(), &y.view());
        assert_eq!(result as f32, 0.06);
    }

    #[test]
    fn test_mae_function() {
        let y_p = array![0.1, 0.4, 0.6].into_dyn();
        let y = array![0.0, 0.5, 1.0].into_dyn();
        let cost = Cost::MAE;
        let result = cost.function(&y_p.view(), &y.view());
        assert_eq!(result as f32, 0.2); // Expected MAE
    }

    #[test]
    fn test_bce_function() {
        let y_p = array![0.07, 0.91, 0.74, 0.23, 0.85, 0.17, 0.94].into_dyn();
        let y = array![0., 1., 1., 0., 0., 1., 1.].into_dyn();
        let cost = Cost::BCE;
        let result = cost.function(&y_p.view(), &y.view());
        assert_eq!(result, 4.460303459760249);
    }

    #[test]
    fn test_mse_derivate() {
        let y_p = array![0.1, 0.4, 0.6].into_dyn();
        let y = array![0.0, 0.5, 1.0].into_dyn();
        let cost = Cost::MSE;
        let result = cost.derivate(&y_p.view(), &y.view());
        let expected = array![0.066666667, -0.066666667, -0.266666668].into_dyn();
        assert_eq!(result.mapv(|v| v as f32), expected.mapv(|v| v as f32));
    }

    #[test]
    fn test_mae_derivate() {
        let y_p = array![0.1, 0.4, 0.6].into_dyn();
        let y = array![0.0, 0.5, 1.0].into_dyn();
        let cost = Cost::MAE;
        let result = cost.derivate(&y_p.view(), &y.view());
        let expected = array![0.33333334, -0.33333334, -0.33333334].into_dyn();
        assert_eq!(result.mapv(|v| v as f32), expected.mapv(|v| v as f32));
    }

    #[test]
    fn test_bce_derivate() {
        let y_p = array![0.9, 0.1, 0.8, 0.2].into_dyn();
        let y = array![1., 0., 1., 0.].into_dyn();
        let cost = Cost::BCE;
        let result = cost.derivate(&y_p.view(), &y.view());
        let expected = array![-0.09999999999999998, 0.1, -0.19999999999999996, 0.2].into_dyn();
        assert_eq!(result.mapv(|v| v as f32), expected.mapv(|v| v as f32));
    }

    #[test]
    fn test_cce_derivate() {
        let y_p = array![0.9, 0.1, 0.8, 0.2].into_dyn();
        let y = array![1., 0., 1., 0.].into_dyn();
        let cost = Cost::BCE;
        let result = cost.derivate(&y_p.view(), &y.view());
        let expected = array![-0.09999999999999998, 0.1, -0.19999999999999996, 0.2].into_dyn();
        assert_eq!(result.mapv(|v| v as f32), expected.mapv(|v| v as f32));
    }

    #[test]
    fn test_custom_cost() {
        #[derive(Debug)]
        struct CustomCost;

        impl CostFunction for CustomCost {
            fn function(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> f64 {
                (y - y_p).abs().mean().unwrap_or(0.)
            }

            fn derivate(&self, y_p: &ArrayViewD<f64>, y: &ArrayViewD<f64>) -> ArrayD<f64> {
                (y_p - y).signum() / y.len() as f64
            }

            fn cost_name(&self) -> &str {
                "Custom Cost"
            }
        }

        let y_p = array![0.1, 0.4, 0.6].into_dyn();
        let y = array![0.0, 0.5, 1.0].into_dyn();

        let cost = CustomCost;

        assert_eq!(cost.cost_name(), "Custom Cost");

        let result = cost.function(&y_p.view(), &y.view());
        assert_eq!(result as f32, 0.2);

        let result = cost.derivate(&y_p.view(), &y.view());
        let expected = array![0.33333334, -0.33333334, -0.33333334];
        assert_eq!(
            result.mapv(|v| v as f32),
            expected.into_dyn().mapv(|v| v as f32)
        );
    }
}
