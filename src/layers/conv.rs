// https://www.ibm.com/topics/convolutional-neural-networks

use crate::error::NNResult;

use ndarray::{Array1, Array2, Array3, ArrayView1};
use serde::{Deserialize, Serialize};

use super::Layer;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Conv {
    input_data: Array3<f64>,
    kernel: Array2<f64>
}

impl Conv {
    pub fn new(input_data: &Array3<f64>, kernel: &Array2<f64>) -> Self {
        todo!()
    }
}

impl Layer for Conv {
    fn layer_type(&self) -> String {
        todo!()
    }

    fn to_json(&self) -> NNResult<String> {
        todo!()
    }

    fn from_json(json: &str) -> NNResult<Box<dyn Layer>> where Self: Sized {
        todo!()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        todo!()
    }

    fn forward(&mut self, input: &Array1<f64>) -> NNResult<Array1<f64>> {
        todo!()
    }

    fn backward(&mut self, output_gradient: ArrayView1<f64>, learning_rate: f64) -> NNResult<Array1<f64>> {
        todo!()
    }
}