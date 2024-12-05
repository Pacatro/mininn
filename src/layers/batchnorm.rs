use ndarray::Array1;
use serde::{Deserialize, Serialize};

// use crate::{layers::Layer, NNMode, NNResult};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BatchNorm {
    input: Array1<f64>,
    gamma: Array1<f64>,
    beta: Array1<f64>,
    epsilon: f64,
    momentum: f64,
    running_mean: Array1<f64>,
    running_var: Array1<f64>,
    mu: f64,
    xmu: Array1<f64>,
    carre: Array1<f64>,
    var: f64,
    sqrtvar: f64,
    invvar: f64,
    va2: Array1<f64>,
    va3: Array1<f64>,
    xbar: Array1<f64>,
    layer_type: String,
}

impl BatchNorm {
    #[inline]
    pub fn _new(
        epsilon: f64,
        momentum: f64,
        running_mean: Option<Array1<f64>>,
        running_var: Option<Array1<f64>>,
    ) -> Self {
        Self {
            input: Array1::zeros(0),
            gamma: Array1::ones(0),
            beta: Array1::zeros(0),
            epsilon,
            momentum,
            running_mean: running_mean.unwrap_or(Array1::zeros(0)),
            running_var: running_var.unwrap_or(Array1::zeros(0)),
            mu: 0.,
            xmu: Array1::zeros(0),
            carre: Array1::zeros(0),
            var: 0.,
            sqrtvar: 0.,
            invvar: 0.,
            va2: Array1::zeros(0),
            va3: Array1::zeros(0),
            xbar: Array1::zeros(0),
            layer_type: "BatchNorm".to_string(),
        }
    }

    #[inline]
    pub fn _gamma(&self) -> Array1<f64> {
        self.gamma.to_owned()
    }

    #[inline]
    pub fn _beta(&self) -> Array1<f64> {
        self.beta.to_owned()
    }

    #[inline]
    pub fn _epsilon(&self) -> f64 {
        self.epsilon
    }

    #[inline]
    pub fn _momentum(&self) -> f64 {
        self.momentum
    }

    #[inline]
    pub fn _running_mean(&self) -> Array1<f64> {
        self.running_mean.to_owned()
    }

    #[inline]
    pub fn _running_var(&self) -> Array1<f64> {
        self.running_var.to_owned()
    }
}

// impl Layer for BatchNorm {
//     #[inline]
//     fn layer_type(&self) -> String {
//         self.layer_type.to_string()
//     }

//     #[inline]
//     fn to_json(&self) -> NNResult<String> {
//         Ok(serde_json::to_string(self)?)
//     }

//     #[inline]
//     fn from_json(json_path: &str) -> NNResult<Box<dyn Layer>> {
//         Ok(Box::new(serde_json::from_str::<Self>(json_path)?))
//     }

//     #[inline]
//     fn as_any(&self) -> &dyn std::any::Any {
//         self
//     }

//     fn forward(&mut self, input: &Array1<f64>, mode: &NNMode) -> NNResult<Array1<f64>> {
//         self.input = input.to_owned();
//         // let d = self.input.len(); // let (n, d) = input.dim();

//         match mode {
//             NNMode::Train => {
//                 self.mu = 1. / 1. * self.input.sum(); // 1 / n * sum(input)
//                 self.xmu = &self.input - self.mu;
//                 self.carre = self.xmu.pow2();
//                 self.var = 1. / 1. * self.carre.sum(); // 1 / n * sum(carre)
//                 self.sqrtvar = (self.var + self.epsilon).sqrt();
//                 self.invvar = 1. / self.sqrtvar;
//                 self.va2 = &self.xmu * self.invvar;
//                 self.va3 = &self.gamma * &self.va2;
//                 let output = &self.va3 + &self.beta;
//                 self.running_mean =
//                     self.momentum * &self.running_mean + (1. - self.momentum) * self.mu;
//                 self.running_var =
//                     self.momentum * &self.running_var + (1. - self.momentum) * self.var;
//                 Ok(output)
//             }
//             NNMode::Test => {
//                 self.xbar =
//                     (input - &self.running_mean) / (&self.running_var + self.epsilon).sqrt();
//                 Ok(&self.gamma * &self.xbar + &self.beta)
//             }
//         }
//     }

//     fn backward(
//         &mut self,
//         output_gradient: &Array1<f64>,
//         _learning_rate: f64,
//         _optimizer: &crate::prelude::Optimizer,
//         _mode: &NNMode,
//     ) -> NNResult<Array1<f64>> {
//         // Step 9
//         let dva3 = output_gradient.to_owned();
//         // let dbeta = output_gradient.sum();

//         // Step 8
//         let dva2 = &self.gamma * &dva3;
//         // let dgamma = (&self.va2 * &dva3).sum();

//         // Step 7
//         let mut dxmu = self.invvar * &dva2;
//         let dinvvar = (&self.xmu * &dva2).sum();

//         // Step 6
//         let dsqrtvar = -1. / (self.sqrtvar.powi(2)) * &dinvvar;

//         // Step 5
//         let dvar = 0.5 * (self.var + self.epsilon).powf(-0.5) * &dsqrtvar;

//         // Step 4
//         let dcarre = 1. / 1. * Array1::ones(self.carre.len()) * dvar; // 1 / n * sum(carre)

//         // Step 3
//         dxmu = dxmu + (2. * &self.xmu * dcarre);

//         // Step 2
//         let mut dx = dxmu.to_owned();
//         let dmu = dxmu.sum();

//         // Step 1
//         dx = dx + (1. / 1. * Array1::ones(dxmu.len()) * dmu);

//         Ok(dx)
//     }
// }
