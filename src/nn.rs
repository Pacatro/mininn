// SEE https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
use ndarray::{Array1, Array2};

use crate::{activation::Activation, cost::Cost, layer::Layer};

/// Represents a linear neural network
/// 
/// ## Atributes
/// 
/// - `layers`: The layers of the neural network
/// - `learning_rate`: The learning rate of the network
/// 
#[derive(Debug)]
pub struct NN {
    layers: Vec<Layer>,
}

impl NN {
    /// Creates a new neural network [`NN`]
    /// 
    /// ## Arguments
    /// 
    /// - `topology`: The topology of the network
    /// - `activations`: The activations for the layers of the network
    /// - `learning_rate`: The learning_rate for train the network
    /// 
    pub fn new(topology: &[usize], activations: &[Activation]) -> NN {
        assert_eq!(
            activations.len(),
            topology.len()-1,
            "The input layer does not have activations, so the number of activations must be topology.len()-1"
        );
        
        let mut layers = Vec::with_capacity(topology.len()-1);

        for i in 0..topology.len()-1 {
            layers.push(Layer::new(topology[i+1], topology[i], activations[i]));
        }

        NN { layers }
    }

    /// Creates a void network, without layers and learning rate equals to `0.0`
    pub fn void() -> NN {
        NN { layers: Vec::new() }
    }

    /// Returns the layers of the network
    pub fn layers(&self) -> &[Layer] {
        &self.layers.as_slice()
    }

    /// Set the layers of the network
    /// 
    /// ## Arguments
    /// 
    /// - `layers`: The new layers of the network
    /// 
    pub fn set_layers(&mut self, layers: Vec<Layer>) {
        self.layers = layers
    }

    /// Insert a new layer to the network
    /// 
    /// ## Arguments
    /// 
    /// - `layer`: The new layer to insert
    /// 
    pub fn push_layer(&mut self, layer: Layer) {
        self.layers.push(layer)
    }

    /// # Forward propagation algorithm
    /// 
    /// Compute the activations of the hidden layers
    /// 
    /// ## Arguments
    /// 
    /// - `input`: A point of the dataset
    /// 
    /// ## Returns
    /// 
    /// A vector of pairs `(z, a)` where `z` is the pre-activation value and `a` is the activation value
    /// 
    pub(crate) fn forward(&self, input: &Array1<f64>) -> Vec<(Array1<f64>, Array1<f64>)> {
        let mut results = Vec::with_capacity(self.layers.len());
        let mut activation = input.clone();

        for layer in &self.layers {
            let z = layer.weights().dot(&activation) + layer.biases();
            let a = layer.activation().function(&z);
            results.push((z.clone(), a.clone()));
            activation = a;
        }

        results
    }

    /// # Backpropagation algorithm
    /// 
    /// Compute the gradients for each layer during backpropagation
    /// 
    /// ## Arguments
    /// 
    /// - `outputs`: A vector of pairs `(z, a)` where `z` is the pre-activation value and `a` is the activation value
    /// - `labels`: The true labels for the input data
    /// - `cost`: The cost function to calculate the loss and its gradient
    /// 
    /// ## Returns
    /// 
    /// A vector of gradients for each layer
    ///
    pub(crate) fn backward(&self, outputs: &Vec<(Array1<f64>, Array1<f64>)>, label: &Array1<f64>, cost: Cost) -> Vec<Array1<f64>> {
        let mut gradients = Vec::with_capacity(self.layers.len());

        // Gradients for last layer
        let (last_z, last_a) = outputs.last().unwrap().to_owned();
        let last_delta = cost.derivate(&last_a, label) * self.layers.last().unwrap().activation().derivate(&last_z);
        gradients.insert(0, last_delta);

        for l in (0..self.layers.len()-1).rev() {
            let (_, a) = outputs[l].to_owned();
            let delta_l = self.layers[l+1].weights().t().dot(&gradients[0]) * self.layers[l+1].activation().derivate(&a);
            gradients.insert(0, delta_l);
        }
        
        gradients
    }

    /// # Gradient descent algorithm
    /// 
    /// Compute the new weights and biases of the network
    /// 
    /// ## Arguments
    /// 
    /// - `gradients`: The derivations computed in the backpropagation algoritm
    /// - `outputs`: A vector of pairs `(z, a)` where `z` is the pre-activation value and `a` is the activation value
    /// - `learning_rate`: The learning rate
    ///  
    pub(crate) fn gradient_descent(
        &mut self,
        gradients: &Vec<Array1<f64>>, 
        outputs: &Vec<(Array1<f64>, Array1<f64>)>,
        learning_rate: f64
    ) {
        for l in 0..self.layers.len() {
            // Update biases
            let new_biases = self.layers[l].biases() - gradients[l].mean().unwrap() * learning_rate;
            self.layers[l].set_biases(&new_biases);

            // Update weights
            let new_weights = self.layers[l].weights() - gradients[l].dot(&outputs[l].1.t()) * learning_rate;
            self.layers[l].set_weights(&new_weights);
        }
    }

    /// Train the model
    /// 
    /// ## Arguments
    /// 
    /// - `epochs`: Number of train epochs
    /// - `data`: The data for the training
    /// - `labels`: The labels of each data
    /// - `cost`: The cost function
    /// - `learning_rate`: The learning rate
    /// 
    pub fn train(&mut self, epochs: u32, data: &Array2<f64>, labels: &Array2<f64>, cost: Cost, learning_rate: f64, verbose: bool) {
        for e in 0..epochs {
            let mut error = 0.0;

            for (x, y) in data.rows().into_iter().zip(labels.rows()) {
                let outputs = self.forward(&x.to_owned());

                error += cost.function(&y.to_owned(), &outputs.last().unwrap().1);
                
                let deltas = self.backward(&outputs, &y.to_owned(), cost);
                self.gradient_descent(&deltas, &outputs, learning_rate);
            }

            error /= data.len() as f64;

            if verbose {
                println!("{}/{}: error = {}", e+1, epochs, error);
            }
        }
    }

    /// Returns the predictions of the neurons of the last layer
    /// 
    /// ## Arguments
    /// 
    /// - `input`: The point to predict
    /// 
    /// ## Returns
    /// 
    /// A array with all the outputs of the last neurons
    /// 
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        self.forward(input).last().unwrap().1.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use  ndarray::array;

    #[test]
    fn test_new_nn() {
        let nn = NN::new(&[2, 3, 2], &[Activation::SIGMOID; 2]);
        assert!(!nn.layers().is_empty());
        assert_eq!(nn.layers().len(), 2);
        assert_eq!(nn.layers()[0].weights().nrows(), 3);
        assert_eq!(nn.layers()[0].weights().ncols(), 2);
        assert_eq!(nn.layers()[0].activation(), &Activation::SIGMOID);
    }

    #[test]
    #[should_panic(expected = "The input layer does not have activations, so the number of activations must be topology.len()-1")]
    fn test_wrong_new_nn() {
        NN::new(&[1, 2, 1], &[Activation::SIGMOID; 4]);
    }

    #[test]
    fn test_void_nn() {
        let nn = NN::void();
        assert!(nn.layers.is_empty());
    }

    #[test]
    fn test_push_layer() {
        let mut nn = NN::void();
        let layer = Layer::new(4, 3, Activation::SIGMOID);
        nn.push_layer(layer.clone());

        assert_eq!(nn.layers.len(), 1);
        assert_eq!(nn.layers()[0], layer);
    }

    #[test]
    fn test_forward() {
        let mut l2 = Layer::new(3, 2, Activation::SIGMOID);

        l2.set_weights(&array![
            [-0.124,  0.871],
            [0.692, -0.036],
            [0.455,  1.239]
        ]);

        l2.set_biases(&array![-0.923, 0.02, -2.918]);

        let mut l3 = Layer::new(2, 3, Activation::SIGMOID);

        l3.set_weights(&array![
            [-0.006,  0.295, 0.207],
            [0.126, 0.399, -0.637],
        ]);

        l3.set_biases(&array![-0.07, 1.912]);

        let mut nn = NN::void();
        nn.set_layers(vec![l2, l3]);

        let results = nn.forward(&array![1.2, 0.7]);

        assert_eq!(results[0].0, array![-0.4621000000000001, 0.8251999999999999, -1.5047000000000001]);
        assert_eq!(results[0].1, array![0.3864877638765807, 0.6953390395346613, 0.18172558150400955]);
        assert_eq!(results[1].0, array![0.17042328545079555, 2.122378539604725]);
        assert_eq!(results[1].1, array![0.5425029993583159, 0.8930593029399653]);
    }

    #[test]
    fn test_backward() {
        let data = array![
            [ 1.2, 0.7],
            [-0.3,-0.5],
            [ 3.0, 0.1],
            [-0.1,-1.0],
            [-0.0, 1.1],
            [ 2.1,-1.3],
            [ 3.1,-1.8],
            [ 1.1,-0.1],
            [ 1.5,-2.2],
            [ 4.0,-1.0]
        ];

        let labels = array![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0];

        let mut l_vec = Vec::new();

        for l in &labels {
            if *l == -1.0 {
                l_vec.extend_from_slice(&[1.0, 0.0]);
            } else {
                l_vec.extend_from_slice(&[0.0, 1.0]);
            }
        }

        let labels: Array2<f64> = Array2::from_shape_vec((labels.len(), 2), l_vec).unwrap();

        let mut l2 = Layer::new(3, 2, Activation::SIGMOID);

        l2.set_weights(&array![
            [-0.124,  0.871],
            [0.692, -0.036],
            [0.455,  1.239]
        ]);

        l2.set_biases(&array![-0.923, 0.02, -2.918]);

        let mut l3 = Layer::new(2, 3, Activation::SIGMOID);

        l3.set_weights(&array![
            [-0.006,  0.295, 0.207],
            [0.126, 0.399, -0.637],
        ]);

        l3.set_biases(&array![-0.07, 1.912]);

        let mut nn = NN::void();
        nn.set_layers(vec![l2, l3]);

        let outputs = nn.forward(&data.row(0).to_owned());
        let deltas = nn.backward(&outputs, &labels.row(0).to_owned(), Cost::MSE);

        // assert_eq!(deltas[0], array![-0., 0.008, 0.005]);
        assert_eq!(deltas[0], array![-0.0005046082755957006, 0.007915401217367644, 0.008523817748346771]);
        assert_eq!(deltas[1], array![0.13464571548343257, -0.010213305437068448]);
    }

    #[test]
    fn test_gradient_descend() {
        let data = array![
            [ 1.2, 0.7],
            [-0.3,-0.5],
            [ 3.0, 0.1],
            [-0.1,-1.0],
            [-0.0, 1.1],
            [ 2.1,-1.3],
            [ 3.1,-1.8],
            [ 1.1,-0.1],
            [ 1.5,-2.2],
            [ 4.0,-1.0]
        ];

        let labels = array![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0];

        let mut l_vec = Vec::new();

        for l in &labels {
            if *l == -1.0 {
                l_vec.extend_from_slice(&[1.0, 0.0]);
            } else {
                l_vec.extend_from_slice(&[0.0, 1.0]);
            }
        }

        let labels: Array2<f64> = Array2::from_shape_vec((labels.len(), 2), l_vec).unwrap();

        let mut l2 = Layer::new(3, 2, Activation::SIGMOID);

        let w2 = array![
            [-0.124,  0.871],
            [0.692, -0.036],
            [0.455,  1.239]
        ];

        let b2 = &array![-0.923, 0.02, -2.918];

        l2.set_weights(&w2);
        l2.set_biases(&b2);

        let mut l3 = Layer::new(2, 3, Activation::SIGMOID);

        let w3 = array![
            [-0.006, 0.295, 0.207],
            [0.126, 0.399, -0.637],
        ];

        let b3 = array![-0.07, 1.912];

        l3.set_weights(&w3);

        l3.set_biases(&b3);

        let mut nn = NN::void();
        nn.set_layers(vec![l2, l3]);

        let outputs = nn.forward(&data.row(0).to_owned());
        let deltas = nn.backward(&outputs, &labels.row(0).to_owned(), Cost::MSE);
        nn.gradient_descent(&deltas, &outputs, 0.1);
        
        assert_ne!(nn.layers()[0].weights(), w2);
        assert_ne!(nn.layers()[0].biases(), b2);
        assert_ne!(nn.layers()[1].weights(), w3);
        assert_ne!(nn.layers()[1].biases(), b3);
    }

    #[test]
    fn test_predict() {
        let mut l2 = Layer::new(3, 2, Activation::SIGMOID);
    
        l2.set_weights(&array![
            [-0.124,  0.871],
            [0.692, -0.036],
            [0.455,  1.239]
        ]);

        l2.set_biases(&array![-0.923, 0.02, -2.918]);

        let mut l3 = Layer::new(2, 3, Activation::SIGMOID);

        l3.set_weights(&array![
            [-0.006,  0.295, 0.207],
            [0.126, 0.399, -0.637],
        ]);

        l3.set_biases(&array![-0.07, 1.912]);

        let mut nn = NN::void();
        nn.set_layers(vec![l2, l3]);

        let predictions = nn.predict(&array![1.2, 0.7]);

        assert_eq!(predictions, array![0.5425029993583159, 0.8930593029399653]);
    }
}