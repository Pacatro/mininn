// See: https://github.com/nmarincic/machineintelligence/blob/master/11.%20Building%20a%20Neural%20Network%20using%20Matrices%20-%20Step%20by%20Step.ipynb
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
    learning_rate: f64,
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
    pub fn new(topology: &[usize], activations: &[Activation], learning_rate: f64) -> NN {
        assert_eq!(
            activations.len(),
            topology.len()-1,
            "The input layer does not have activations, so the number of activations must be topology.len()-1"
        );
        
        let mut layers = Vec::with_capacity(topology.len()-1);

        for i in 0..topology.len()-1 {
            layers.push(Layer::new(topology[i+1], topology[i], activations[i]));
        }

        NN { layers, learning_rate }
    }

    /// Creates a void network, without layers and learning rate equals to `0.0`
    pub fn void() -> NN {
        NN { layers: Vec::new(), learning_rate: 0.0 }
    }

    /// Returns the layers of the network
    pub fn layers(&self) -> &[Layer] {
        &self.layers.as_slice()
    }

    /// Returns the learning rate of the network
    pub fn learning_rate(&self) -> &f64 {
        &self.learning_rate
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

    /// Set the learning rate of the network
    /// 
    /// ## Arguments
    /// 
    /// - `learning_rate`: The new learning rate of the network
    /// 
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate
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
            // println!("Weights: {}\n", layer.weights());
            // println!("Biases: {}", layer.biases());

            let layer_act = layer.activation().function();
            let z = layer.weights().dot(&activation) + layer.biases();
            let a = layer_act(&z);
            results.push((z.clone(), a.clone()));
            activation = a;
        }

        results
    }

    /// # Backpropagation algorithm
    /// 
    /// Compute the deltas (gradients) for each layer during backpropagation
    /// 
    /// ## Arguments
    /// 
    /// - `outputs`: A vector of pairs `(z, a)` where `z` is the pre-activation value and `a` is the activation value
    /// - `labels`: The true labels for the input data
    /// - `cost`: The cost function to calculate the loss and its gradient
    /// 
    /// ## Returns
    /// 
    /// A vector of deltas for each layer
    ///
    pub(crate) fn backward(&self, outputs: &Vec<(Array1<f64>, Array1<f64>)>, label: &f64, cost: Cost) -> Vec<Array1<f64>> {
        let mut deltas = Vec::with_capacity(self.layers.len());
        let d_cost = cost.derivate();

        // Deltas for the last layer
        let last_layer = self.layers.last().unwrap();
        let (last_z, last_a) = outputs.last().unwrap();
        let last_dc_a: Array1<f64> = last_a.iter().map(|a| d_cost(a, label)).collect();
        deltas.insert(0, last_dc_a * &last_layer.activation().derivate()(last_z));

        // Deltas for the other layers
        for l in (0..self.layers.len()-1).rev() {
            let z = &outputs[l].0;
            let da_dz = self.layers[l].activation().derivate()(z);
            let delta_l_1 = self.layers[l+1].weights().t().dot(&deltas[0]) * &da_dz;
            deltas.insert(0, delta_l_1);
        }
        
        deltas
    }

    /// # Gradient descent algorithm
    /// 
    /// Compute the new weights and biases of the network
    /// 
    /// ## Arguments
    /// 
    /// - `deltas`: The derivations computed in the backpropagation algoritm
    /// - `outputs`: A vector of pairs `(z, a)` where `z` is the pre-activation value and `a` is the activation value
    ///  
    pub(crate) fn gradient_descent(&mut self, deltas: &Vec<Array1<f64>>, outputs: &Vec<(Array1<f64>, Array1<f64>)>) {
        for l in 0..self.layers.len() {
            // Update biases
            let new_biases = self.layers[l].biases() - &deltas[0].mean().unwrap() * self.learning_rate;
            self.layers[l].set_biases(&new_biases);
            
            // Update weights
            // let prev_activation = if l == 0 {
            //     Array2::from_shape_vec((outputs[l].1.len(), 1), outputs[l].1.to_vec()).unwrap()
            // } else {
            //     Array2::from_shape_vec((outputs[l-1].1.len(), 1), outputs[l-1].1.to_vec()).unwrap()
            // };

            let prev_activation = if l == 0 { &outputs[l].1 } else { &outputs[l-1].1 };

            // let delta = Array2::from_shape_vec((deltas[0].len(), 1), deltas[0].to_vec()).unwrap();

            let new_weights = self.layers[l].weights() - prev_activation.t().dot(&deltas[0]) * self.learning_rate;

            self.layers[l].set_weights(&new_weights);
        }
    }

    // Train the model with the data
    pub fn train(&mut self, epochs: u32, inputs: &Array2<f64>, labels: &Array1<f64>, cost: Cost) {
        // TODO: DO MINI BATCH
        for _ in 0..epochs {
            for (input, label) in inputs.rows().into_iter().zip(labels) {
                let outputs = self.forward(&input.to_owned());
                let deltas = self.backward(&outputs, &label, cost);
                self.gradient_descent(&deltas, &outputs);
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
        let results = self.forward(input);
        // println!("{}\n", results[self.layers.len()-1].0);
        // println!("{}", results[self.layers.len()-1].1);
        results[self.layers.len()-1].1.clone()
    }

    /// Calc the cost of the network
    /// 
    /// ## Arguments
    /// 
    /// - `labels`: The labels of the dataset
    /// - `predictions`: The predictions of the network
    /// - `cost`: The cost fucntion (MSE, CCE, EXP, ...)
    /// 
    /// ## Returns
    /// 
    /// The average network cost
    /// 
    pub fn cost(&self, labels: &Array1<f64>, predictions: &Array1<f64>, cost: Cost) -> f64 {
        predictions
            .iter()
            .zip(labels)
            .map(|(p, l)| cost.function()(p, l))
            .collect::<Array1<f64>>()
            .mean()
            .unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use  ndarray::array;

    #[test]
    fn test_new_nn() {
        let nn = NN::new(&[2, 3, 2], &[Activation::SIGMOID; 2], 0.5);
        assert!(!nn.layers().is_empty());
        assert_eq!(nn.layers().len(), 2);
        assert_eq!(nn.layers()[0].weights().nrows(), 3);
        assert_eq!(nn.layers()[0].weights().ncols(), 2);
        assert_eq!(nn.layers()[0].activation(), &Activation::SIGMOID);
        assert_eq!(nn.learning_rate(), &0.5);
    }

    #[test]
    #[should_panic(expected = "The input layer does not have activations, so the number of activations must be topology.len()-1")]
    fn test_wrong_new_nn() {
        NN::new(&[1, 2, 1], &[Activation::SIGMOID; 4], 0.5);
    }

    #[test]
    fn test_void_nn() {
        let nn = NN::void();
        assert!(nn.layers.is_empty());
        assert_eq!(nn.learning_rate, 0.0);
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
        // let mut l2 = Layer::new(3, 2, Activation::SIGMOID);
    
        // l2.set_weights(array![
        //     [-0.124,  0.871],
        //     [0.692, -0.036],
        //     [0.455,  1.239]
        // ]);

        // l2.set_biases(array![-0.923, 0.02, -2.918]);

        // let mut l3 = Layer::new(2, 3, Activation::SIGMOID);

        // l3.set_weights(array![
        //     [-0.006,  0.295, 0.207],
        //     [0.126, 0.399, -0.637],
        // ]);

        // l3.set_biases(array![-0.07, 1.912]);

        // let mut nn = NN::void();
        // nn.set_layers(vec![l2, l3]);

        // let outputs = nn.forward(&array![1.2, 0.7]);
        // let deltas = nn.backward(&outputs, &array![0., 1.], Cost::MSE);

        // assert_eq!(deltas[0], array![0.07669863420246847, 0.2547447312050006]);
        // assert_eq!(deltas[1], array![-0.02137554443369121, 0.017925212251251512, 0.5517396363471382]);
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

    #[test]
    fn test_train() {
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

        let data = array![
            [1.2, 0.7],
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

        let labels = array![1., -1., 1., -1.,-1., 1., -1., 1., -1., -1.];

        let predictions = nn.predict(&data.row(0).to_owned());
        let old_cost = nn.cost(&labels, &predictions, Cost::MSE);

        nn.train(1, &data, &labels, Cost::MSE);

        let predictions = nn.predict(&data.row(0).to_owned());

        assert!(nn.cost(&labels, &predictions, Cost::MSE) < old_cost)

    }

}