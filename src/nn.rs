use crate::{activation::Activation, layer::Layer};

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

    /// Creates a void network, without layers and learning rate equals to 0
    pub fn void() -> NN {
        NN { layers: Vec::new(), learning_rate: 0.0 }
    }

    /// Returns the layers of the network
    pub fn layers(&self) -> &[Layer] {
        &self.layers.as_slice()
    }

    /// Returns an specific hidden layer, it is the same as `nn.layers()[idx]`
    /// 
    /// ## Arguments
    /// 
    /// - `idx`: Index of the layer
    /// 
    pub fn layer(&self, idx: usize) -> &Layer {
        assert!(idx < self.layers.len(), "Invalid layer index");
        &self.layers[idx]
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
    pub fn insert_layer(&mut self, layer: Layer) {
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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new_nn() {
        let nn = NN::new(&[1, 2, 1], &[Activation::SIGMOID; 2], 0.5);
        assert!(!nn.layers().is_empty());
        assert_eq!(nn.layers().len(), 2);
        assert_eq!(nn.layer(0).weights().nrows(), 1);
        assert_eq!(nn.layer(0).weights().ncols(), 2);
        assert_eq!(nn.layer(0).activation(), &Activation::SIGMOID);
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
    #[should_panic(expected = "Invalid layer index")]
    fn test_invalid_layer_index() {
        let nn = NN::void();
        nn.layer(0); // This should panic because there are no layers
    }

    #[test]
    fn test_insert_layer() {
        let mut nn = NN::void();
        let layer = Layer::new(4, 3, Activation::SIGMOID);
        nn.insert_layer(layer.clone());

        assert_eq!(nn.layers.len(), 1);
        assert_eq!(nn.layer(0), &layer);
    }
}