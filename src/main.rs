use ndarray::{array, Array1, Array2};
use rs_nn::{activation::Activation, layer::Layer, nn::NN};

fn main() {    
    let data: Array2<f64> = array![
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
        
    let labels: Array1<f64> = array![1., -1., 1., -1., -1., 1., -1., 1., -1., -1.];

    let mut l2 = Layer::new(3, 2, Activation::SIGMOID);
    
    l2.set_weights(array![
        [-0.124,  0.871],
        [0.692, -0.036],
        [0.455,  1.239]
    ]);

    l2.set_biases(array![-0.923, 0.02, -2.918]);

    let mut l3 = Layer::new(2, 3, Activation::SIGMOID);

    l3.set_weights(array![
        [-0.006,  0.295, 0.207],
        [0.126, 0.399, -0.637],
    ]);

    l3.set_biases(array![-0.07, 1.912]);

    let mut nn = NN::void();
    nn.set_layers(vec![l2, l3]);
    
    let input = data.row(0).to_owned();

    let results = nn.predict(&input);

    println!("{}", results);

    nn.train(20, data, labels, rs_nn::cost::Cost::MSE);

    let results = nn.predict(&input);

    println!("{}", results);

}