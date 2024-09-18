use ndarray::array;
use rs_nn::NN;

fn main() {
    let train_data = array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];
    
    let mut nn = NN::load("test.toml").unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    for input in train_data.rows() {
        let pred = nn.predict(&input.to_owned());
        let out = if pred[(0, 0)] < 0.5 { 0 } else { 1 };
        println!("{} --> {}", input, out)
    }
}