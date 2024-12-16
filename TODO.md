# 🏁 TODOs

## IMPORTANT

<!--- v0.1.3 --->
- [x] New costs functions
- [x] Batch in train algorithm
- [x] Add optimizers
- [x] Add Dropout layer

<!--- v0.1.4 API --->
- [x] FIX INPUTS/OUTPUTS DIMENSIONS
- [x] Create custom Cost
- [x] Create custom Activation functions
- [x] Fix problems with activation register
- [x] Add early stopping
- [x] Check if clone is necessary
- [x] Change train API to use config struct instead of arguments (TrainConfig)
- [x] Flatten layer
- [x] Separates from/to MSGPACK logic from Layer trait
- [x] Try to use only one global register isntead of three
- [x] Separates builder and gloabl register logic
- [x] Create `register!` macro to register all layers, activations and costs
- [x] Create derive macro for CostFunction and ActivationFunction
- [x] Create derive macro for `MSGPackFormat`
- [x] Generalize CostFunction and ActivationFunction traits (NNUtils)
- [ ] Check docs
- [ ] Check `README`

<!--- v0.1.5 LAYERS --->
[Layers implementations](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/)

- [ ] Add BatchNorm layer
- [ ] Add Conv layer
- [ ] Add Pooling layer
- [ ] Add Deconv layer
- [ ] Add Embedding layer
- [ ] Add Recurrent layer

<!--- v0.2.0 OPTIMIZATIONS --->
- [ ] Improve backpropagation ([resilient propagation](https://medium.com/@Ahmad_AM0/resilient-propagation-e76b569beea2))
- [ ] [Multithreading](https://www.heatonresearch.com/encog/mprop/compare.html)
- [ ] Try to use GPU (WGPU, torch, etc)

## NOT IMPORTANT

- [ ] See erased-serde
- [ ] Refactoring train algorithm
- [ ] Fix Adam optimizer
- [ ] New docs
- [ ] Allow users to set type of the numbers used in the neural network (f32 or f64)
