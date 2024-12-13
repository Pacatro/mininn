# 🏁 TODOs

## IMPORTANT

<!--- v0.1.3 --->
- [x] New costs functions
- [x] Batch in train algorithm
- [x] Add optimizers
- [x] Add Dropout layer
<!--- v0.1.4 --->
- [x] FIX INPUTS/OUTPUTS DIMENSIONS
- [x] Create custom Cost
- [x] Create custom Activation functions
- [x] Fix problems with activation register
- [x] Add early stopping
- [x] Check if clone is necessary
- [x] Change train API to use config struct instead of arguments (TrainConfig)
- [ ] Set layer dimensionality in the layer trait as a generic `Layer<D>`
- [ ] Dense layer should have a 1D input
- [ ] Every layer should have his own dimensionality (Flatten layer)
- [ ] Allow user to set format to save/load (JSON, MessagePack, etc) --> Use another trait for serialization (SerdeLayer/Uses serialize and deserialize traits)
- [ ] Check docs

<!--- v0.1.5 --->
<!--- https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/ --->
- [ ] Add BatchNorm layer
- [ ] Add Conv layer
- [ ] Add Pooling layer
- [ ] Add Deconv layer
- [ ] Add Embedding layer
- [ ] Add Recurrent layer

<!--- v0.2.0 --->
- [ ] Try to use GPU (WGPU, torch, etc)

## NOT IMPORTANT

- [ ] Refactoring train algorithm
- [ ] Fix Adam optimizer
- [ ] New docs
- [ ] Allow users to set type of the numbers used in the neural network (f32 or f64)
