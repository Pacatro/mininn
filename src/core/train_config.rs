use mininn_derive::MSGPackFormatting;
use serde::{Deserialize, Serialize};

use crate::core::NNResult;
use crate::utils::{Cost, CostFunction, MSGPackFormatting, Optimizer};

/// Training configuration for [`NN`](crate::core::NN)
#[derive(Debug, Clone, Serialize, Deserialize, MSGPackFormatting)]
pub struct TrainConfig {
    cost: Box<dyn CostFunction>,
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
    optimizer: Optimizer,
    early_stopping: bool,
    patience: usize,
    tolerance: f32,
    verbose: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            cost: Box::new(Cost::MSE),
            epochs: 100,
            learning_rate: 0.1,
            batch_size: 1,
            optimizer: Optimizer::GD,
            early_stopping: false,
            patience: 0,
            tolerance: 0.0,
            verbose: true,
        }
    }
}

impl TrainConfig {
    /// Creates a new empty [`TrainConfig`].
    ///
    /// ## Returns
    ///
    /// A new configuration instance with empty values.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use mininn::prelude::*;
    /// let train_config = TrainConfig::new();
    /// assert_eq!(train_config.cost().name(), "MSE");
    /// assert_eq!(train_config.epochs(), 0);
    /// assert_eq!(train_config.learning_rate(), 0.0);
    /// assert_eq!(train_config.batch_size(), 1);
    /// assert_eq!(train_config.optimizer(), &Optimizer::GD);
    /// assert_eq!(train_config.early_stopping(), false);
    /// assert_eq!(train_config.patience(), 0);
    /// assert_eq!(train_config.tolerance(), 0.0);
    /// assert_eq!(train_config.verbose(), false);
    /// ```
    ///
    pub fn new() -> Self {
        Self {
            cost: Box::new(Cost::MSE),
            epochs: 0,
            learning_rate: 0.0,
            batch_size: 1,
            optimizer: Optimizer::GD,
            early_stopping: false,
            patience: 0,
            tolerance: 0.0,
            verbose: false,
        }
    }

    /// Sets the number of epochs to train the network.
    ///
    /// The number of epochs determines the number of times the network will be trained on the
    /// entire training dataset.
    ///
    /// ## Arguments
    ///
    /// * `epochs` - The number of epochs to train the network.
    ///
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the cost function to be used during training.
    ///
    /// The cost function is responsible for calculating the loss of the network during training.
    /// It takes the predicted output and the actual output as input and returns a scalar value
    /// representing the loss.
    ///
    /// ## Arguments
    ///
    /// * `cost` - The cost function to be used during training.
    ///
    pub fn with_cost(mut self, cost: impl CostFunction + 'static) -> Self {
        self.cost = Box::new(cost);
        self
    }

    /// Sets the learning rate of the optimizer.
    ///
    /// The learning rate determines the step size of the optimization algorithm. A higher learning
    /// rate means that the optimizer will move faster, but may also lead to unstable training.
    ///
    /// ## Arguments
    ///
    /// * `learning_rate` - The learning rate of the optimizer.
    ///
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the batch size of the training dataset.
    ///
    /// The batch size determines the number of samples that are processed at a time during training.
    /// A larger batch size means that the network will be able to learn more quickly, but may also
    /// lead to unstable training.
    ///
    /// ## Arguments
    ///
    /// * `batch_size` - The batch size of the training dataset.
    ///
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the optimizer to be used during training.
    ///
    /// The optimizer is responsible for updating the weights of the network during training. It
    /// takes the current weights and the gradients of the loss function as input and returns the
    /// updated weights.
    ///
    /// ## Arguments
    ///
    /// * `optimizer` - The optimizer to be used during training.
    ///
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Sets whether the training process should stop early.
    ///
    /// If set to `true`, the training process will stop early if the validation loss does not
    /// improve for a certain number of epochs.
    ///
    /// ## Arguments
    ///
    /// * `early_stopping` - Whether to stop early or not.
    /// * `patience` - The limit of epochs without improvement before the training process stops.
    /// * `tolerance` - The minimum improvement required to continue training.
    ///
    pub fn with_early_stopping(mut self, patience: usize, tolerance: f32) -> Self {
        if patience > 0 && tolerance > 0.0 {
            self.early_stopping = true;
            self.patience = patience;
            self.tolerance = tolerance;
        }
        self
    }

    /// Sets whether the training process should be verbose.
    ///
    /// If set to `true`, the training process will print out information about the training
    /// process, such as the current loss and the current epoch.
    ///
    /// ## Arguments
    ///
    /// * `verbose` - Whether the training process should be verbose.
    ///
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Returns the cost function used for training.
    #[inline]
    pub fn cost(&self) -> &Box<dyn CostFunction> {
        &self.cost
    }

    /// Returns the number of epochs to train the model.
    #[inline]
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// Returns the learning rate used for training.
    #[inline]
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the batch size used for training.
    #[inline]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the optimizer used for training.
    #[inline]
    pub fn optimizer(&self) -> &Optimizer {
        &self.optimizer
    }

    /// Returns whether early stopping is enabled.
    #[inline]
    pub fn early_stopping(&self) -> bool {
        self.early_stopping
    }

    /// Returns the patience used for early stopping.
    #[inline]
    pub fn patience(&self) -> usize {
        self.patience
    }

    /// Returns the tolerance used for early stopping.
    #[inline]
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Returns whether the neural network is in verbose mode.
    #[inline]
    pub fn verbose(&self) -> bool {
        self.verbose
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        core::TrainConfig,
        utils::{Cost, Optimizer},
    };

    #[test]
    fn test_train_config_new() {
        let train_config = TrainConfig::new();
        assert_eq!(train_config.cost().name(), "MSE");
        assert_eq!(train_config.epochs(), 0);
        assert_eq!(train_config.learning_rate(), 0.0);
        assert_eq!(train_config.batch_size(), 1);
        assert_eq!(train_config.optimizer(), &Optimizer::GD);
        assert_eq!(train_config.early_stopping(), false);
        assert_eq!(train_config.patience(), 0);
        assert_eq!(train_config.tolerance(), 0.0);
        assert_eq!(train_config.verbose(), false);
    }

    #[test]
    fn test_train_config_default() {
        let train_config = TrainConfig::default();
        assert_eq!(train_config.cost().name(), "MSE");
        assert_eq!(train_config.epochs(), 100);
        assert_eq!(train_config.learning_rate(), 0.1);
        assert_eq!(train_config.batch_size(), 1);
        assert_eq!(train_config.optimizer(), &Optimizer::GD);
        assert_eq!(train_config.verbose(), true);
    }

    #[test]
    fn test_custom_train_config() {
        let train_config = TrainConfig::new()
            .with_epochs(1000)
            .with_cost(Cost::CCE)
            .with_learning_rate(0.01)
            .with_batch_size(32)
            .with_optimizer(Optimizer::default_momentum())
            .with_verbose(true);

        assert_eq!(train_config.cost().name(), "CCE");
        assert_eq!(train_config.epochs(), 1000);
        assert_eq!(train_config.learning_rate(), 0.01);
        assert_eq!(train_config.batch_size(), 32);
        assert_eq!(train_config.optimizer(), &Optimizer::default_momentum());
        assert_eq!(train_config.verbose(), true);
    }
}
