use ndarray::{Array1, Array2};

/// Calculate the metrics for a classification model based on the labels and the predictions.
#[derive(Debug)]
pub struct MetricsCalculator {
    confusion_matrix: Array2<f64>,
}

impl MetricsCalculator {
    /// Creates a new `MetricsCalculator` instance with the given true labels and predicted labels.
    ///
    /// ## Arguments
    ///
    /// * `labels` -  True labels for the classification problem.
    /// * `predictions` -  Predicted labels for the classification problem.
    ///
    pub fn new(labels: &Array2<f64>, predictions: &Array1<f64>) -> Self {
        if labels.shape()[0] != predictions.shape()[0] {
            return Self {
                confusion_matrix: Array2::zeros((0, 0)),
            };
        }

        if labels.is_empty() || predictions.is_empty() {
            return Self {
                confusion_matrix: Array2::zeros((0, 0)),
            };
        }

        let num_classes = labels.iter().map(|e| *e as usize).max().unwrap_or(0) + 1;

        let mut confusion_matrix = Array2::zeros((num_classes, num_classes));

        for (true_label, pred_label) in labels.iter().zip(predictions.iter()) {
            confusion_matrix[(*true_label as usize, *pred_label as usize)] += 1.0;
        }

        Self { confusion_matrix }
    }

    /// Calculates the confusion matrix for the classification problem.
    ///
    /// The confusion matrix is a 2D array where the rows represent the true labels and the columns
    /// represent the predicted labels. The value at each cell represents the number of instances
    /// that were classified as the predicted label when the true label was the row label.
    ///
    /// For binary classificaton, the confussion matrix will be like this:
    /// 
    /// ```text
    /// +----+----+
    /// | TP | FP |
    /// +-----+---+
    /// | FN | TN |
    /// +----+----+
    /// ```
    /// 
    /// Where:
    ///
    /// - `TP`: True Positive
    /// - `FP`: False Positive
    /// - `TN`: True Negative
    /// - `FN`: False Negative
    ///
    /// ## Returns
    ///
    /// Returns a 2D array (`Array2<f64>`) representing the confusion matrix.
    ///
    #[inline]
    pub fn confusion_matrix(&self) -> &Array2<f64> {
        &self.confusion_matrix
    }

    /// Calculates the accuracy of the classification model.
    ///
    /// Accuracy is the ratio of correctly predicted instances to the total number of instances.
    ///
    /// ## Returns
    ///
    /// The accuracy of the classification model.
    ///
    pub fn accuracy(&self) -> f64 {
        if self.confusion_matrix.is_empty() {
            return 0.0;
        }

        let total_examples = self.confusion_matrix.sum();
        let correct_predictions = self.confusion_matrix.diag().sum();
        correct_predictions / total_examples
    }

    /// Calculates the precision of the classification model for multiple classes.
    ///
    /// Precision is the ratio of correctly predicted positive instances to the total number of
    /// predicted positive instances, averaged across all classes.
    ///
    /// ## Returns
    ///
    /// The precision of the classification model.
    ///
    pub fn precision(&self) -> f64 {
        if self.confusion_matrix.is_empty() {
            return 0.0;
        }

        let num_classes = self.confusion_matrix.shape()[0];
        let mut precision_sum = 0.0;

        for i in 0..num_classes {
            let true_positives = self.confusion_matrix[[i, i]];
            let predicted_positives = self.confusion_matrix.column(i).sum();
            precision_sum += true_positives / predicted_positives;
        }

        precision_sum / num_classes as f64
    }

    /// Calculates the recall of the classification model for multiple classes.
    ///
    /// Recall is the ratio of correctly predicted positive instances to the total number of
    /// actual positive instances, averaged across all classes.
    ///
    /// ## Returns
    ///
    /// The recall of the classification model.
    ///
    pub fn recall(&self) -> f64 {
        if self.confusion_matrix.is_empty() {
            return 0.0;
        }

        let num_classes = self.confusion_matrix.shape()[0];
        let mut recall_sum = 0.0;

        for i in 0..num_classes {
            let true_positives = self.confusion_matrix[[i, i]];
            let actual_positives = self.confusion_matrix.row(i).sum();
            recall_sum += true_positives / actual_positives;
        }

        recall_sum / num_classes as f64
    }

    /// Calculates the F1-score of the classification model for multiple classes.
    ///
    /// The F1-score is the harmonic mean of precision and recall, averaged across all classes.
    ///
    /// ## Returns
    ///
    /// The F1-score of the classification model.
    ///
    pub fn f1_score(&self) -> f64 {
        if self.confusion_matrix.is_empty() {
            return 0.0;
        }

        let num_classes = self.confusion_matrix.shape()[0];
        let mut f1_sum = 0.0;

        for i in 0..num_classes {
            let true_positives = self.confusion_matrix[[i, i]];
            let predicted_positives = self.confusion_matrix.column(i).sum();
            let actual_positives = self.confusion_matrix.row(i).sum();

            let precision = true_positives / predicted_positives;
            let recall = true_positives / actual_positives;

            let f1 = (2.0 * precision * recall) / (precision + recall);
            f1_sum += f1;
        }

        f1_sum / num_classes as f64
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_confusion_matrix_binary_classification() {
        let labels = array![[0.], [0.], [1.], [0.], [1.], [1.]];
        let predictions = array![0., 1., 1., 0., 0., 1.];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let confusion_matrix = class_metrics.confusion_matrix();

        assert_eq!(confusion_matrix[[0, 0]], 2.0); // TP
        assert_eq!(confusion_matrix[[0, 1]], 1.0); // FP
        assert_eq!(confusion_matrix[[1, 0]], 1.0); // FN
        assert_eq!(confusion_matrix[[1, 1]], 2.0); // TN
    }

    #[test]
    fn test_confusion_matrix_multi_class_classification() {
        let labels = array![[0.], [1.], [2.], [0.], [1.], [2.]];
        let predictions = array![0., 2., 1., 0., 1., 2.];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let confusion_matrix = class_metrics.confusion_matrix();

        assert_eq!(confusion_matrix[[0, 0]], 2.0);
        assert_eq!(confusion_matrix[[0, 1]], 0.0);
        assert_eq!(confusion_matrix[[0, 2]], 0.0);
        assert_eq!(confusion_matrix[[1, 0]], 0.0);
        assert_eq!(confusion_matrix[[1, 1]], 1.0);
        assert_eq!(confusion_matrix[[1, 2]], 1.0);
        assert_eq!(confusion_matrix[[2, 0]], 0.0);
        assert_eq!(confusion_matrix[[2, 1]], 1.0);
        assert_eq!(confusion_matrix[[2, 2]], 1.0);
    }

    #[test]
    fn test_empty_confusion_matrix() {
        let labels = Array2::<f64>::zeros((0, 0));
        let predictions = array![];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let confusion_matrix = class_metrics.confusion_matrix();

        assert_eq!(confusion_matrix.shape(), [0, 0]);
        assert_eq!(confusion_matrix, Array2::<f64>::zeros((0, 0)));
    }

    #[test]
    fn test_accuracy() {
        let labels = array![[0.], [0.], [1.], [0.], [1.], [1.]];
        let predictions = array![0., 1., 1., 0., 0., 1.];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let accuracy = class_metrics.accuracy();

        assert_eq!(accuracy, 0.6666666666666666);
    }

    #[test]
    fn test_precision() {
        let labels = array![[0.], [0.], [1.], [0.], [1.], [1.]];
        let predictions = array![0., 1., 1., 0., 0., 1.];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let precision = class_metrics.precision();

        assert_eq!(precision, 0.6666666666666666);
    }

    #[test]
    fn test_recall() {
        let labels = array![[0.], [0.], [1.], [0.], [1.], [1.]];
        let predictions = array![0., 1., 1., 0., 0., 1.];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let recall = class_metrics.recall();

        assert_eq!(recall, 0.6666666666666666);
    }

    #[test]
    fn test_f1_score() {
        let labels = array![[0.], [0.], [1.], [0.], [1.], [1.]];
        let predictions = array![0., 1., 1., 0., 0., 1.];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let f1_score = class_metrics.f1_score();

        assert_eq!(f1_score, 0.6666666666666666);
    }

    //new

    #[test]
    fn test_empty_cm_accuracy() {
        let labels = Array2::<f64>::zeros((0, 0));
        let predictions = array![];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let accuracy = class_metrics.accuracy();

        assert_eq!(accuracy, 0.0);
    }

    #[test]
    fn test_empty_cm_precision() {
        let labels = Array2::<f64>::zeros((0, 0));
        let predictions = array![];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let precision = class_metrics.precision();

        assert_eq!(precision, 0.0);
    }

    #[test]
    fn test_empty_cm_recall() {
        let labels = Array2::<f64>::zeros((0, 0));
        let predictions = array![];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let recall = class_metrics.recall();

        assert_eq!(recall, 0.0);
    }

    #[test]
    fn test_empty_cm_f1_score() {
        let labels = Array2::<f64>::zeros((0, 0));
        let predictions = array![];

        let class_metrics = MetricsCalculator::new(&labels, &predictions);
        let f1_score = class_metrics.f1_score();

        assert_eq!(f1_score, 0.0);
    }
}
