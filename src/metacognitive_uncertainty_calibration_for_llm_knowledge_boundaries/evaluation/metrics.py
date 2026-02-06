"""
Comprehensive evaluation metrics for metacognitive uncertainty calibration.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import scipy.stats
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class UncertaintyMetrics:
    """
    Comprehensive uncertainty quantification and calibration metrics
    for metacognitive models.
    """

    def __init__(self, n_bins: int = 15) -> None:
        """
        Initialize metrics calculator.

        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins

    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        uncertainty_types: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive uncertainty metrics.

        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            confidences: Prediction confidences [0, 1]
            uncertainty_types: Uncertainty type predictions

        Returns:
            Dictionary of all computed metrics
        """
        metrics = {}

        # Basic accuracy metrics
        metrics["accuracy"] = accuracy_score(labels, predictions)

        # Calibration metrics
        metrics.update(self.compute_calibration_metrics(predictions, labels, confidences))

        # Uncertainty quantification metrics
        metrics.update(self.compute_uncertainty_metrics(predictions, labels, confidences))

        # Selective prediction metrics
        metrics.update(self.compute_selective_prediction_metrics(
            predictions, labels, confidences
        ))

        # Reliability metrics
        metrics.update(self.compute_reliability_metrics(predictions, labels, confidences))

        # If uncertainty types are provided
        if uncertainty_types is not None:
            metrics.update(self.compute_uncertainty_type_metrics(
                predictions, labels, confidences, uncertainty_types
            ))

        return metrics

    def compute_calibration_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute calibration metrics including ECE, MCE, and Brier score.

        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            confidences: Prediction confidences

        Returns:
            Dictionary of calibration metrics
        """
        # Expected Calibration Error (ECE)
        ece = self._compute_expected_calibration_error(predictions, labels, confidences)

        # Maximum Calibration Error (MCE)
        mce = self._compute_maximum_calibration_error(predictions, labels, confidences)

        # Brier Score
        # Convert to binary format for Brier score
        correct = (predictions == labels).astype(float)
        brier_score = brier_score_loss(correct, confidences)

        # Reliability and Resolution
        reliability, resolution = self._compute_reliability_resolution(
            predictions, labels, confidences
        )

        # Calibration curve correlation
        cal_correlation = self._compute_calibration_correlation(
            predictions, labels, confidences
        )

        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "brier_score": brier_score,
            "reliability": reliability,
            "resolution": resolution,
            "calibration_correlation": cal_correlation
        }

    def _compute_expected_calibration_error(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _compute_maximum_calibration_error(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray
    ) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return mce

    def _compute_reliability_resolution(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[float, float]:
        """Compute reliability and resolution components of Brier score."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        correct = (predictions == labels).astype(float)
        overall_accuracy = correct.mean()

        reliability = 0.0
        resolution = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                resolution += prop_in_bin * (accuracy_in_bin - overall_accuracy) ** 2

        return reliability, resolution

    def _compute_calibration_correlation(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray
    ) -> float:
        """Compute correlation between confidence and accuracy."""
        correct = (predictions == labels).astype(float)

        if len(np.unique(confidences)) < 2 or len(np.unique(correct)) < 2:
            return 0.0

        correlation, _ = scipy.stats.pearsonr(confidences, correct)
        return correlation if not np.isnan(correlation) else 0.0

    def compute_uncertainty_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute uncertainty quantification metrics.

        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            confidences: Prediction confidences

        Returns:
            Dictionary of uncertainty metrics
        """
        correct = (predictions == labels).astype(float)

        # Uncertainty (1 - confidence) for incorrect predictions should be higher
        uncertainties = 1 - confidences

        # AUROC for uncertainty as a binary classification problem
        # (predicting whether the model is correct or not)
        try:
            uncertainty_auroc = roc_auc_score(1 - correct, uncertainties)
        except ValueError:
            uncertainty_auroc = 0.5  # Random performance if all predictions are same

        # AUPR for uncertainty
        try:
            uncertainty_aupr = average_precision_score(1 - correct, uncertainties)
        except ValueError:
            uncertainty_aupr = (1 - correct).mean()  # Prior if all predictions are same

        # Entropy-based metrics
        entropy = self._compute_predictive_entropy(confidences)

        # Mutual information between predictions and correctness
        mutual_info = self._compute_mutual_information(correct, uncertainties)

        return {
            "uncertainty_auroc": uncertainty_auroc,
            "uncertainty_aupr": uncertainty_aupr,
            "predictive_entropy": entropy,
            "uncertainty_mutual_info": mutual_info
        }

    def _compute_predictive_entropy(self, confidences: np.ndarray) -> float:
        """Compute average predictive entropy."""
        # Assuming binary case for simplicity
        p_correct = confidences
        p_incorrect = 1 - confidences

        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        p_correct = np.clip(p_correct, epsilon, 1 - epsilon)
        p_incorrect = np.clip(p_incorrect, epsilon, 1 - epsilon)

        entropy = -(p_correct * np.log(p_correct) + p_incorrect * np.log(p_incorrect))
        return entropy.mean()

    def _compute_mutual_information(
        self,
        correct: np.ndarray,
        uncertainties: np.ndarray
    ) -> float:
        """Compute mutual information between correctness and uncertainty."""
        # Discretize uncertainties into bins
        uncertainty_bins = np.digitize(uncertainties, np.linspace(0, 1, 5))

        # Compute joint and marginal distributions
        joint_counts = np.zeros((2, 5))  # correct/incorrect x uncertainty_bins
        for i in range(len(correct)):
            joint_counts[int(correct[i]), uncertainty_bins[i] - 1] += 1

        joint_prob = joint_counts / joint_counts.sum()
        marginal_correct = joint_prob.sum(axis=1, keepdims=True)
        marginal_uncertainty = joint_prob.sum(axis=0, keepdims=True)

        # Avoid log(0)
        epsilon = 1e-10
        joint_prob = np.clip(joint_prob, epsilon, 1)
        marginal_product = np.clip(marginal_correct @ marginal_uncertainty, epsilon, 1)

        # Compute mutual information
        mi = (joint_prob * np.log(joint_prob / marginal_product)).sum()
        return mi

    def compute_selective_prediction_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        coverage: float = 0.8
    ) -> Dict[str, float]:
        """
        Compute selective prediction metrics at given coverage.

        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            confidences: Prediction confidences
            coverage: Desired coverage level

        Returns:
            Selective prediction metrics
        """
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        n_samples = len(predictions)

        if n_samples == 0:
            return {
                "selective_accuracy": 0.0,
                "selective_auc": 0.5,
                "risk": 1.0,
                "coverage": 0.0
            }

        n_select = max(1, int(coverage * n_samples))

        # Select top confident predictions
        selected_indices = sorted_indices[:n_select]
        selected_predictions = predictions[selected_indices]
        selected_labels = labels[selected_indices]
        selected_confidences = confidences[selected_indices]

        # Compute metrics on selected subset
        selective_accuracy = accuracy_score(selected_labels, selected_predictions)

        # Compute AUC for selective prediction
        # This measures how well confidence ranks the examples
        correct = (predictions == labels).astype(float)
        try:
            selective_auc = roc_auc_score(correct, confidences)
        except ValueError:
            selective_auc = 0.5

        # Risk-coverage trade-off
        risk = 1 - selective_accuracy
        actual_coverage = len(selected_indices) / n_samples

        return {
            "selective_accuracy": selective_accuracy,
            "selective_risk": risk,
            "selective_coverage": actual_coverage,
            "selective_prediction_auc": selective_auc
        }

    def compute_reliability_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute reliability and trustworthiness metrics.

        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            confidences: Prediction confidences

        Returns:
            Reliability metrics
        """
        correct = (predictions == labels).astype(float)

        # Overconfidence: how often high-confidence predictions are wrong
        high_confidence_mask = confidences > 0.8
        if high_confidence_mask.sum() > 0:
            overconfidence_rate = 1 - correct[high_confidence_mask].mean()
        else:
            overconfidence_rate = 0.0

        # Underconfidence: how often low-confidence predictions are correct
        low_confidence_mask = confidences < 0.6
        if low_confidence_mask.sum() > 0:
            underconfidence_rate = correct[low_confidence_mask].mean()
        else:
            underconfidence_rate = 0.0

        # Confidence-accuracy alignment
        confidence_accuracy_corr = self._compute_calibration_correlation(
            predictions, labels, confidences
        )

        # Trust calibration: how well confidence predicts correctness
        trust_calibration = self._compute_trust_calibration(correct, confidences)

        return {
            "overconfidence_rate": overconfidence_rate,
            "underconfidence_rate": underconfidence_rate,
            "confidence_accuracy_correlation": confidence_accuracy_corr,
            "trust_calibration": trust_calibration
        }

    def _compute_trust_calibration(
        self,
        correct: np.ndarray,
        confidences: np.ndarray
    ) -> float:
        """
        Compute trust calibration metric.

        This measures how well human trust (based on confidence) aligns
        with actual model performance.
        """
        # Create bins based on confidence
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        trust_alignment = 0.0
        total_weight = 0.0

        for i in range(self.n_bins):
            bin_mask = bin_indices == i
            if bin_mask.sum() > 0:
                bin_accuracy = correct[bin_mask].mean()
                bin_confidence = confidences[bin_mask].mean()
                bin_weight = bin_mask.sum() / len(correct)

                # Trust alignment: how well confidence predicts accuracy
                trust_alignment += bin_weight * np.abs(bin_confidence - bin_accuracy)
                total_weight += bin_weight

        return 1 - trust_alignment if total_weight > 0 else 0.0

    def compute_uncertainty_type_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        uncertainty_types: np.ndarray,
        true_uncertainty_types: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute metrics specific to uncertainty type classification.

        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            confidences: Prediction confidences
            uncertainty_types: Predicted uncertainty types
            true_uncertainty_types: Ground truth uncertainty types

        Returns:
            Uncertainty type classification metrics
        """
        metrics = {}

        # If ground truth uncertainty types are available
        if true_uncertainty_types is not None:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_uncertainty_types,
                uncertainty_types,
                average='weighted'
            )

            metrics.update({
                "uncertainty_type_precision": precision,
                "uncertainty_type_recall": recall,
                "uncertainty_type_f1": f1
            })

        # Analyze uncertainty type distributions
        unique_types, counts = np.unique(uncertainty_types, return_counts=True)
        total = len(uncertainty_types)

        for utype, count in zip(unique_types, counts):
            metrics[f"uncertainty_type_{utype}_proportion"] = count / total

        # Analyze uncertainty type effectiveness
        correct = (predictions == labels).astype(float)

        for utype in unique_types:
            mask = uncertainty_types == utype
            if mask.sum() > 0:
                type_accuracy = correct[mask].mean()
                type_confidence = confidences[mask].mean()

                metrics.update({
                    f"uncertainty_type_{utype}_accuracy": type_accuracy,
                    f"uncertainty_type_{utype}_confidence": type_confidence
                })

        return metrics

    def compute_domain_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        domains: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics broken down by domain.

        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            confidences: Prediction confidences
            domains: List of domain labels

        Returns:
            Dictionary of metrics for each domain
        """
        domain_metrics = {}
        unique_domains = list(set(domains))

        for domain in unique_domains:
            domain_mask = np.array([d == domain for d in domains])

            if domain_mask.sum() > 0:
                domain_preds = predictions[domain_mask]
                domain_labels = labels[domain_mask]
                domain_confidences = confidences[domain_mask]

                # Compute standard metrics for this domain
                domain_metrics[domain] = {
                    "accuracy": accuracy_score(domain_labels, domain_preds),
                    "confidence_mean": domain_confidences.mean(),
                    "confidence_std": domain_confidences.std(),
                    "sample_count": domain_mask.sum()
                }

                # Compute calibration metrics if enough samples
                if domain_mask.sum() >= 10:
                    ece = self._compute_expected_calibration_error(
                        domain_preds, domain_labels, domain_confidences
                    )
                    domain_metrics[domain]["expected_calibration_error"] = ece

        return domain_metrics

    def plot_calibration_curve(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot calibration curve.

        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            confidences: Prediction confidences
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        correct = (predictions == labels).astype(float)

        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            correct, confidences, n_bins=self.n_bins
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

        # Add histogram
        ax.hist(confidences, bins=30, alpha=0.3, density=True, color='blue')

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_uncertainty_distribution(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        uncertainty_types: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot uncertainty distribution by type and correctness.

        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            confidences: Prediction confidences
            uncertainty_types: Uncertainty type labels
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        correct = (predictions == labels).astype(float)
        uncertainties = 1 - confidences

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Uncertainty by correctness
        axes[0, 0].hist(uncertainties[correct == 1], alpha=0.7, label='Correct', bins=20)
        axes[0, 0].hist(uncertainties[correct == 0], alpha=0.7, label='Incorrect', bins=20)
        axes[0, 0].set_xlabel('Uncertainty')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Uncertainty Distribution by Correctness')
        axes[0, 0].legend()

        # Plot 2: Confidence by uncertainty type
        unique_types = np.unique(uncertainty_types)
        for i, utype in enumerate(unique_types):
            mask = uncertainty_types == utype
            axes[0, 1].boxplot(
                confidences[mask],
                positions=[i],
                labels=[utype],
                patch_artist=True
            )
        axes[0, 1].set_xlabel('Uncertainty Type')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].set_title('Confidence by Uncertainty Type')

        # Plot 3: Accuracy by uncertainty type
        accuracy_by_type = []
        for utype in unique_types:
            mask = uncertainty_types == utype
            if mask.sum() > 0:
                accuracy_by_type.append(correct[mask].mean())
            else:
                accuracy_by_type.append(0)

        axes[1, 0].bar(unique_types, accuracy_by_type)
        axes[1, 0].set_xlabel('Uncertainty Type')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy by Uncertainty Type')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 4: Uncertainty type distribution
        type_counts = [np.sum(uncertainty_types == utype) for utype in unique_types]
        axes[1, 1].pie(type_counts, labels=unique_types, autopct='%1.1f%%')
        axes[1, 1].set_title('Uncertainty Type Distribution')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig