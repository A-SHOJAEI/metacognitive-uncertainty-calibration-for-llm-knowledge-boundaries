#!/usr/bin/env python3
"""
Comprehensive evaluation script for metacognitive uncertainty calibration model.

This script provides detailed analysis including calibration plots, uncertainty
distributions, domain-specific metrics, and selective prediction analysis.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.loader import MMLUDataLoader
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing import MMLUPreprocessor
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model import MetacognitiveUncertaintyModel
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.evaluation.metrics import UncertaintyMetrics
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.utils.config import Config

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluator with detailed analysis and visualization."""

    def __init__(
        self,
        model: MetacognitiveUncertaintyModel,
        config: Config,
        output_dir: Path
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model to evaluate
            config: Configuration object
            output_dir: Directory to save evaluation outputs
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.metrics = UncertaintyMetrics(n_bins=config.evaluation.n_bins)

        # Setup device
        self.device = torch.device(config.get_device())
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Initialized evaluator with device: {self.device}")

    def evaluate_dataset(
        self,
        data_loader: DataLoader,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: Data loader for the dataset
            dataset_name: Name of the dataset for logging

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating on {dataset_name} dataset...")

        all_predictions = []
        all_labels = []
        all_confidences = []
        all_uncertainty_types = []
        all_uncertainty_confidences = []
        all_domains = []
        all_subjects = []
        all_explanations = []

        # Collect predictions
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get model outputs
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                # Extract predictions
                predictions = outputs.answer_logits.argmax(dim=-1)
                confidences = torch.softmax(outputs.answer_logits, dim=-1).max(dim=-1)[0]
                uncertainty_predictions = outputs.uncertainty_logits.argmax(dim=-1)
                uncertainty_confidences = torch.softmax(outputs.uncertainty_logits, dim=-1).max(dim=-1)[0]

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["answer_labels"].cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_uncertainty_types.extend(uncertainty_predictions.cpu().numpy())
                all_uncertainty_confidences.extend(uncertainty_confidences.cpu().numpy())

                # Store metadata if available
                if "domain" in batch:
                    all_domains.extend(batch["domain"])
                if "subject" in batch:
                    all_subjects.extend(batch["subject"])

                # Generate explanations for sample
                if batch_idx == 0:  # Only for first batch to save time
                    explanations = self.model._generate_explanations(
                        [self.model.UNCERTAINTY_TYPES[pred] for pred in uncertainty_predictions.cpu().numpy()],
                        confidences.cpu().numpy().tolist(),
                        batch.get("domain", ["unknown"] * len(predictions))
                    )
                    all_explanations.extend(explanations)

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        confidences = np.array(all_confidences)
        uncertainty_types = np.array(all_uncertainty_types)

        # Compute comprehensive metrics
        logger.info("Computing metrics...")
        eval_results = self.metrics.compute_all_metrics(
            predictions, labels, confidences, uncertainty_types
        )

        # Add dataset-specific info
        eval_results.update({
            "dataset_name": dataset_name,
            "num_samples": len(predictions),
            "accuracy": (predictions == labels).mean(),
            "mean_confidence": confidences.mean(),
            "std_confidence": confidences.std()
        })

        # Domain-specific analysis
        if all_domains:
            domain_metrics = self.metrics.compute_domain_metrics(
                predictions, labels, confidences, all_domains
            )
            eval_results["domain_metrics"] = domain_metrics

        # Subject-specific analysis
        if all_subjects:
            subject_metrics = self._compute_subject_metrics(
                predictions, labels, confidences, all_subjects
            )
            eval_results["subject_metrics"] = subject_metrics

        # Uncertainty type analysis
        uncertainty_analysis = self._analyze_uncertainty_types(
            predictions, labels, confidences, uncertainty_types, all_domains
        )
        eval_results["uncertainty_analysis"] = uncertainty_analysis

        # Sample explanations
        if all_explanations:
            eval_results["sample_explanations"] = all_explanations[:10]

        # Save detailed results
        self._save_detailed_results(eval_results, predictions, labels, confidences,
                                  uncertainty_types, all_domains, all_subjects, dataset_name)

        logger.info(f"Evaluation on {dataset_name} complete")
        return eval_results

    def _compute_subject_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        subjects: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by subject."""
        subject_metrics = {}
        unique_subjects = list(set(subjects))

        for subject in unique_subjects:
            subject_mask = np.array([s == subject for s in subjects])

            if subject_mask.sum() > 0:
                subject_preds = predictions[subject_mask]
                subject_labels = labels[subject_mask]
                subject_confidences = confidences[subject_mask]

                subject_metrics[subject] = {
                    "accuracy": (subject_preds == subject_labels).mean(),
                    "confidence_mean": subject_confidences.mean(),
                    "confidence_std": subject_confidences.std(),
                    "sample_count": subject_mask.sum()
                }

                # Calibration if enough samples
                if subject_mask.sum() >= 10:
                    ece = self.metrics._compute_expected_calibration_error(
                        subject_preds, subject_labels, subject_confidences
                    )
                    subject_metrics[subject]["expected_calibration_error"] = ece

        return subject_metrics

    def _analyze_uncertainty_types(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        uncertainty_types: np.ndarray,
        domains: List[str]
    ) -> Dict[str, Any]:
        """Analyze uncertainty type predictions and effectiveness."""
        correct = (predictions == labels).astype(float)
        uncertainty_type_names = [self.model.UNCERTAINTY_TYPES[ut] for ut in uncertainty_types]

        analysis = {
            "type_distribution": {},
            "type_effectiveness": {},
            "domain_type_correlation": {}
        }

        # Type distribution
        unique_types, counts = np.unique(uncertainty_types, return_counts=True)
        total = len(uncertainty_types)
        for utype, count in zip(unique_types, counts):
            type_name = self.model.UNCERTAINTY_TYPES[utype]
            analysis["type_distribution"][type_name] = {
                "count": int(count),
                "proportion": float(count / total)
            }

        # Type effectiveness
        for utype in unique_types:
            type_name = self.model.UNCERTAINTY_TYPES[utype]
            mask = uncertainty_types == utype

            if mask.sum() > 0:
                analysis["type_effectiveness"][type_name] = {
                    "accuracy": float(correct[mask].mean()),
                    "confidence_mean": float(confidences[mask].mean()),
                    "confidence_std": float(confidences[mask].std()),
                    "sample_count": int(mask.sum())
                }

        # Domain-type correlation if domains available
        if domains:
            domain_type_counts = {}
            unique_domains = list(set(domains))

            for domain in unique_domains:
                domain_mask = np.array([d == domain for d in domains])
                domain_types = uncertainty_types[domain_mask]

                if len(domain_types) > 0:
                    unique_dtypes, dcounts = np.unique(domain_types, return_counts=True)
                    domain_total = len(domain_types)

                    domain_type_counts[domain] = {}
                    for dtype, dcount in zip(unique_dtypes, dcounts):
                        type_name = self.model.UNCERTAINTY_TYPES[dtype]
                        domain_type_counts[domain][type_name] = float(dcount / domain_total)

            analysis["domain_type_correlation"] = domain_type_counts

        return analysis

    def _save_detailed_results(
        self,
        eval_results: Dict[str, Any],
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        uncertainty_types: np.ndarray,
        domains: List[str],
        subjects: List[str],
        dataset_name: str
    ) -> None:
        """Save detailed evaluation results and create visualizations."""
        # Save JSON results
        json_path = self.output_dir / f"{dataset_name}_results.json"
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = self._make_json_serializable(eval_results)
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved detailed results to {json_path}")

        # Create CSV with individual predictions
        results_df = pd.DataFrame({
            "prediction": predictions,
            "label": labels,
            "confidence": confidences,
            "uncertainty_type": [self.model.UNCERTAINTY_TYPES[ut] for ut in uncertainty_types],
            "correct": (predictions == labels).astype(int)
        })

        if domains:
            results_df["domain"] = domains
        if subjects:
            results_df["subject"] = subjects

        csv_path = self.output_dir / f"{dataset_name}_predictions.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")

        # Create visualizations
        self._create_visualizations(
            predictions, labels, confidences, uncertainty_types,
            domains, subjects, dataset_name
        )

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _create_visualizations(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        uncertainty_types: np.ndarray,
        domains: List[str],
        subjects: List[str],
        dataset_name: str
    ) -> None:
        """Create comprehensive visualization plots."""
        logger.info("Creating visualizations...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Calibration curve
        fig = self.metrics.plot_calibration_curve(
            predictions, labels, confidences,
            save_path=str(self.output_dir / f"{dataset_name}_calibration_curve.png")
        )
        plt.close(fig)

        # 2. Uncertainty distribution
        fig = self.metrics.plot_uncertainty_distribution(
            predictions, labels, confidences, uncertainty_types,
            save_path=str(self.output_dir / f"{dataset_name}_uncertainty_distribution.png")
        )
        plt.close(fig)

        # 3. Domain performance
        if domains:
            self._plot_domain_performance(
                predictions, labels, confidences, domains, dataset_name
            )

        # 4. Subject performance (top subjects only)
        if subjects:
            self._plot_subject_performance(
                predictions, labels, confidences, subjects, dataset_name
            )

        # 5. Confidence vs accuracy scatter
        self._plot_confidence_accuracy_scatter(
            predictions, labels, confidences, dataset_name
        )

        # 6. Selective prediction curve
        self._plot_selective_prediction_curve(
            predictions, labels, confidences, dataset_name
        )

        logger.info("Visualizations created")

    def _plot_domain_performance(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        domains: List[str],
        dataset_name: str
    ) -> None:
        """Plot performance metrics by domain."""
        unique_domains = list(set(domains))
        domain_data = []

        for domain in unique_domains:
            domain_mask = np.array([d == domain for d in domains])
            if domain_mask.sum() > 0:
                domain_acc = (predictions[domain_mask] == labels[domain_mask]).mean()
                domain_conf = confidences[domain_mask].mean()
                domain_count = domain_mask.sum()

                domain_data.append({
                    "domain": domain,
                    "accuracy": domain_acc,
                    "confidence": domain_conf,
                    "count": domain_count
                })

        df = pd.DataFrame(domain_data)
        df = df.sort_values("accuracy", ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy by domain
        bars1 = ax1.bar(df["domain"], df["accuracy"])
        ax1.set_xlabel("Domain")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy by Domain")
        ax1.tick_params(axis='x', rotation=45)

        # Add count annotations
        for bar, count in zip(bars1, df["count"]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)

        # Confidence by domain
        bars2 = ax2.bar(df["domain"], df["confidence"])
        ax2.set_xlabel("Domain")
        ax2.set_ylabel("Mean Confidence")
        ax2.set_title("Confidence by Domain")
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{dataset_name}_domain_performance.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_subject_performance(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        subjects: List[str],
        dataset_name: str
    ) -> None:
        """Plot performance for top subjects by sample count."""
        subject_counts = pd.Series(subjects).value_counts()
        top_subjects = subject_counts.head(15).index.tolist()

        subject_data = []
        for subject in top_subjects:
            subject_mask = np.array([s == subject for s in subjects])
            if subject_mask.sum() > 0:
                subject_acc = (predictions[subject_mask] == labels[subject_mask]).mean()
                subject_conf = confidences[subject_mask].mean()
                subject_count = subject_mask.sum()

                subject_data.append({
                    "subject": subject,
                    "accuracy": subject_acc,
                    "confidence": subject_conf,
                    "count": subject_count
                })

        df = pd.DataFrame(subject_data)
        df = df.sort_values("accuracy", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(df["subject"], df["accuracy"])
        ax.set_xlabel("Accuracy")
        ax.set_title("Accuracy by Subject (Top 15 by Sample Count)")

        # Add count annotations
        for bar, count in zip(bars, df["count"]):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'n={count}', ha='left', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{dataset_name}_subject_performance.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confidence_accuracy_scatter(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        dataset_name: str
    ) -> None:
        """Plot confidence vs accuracy scatter plot."""
        correct = (predictions == labels).astype(float)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create bins for better visualization
        conf_bins = np.linspace(0, 1, 21)
        bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []

        for i in range(len(conf_bins) - 1):
            mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i + 1])
            if mask.sum() > 0:
                bin_accuracies.append(correct[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)

        # Scatter plot with size proportional to count
        scatter = ax.scatter(bin_centers, bin_accuracies, s=[c*2 for c in bin_counts],
                           alpha=0.6, c=bin_counts, cmap='viridis')

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Confidence vs Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Colorbar for counts
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sample Count')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{dataset_name}_confidence_accuracy_scatter.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_selective_prediction_curve(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        dataset_name: str
    ) -> None:
        """Plot selective prediction (risk-coverage) curve."""
        # Sort by confidence
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_correct = (predictions[sorted_indices] == labels[sorted_indices]).astype(float)

        # Compute cumulative accuracy at different coverage levels
        coverages = np.linspace(0.1, 1.0, 100)
        accuracies = []
        risks = []

        for coverage in coverages:
            n_select = int(coverage * len(sorted_correct))
            if n_select > 0:
                selected_correct = sorted_correct[:n_select]
                accuracy = selected_correct.mean()
                risk = 1 - accuracy
            else:
                accuracy = 0
                risk = 1

            accuracies.append(accuracy)
            risks.append(risk)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Risk-Coverage curve
        ax1.plot(coverages, risks, 'b-', linewidth=2)
        ax1.set_xlabel("Coverage")
        ax1.set_ylabel("Risk (1 - Accuracy)")
        ax1.set_title("Risk-Coverage Curve")
        ax1.grid(True, alpha=0.3)

        # Accuracy-Coverage curve
        ax2.plot(coverages, accuracies, 'g-', linewidth=2)
        ax2.axhline(y=np.mean(predictions == labels), color='r', linestyle='--',
                   label='Overall Accuracy')
        ax2.set_xlabel("Coverage")
        ax2.set_ylabel("Selective Accuracy")
        ax2.set_title("Selective Accuracy vs Coverage")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{dataset_name}_selective_prediction.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, eval_results: Dict[str, Any], dataset_name: str) -> None:
        """Generate a comprehensive evaluation report."""
        report_path = self.output_dir / f"{dataset_name}_evaluation_report.md"

        with open(report_path, 'w') as f:
            f.write(f"# Metacognitive Uncertainty Evaluation Report - {dataset_name.title()}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Dataset**: {eval_results['dataset_name']}\n")
            f.write(f"- **Samples**: {eval_results['num_samples']:,}\n")
            f.write(f"- **Accuracy**: {eval_results['accuracy']:.3f}\n")
            f.write(f"- **Mean Confidence**: {eval_results['mean_confidence']:.3f}\n")
            f.write(f"- **Expected Calibration Error**: {eval_results.get('expected_calibration_error', 'N/A'):.3f}\n")
            f.write(f"- **Uncertainty AUROC**: {eval_results.get('uncertainty_auroc', 'N/A'):.3f}\n\n")

            # Key Metrics
            f.write("## Key Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")

            key_metrics = [
                "accuracy", "expected_calibration_error", "maximum_calibration_error",
                "brier_score", "uncertainty_auroc", "uncertainty_aupr",
                "selective_prediction_auc", "confidence_accuracy_correlation"
            ]

            for metric in key_metrics:
                if metric in eval_results:
                    f.write(f"| {metric.replace('_', ' ').title()} | {eval_results[metric]:.4f} |\n")

            # Target Metrics Comparison
            f.write("\n## Target Metrics Comparison\n\n")
            f.write("| Target Metric | Target | Achieved | Status |\n")
            f.write("|---------------|--------|----------|--------|\n")

            targets = {
                'expected_calibration_error': 0.05,
                'uncertainty_type_classification_f1': 0.82,
                'selective_prediction_auc': 0.91,
                'human_trust_calibration_correlation': 0.78
            }

            for target_name, target_value in targets.items():
                achieved = eval_results.get(target_name, 0.0)
                status = "✅ Met" if achieved <= target_value else "❌ Not Met"
                f.write(f"| {target_name.replace('_', ' ').title()} | {target_value:.3f} | {achieved:.3f} | {status} |\n")

            # Domain Analysis
            if "domain_metrics" in eval_results:
                f.write("\n## Domain Performance\n\n")
                f.write("| Domain | Accuracy | Confidence | Samples | ECE |\n")
                f.write("|--------|----------|------------|---------|-----|\n")

                for domain, metrics in eval_results["domain_metrics"].items():
                    acc = metrics.get("accuracy", 0)
                    conf = metrics.get("confidence_mean", 0)
                    count = metrics.get("sample_count", 0)
                    ece = metrics.get("expected_calibration_error", "N/A")
                    ece_str = f"{ece:.3f}" if isinstance(ece, (int, float)) else "N/A"
                    f.write(f"| {domain} | {acc:.3f} | {conf:.3f} | {count} | {ece_str} |\n")

            # Uncertainty Type Analysis
            if "uncertainty_analysis" in eval_results:
                f.write("\n## Uncertainty Type Analysis\n\n")

                f.write("### Distribution\n\n")
                for utype, data in eval_results["uncertainty_analysis"]["type_distribution"].items():
                    f.write(f"- **{utype}**: {data['count']} samples ({data['proportion']:.1%})\n")

                f.write("\n### Effectiveness\n\n")
                f.write("| Type | Accuracy | Confidence | Samples |\n")
                f.write("|------|----------|------------|---------|\n")

                for utype, data in eval_results["uncertainty_analysis"]["type_effectiveness"].items():
                    f.write(f"| {utype} | {data['accuracy']:.3f} | {data['confidence_mean']:.3f} | {data['sample_count']} |\n")

            # Sample Explanations
            if "sample_explanations" in eval_results:
                f.write("\n## Sample Explanations\n\n")
                for i, explanation in enumerate(eval_results["sample_explanations"][:5]):
                    f.write(f"{i+1}. {explanation}\n\n")

            f.write("\n## Files Generated\n\n")
            f.write("- Detailed results: `results.json`\n")
            f.write("- Predictions CSV: `predictions.csv`\n")
            f.write("- Calibration curve: `calibration_curve.png`\n")
            f.write("- Uncertainty distribution: `uncertainty_distribution.png`\n")
            f.write("- Domain performance: `domain_performance.png`\n")
            f.write("- Subject performance: `subject_performance.png`\n")
            f.write("- Confidence-accuracy scatter: `confidence_accuracy_scatter.png`\n")
            f.write("- Selective prediction: `selective_prediction.png`\n")

        logger.info(f"Generated evaluation report: {report_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate metacognitive uncertainty model")

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["test", "validation", "all"],
        default="test",
        help="Dataset to evaluate on"
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )

    # Options
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to evaluate")
    parser.add_argument("--batch_size", type=int, help="Evaluation batch size")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Override with command line arguments
    if args.subjects:
        config.data.subjects = args.subjects
    if args.batch_size:
        config.data.batch_size = args.batch_size

    # Setup logging
    config.setup_logging(args.log_level)

    logger.info("Starting model evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Load model
        logger.info("Loading model...")
        device = torch.device(config.get_device())

        # Initialize model
        model = MetacognitiveUncertaintyModel(
            base_model_name=config.model.base_model_name,
            num_choices=config.model.num_choices,
            uncertainty_weight=config.model.uncertainty_weight,
            explanation_weight=config.model.explanation_weight,
            use_epistemic_estimation=config.model.use_epistemic_estimation,
            freeze_base_model=config.model.freeze_base_model
        )

        # Load checkpoint
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model loaded successfully")

        # Setup data loader
        data_loader = MMLUDataLoader(
            cache_dir=config.data.cache_dir,
            seed=config.seed,
            uncertainty_augmentation=config.data.uncertainty_augmentation
        )

        # Setup preprocessor
        preprocessor = MMLUPreprocessor(
            tokenizer_name=config.model.base_model_name,
            max_length=config.data.max_length,
            include_uncertainty_prompt=True
        )

        # Create evaluator
        evaluator = ModelEvaluator(
            model=model,
            config=config,
            output_dir=Path(args.output_dir)
        )

        # Evaluate datasets
        datasets_to_eval = []
        if args.dataset == "all":
            datasets_to_eval = [("test", config.data.test_split),
                              ("validation", config.data.val_split)]
        else:
            split = config.data.test_split if args.dataset == "test" else config.data.val_split
            datasets_to_eval = [(args.dataset, split)]

        all_results = {}

        for dataset_name, split in datasets_to_eval:
            logger.info(f"Evaluating {dataset_name} dataset...")

            # Load dataset
            dataset = data_loader.load_dataset(split=split, subjects=config.data.subjects)

            # Create data loader
            def collate_fn(batch):
                examples = [
                    {
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": item["answer"],
                        "subject": item["subject"],
                        "domain": item["domain"],
                        "uncertainty_type": item.get("uncertainty_type")
                    }
                    for item in batch
                ]

                return preprocessor.batch_tokenize(examples)

            eval_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available()
            )

            # Evaluate
            results = evaluator.evaluate_dataset(eval_loader, dataset_name)
            all_results[dataset_name] = results

            # Generate report
            evaluator.generate_report(results, dataset_name)

        # Save combined results
        combined_path = Path(args.output_dir) / "combined_results.json"
        with open(combined_path, 'w') as f:
            serializable_results = evaluator._make_json_serializable(all_results)
            json.dump(serializable_results, f, indent=2)

        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()