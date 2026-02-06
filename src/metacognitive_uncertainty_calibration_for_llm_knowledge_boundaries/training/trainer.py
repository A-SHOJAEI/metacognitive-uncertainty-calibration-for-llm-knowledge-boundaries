"""
Training framework for metacognitive uncertainty calibration with MLflow tracking.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import classification_report, f1_score

from ..models.model import MetacognitiveUncertaintyModel, MetacognitiveOutput
from ..evaluation.metrics import UncertaintyMetrics

logger = logging.getLogger(__name__)


class MetacognitiveTrainer:
    """
    Comprehensive trainer for metacognitive uncertainty calibration with
    MLflow tracking, checkpointing, and advanced training strategies.
    """

    def __init__(
        self,
        model: MetacognitiveUncertaintyModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        experiment_name: str = "metacognitive_uncertainty",
        checkpoint_dir: str = "./checkpoints"
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Metacognitive uncertainty model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            experiment_name: MLflow experiment name
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = self._setup_optimizer()

        # Scheduler
        self.scheduler = self._setup_scheduler()

        # Evaluation metrics
        self.metrics = UncertaintyMetrics()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []

        # MLflow setup
        self.experiment_name = experiment_name
        self._setup_mlflow()

        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with different learning rates for different components."""
        # Different learning rates for different model components
        base_lr = self.config["learning_rate"]

        param_groups = [
            {
                "params": self.model.base_model.parameters(),
                "lr": base_lr * 0.1,  # Lower LR for pre-trained components
                "weight_decay": self.config.get("weight_decay", 0.01)
            },
            {
                "params": self.model.answer_head.parameters(),
                "lr": base_lr,
                "weight_decay": self.config.get("weight_decay", 0.01)
            },
            {
                "params": self.model.uncertainty_head.parameters(),
                "lr": base_lr,
                "weight_decay": self.config.get("weight_decay", 0.01)
            }
        ]

        if hasattr(self.model, 'epistemic_estimator'):
            param_groups.append({
                "params": self.model.epistemic_estimator.parameters(),
                "lr": base_lr * 0.05,  # Even lower for epistemic estimator
                "weight_decay": self.config.get("weight_decay", 0.01)
            })

        optimizer_name = self.config.get("optimizer", "adamw")
        if optimizer_name == "adamw":
            return AdamW(param_groups, eps=1e-6, betas=(0.9, 0.999))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _setup_scheduler(self) -> Union[torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler.CosineAnnealingLR]:
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", "reduce_on_plateau")

        if scheduler_name == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.get("scheduler_patience", 3)
            )
        elif scheduler_name == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"],
                eta_min=self.config["learning_rate"] * 0.01
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        mlflow.set_experiment(self.experiment_name)

        # Start new run
        mlflow.start_run()

        # Log configuration
        mlflow.log_params(self.config)

        # Log model info
        model_info = self.model.get_model_info()
        for key, value in model_info.items():
            mlflow.log_param(f"model_{key}", value)

        logger.info(f"Started MLflow run: {mlflow.active_run().info.run_id}")

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Main training loop with comprehensive tracking and early stopping.

        Args:
            num_epochs: Number of epochs to train (uses config if None)

        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config["num_epochs"]

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self._train_epoch(epoch)

            # Validation phase
            val_metrics = self._validate_epoch(epoch)

            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_loss"])
            else:
                self.scheduler.step()

            # Record metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_history.append(epoch_metrics)

            # Log to MLflow
            mlflow.log_metrics(epoch_metrics, step=epoch)

            # Early stopping check
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            # Regular checkpoint
            if epoch % self.config.get("save_every", 5) == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                f"Time: {epoch_time:.1f}s"
            )

            # Early stopping
            if self.patience_counter >= self.config.get("patience", 10):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Final evaluation and calibration
        self._final_evaluation()

        return self._format_training_history()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with comprehensive logging."""
        start_time = time.time()
        self.model.train()

        # Initialize metrics tracking
        total_loss = 0.0
        total_answer_loss = 0.0
        total_uncertainty_loss = 0.0
        total_calibration_loss = 0.0
        num_correct = 0
        num_samples = 0
        batch_times = []
        gradient_norms = []
        learning_rates = []

        logger.info(f"Starting training epoch {epoch+1}")
        logger.debug(f"Train loader contains {len(self.train_loader)} batches")

        if hasattr(self.optimizer, 'param_groups'):
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.debug(f"Current learning rate: {current_lr:.2e}")

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch+1}",
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()

            # Debug: Log batch details periodically
            if batch_idx == 0 or (batch_idx + 1) % self.config.get("log_every", 100) == 0:
                batch_size = batch["input_ids"].size(0) if "input_ids" in batch else 0
                seq_len = batch["input_ids"].size(1) if "input_ids" in batch else 0
                logger.debug(
                    f"Processing batch {batch_idx + 1}/{len(self.train_loader)}: "
                    f"batch_size={batch_size}, seq_len={seq_len}"
                )
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass with timing
                forward_start = time.time()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    answer_labels=batch.get("answer_labels"),
                    uncertainty_labels=batch.get("uncertainty_labels")
                )
                forward_time = time.time() - forward_start

                # Compute losses
                loss_start = time.time()
                losses = self.model.compute_loss(
                    outputs,
                    batch["answer_labels"],
                    batch.get("uncertainty_labels")
                )
                loss_time = time.time() - loss_start

                loss = losses["total_loss"]

                # Debug: Log loss components periodically
                if batch_idx == 0 or (batch_idx + 1) % self.config.get("log_every", 100) == 0:
                    logger.debug(
                        f"Batch {batch_idx + 1} losses: "
                        f"total={loss.item():.4f}, "
                        f"answer={losses['answer_loss'].item():.4f}, "
                        f"uncertainty={losses['uncertainty_loss'].item():.4f}, "
                        f"calibration={losses['calibration_loss'].item():.4f}"
                    )
                    logger.debug(
                        f"Batch {batch_idx + 1} timing: "
                        f"forward={forward_time:.3f}s, loss_comp={loss_time:.3f}s"
                    )

                    # Log tensor shapes for debugging
                    logger.debug(
                        f"Batch {batch_idx + 1} tensor shapes: "
                        f"input_ids={batch['input_ids'].shape}, "
                        f"answer_logits={outputs.answer_logits.shape}, "
                        f"uncertainty_logits={outputs.uncertainty_logits.shape}"
                    )

                # Backward pass
                backward_start = time.time()
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping with norm tracking
                max_grad_norm = self.config.get("max_grad_norm", 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm
                )
                gradient_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

                self.optimizer.step()
                backward_time = time.time() - backward_start

                # Track learning rate
                if hasattr(self.optimizer, 'param_groups'):
                    current_lr = self.optimizer.param_groups[0]['lr']
                    learning_rates.append(current_lr)

                self.global_step += 1

                # Debug: Log gradient and optimization details periodically
                if batch_idx == 0 or (batch_idx + 1) % self.config.get("log_every", 100) == 0:
                    logger.debug(
                        f"Batch {batch_idx + 1} optimization: "
                        f"grad_norm={grad_norm:.4f}, "
                        f"lr={current_lr:.2e}, "
                        f"backward_time={backward_time:.3f}s"
                    )

                # Accumulate metrics
                total_loss += loss.item()
                total_answer_loss += losses["answer_loss"].item()
                total_uncertainty_loss += losses["uncertainty_loss"].item()
                total_calibration_loss += losses["calibration_loss"].item()

                # Calculate accuracy
                predictions = outputs.answer_logits.argmax(dim=-1)
                num_correct += (predictions == batch["answer_labels"]).sum().item()
                num_samples += batch["answer_labels"].size(0)

                # Track batch timing
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx + 1}: {e}", exc_info=True)
                # Skip this batch and continue
                continue

            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = num_correct / num_samples
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}"
            })

            # Log intermediate metrics
            if self.global_step % self.config.get("log_every", 100) == 0:
                mlflow.log_metrics({
                    "step_train_loss": loss.item(),
                    "step_answer_loss": losses["answer_loss"].item(),
                    "step_uncertainty_loss": losses["uncertainty_loss"].item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                }, step=self.global_step)

        # Calculate epoch metrics
        num_processed_batches = len(self.train_loader)
        avg_loss = total_loss / num_processed_batches
        avg_answer_loss = total_answer_loss / num_processed_batches
        avg_uncertainty_loss = total_uncertainty_loss / num_processed_batches
        avg_calibration_loss = total_calibration_loss / num_processed_batches
        accuracy = num_correct / num_samples

        # Calculate timing statistics
        epoch_time = time.time() - start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0.0
        samples_per_second = num_samples / epoch_time if epoch_time > 0 else 0.0

        # Calculate gradient statistics
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        max_grad_norm = np.max(gradient_norms) if gradient_norms else 0.0

        # Calculate learning rate statistics
        final_lr = learning_rates[-1] if learning_rates else 0.0
        lr_change = (learning_rates[-1] - learning_rates[0]) if len(learning_rates) > 1 else 0.0

        # Comprehensive epoch summary logging
        logger.info(
            f"Training epoch {epoch+1} completed: "
            f"avg_loss={avg_loss:.4f}, accuracy={accuracy:.4f}, "
            f"time={epoch_time:.2f}s, samples/s={samples_per_second:.1f}"
        )

        logger.debug(
            f"Epoch {epoch+1} detailed metrics: "
            f"batches_processed={num_processed_batches}, "
            f"total_samples={num_samples}, "
            f"avg_batch_time={avg_batch_time:.3f}s"
        )

        logger.debug(
            f"Epoch {epoch+1} loss breakdown: "
            f"answer_loss={avg_answer_loss:.4f}, "
            f"uncertainty_loss={avg_uncertainty_loss:.4f}, "
            f"calibration_loss={avg_calibration_loss:.4f}"
        )

        logger.debug(
            f"Epoch {epoch+1} optimization stats: "
            f"avg_grad_norm={avg_grad_norm:.4f}, "
            f"max_grad_norm={max_grad_norm:.4f}, "
            f"final_lr={final_lr:.2e}, "
            f"lr_change={lr_change:.2e}"
        )

        # Log memory usage if available
        try:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                logger.debug(
                    f"Epoch {epoch+1} GPU memory: "
                    f"allocated={memory_allocated:.2f}GB, "
                    f"reserved={memory_reserved:.2f}GB"
                )
        except Exception as e:
            logger.debug(f"Could not log GPU memory usage: {e}")

        return {
            "train_loss": avg_loss,
            "train_answer_loss": avg_answer_loss,
            "train_uncertainty_loss": avg_uncertainty_loss,
            "train_calibration_loss": avg_calibration_loss,
            "train_accuracy": accuracy
        }

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        # For uncertainty metrics
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_uncertainty_predictions = []
        all_uncertainty_labels = []

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {epoch+1}",
            leave=False
        )

        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    answer_labels=batch.get("answer_labels"),
                    uncertainty_labels=batch.get("uncertainty_labels")
                )

                # Compute losses
                losses = self.model.compute_loss(
                    outputs,
                    batch["answer_labels"],
                    batch.get("uncertainty_labels")
                )

                loss = losses["total_loss"]
                total_loss += loss.item()

                # Calculate accuracy
                predictions = outputs.answer_logits.argmax(dim=-1)
                num_correct += (predictions == batch["answer_labels"]).sum().item()
                num_samples += batch["answer_labels"].size(0)

                # Collect for uncertainty metrics
                confidences = torch.softmax(outputs.answer_logits, dim=-1).max(dim=-1)[0]
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["answer_labels"].cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

                if "uncertainty_labels" in batch:
                    uncertainty_preds = outputs.uncertainty_logits.argmax(dim=-1)
                    all_uncertainty_predictions.extend(uncertainty_preds.cpu().numpy())
                    all_uncertainty_labels.extend(batch["uncertainty_labels"].cpu().numpy())

        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = num_correct / num_samples

        # Calculate uncertainty metrics
        uncertainty_metrics = self.metrics.compute_all_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_confidences)
        )

        val_metrics = {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            **{f"val_{k}": v for k, v in uncertainty_metrics.items()}
        }

        # Add uncertainty classification metrics if available
        if all_uncertainty_predictions and all_uncertainty_labels:
            uncertainty_f1 = f1_score(
                all_uncertainty_labels,
                all_uncertainty_predictions,
                average='weighted'
            )
            val_metrics["val_uncertainty_f1"] = uncertainty_f1

        return val_metrics

    def _final_evaluation(self) -> None:
        """Perform comprehensive final evaluation."""
        logger.info("Performing final evaluation...")

        # Load best checkpoint
        best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        if best_checkpoint_path.exists():
            self.load_checkpoint(str(best_checkpoint_path))

        # Calibrate model
        self.model.calibrate_temperature(self.val_loader)

        # Comprehensive evaluation
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_domains = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                predictions = outputs.answer_logits.argmax(dim=-1)
                confidences = torch.softmax(outputs.answer_logits, dim=-1).max(dim=-1)[0]

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["answer_labels"].cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

                if "domain" in batch:
                    all_domains.extend(batch["domain"])

        # Final metrics
        final_metrics = self.metrics.compute_all_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_confidences)
        )

        # Log final metrics
        final_metrics = {f"final_{k}": v for k, v in final_metrics.items()}
        mlflow.log_metrics(final_metrics)

        # Domain-specific analysis
        if all_domains:
            domain_metrics = self.metrics.compute_domain_metrics(
                np.array(all_predictions),
                np.array(all_labels),
                np.array(all_confidences),
                all_domains
            )

            for domain, metrics in domain_metrics.items():
                domain_metrics_prefixed = {f"final_{domain}_{k}": v for k, v in metrics.items()}
                mlflow.log_metrics(domain_metrics_prefixed)

        logger.info("Final evaluation completed")

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
            "training_history": self.training_history
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            mlflow.log_artifact(str(best_path))

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.global_step = checkpoint["global_step"]
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Loaded checkpoint from: {checkpoint_path}")

    def _format_training_history(self) -> Dict[str, List[float]]:
        """Format training history for easy plotting."""
        if not self.training_history:
            return {}

        history = {}
        for key in self.training_history[0].keys():
            history[key] = [epoch[key] for epoch in self.training_history]

        return history

    def evaluate_selective_prediction(
        self,
        test_loader: DataLoader,
        coverage_levels: List[float] = [0.5, 0.7, 0.8, 0.9]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate selective prediction at different coverage levels.

        Args:
            test_loader: Test data loader
            coverage_levels: List of coverage levels to evaluate

        Returns:
            Metrics at each coverage level
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                predictions = outputs.answer_logits.argmax(dim=-1)
                confidences = torch.softmax(outputs.answer_logits, dim=-1).max(dim=-1)[0]

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["answer_labels"].cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

        # Calculate selective prediction metrics
        results = {}
        for coverage in coverage_levels:
            metrics = self.metrics.compute_selective_prediction_metrics(
                np.array(all_predictions),
                np.array(all_labels),
                np.array(all_confidences),
                coverage=coverage
            )
            results[f"coverage_{coverage}"] = metrics

        return results

    def close(self) -> None:
        """Clean up MLflow run."""
        mlflow.end_run()
        logger.info("Training session closed")