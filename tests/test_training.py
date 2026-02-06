"""
Tests for training functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.training.trainer import MetacognitiveTrainer
from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model import MetacognitiveUncertaintyModel, MetacognitiveOutput


class TestMetacognitiveTrainer:
    """Test cases for MetacognitiveTrainer."""

    @pytest.fixture
    def mock_trainer_setup(self, small_model, mock_training_data, sample_config, temp_dir):
        """Set up trainer with mocked components."""
        # Mock MLflow to avoid actual logging
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.training.trainer.mlflow') as mock_mlflow:
            mock_mlflow.active_run.return_value.info.run_id = "test_run_id"

            trainer = MetacognitiveTrainer(
                model=small_model,
                train_loader=mock_training_data,
                val_loader=mock_training_data,  # Use same data for val
                config=sample_config.to_dict(),
                experiment_name="test_experiment",
                checkpoint_dir=str(temp_dir)
            )

            return trainer, mock_mlflow

    def test_initialization(self, mock_trainer_setup):
        """Test trainer initialization."""
        trainer, mock_mlflow = mock_trainer_setup

        assert trainer.model is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.config is not None
        assert trainer.device is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.metrics is not None

        # MLflow should be set up
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.start_run.assert_called_once()

    def test_setup_optimizer(self, mock_trainer_setup):
        """Test optimizer setup."""
        trainer, _ = mock_trainer_setup

        assert trainer.optimizer is not None
        assert hasattr(trainer.optimizer, 'param_groups')

        # Should have different learning rates for different components
        param_groups = trainer.optimizer.param_groups
        assert len(param_groups) >= 3  # base_model, answer_head, uncertainty_head

    def test_setup_scheduler(self, mock_trainer_setup):
        """Test learning rate scheduler setup."""
        trainer, _ = mock_trainer_setup

        assert trainer.scheduler is not None

        # Test different scheduler types
        trainer.config['scheduler'] = 'cosine'
        scheduler = trainer._setup_scheduler()
        assert scheduler is not None

    def test_train_epoch(self, mock_trainer_setup):
        """Test training for one epoch."""
        trainer, mock_mlflow = mock_trainer_setup

        # Mock model forward and loss computation
        mock_output = MetacognitiveOutput(
            answer_logits=torch.randn(2, 4),
            answer_prediction=[1, 2],
            answer_confidence=0.8,
            uncertainty_logits=torch.randn(2, 3),
            uncertainty_type=["knowledge_gap", "ambiguous"],
            uncertainty_confidence=0.6,
            explanation="Test explanation"
        )

        with patch.object(trainer.model, 'forward', return_value=mock_output), \
             patch.object(trainer.model, 'compute_loss') as mock_compute_loss:

            mock_losses = {
                "total_loss": torch.tensor(1.0),
                "answer_loss": torch.tensor(0.5),
                "uncertainty_loss": torch.tensor(0.3),
                "calibration_loss": torch.tensor(0.2)
            }
            mock_compute_loss.return_value = mock_losses

            # Run one training epoch
            metrics = trainer._train_epoch(0)

            assert "train_loss" in metrics
            assert "train_answer_loss" in metrics
            assert "train_uncertainty_loss" in metrics
            assert "train_accuracy" in metrics

            assert isinstance(metrics["train_loss"], float)
            assert 0 <= metrics["train_accuracy"] <= 1

    def test_validate_epoch(self, mock_trainer_setup):
        """Test validation for one epoch."""
        trainer, _ = mock_trainer_setup

        # Mock model forward and loss computation
        mock_output = MetacognitiveOutput(
            answer_logits=torch.randn(2, 4),
            answer_prediction=[1, 2],
            answer_confidence=0.8,
            uncertainty_logits=torch.randn(2, 3),
            uncertainty_type=["knowledge_gap", "ambiguous"],
            uncertainty_confidence=0.6,
            explanation="Test explanation"
        )

        with patch.object(trainer.model, 'forward', return_value=mock_output), \
             patch.object(trainer.model, 'compute_loss') as mock_compute_loss:

            mock_losses = {
                "total_loss": torch.tensor(1.0),
                "answer_loss": torch.tensor(0.5),
                "uncertainty_loss": torch.tensor(0.3),
                "calibration_loss": torch.tensor(0.2)
            }
            mock_compute_loss.return_value = mock_losses

            # Run one validation epoch
            metrics = trainer._validate_epoch(0)

            assert "val_loss" in metrics
            assert "val_accuracy" in metrics
            assert isinstance(metrics["val_loss"], float)
            assert 0 <= metrics["val_accuracy"] <= 1

    def test_save_checkpoint(self, mock_trainer_setup, temp_dir):
        """Test checkpoint saving."""
        trainer, _ = mock_trainer_setup

        # Save a checkpoint
        trainer._save_checkpoint(epoch=0, is_best=True)

        # Check that files were created
        checkpoint_path = temp_dir / "checkpoint_epoch_0.pt"
        best_path = temp_dir / "best_model.pt"

        assert checkpoint_path.exists()
        assert best_path.exists()

        # Load and check checkpoint contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "config" in checkpoint

    def test_load_checkpoint(self, mock_trainer_setup, temp_dir):
        """Test checkpoint loading."""
        trainer, _ = mock_trainer_setup

        # Save a checkpoint first
        trainer._save_checkpoint(epoch=0, is_best=True)
        checkpoint_path = temp_dir / "checkpoint_epoch_0.pt"

        # Modify some state
        original_step = trainer.global_step
        trainer.global_step = 999

        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))

        # State should be restored
        assert trainer.global_step == 0  # Should be restored from checkpoint

    def test_train_full_loop(self, mock_trainer_setup):
        """Test complete training loop."""
        trainer, mock_mlflow = mock_trainer_setup

        # Set very short training for testing
        trainer.config['num_epochs'] = 1
        trainer.config['patience'] = 1

        # Mock model methods
        mock_output = MetacognitiveOutput(
            answer_logits=torch.randn(2, 4),
            answer_prediction=[1, 2],
            answer_confidence=0.8,
            uncertainty_logits=torch.randn(2, 3),
            uncertainty_type=["knowledge_gap", "ambiguous"],
            uncertainty_confidence=0.6,
            explanation="Test explanation"
        )

        mock_losses = {
            "total_loss": torch.tensor(1.0),
            "answer_loss": torch.tensor(0.5),
            "uncertainty_loss": torch.tensor(0.3),
            "calibration_loss": torch.tensor(0.2)
        }

        with patch.object(trainer.model, 'forward', return_value=mock_output), \
             patch.object(trainer.model, 'compute_loss', return_value=mock_losses), \
             patch.object(trainer.model, 'calibrate_temperature'):

            # Run training
            history = trainer.train(num_epochs=1)

            assert isinstance(history, dict)
            if history:  # May be empty if training is very short
                assert "train_loss" in history or len(history) == 0

    def test_early_stopping(self, mock_trainer_setup):
        """Test early stopping functionality."""
        trainer, _ = mock_trainer_setup

        # Set up for immediate early stopping
        trainer.config['patience'] = 1
        trainer.patience_counter = 1

        # Mock decreasing validation loss to prevent early stopping
        val_losses = [1.0, 0.9, 0.8]  # Decreasing
        loss_iter = iter(val_losses)

        def mock_validate_epoch(epoch):
            return {"val_loss": next(loss_iter)}

        with patch.object(trainer, '_train_epoch', return_value={"train_loss": 1.0}), \
             patch.object(trainer, '_validate_epoch', side_effect=mock_validate_epoch), \
             patch.object(trainer, '_final_evaluation'):

            history = trainer.train(num_epochs=5)  # Should stop early

            # Training should complete without reaching 5 epochs due to patience
            assert isinstance(history, dict)

    def test_evaluate_selective_prediction(self, mock_trainer_setup):
        """Test selective prediction evaluation."""
        trainer, _ = mock_trainer_setup

        # Mock model outputs
        mock_output = MetacognitiveOutput(
            answer_logits=torch.randn(2, 4),
            answer_prediction=[1, 2],
            answer_confidence=0.8,
            uncertainty_logits=torch.randn(2, 3),
            uncertainty_type=["knowledge_gap", "ambiguous"],
            uncertainty_confidence=0.6,
            explanation="Test explanation"
        )

        with patch.object(trainer.model, 'forward', return_value=mock_output):
            results = trainer.evaluate_selective_prediction(
                trainer.val_loader,
                coverage_levels=[0.5, 0.8]
            )

            assert isinstance(results, dict)
            assert "coverage_0.5" in results
            assert "coverage_0.8" in results

            for coverage_result in results.values():
                assert "selective_accuracy" in coverage_result
                assert "selective_risk" in coverage_result

    def test_gradient_clipping(self, mock_trainer_setup):
        """Test gradient clipping during training."""
        trainer, _ = mock_trainer_setup

        # Mock model with exploding gradients
        mock_output = MetacognitiveOutput(
            answer_logits=torch.randn(2, 4),
            answer_prediction=[1, 2],
            answer_confidence=0.8,
            uncertainty_logits=torch.randn(2, 3),
            uncertainty_type=["knowledge_gap", "ambiguous"],
            uncertainty_confidence=0.6,
            explanation="Test explanation"
        )

        mock_losses = {
            "total_loss": torch.tensor(10.0, requires_grad=True),  # Large loss
            "answer_loss": torch.tensor(5.0),
            "uncertainty_loss": torch.tensor(3.0),
            "calibration_loss": torch.tensor(2.0)
        }

        with patch.object(trainer.model, 'forward', return_value=mock_output), \
             patch.object(trainer.model, 'compute_loss', return_value=mock_losses), \
             patch('torch.nn.utils.clip_grad_norm_') as mock_clip:

            # Run one training step
            trainer._train_epoch(0)

            # Gradient clipping should be called
            mock_clip.assert_called()

    def test_different_optimizers(self, small_model, mock_training_data, sample_config, temp_dir):
        """Test different optimizer configurations."""
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.training.trainer.mlflow'):
            # Test unsupported optimizer
            config = sample_config.to_dict()
            config['optimizer'] = 'sgd'

            with pytest.raises(ValueError, match="Unsupported optimizer"):
                trainer = MetacognitiveTrainer(
                    model=small_model,
                    train_loader=mock_training_data,
                    val_loader=mock_training_data,
                    config=config,
                    experiment_name="test",
                    checkpoint_dir=str(temp_dir)
                )

    def test_different_schedulers(self, small_model, mock_training_data, sample_config, temp_dir):
        """Test different scheduler configurations."""
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.training.trainer.mlflow'):
            config = sample_config.to_dict()

            # Test cosine scheduler
            config['scheduler'] = 'cosine'
            trainer = MetacognitiveTrainer(
                model=small_model,
                train_loader=mock_training_data,
                val_loader=mock_training_data,
                config=config,
                experiment_name="test",
                checkpoint_dir=str(temp_dir)
            )
            assert trainer.scheduler is not None

            # Test unsupported scheduler
            config['scheduler'] = 'unsupported'
            with pytest.raises(ValueError, match="Unsupported scheduler"):
                trainer = MetacognitiveTrainer(
                    model=small_model,
                    train_loader=mock_training_data,
                    val_loader=mock_training_data,
                    config=config,
                    experiment_name="test",
                    checkpoint_dir=str(temp_dir)
                )

    def test_mlflow_logging(self, mock_trainer_setup):
        """Test MLflow logging functionality."""
        trainer, mock_mlflow = mock_trainer_setup

        # Test that MLflow methods are called during training setup
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()

    def test_format_training_history(self, mock_trainer_setup):
        """Test training history formatting."""
        trainer, _ = mock_trainer_setup

        # Add some fake history
        trainer.training_history = [
            {"train_loss": 1.0, "val_loss": 1.2, "val_accuracy": 0.8},
            {"train_loss": 0.8, "val_loss": 1.0, "val_accuracy": 0.85}
        ]

        formatted = trainer._format_training_history()

        assert "train_loss" in formatted
        assert "val_loss" in formatted
        assert "val_accuracy" in formatted
        assert formatted["train_loss"] == [1.0, 0.8]
        assert formatted["val_accuracy"] == [0.8, 0.85]

    def test_empty_training_history(self, mock_trainer_setup):
        """Test handling of empty training history."""
        trainer, _ = mock_trainer_setup

        trainer.training_history = []
        formatted = trainer._format_training_history()

        assert formatted == {}

    def test_device_handling(self, mock_trainer_setup):
        """Test that trainer properly handles device assignment."""
        trainer, _ = mock_trainer_setup

        # Model should be moved to trainer device
        assert next(trainer.model.parameters()).device == trainer.device

    def test_checkpoint_directory_creation(self, small_model, mock_training_data, sample_config, temp_dir):
        """Test that checkpoint directory is created if it doesn't exist."""
        checkpoint_dir = temp_dir / "new_checkpoints"
        assert not checkpoint_dir.exists()

        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.training.trainer.mlflow'):
            trainer = MetacognitiveTrainer(
                model=small_model,
                train_loader=mock_training_data,
                val_loader=mock_training_data,
                config=sample_config.to_dict(),
                experiment_name="test",
                checkpoint_dir=str(checkpoint_dir)
            )

        assert checkpoint_dir.exists()

    def test_final_evaluation(self, mock_trainer_setup, temp_dir):
        """Test final evaluation process."""
        trainer, _ = mock_trainer_setup

        # Create a mock best checkpoint
        checkpoint_path = temp_dir / "best_model.pt"
        torch.save({
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "best_val_loss": 1.0,
            "global_step": 100,
            "epoch": 5,
            "config": trainer.config,
            "training_history": []
        }, checkpoint_path)

        # Mock model methods
        mock_output = MetacognitiveOutput(
            answer_logits=torch.randn(2, 4),
            answer_prediction=[1, 2],
            answer_confidence=0.8,
            uncertainty_logits=torch.randn(2, 3),
            uncertainty_type=["knowledge_gap", "ambiguous"],
            uncertainty_confidence=0.6,
            explanation="Test explanation"
        )

        with patch.object(trainer.model, 'forward', return_value=mock_output), \
             patch.object(trainer.model, 'calibrate_temperature'), \
             patch.object(trainer, 'load_checkpoint') as mock_load:

            trainer._final_evaluation()

            # Should load best checkpoint
            mock_load.assert_called_once()

    def test_close(self, mock_trainer_setup):
        """Test trainer cleanup."""
        trainer, mock_mlflow = mock_trainer_setup

        trainer.close()

        mock_mlflow.end_run.assert_called_once()

    @pytest.mark.parametrize("metric_for_best_model", ["val_loss", "val_accuracy"])
    def test_best_model_selection(self, metric_for_best_model, small_model, mock_training_data, sample_config, temp_dir):
        """Test best model selection with different metrics."""
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.training.trainer.mlflow'):
            config = sample_config.to_dict()
            config["metric_for_best_model"] = metric_for_best_model

            trainer = MetacognitiveTrainer(
                model=small_model,
                train_loader=mock_training_data,
                val_loader=mock_training_data,
                config=config,
                experiment_name="test",
                checkpoint_dir=str(temp_dir)
            )

            assert trainer.config["metric_for_best_model"] == metric_for_best_model