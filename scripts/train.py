#!/usr/bin/env python3
"""
Training script for metacognitive uncertainty calibration model.

This script provides a complete training pipeline with MLflow tracking,
checkpointing, and comprehensive evaluation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
import mlflow

from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.loader import MMLUDataLoader
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing import MMLUPreprocessor
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model import MetacognitiveUncertaintyModel
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.training.trainer import MetacognitiveTrainer
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.utils.config import Config

logger = logging.getLogger(__name__)


def setup_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Setup train, validation, and test data loaders.

    Args:
        config: Configuration object

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Setting up data loaders...")

    # Initialize data loader
    data_loader = MMLUDataLoader(
        cache_dir=config.data.cache_dir,
        seed=config.seed,
        uncertainty_augmentation=config.data.uncertainty_augmentation
    )

    # Load datasets
    train_dataset = data_loader.load_dataset(
        split=config.data.train_split,
        subjects=config.data.subjects
    )

    val_dataset = data_loader.load_dataset(
        split=config.data.val_split,
        subjects=config.data.subjects
    )

    test_dataset = data_loader.load_dataset(
        split=config.data.test_split,
        subjects=config.data.subjects
    )

    logger.info(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test examples")

    # Setup preprocessor
    preprocessor = MMLUPreprocessor(
        tokenizer_name=config.model.base_model_name,
        max_length=config.data.max_length,
        include_uncertainty_prompt=True
    )

    # Get few-shot examples if enabled
    few_shot_examples = None
    if config.data.use_few_shot:
        few_shot_examples = data_loader.sample_few_shot_examples(
            train_dataset,
            n_examples=config.data.few_shot_examples,
            per_domain=True
        )

    # Create data loaders
    def create_dataloader(dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader with proper preprocessing."""

        def collate_fn(batch):
            # Convert dataset batch to list of dictionaries
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

            # Tokenize batch
            tokenized = preprocessor.batch_tokenize(
                examples,
                use_few_shot=config.data.use_few_shot,
                few_shot_examples=[ex.__dict__ for ex in few_shot_examples] if few_shot_examples else None
            )

            return tokenized

        return DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )

    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset, shuffle=False)
    test_loader = create_dataloader(test_dataset, shuffle=False)

    logger.info("Data loaders setup complete")
    return train_loader, val_loader, test_loader


def setup_model(config: Config) -> MetacognitiveUncertaintyModel:
    """
    Setup and initialize the metacognitive uncertainty model.

    Args:
        config: Configuration object

    Returns:
        Initialized model
    """
    logger.info("Setting up model...")

    model = MetacognitiveUncertaintyModel(
        base_model_name=config.model.base_model_name,
        num_choices=config.model.num_choices,
        uncertainty_weight=config.model.uncertainty_weight,
        explanation_weight=config.model.explanation_weight,
        use_epistemic_estimation=config.model.use_epistemic_estimation,
        freeze_base_model=config.model.freeze_base_model
    )

    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")

    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train metacognitive uncertainty model")

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    # Override options
    parser.add_argument("--data_batch_size", type=int, help="Training batch size")
    parser.add_argument("--training_learning_rate", type=float, help="Learning rate")
    parser.add_argument("--training_num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--model_base_model_name", type=str, help="Base model name")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--experiment_run_name", type=str, help="Run name")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Execution options
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--evaluate_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Merge with command line arguments
    config = config.merge_with_args(args)

    # Setup logging
    config.setup_logging(args.log_level)

    # Debug mode adjustments
    if args.debug:
        config.data.batch_size = 2
        config.training.num_epochs = 2
        config.training.log_every = 1
        config.data.num_workers = 0
        logger.info("Debug mode enabled - reduced batch size and epochs")

    # Validate configuration
    issues = config.validate()
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        sys.exit(1)

    logger.info("Starting metacognitive uncertainty training")
    logger.info(f"Configuration: {config}")

    try:
        # Setup data
        train_loader, val_loader, test_loader = setup_data_loaders(config)

        # Setup model
        model = setup_model(config)

        # Setup trainer
        trainer = MetacognitiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.to_dict().get("training", config.to_dict()),
            experiment_name=config.experiment.name,
            checkpoint_dir=str(config.get_cache_dir() / "checkpoints")
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Evaluate only mode
        if args.evaluate_only:
            logger.info("Evaluation only mode")

            # Load best checkpoint if exists
            best_checkpoint = config.get_cache_dir() / "checkpoints" / "best_model.pt"
            if best_checkpoint.exists():
                trainer.load_checkpoint(str(best_checkpoint))

            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_metrics = trainer._validate_epoch(-1)  # Use validation function

            logger.info("Test Results:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            # Selective prediction evaluation
            selective_results = trainer.evaluate_selective_prediction(test_loader)
            logger.info("Selective Prediction Results:")
            for coverage, metrics in selective_results.items():
                logger.info(f"  {coverage}: {metrics}")

        else:
            # Training mode
            logger.info("Starting training...")

            # Train model
            training_history = trainer.train(config.training.num_epochs)

            logger.info("Training completed!")
            logger.info("Training History Summary:")
            if training_history:
                for metric in ["train_loss", "val_loss", "val_accuracy"]:
                    if metric in training_history:
                        values = training_history[metric]
                        logger.info(f"  {metric}: final={values[-1]:.4f}, best={min(values) if 'loss' in metric else max(values):.4f}")

            # Final evaluation on test set
            logger.info("Final evaluation on test set...")
            model.eval()

            # Load best checkpoint
            best_checkpoint = config.get_cache_dir() / "checkpoints" / "best_model.pt"
            if best_checkpoint.exists():
                trainer.load_checkpoint(str(best_checkpoint))

            # Evaluate
            test_metrics = trainer._validate_epoch(-1)
            logger.info("Final Test Results:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

        # Save final configuration
        config.to_yaml(config.get_cache_dir() / "final_config.yaml")

        # Close trainer (ends MLflow run)
        trainer.close()

        logger.info("Training script completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()