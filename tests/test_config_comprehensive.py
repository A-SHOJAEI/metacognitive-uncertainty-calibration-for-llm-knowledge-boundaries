"""
Comprehensive tests for configuration module.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.utils.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    ExperimentConfig,
    InferenceConfig
)


class TestDataConfig:
    """Test DataConfig dataclass."""

    def test_data_config_defaults(self):
        """Test DataConfig with default values."""
        config = DataConfig()

        assert config.cache_dir == "data/cache"
        assert config.dataset_name == "mmlu"
        assert config.train_split == "auxiliary_train"
        assert config.val_split == "validation"
        assert config.test_split == "test"
        assert config.subjects is None
        assert config.max_length == 512
        assert config.batch_size == 16
        assert config.num_workers == 4
        assert config.use_few_shot is False
        assert config.few_shot_examples == 5
        assert config.uncertainty_augmentation is False
        assert config.domain_split_strategy == "balanced"

    def test_data_config_custom_values(self):
        """Test DataConfig with custom values."""
        config = DataConfig(
            cache_dir="/custom/path",
            dataset_name="custom_dataset",
            train_split="train",
            val_split="val",
            test_split="test",
            subjects=["math", "physics"],
            max_length=256,
            batch_size=32,
            num_workers=8,
            use_few_shot=True,
            few_shot_examples=10,
            uncertainty_augmentation=True,
            domain_split_strategy="random"
        )

        assert config.cache_dir == "/custom/path"
        assert config.dataset_name == "custom_dataset"
        assert config.subjects == ["math", "physics"]
        assert config.max_length == 256
        assert config.batch_size == 32
        assert config.num_workers == 8
        assert config.use_few_shot is True
        assert config.few_shot_examples == 10
        assert config.uncertainty_augmentation is True
        assert config.domain_split_strategy == "random"


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = ModelConfig()

        assert config.base_model_name == "microsoft/DialoGPT-medium"
        assert config.num_choices == 4
        assert config.uncertainty_weight == 0.3
        assert config.explanation_weight == 0.2
        assert config.use_epistemic_estimation is True
        assert config.freeze_base_model is False
        assert config.dropout_rate == 0.1
        assert config.temperature == 1.0

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            base_model_name="bert-base-uncased",
            num_choices=5,
            uncertainty_weight=0.5,
            explanation_weight=0.3,
            use_epistemic_estimation=False,
            freeze_base_model=True,
            dropout_rate=0.2,
            temperature=2.0
        )

        assert config.base_model_name == "bert-base-uncased"
        assert config.num_choices == 5
        assert config.uncertainty_weight == 0.5
        assert config.explanation_weight == 0.3
        assert config.use_epistemic_estimation is False
        assert config.freeze_base_model is True
        assert config.dropout_rate == 0.2
        assert config.temperature == 2.0


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_training_config_defaults(self):
        """Test TrainingConfig with default values."""
        config = TrainingConfig()

        assert config.num_epochs == 10
        assert config.learning_rate == 2e-5
        assert config.weight_decay == 0.01
        assert config.warmup_steps == 500
        assert config.max_grad_norm == 1.0
        assert config.lr_scheduler == "cosine"
        assert config.save_every == 1000
        assert config.eval_every == 500
        assert config.log_every == 100
        assert config.early_stopping_patience == 3
        assert config.accumulate_grad_batches == 1

    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            num_epochs=20,
            learning_rate=1e-4,
            weight_decay=0.02,
            warmup_steps=1000,
            max_grad_norm=2.0,
            lr_scheduler="linear",
            save_every=2000,
            eval_every=1000,
            log_every=200,
            early_stopping_patience=5,
            accumulate_grad_batches=4
        )

        assert config.num_epochs == 20
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.02
        assert config.warmup_steps == 1000
        assert config.max_grad_norm == 2.0
        assert config.lr_scheduler == "linear"
        assert config.save_every == 2000
        assert config.eval_every == 1000
        assert config.log_every == 200
        assert config.early_stopping_patience == 5
        assert config.accumulate_grad_batches == 4


class TestConfig:
    """Test main Config class."""

    def test_config_initialization(self):
        """Test Config initialization with default values."""
        config = Config()

        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert isinstance(config.experiment, ExperimentConfig)
        assert isinstance(config.inference, InferenceConfig)
        assert config.seed == 42
        assert config.device == "auto"

    def test_config_custom_initialization(self):
        """Test Config initialization with custom values."""
        custom_data_config = DataConfig(batch_size=64)
        custom_model_config = ModelConfig(num_choices=5)

        config = Config(
            data=custom_data_config,
            model=custom_model_config,
            seed=123,
            device="cpu"
        )

        assert config.data.batch_size == 64
        assert config.model.num_choices == 5
        assert config.seed == 123
        assert config.device == "cpu"

    def test_config_to_dict(self):
        """Test Config conversion to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
        assert "evaluation" in config_dict
        assert "experiment" in config_dict
        assert "inference" in config_dict
        assert "seed" in config_dict
        assert "device" in config_dict

        # Check nested structure
        assert isinstance(config_dict["data"], dict)
        assert "batch_size" in config_dict["data"]
        assert config_dict["data"]["batch_size"] == 16

    def test_config_from_dict(self):
        """Test Config creation from dictionary."""
        config_dict = {
            "data": {"batch_size": 32, "max_length": 256},
            "model": {"num_choices": 5, "uncertainty_weight": 0.4},
            "training": {"num_epochs": 15, "learning_rate": 1e-4},
            "seed": 999,
            "device": "cuda"
        }

        config = Config.from_dict(config_dict)

        assert config.data.batch_size == 32
        assert config.data.max_length == 256
        assert config.model.num_choices == 5
        assert config.model.uncertainty_weight == 0.4
        assert config.training.num_epochs == 15
        assert config.training.learning_rate == 1e-4
        assert config.seed == 999
        assert config.device == "cuda"

    def test_config_to_yaml(self):
        """Test Config saving to YAML file."""
        config = Config()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name

        try:
            config.to_yaml(yaml_path)

            # Read back and verify
            with open(yaml_path, 'r') as f:
                loaded_data = yaml.safe_load(f)

            assert isinstance(loaded_data, dict)
            assert "data" in loaded_data
            assert "model" in loaded_data
            assert "seed" in loaded_data
            assert loaded_data["seed"] == 42

        finally:
            os.unlink(yaml_path)

    def test_config_from_yaml(self):
        """Test Config loading from YAML file."""
        config_data = {
            "data": {"batch_size": 64, "max_length": 1024},
            "model": {"base_model_name": "custom-model", "num_choices": 6},
            "training": {"num_epochs": 25},
            "seed": 888
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name

        try:
            config = Config.from_yaml(yaml_path)

            assert config.data.batch_size == 64
            assert config.data.max_length == 1024
            assert config.model.base_model_name == "custom-model"
            assert config.model.num_choices == 6
            assert config.training.num_epochs == 25
            assert config.seed == 888

        finally:
            os.unlink(yaml_path)

    def test_config_from_yaml_file_not_found(self):
        """Test Config loading when YAML file doesn't exist."""
        # Should return default config when file doesn't exist
        config = Config.from_yaml("non_existent_file.yaml")

        assert isinstance(config, Config)
        assert config.seed == 42  # Default value

    def test_config_from_yaml_invalid_yaml(self):
        """Test Config loading with invalid YAML content."""
        invalid_yaml_content = "invalid: yaml: content: ["

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml_content)
            yaml_path = f.name

        try:
            # Should return default config on invalid YAML
            config = Config.from_yaml(yaml_path)
            assert isinstance(config, Config)
            assert config.seed == 42  # Default value

        finally:
            os.unlink(yaml_path)

    def test_config_validation_success(self):
        """Test configuration validation with valid config."""
        config = Config()
        issues = config.validate()

        assert isinstance(issues, list)
        assert len(issues) == 0  # Should have no validation issues

    def test_config_validation_batch_size_negative(self):
        """Test validation of negative batch size."""
        config = Config()
        config.data.batch_size = -1

        issues = config.validate()

        assert len(issues) > 0
        assert any("batch_size must be positive" in issue for issue in issues)

    def test_config_validation_learning_rate_negative(self):
        """Test validation of negative learning rate."""
        config = Config()
        config.training.learning_rate = -0.01

        issues = config.validate()

        assert len(issues) > 0
        assert any("learning_rate must be positive" in issue for issue in issues)

    def test_config_validation_num_epochs_zero(self):
        """Test validation of zero epochs."""
        config = Config()
        config.training.num_epochs = 0

        issues = config.validate()

        assert len(issues) > 0
        assert any("num_epochs must be positive" in issue for issue in issues)

    def test_config_validation_max_length_too_large(self):
        """Test validation of excessive max_length."""
        config = Config()
        config.data.max_length = 10000

        issues = config.validate()

        assert len(issues) > 0
        assert any("max_length should be reasonable" in issue for issue in issues)

    def test_config_validation_num_choices_invalid(self):
        """Test validation of invalid num_choices."""
        config = Config()
        config.model.num_choices = 1

        issues = config.validate()

        assert len(issues) > 0
        assert any("num_choices should be at least 2" in issue for issue in issues)

    def test_config_validation_weights_out_of_range(self):
        """Test validation of weights outside [0,1] range."""
        config = Config()
        config.model.uncertainty_weight = 1.5
        config.model.explanation_weight = -0.1

        issues = config.validate()

        assert len(issues) > 0
        assert any("uncertainty_weight should be in [0, 1]" in issue for issue in issues)
        assert any("explanation_weight should be in [0, 1]" in issue for issue in issues)

    def test_config_validation_device_invalid(self):
        """Test validation of invalid device specification."""
        config = Config()
        config.device = "invalid_device"

        issues = config.validate()

        assert len(issues) > 0
        assert any("device should be 'auto', 'cpu', 'cuda', or 'mps'" in issue for issue in issues)

    def test_config_merge_with_args(self):
        """Test merging config with command line arguments."""
        config = Config()
        original_batch_size = config.data.batch_size
        original_learning_rate = config.training.learning_rate

        # Mock argparse namespace
        class MockArgs:
            data_batch_size = 128
            training_learning_rate = 1e-3
            seed = 999
            non_existent_field = "ignored"

        args = MockArgs()
        merged_config = config.merge_with_args(args)

        assert merged_config.data.batch_size == 128
        assert merged_config.training.learning_rate == 1e-3
        assert merged_config.seed == 999
        # Non-existent fields should be ignored
        assert not hasattr(merged_config, 'non_existent_field')

    def test_config_update_from_dict(self):
        """Test updating config from dictionary."""
        config = Config()
        original_batch_size = config.data.batch_size

        update_dict = {
            "data": {"batch_size": 256},
            "training": {"num_epochs": 50}
        }

        config.update_from_dict(update_dict)

        assert config.data.batch_size == 256
        assert config.training.num_epochs == 50
        # Other fields should remain unchanged
        assert config.model.num_choices == 4  # Default value

    def test_config_get_cache_dir(self):
        """Test cache directory path resolution."""
        config = Config()

        cache_dir = config.get_cache_dir()
        assert isinstance(cache_dir, Path)
        assert str(cache_dir).endswith("data/cache")

    def test_config_setup_logging(self):
        """Test logging setup."""
        config = Config()

        # Test with different log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            try:
                config.setup_logging(level)
                # Should not raise exception
                assert True
            except Exception as e:
                pytest.fail(f"setup_logging failed for level {level}: {e}")

    def test_config_setup_logging_invalid_level(self):
        """Test logging setup with invalid level."""
        config = Config()

        # Should handle invalid log level gracefully
        try:
            config.setup_logging("INVALID_LEVEL")
            # Should default to INFO level
            assert True
        except Exception as e:
            pytest.fail(f"setup_logging should handle invalid levels gracefully: {e}")

    def test_config_str_representation(self):
        """Test string representation of config."""
        config = Config()
        config_str = str(config)

        assert isinstance(config_str, str)
        assert len(config_str) > 0
        # Should contain some key information
        assert "data" in config_str.lower() or "model" in config_str.lower()

    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = Config()
        config2 = Config()

        # Should be equal with same values
        assert config1.to_dict() == config2.to_dict()

        # Should be different after modification
        config2.data.batch_size = 999
        assert config1.to_dict() != config2.to_dict()

    def test_config_device_auto_resolution(self):
        """Test automatic device resolution."""
        config = Config()
        assert config.device == "auto"

        # Test device resolution logic (if implemented)
        # This would depend on the actual implementation of device resolution
        pass

    def test_config_partial_yaml_loading(self):
        """Test loading partial YAML config (only some sections)."""
        partial_config = {
            "data": {"batch_size": 128},
            "seed": 777
            # Missing other sections
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(partial_config, f)
            yaml_path = f.name

        try:
            config = Config.from_yaml(yaml_path)

            # Specified values should be loaded
            assert config.data.batch_size == 128
            assert config.seed == 777

            # Missing sections should use defaults
            assert config.model.num_choices == 4  # Default value
            assert config.training.num_epochs == 10  # Default value

        finally:
            os.unlink(yaml_path)

    def test_config_nested_validation(self):
        """Test validation of nested config structures."""
        config = Config()

        # Set multiple invalid values
        config.data.batch_size = -5
        config.data.num_workers = -2
        config.training.learning_rate = -0.1
        config.training.num_epochs = -10
        config.model.num_choices = 0

        issues = config.validate()

        # Should catch multiple issues
        assert len(issues) >= 4
        assert any("batch_size" in issue for issue in issues)
        assert any("learning_rate" in issue for issue in issues)
        assert any("num_epochs" in issue for issue in issues)
        assert any("num_choices" in issue for issue in issues)