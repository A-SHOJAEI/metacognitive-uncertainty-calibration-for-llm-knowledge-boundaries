"""
Configuration management for metacognitive uncertainty calibration.
"""

import logging
import logging.handlers
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import yaml

from .constants import (
    DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY, DEFAULT_WARMUP_STEPS,
    DEFAULT_MAX_GRAD_NORM, DEFAULT_DROPOUT_RATE, DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS, DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_NUM_CHOICES,
    DEFAULT_UNCERTAINTY_WEIGHT, DEFAULT_EXPLANATION_WEIGHT,
    MIN_LEARNING_RATE, MAX_LEARNING_RATE, MIN_DROPOUT_RATE, MAX_DROPOUT_RATE,
    MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY, MIN_BATCH_SIZE, MAX_BATCH_SIZE,
    MAX_SEQUENCE_LENGTH_HARD_LIMIT, DEVICE_OPTIONS, LR_SCHEDULER_OPTIONS,
    DOMAIN_SPLIT_STRATEGIES, MIN_TEMPERATURE, MAX_TEMPERATURE
)

logger = logging.getLogger(__name__)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        # Basic log data
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from LoggerAdapter or custom fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'exc_info', 'exc_text', 'stack_info'
            }:
                log_entry[key] = value

        try:
            return json.dumps(log_entry, default=str)
        except (TypeError, ValueError):
            # Fallback to basic format if JSON serialization fails
            return f"{log_entry['timestamp']} - {log_entry['logger']} - {log_entry['level']} - {log_entry['message']}"


class ContextualLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds contextual information to log records."""

    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """Initialize adapter with context.

        Args:
            logger: Base logger
            context: Context information to add to all log records
        """
        super().__init__(logger, context)

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add context.

        Args:
            msg: Log message
            kwargs: Additional keyword arguments

        Returns:
            Tuple of (message, kwargs) with added context
        """
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        kwargs['extra'].update(self.extra)
        return msg, kwargs

    def debug_with_timing(self, msg: str, start_time: float, **kwargs) -> None:
        """Log debug message with execution timing.

        Args:
            msg: Log message
            start_time: Start time (from time.time())
            **kwargs: Additional keyword arguments
        """
        duration = time.time() - start_time
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['duration_seconds'] = round(duration, 4)
        self.debug(f"{msg} (took {duration:.4f}s)", **kwargs)

    def log_model_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, **kwargs) -> None:
        """Log model metrics with structured format.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional training step
            **kwargs: Additional keyword arguments
        """
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        kwargs['extra']['metrics'] = metrics
        if step is not None:
            kwargs['extra']['step'] = step

        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        step_str = f" [step {step}]" if step is not None else ""
        self.info(f"Metrics{step_str}: {metric_str}", **kwargs)


@dataclass
class DataConfig:
    """Data-related configuration."""
    dataset_name: str = "cais/mmlu"
    cache_dir: Optional[str] = None
    max_length: int = 512
    batch_size: int = 16
    num_workers: int = 4
    train_split: str = "auxiliary_train"
    val_split: str = "validation"
    test_split: str = "test"
    subjects: Optional[List[str]] = None
    use_few_shot: bool = True
    few_shot_examples: int = 5
    uncertainty_augmentation: bool = True
    domain_split_ratio: float = 0.8


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    base_model_name: str = "microsoft/DialoGPT-medium"
    num_choices: int = 4
    uncertainty_weight: float = 0.3
    explanation_weight: float = 0.2
    use_epistemic_estimation: bool = True
    freeze_base_model: bool = False
    epistemic_samples: int = 10
    dropout_rate: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    temperature_init: float = 1.0


@dataclass
class TrainingConfig:
    """Training procedure configuration."""
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 3
    patience: int = 10
    save_every: int = 5
    log_every: int = 100
    eval_every: int = 1
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    early_stopping: bool = True
    metric_for_best_model: str = "val_loss"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    n_bins: int = 15
    coverage_levels: List[float] = None
    uncertainty_types: List[str] = None
    compute_domain_metrics: bool = True
    compute_calibration_plots: bool = True
    compute_selective_prediction: bool = True
    selective_coverage: float = 0.8
    statistical_tests: bool = True
    bootstrap_samples: int = 1000

    def __post_init__(self):
        if self.coverage_levels is None:
            self.coverage_levels = [0.5, 0.7, 0.8, 0.9]
        if self.uncertainty_types is None:
            self.uncertainty_types = ["knowledge_gap", "ambiguous", "reasoning_error"]


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    name: str = "metacognitive_uncertainty"
    run_name: Optional[str] = None
    tags: Dict[str, str] = None
    notes: str = ""
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    log_model: bool = True
    log_artifacts: bool = True
    log_figures: bool = True

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class InferenceConfig:
    """Inference configuration."""
    batch_size: int = 32
    max_length: int = 512
    return_explanations: bool = True
    return_uncertainties: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    num_beams: Optional[int] = None
    do_sample: bool = False


class Config:
    """
    Main configuration class that combines all configuration components
    with YAML loading/saving capabilities.
    """

    def __init__(
        self,
        data: Optional[DataConfig] = None,
        model: Optional[ModelConfig] = None,
        training: Optional[TrainingConfig] = None,
        evaluation: Optional[EvaluationConfig] = None,
        experiment: Optional[ExperimentConfig] = None,
        inference: Optional[InferenceConfig] = None
    ) -> None:
        """
        Initialize configuration.

        Args:
            data: Data configuration
            model: Model configuration
            training: Training configuration
            evaluation: Evaluation configuration
            experiment: Experiment configuration
            inference: Inference configuration
        """
        self.data = data or DataConfig()
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.experiment = experiment or ExperimentConfig()
        self.inference = inference or InferenceConfig()

        # Set random seed for reproducibility
        self.seed = 42
        self.set_seed()

        logger.info("Configuration initialized")

    def set_seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducibility."""
        if seed is not None:
            self.seed = seed

        import random
        import numpy as np
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(f"Set random seed to {self.seed}")

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            logger.warning(f"Config file {yaml_path} not found. Using default configuration.")
            return cls()

        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            # Parse each configuration section
            data_config = DataConfig(**config_dict.get('data', {}))
            model_config = ModelConfig(**config_dict.get('model', {}))
            training_config = TrainingConfig(**config_dict.get('training', {}))
            evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
            experiment_config = ExperimentConfig(**config_dict.get('experiment', {}))
            inference_config = InferenceConfig(**config_dict.get('inference', {}))

            config = cls(
                data=data_config,
                model=model_config,
                training=training_config,
                evaluation=evaluation_config,
                experiment=experiment_config,
                inference=inference_config
            )

            # Set seed if specified in config
            if 'seed' in config_dict:
                config.set_seed(config_dict['seed'])

            logger.info(f"Loaded configuration from {yaml_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration from {yaml_path}: {e}")
            logger.info("Using default configuration")
            return cls()

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'evaluation': asdict(self.evaluation),
            'experiment': asdict(self.experiment),
            'inference': asdict(self.inference),
            'seed': self.seed
        }

        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Saved configuration to {yaml_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {yaml_path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'evaluation': asdict(self.evaluation),
            'experiment': asdict(self.experiment),
            'inference': asdict(self.inference),
            'seed': self.seed
        }

    def update_from_dict(self, update_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.

        Args:
            update_dict: Dictionary with configuration updates
        """
        def update_dataclass(dataclass_instance: Any, updates: Dict[str, Any]) -> None:
            for key, value in updates.items():
                if hasattr(dataclass_instance, key):
                    setattr(dataclass_instance, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")

        for section, updates in update_dict.items():
            if section == 'data' and isinstance(updates, dict):
                update_dataclass(self.data, updates)
            elif section == 'model' and isinstance(updates, dict):
                update_dataclass(self.model, updates)
            elif section == 'training' and isinstance(updates, dict):
                update_dataclass(self.training, updates)
            elif section == 'evaluation' and isinstance(updates, dict):
                update_dataclass(self.evaluation, updates)
            elif section == 'experiment' and isinstance(updates, dict):
                update_dataclass(self.experiment, updates)
            elif section == 'inference' and isinstance(updates, dict):
                update_dataclass(self.inference, updates)
            elif section == 'seed':
                self.set_seed(updates)
            else:
                logger.warning(f"Unknown configuration section: {section}")

    def validate(self) -> List[str]:
        """Validate configuration comprehensively using defined constants.

        Returns:
            List of validation issues with detailed descriptions
        """
        issues = []

        # Data validation with range checks
        if not MIN_BATCH_SIZE <= self.data.batch_size <= MAX_BATCH_SIZE:
            issues.append(
                f"data.batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}, "
                f"got {self.data.batch_size}"
            )

        if self.data.max_length <= 0:
            issues.append("data.max_length must be positive")
        elif self.data.max_length > MAX_SEQUENCE_LENGTH_HARD_LIMIT:
            issues.append(
                f"data.max_length should be <= {MAX_SEQUENCE_LENGTH_HARD_LIMIT} for reasonable memory usage, "
                f"got {self.data.max_length}"
            )

        if hasattr(self.data, 'domain_split_ratio') and not (0 < self.data.domain_split_ratio < 1):
            issues.append(
                f"data.domain_split_ratio must be between 0 and 1, "
                f"got {self.data.domain_split_ratio}"
            )

        if self.data.num_workers < 0:
            issues.append("data.num_workers must be non-negative")

        if hasattr(self.data, 'domain_split_strategy'):
            if self.data.domain_split_strategy not in DOMAIN_SPLIT_STRATEGIES:
                issues.append(
                    f"data.domain_split_strategy must be one of {DOMAIN_SPLIT_STRATEGIES}, "
                    f"got {self.data.domain_split_strategy}"
                )

        # Model validation with comprehensive checks
        if self.model.num_choices < 2:
            issues.append("model.num_choices should be at least 2 for meaningful multiple choice")
        elif self.model.num_choices > 6:
            issues.append(
                f"model.num_choices should be <= 6 for practical purposes, "
                f"got {self.model.num_choices}"
            )

        if not (0 <= self.model.uncertainty_weight <= 1):
            issues.append(
                f"model.uncertainty_weight should be in [0, 1], "
                f"got {self.model.uncertainty_weight}"
            )

        if not (0 <= self.model.explanation_weight <= 1):
            issues.append(
                f"model.explanation_weight should be in [0, 1], "
                f"got {self.model.explanation_weight}"
            )

        if not MIN_DROPOUT_RATE <= self.model.dropout_rate <= MAX_DROPOUT_RATE:
            issues.append(
                f"model.dropout_rate should be in [{MIN_DROPOUT_RATE}, {MAX_DROPOUT_RATE}], "
                f"got {self.model.dropout_rate}"
            )

        if hasattr(self.model, 'temperature'):
            if not MIN_TEMPERATURE <= self.model.temperature <= MAX_TEMPERATURE:
                issues.append(
                    f"model.temperature should be in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}], "
                    f"got {self.model.temperature}"
                )

        # Training validation with range checks
        if not MIN_LEARNING_RATE <= self.training.learning_rate <= MAX_LEARNING_RATE:
            issues.append(
                f"training.learning_rate should be in [{MIN_LEARNING_RATE}, {MAX_LEARNING_RATE}], "
                f"got {self.training.learning_rate}"
            )

        if self.training.num_epochs <= 0:
            issues.append("training.num_epochs must be positive")
        elif self.training.num_epochs > 1000:
            issues.append(
                f"training.num_epochs seems excessive (>{1000}), "
                f"got {self.training.num_epochs}"
            )

        if not MIN_WEIGHT_DECAY <= self.training.weight_decay <= MAX_WEIGHT_DECAY:
            issues.append(
                f"training.weight_decay should be in [{MIN_WEIGHT_DECAY}, {MAX_WEIGHT_DECAY}], "
                f"got {self.training.weight_decay}"
            )

        if self.training.warmup_steps < 0:
            issues.append("training.warmup_steps must be non-negative")

        if self.training.max_grad_norm <= 0:
            issues.append("training.max_grad_norm must be positive")

        if hasattr(self.training, 'early_stopping_patience'):
            if self.training.early_stopping_patience <= 0:
                issues.append("training.early_stopping_patience must be positive")

        if hasattr(self.training, 'lr_scheduler'):
            if self.training.lr_scheduler not in LR_SCHEDULER_OPTIONS:
                issues.append(
                    f"training.lr_scheduler must be one of {LR_SCHEDULER_OPTIONS}, "
                    f"got {self.training.lr_scheduler}"
                )

        # Evaluation validation
        if hasattr(self.evaluation, 'n_bins'):
            if self.evaluation.n_bins <= 0:
                issues.append("evaluation.n_bins must be positive")
            elif self.evaluation.n_bins > 100:
                issues.append(
                    f"evaluation.n_bins should be <= 100 for meaningful calibration, "
                    f"got {self.evaluation.n_bins}"
                )

        if hasattr(self.evaluation, 'coverage_levels'):
            if not all(0 <= c <= 1 for c in self.evaluation.coverage_levels):
                issues.append("evaluation.coverage_levels must be between 0 and 1")

        # Inference validation
        if hasattr(self.inference, 'batch_size'):
            if self.inference.batch_size <= 0:
                issues.append("inference.batch_size must be positive")

        if hasattr(self.inference, 'temperature'):
            if self.inference.temperature <= 0:
                issues.append("inference.temperature must be positive")

        # Device validation
        if hasattr(self, 'device') and self.device not in DEVICE_OPTIONS:
            issues.append(
                f"device should be one of {DEVICE_OPTIONS}, "
                f"got {self.device}"
            )

        # Cross-parameter validation
        total_weight = self.model.uncertainty_weight + self.model.explanation_weight
        if total_weight > 1.0:
            issues.append(
                f"Sum of model.uncertainty_weight and model.explanation_weight should be <= 1.0, "
                f"got {total_weight:.3f}"
            )

        # Performance-related warnings (not errors)
        if self.data.batch_size > 64 and self.data.max_length > 1024:
            issues.append(
                "WARNING: Large batch_size with long sequences may cause memory issues"
            )

        if hasattr(self.training, 'accumulate_grad_batches'):
            effective_batch_size = self.data.batch_size * self.training.accumulate_grad_batches
            if effective_batch_size > 512:
                issues.append(
                    f"WARNING: Effective batch size ({effective_batch_size}) is very large, "
                    "may affect training stability"
                )

        # Logging configuration validation
        if hasattr(self.training, 'log_every') and self.training.log_every <= 0:
            issues.append("training.log_every must be positive")

        if hasattr(self.training, 'eval_every') and self.training.eval_every <= 0:
            issues.append("training.eval_every must be positive")

        if hasattr(self.training, 'save_every') and self.training.save_every <= 0:
            issues.append("training.save_every must be positive")

        return issues

    def get_device(self) -> str:
        """Get device string based on availability."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_cache_dir(self) -> Path:
        """Get cache directory path."""
        if self.data.cache_dir:
            return Path(self.data.cache_dir)
        else:
            return Path.home() / ".cache" / "metacognitive_uncertainty"

    def setup_logging(self, log_level: str = "INFO", use_json: bool = False) -> None:
        """Setup comprehensive logging configuration with structured JSON support.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_json: Whether to use JSON structured logging format

        Raises:
            ValueError: If invalid log level provided
        """
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            logger.warning(f'Invalid log level: {log_level}, defaulting to INFO')
            numeric_level = logging.INFO
            log_level = "INFO"

        # Clear existing handlers to avoid duplication
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create cache directory if it doesn't exist
        cache_dir = self.get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup formatters
        if use_json:
            json_formatter = JSONFormatter()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)-20s - %(levelname)-8s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )

        handlers = []

        # Console handler (always use human-readable format)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(max(numeric_level, logging.INFO))  # Console shows INFO+
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

        # File handlers
        log_files = {
            "training.log": numeric_level,  # All logs
            "debug.log": logging.DEBUG,     # Debug and above
            "error.log": logging.ERROR      # Errors only
        }

        for log_file, file_level in log_files.items():
            try:
                # Use rotating file handler to manage file size
                file_handler = logging.handlers.RotatingFileHandler(
                    cache_dir / log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
                file_handler.setLevel(file_level)

                if use_json and log_file == "training.log":
                    file_handler.setFormatter(json_formatter)
                else:
                    file_handler.setFormatter(file_formatter)

                handlers.append(file_handler)

            except Exception as e:
                # Fallback to console if file logging fails
                logger.warning(f"Failed to setup file logging for {log_file}: {e}")

        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,  # Capture all levels, let handlers filter
            handlers=handlers,
            force=True  # Override existing configuration
        )

        # Set library logger levels to reduce noise
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # Log setup completion
        setup_logger = logging.getLogger(__name__)
        setup_logger.info(
            f"Logging setup complete - Level: {log_level}, JSON: {use_json}, "
            f"Handlers: {len(handlers)}, Cache: {cache_dir}"
        )

        # Log system information for debugging
        if numeric_level <= logging.DEBUG:
            import platform
            try:
                import psutil
                import torch

                setup_logger.debug("=== System Information ===")
                setup_logger.debug(f"Platform: {platform.platform()}")
                setup_logger.debug(f"Python: {platform.python_version()}")
                setup_logger.debug(f"PyTorch: {torch.__version__}")
                setup_logger.debug(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    setup_logger.debug(f"CUDA devices: {torch.cuda.device_count()}")
                    setup_logger.debug(f"Current device: {torch.cuda.current_device()}")

                memory = psutil.virtual_memory()
                setup_logger.debug(f"Memory: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
                setup_logger.debug("=== End System Information ===")
            except ImportError as e:
                setup_logger.debug(f"Could not load system info modules: {e}")

    def get_contextual_logger(self, name: str, context: Optional[Dict[str, Any]] = None) -> ContextualLoggerAdapter:
        """Create a contextual logger with additional metadata.

        Args:
            name: Logger name
            context: Additional context to include in all log messages

        Returns:
            ContextualLoggerAdapter with added context
        """
        base_logger = logging.getLogger(name)
        default_context = {
            "experiment_name": self.experiment.name,
            "model": self.model.base_model_name,
            "device": self.device,
            "seed": self.seed
        }

        if context:
            default_context.update(context)

        return ContextualLoggerAdapter(base_logger, default_context)

    def get_experiment_name(self) -> str:
        """Generate experiment name with timestamp if not provided."""
        if self.experiment.run_name:
            return self.experiment.run_name

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment.name}_{timestamp}"

    def merge_with_args(self, args: Any) -> 'Config':
        """
        Merge configuration with command line arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            Updated configuration
        """
        # Create a copy of current config
        import copy
        new_config = copy.deepcopy(self)

        # Update with arguments that exist
        for section_name in ['data', 'model', 'training', 'evaluation', 'experiment', 'inference']:
            section = getattr(new_config, section_name)
            for field_name in section.__dataclass_fields__.keys():
                arg_name = f"{section_name}_{field_name}"
                if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                    setattr(section, field_name, getattr(args, arg_name))

        # Handle special arguments
        if hasattr(args, 'seed') and args.seed is not None:
            new_config.set_seed(args.seed)

        return new_config

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"""Config(
    data={self.data},
    model={self.model},
    training={self.training},
    evaluation={self.evaluation},
    experiment={self.experiment},
    inference={self.inference},
    seed={self.seed}
)"""