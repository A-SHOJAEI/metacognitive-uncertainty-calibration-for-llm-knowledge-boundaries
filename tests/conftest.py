"""
Shared test fixtures and configuration for metacognitive uncertainty calibration tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import os

from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model import MetacognitiveUncertaintyModel
from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.loader import MMLUDataLoader
from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing import MMLUPreprocessor
from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.utils.config import Config


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config = Config()

    # Reduce sizes for testing
    config.data.batch_size = 2
    config.data.max_length = 128
    config.data.num_workers = 0
    config.training.num_epochs = 1
    config.training.log_every = 1
    config.evaluation.n_bins = 5

    return config


@pytest.fixture
def mock_mmlu_data():
    """Create mock MMLU data for testing."""
    return [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Berlin", "Paris", "Madrid"],
            "answer": 2,
            "subject": "geography",
            "domain": "geography",
            "uncertainty_type": "knowledge_gap"
        },
        {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
            "subject": "elementary_mathematics",
            "domain": "mathematics",
            "uncertainty_type": "reasoning_error"
        },
        {
            "question": "Which gas is most abundant in Earth's atmosphere?",
            "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Argon"],
            "answer": 1,
            "subject": "chemistry",
            "domain": "chemistry",
            "uncertainty_type": "knowledge_gap"
        }
    ]


@pytest.fixture
def mock_tokenized_batch():
    """Create a mock tokenized batch for testing."""
    batch_size = 2
    seq_length = 64
    vocab_size = 1000

    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long),
        "answer_labels": torch.randint(0, 4, (batch_size,)),
        "uncertainty_labels": torch.randint(0, 3, (batch_size,)),
        "domain": ["mathematics", "physics"],
        "subject": ["algebra", "mechanics"]
    }


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    model = MetacognitiveUncertaintyModel(
        base_model_name="distilbert-base-uncased",
        num_choices=4,
        uncertainty_weight=0.3,
        explanation_weight=0.2,
        use_epistemic_estimation=False,  # Disable for faster testing
        freeze_base_model=True  # Freeze for faster testing
    )
    return model


@pytest.fixture
def mock_predictions():
    """Create mock predictions for testing metrics."""
    np.random.seed(42)
    n_samples = 100

    predictions = np.random.randint(0, 4, n_samples)
    labels = np.random.randint(0, 4, n_samples)
    confidences = np.random.uniform(0.1, 0.9, n_samples)
    uncertainty_types = np.random.randint(0, 3, n_samples)

    return {
        "predictions": predictions,
        "labels": labels,
        "confidences": confidences,
        "uncertainty_types": uncertainty_types
    }


@pytest.fixture
def mock_training_data():
    """Create mock training data loader."""
    class MockDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    data = [
        {
            "input_ids": torch.randint(0, 1000, (64,)),
            "attention_mask": torch.ones(64, dtype=torch.long),
            "answer_labels": torch.tensor(1),
            "uncertainty_labels": torch.tensor(0)
        }
        for _ in range(4)
    ]

    dataset = MockDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture(autouse=True)
def set_deterministic():
    """Set deterministic behavior for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")  # Use CPU for testing to avoid GPU dependencies


class MockMMLUDataset:
    """Mock MMLU dataset for testing."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def filter(self, func):
        filtered_data = [item for item in self.data if func(item)]
        return MockMMLUDataset(filtered_data)

    def map(self, func, desc=None):
        mapped_data = [func(item) for item in self.data]
        return MockMMLUDataset(mapped_data)


@pytest.fixture
def mock_dataset_loader(mock_mmlu_data):
    """Create mock dataset loader."""
    def mock_load_dataset(split="test", subjects=None):
        data = mock_mmlu_data.copy()
        if subjects:
            data = [item for item in data if item["subject"] in subjects]
        return MockMMLUDataset(data)

    return mock_load_dataset


@pytest.fixture
def sample_uncertainty_metrics_data():
    """Sample data for testing uncertainty metrics."""
    np.random.seed(42)
    n_samples = 50

    # Create realistic test data
    predictions = np.random.randint(0, 4, n_samples)
    labels = np.random.randint(0, 4, n_samples)

    # Make some predictions correct to test calibration
    correct_mask = np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    predictions[correct_mask] = labels[correct_mask]

    # Generate confidences that are somewhat correlated with correctness
    base_confidences = np.random.uniform(0.2, 0.8, n_samples)
    correct_boost = correct_mask * 0.2
    confidences = np.clip(base_confidences + correct_boost, 0.1, 0.95)

    uncertainty_types = np.random.randint(0, 3, n_samples)
    domains = np.random.choice(["math", "science", "history"], n_samples)

    return {
        "predictions": predictions,
        "labels": labels,
        "confidences": confidences,
        "uncertainty_types": uncertainty_types,
        "domains": list(domains)
    }


@pytest.fixture(scope="session", autouse=True)
def disable_mlflow():
    """Disable MLflow logging during tests."""
    os.environ["MLFLOW_TRACKING_URI"] = ""
    os.environ["DISABLE_MLFLOW"] = "1"


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)

        # Mark slow tests
        if "slow" in item.nodeid.lower() or "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Mark tests with model loading as slow
        if "model" in item.nodeid.lower() and "load" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)