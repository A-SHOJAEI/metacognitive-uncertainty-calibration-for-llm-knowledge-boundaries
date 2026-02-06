"""
Metacognitive Uncertainty Calibration for LLM Knowledge Boundaries.

A novel framework that teaches language models to recognize and articulate their own
knowledge boundaries by training uncertainty estimators on MMLU's multi-domain structure.
"""

__version__ = "0.1.0"
__author__ = "A-SHOJAEI"
__email__ = ""

from .models.model import MetacognitiveUncertaintyModel
from .data.loader import MMLUDataLoader
from .training.trainer import MetacognitiveTrainer
from .evaluation.metrics import UncertaintyMetrics

__all__ = [
    "MetacognitiveUncertaintyModel",
    "MMLUDataLoader",
    "MetacognitiveTrainer",
    "UncertaintyMetrics"
]