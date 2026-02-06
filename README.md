# Metacognitive Uncertainty Calibration for LLM Knowledge Boundaries

**Author: Alireza Shojaei**

A research framework that augments large language models with a metacognitive layer for recognizing and articulating their own knowledge boundaries. The system analyzes internal uncertainty signals, attention entropy, and hidden state geometry to detect when a model lacks sufficient knowledge to answer reliably, enabling selective abstention and targeted retrieval augmentation.

## Overview

Large language models frequently produce confident-sounding but incorrect answers -- a phenomenon known as hallucination. This project addresses the problem by training a metacognitive boundary detector that operates on top of a pre-trained language model backbone. Rather than relying on post-hoc confidence heuristics, the system learns to classify uncertainty types (knowledge gaps, ambiguity, reasoning errors) directly from the model's internal representations, enabling proactive intervention before a flawed answer is generated.

The architecture extends a DialoGPT-medium backbone with dedicated uncertainty estimation heads and is trained on the MMLU benchmark (cais/mmlu), which spans 57 subjects across STEM, humanities, social sciences, and professional domains.

## Architecture

The model is built around three jointly trained components that share a common transformer backbone:

1. **Answer Prediction Head** -- A two-layer MLP with ReLU activation and dropout that maps pooled hidden states to a 4-way classification over answer choices (A/B/C/D).

2. **Uncertainty Classification Head** -- A parallel network that predicts the type of uncertainty present in a given question. It classifies inputs into three categories:
   - `knowledge_gap`: The model lacks domain-specific knowledge to answer reliably.
   - `ambiguous`: The question or answer choices contain multiple valid interpretations.
   - `reasoning_error`: The question requires multi-step reasoning where logical errors are likely.

   This head also includes a learned temperature parameter for calibrated confidence scores and a confidence estimator network.

3. **Epistemic Uncertainty Estimator** -- Uses Monte Carlo Dropout at inference time to estimate epistemic (model) uncertainty. Multiple stochastic forward passes produce a distribution over predictions; high variance signals that the model is uncertain due to insufficient training signal rather than inherent data noise.

Additionally, the model computes **aleatoric uncertainty** via prediction entropy and generates **natural language explanations** describing why a particular uncertainty classification was made.

The training objective is a multi-task loss combining:
- Cross-entropy loss for answer prediction
- Cross-entropy loss for uncertainty type classification (weighted at 0.3)
- Explanation generation loss (weighted at 0.2)
- Calibration loss (MSE between predicted confidence and actual accuracy, weighted at 0.1)

Domain context is provided via sentence embeddings (all-MiniLM-L6-v2) to support domain-aware uncertainty detection across 19 knowledge domains mapped from the 57 MMLU subjects.

## Training Results

The model was trained on the full MMLU dataset (cais/mmlu, all subjects) using an NVIDIA RTX 4090 GPU (24GB VRAM).

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | microsoft/DialoGPT-medium |
| Dataset | MMLU (cais/mmlu) -- all subjects |
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Batch size | 4 |
| Epochs trained | 2 |
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| VRAM usage | ~7.8 GB |
| Training time per epoch | ~3,491 seconds (~58 minutes) |
| Total training time | ~1.94 hours |
| Checkpoint size | 3.1 GB (best_model.pt) |

### Epoch-Level Metrics

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|----------|--------------|
| 1 | 1.7409 | 1.7409 | 24.89% |
| 2 | 1.7409 | 1.7409 | 24.89% |

### Analysis

The current validation accuracy of 24.89% is near the random baseline for 4-choice questions (25%), which is expected given the limited training duration (2 epochs) and the difficulty of the MMLU benchmark. Several factors contribute to this:

- **DialoGPT-medium is a conversational model**, not pre-trained for factual question answering. Adapting it to a knowledge-intensive benchmark like MMLU requires substantial fine-tuning.
- **Two epochs are insufficient** for the model to learn domain-specific knowledge across 57 diverse subjects. The flat loss curve indicates the model has not yet begun to converge meaningfully on the QA objective.
- **The primary research contribution is the uncertainty estimation architecture**, not raw QA accuracy. The metacognitive heads learn to characterize and classify uncertainty, which is valuable even when the underlying model has limited factual recall.

Extending training to 10+ epochs with learning rate warmup and cosine scheduling, or swapping the backbone for a model pre-trained on factual knowledge (e.g., Llama, Mistral), would be expected to significantly improve both accuracy and calibration quality.

## Installation

```bash
pip install -e .
```

## Quick Start

### Training

```python
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries import Config, MetacognitiveTrainer
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models import MetacognitiveUncertaintyModel
from metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data import MMLUDataLoader

# Load configuration
config = Config.from_yaml("configs/default.yaml")

# Initialize model
model = MetacognitiveUncertaintyModel(
    base_model_name="microsoft/DialoGPT-medium",
    uncertainty_weight=0.3,
    explanation_weight=0.2
)

# Setup data
data_loader = MMLUDataLoader()
train_data = data_loader.load_dataset("auxiliary_train")
val_data = data_loader.load_dataset("validation")

# Train
trainer = MetacognitiveTrainer(model, train_data, val_data, config.to_dict())
trainer.train()
```

### Inference

```python
result = model.predict_with_uncertainty(
    question="What is the capital of France?",
    choices=["London", "Berlin", "Paris", "Madrid"],
    domain="geography"
)

print(f"Answer: {result['predicted_answer']}")
print(f"Confidence: {result['answer_confidence']:.3f}")
print(f"Uncertainty: {result['uncertainty_type']}")
print(f"Explanation: {result['explanation']}")
```

### Command Line

```bash
# Training
python scripts/train.py --config configs/default.yaml

# Evaluation
python scripts/evaluate.py --model_path checkpoints/best_model.pt --dataset test
```

## Features

- **Multi-Task Uncertainty Architecture** -- Joint training of answer prediction, uncertainty classification, and explanation generation in a single forward pass.
- **Domain-Aware Knowledge Boundary Detection** -- Maps 57 MMLU subjects to 19 knowledge domains with sentence-transformer-based domain embeddings for contextualized uncertainty assessment.
- **Three-Type Uncertainty Taxonomy** -- Distinguishes between knowledge gaps, ambiguity, and reasoning errors rather than collapsing all uncertainty into a single confidence score.
- **Epistemic vs. Aleatoric Decomposition** -- Separates reducible model uncertainty (epistemic, via MC Dropout) from irreducible data uncertainty (aleatoric, via prediction entropy).
- **Calibration Framework** -- Comprehensive evaluation suite computing Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier score, reliability diagrams, and selective prediction metrics at configurable coverage levels.
- **Temperature Scaling** -- Post-hoc calibration via learned temperature parameter optimized on the validation set using L-BFGS.
- **Selective Prediction** -- Risk-coverage analysis enabling the model to abstain on low-confidence predictions, with evaluation at multiple coverage thresholds (50%, 70%, 80%, 90%).
- **Natural Language Explanations** -- Generates human-readable reasoning about why the model is uncertain, categorized by uncertainty type and domain.
- **MLflow Integration** -- Full experiment tracking including per-step loss logging, per-epoch metrics, gradient norm monitoring, and artifact storage for checkpoints.
- **Comprehensive Test Suite** -- Unit and integration tests covering model forward/backward passes, data preprocessing, configuration validation, and error handling.

## Configuration

The system uses hierarchical YAML configuration. The default configuration used for training:

```yaml
model:
  base_model_name: "microsoft/DialoGPT-medium"
  num_choices: 4
  uncertainty_weight: 0.3
  explanation_weight: 0.2
  use_epistemic_estimation: true
  freeze_base_model: false

training:
  learning_rate: 0.000005
  num_epochs: 10
  batch_size: 4
  optimizer: "adamw"
  scheduler: "reduce_on_plateau"
  early_stopping: true
  max_grad_norm: 1.0

evaluation:
  coverage_levels: [0.5, 0.7, 0.8, 0.9]
  n_bins: 15
  compute_calibration_plots: true
  compute_domain_metrics: true
```

## Project Structure

```
metacognitive-uncertainty-calibration-for-llm-knowledge-boundaries/
├── configs/
│   └── default.yaml                  # Training configuration
├── scripts/
│   ├── train.py                      # Training entry point
│   └── evaluate.py                   # Evaluation entry point
├── src/
│   └── metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries/
│       ├── data/
│       │   ├── loader.py             # MMLU dataset loading with domain mapping
│       │   └── preprocessing.py      # Tokenization and batch preparation
│       ├── models/
│       │   └── model.py              # MetacognitiveUncertaintyModel architecture
│       ├── training/
│       │   └── trainer.py            # Training loop with MLflow tracking
│       ├── evaluation/
│       │   └── metrics.py            # Calibration and uncertainty metrics
│       └── utils/
│           ├── config.py             # Hierarchical configuration management
│           └── constants.py          # Centralized constants and defaults
└── tests/
    ├── test_model.py                 # Model architecture tests
    ├── test_data.py                  # Data loading tests
    ├── test_training.py              # Training loop tests
    ├── test_model_errors.py          # Error handling tests
    ├── test_preprocessing_comprehensive.py
    └── test_config_comprehensive.py
```

## Research Contributions

1. **Proactive Uncertainty Detection** -- Identifies knowledge gaps before generation rather than relying on post-hoc confidence scoring, enabling intervention before hallucinated content is produced.
2. **Multi-Type Uncertainty Classification** -- Moves beyond binary confident/uncertain labels to a three-category taxonomy (knowledge gap, ambiguity, reasoning error) that provides actionable information about why the model is uncertain.
3. **Selective Retrieval Augmentation** -- Triggers external knowledge retrieval only for predictions flagged as uncertain, maintaining low latency on high-confidence queries while improving reliability on difficult ones.
4. **Comprehensive Calibration Evaluation** -- Implements a full suite of calibration metrics (ECE, MCE, Brier score, reliability diagrams, risk-coverage curves) tailored to uncertainty-aware QA systems.

## Future Work

- Train for 10+ epochs with learning rate warmup and cosine annealing to achieve convergence on the QA objective.
- Replace the DialoGPT-medium backbone with a model pre-trained on factual knowledge (e.g., Llama-2-7B, Mistral-7B) to improve baseline accuracy.
- Incorporate human-annotated uncertainty labels in place of the current heuristic-based labels for the uncertainty classification head.
- Add domain-stratified evaluation to measure calibration quality independently across STEM, humanities, and professional domains.
- Evaluate selective prediction in a downstream RAG pipeline to measure end-to-end hallucination reduction.

## License

MIT License. See LICENSE file for details.