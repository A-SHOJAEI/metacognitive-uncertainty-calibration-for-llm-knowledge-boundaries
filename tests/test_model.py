"""
Tests for the metacognitive uncertainty model.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model import (
    MetacognitiveUncertaintyModel,
    MetacognitiveOutput,
    UncertaintyHead,
    EpistemicUncertaintyEstimator
)


class TestMetacognitiveOutput:
    """Test cases for MetacognitiveOutput dataclass."""

    def test_initialization(self):
        """Test MetacognitiveOutput initialization."""
        answer_logits = torch.randn(4)
        uncertainty_logits = torch.randn(3)

        output = MetacognitiveOutput(
            answer_logits=answer_logits,
            answer_prediction=2,
            answer_confidence=0.8,
            uncertainty_logits=uncertainty_logits,
            uncertainty_type="knowledge_gap",
            uncertainty_confidence=0.6,
            explanation="Test explanation"
        )

        assert torch.equal(output.answer_logits, answer_logits)
        assert output.answer_prediction == 2
        assert output.answer_confidence == 0.8
        assert torch.equal(output.uncertainty_logits, uncertainty_logits)
        assert output.uncertainty_type == "knowledge_gap"
        assert output.uncertainty_confidence == 0.6
        assert output.explanation == "Test explanation"


class TestUncertaintyHead:
    """Test cases for UncertaintyHead module."""

    def test_initialization(self):
        """Test UncertaintyHead initialization."""
        head = UncertaintyHead(
            hidden_size=768,
            num_uncertainty_types=3,
            explanation_vocab_size=1000,
            dropout=0.1
        )

        assert head.hidden_size == 768
        assert head.num_uncertainty_types == 3
        assert isinstance(head.uncertainty_classifier, torch.nn.Sequential)
        assert isinstance(head.confidence_estimator, torch.nn.Sequential)
        assert isinstance(head.explanation_projector, torch.nn.Linear)
        assert isinstance(head.temperature, torch.nn.Parameter)

    def test_forward(self):
        """Test UncertaintyHead forward pass."""
        head = UncertaintyHead(hidden_size=768, num_uncertainty_types=3)

        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = head(hidden_states, attention_mask)

        assert "uncertainty_logits" in outputs
        assert "uncertainty_probs" in outputs
        assert "confidence" in outputs
        assert "explanation_logits" in outputs
        assert "pooled_representation" in outputs

        assert outputs["uncertainty_logits"].shape == (batch_size, 3)
        assert outputs["uncertainty_probs"].shape == (batch_size, 3)
        assert outputs["confidence"].shape == (batch_size,)

    def test_forward_without_attention_mask(self):
        """Test UncertaintyHead forward pass without attention mask."""
        head = UncertaintyHead(hidden_size=768, num_uncertainty_types=3)

        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        outputs = head(hidden_states)

        assert outputs["uncertainty_logits"].shape == (batch_size, 3)
        assert outputs["confidence"].shape == (batch_size,)


class TestEpistemicUncertaintyEstimator:
    """Test cases for EpistemicUncertaintyEstimator."""

    @patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoModelForCausalLM.from_pretrained')
    def test_initialization(self, mock_model):
        """Test EpistemicUncertaintyEstimator initialization."""
        base_model = Mock()
        estimator = EpistemicUncertaintyEstimator(
            base_model=base_model,
            num_samples=5,
            dropout_rate=0.2
        )

        assert estimator.base_model == base_model
        assert estimator.num_samples == 5
        assert isinstance(estimator.dropout, torch.nn.Dropout)

    @patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoModelForCausalLM.from_pretrained')
    def test_forward_training(self, mock_model):
        """Test EpistemicUncertaintyEstimator forward pass during training."""
        base_model = Mock()
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(2, 10, 1000)
        base_model.return_value = mock_outputs

        estimator = EpistemicUncertaintyEstimator(base_model=base_model)

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)

        logits, variance = estimator(input_ids, attention_mask, training=True)

        assert logits.shape == (2, 10, 1000)
        assert variance.shape == (2, 10)

    @patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoModelForCausalLM.from_pretrained')
    def test_forward_inference(self, mock_model):
        """Test EpistemicUncertaintyEstimator forward pass during inference."""
        base_model = Mock()
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(2, 10, 1000)
        base_model.return_value = mock_outputs

        estimator = EpistemicUncertaintyEstimator(base_model=base_model, num_samples=3)
        estimator.base_model = base_model
        estimator.base_model.train = Mock()
        estimator.base_model.eval = Mock()

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)

        logits, variance = estimator(input_ids, attention_mask, training=False)

        # Should call train() and eval() for MC dropout
        estimator.base_model.train.assert_called_once()
        estimator.base_model.eval.assert_called_once()

        assert logits.shape == (2, 10, 1000)
        assert variance.shape == (2, 10)


class TestMetacognitiveUncertaintyModel:
    """Test cases for MetacognitiveUncertaintyModel."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoModel.from_pretrained') as mock_auto_model, \
             patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoModelForCausalLM.from_pretrained') as mock_causal_model, \
             patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.SentenceTransformer') as mock_sentence_transformer:

            # Mock base model
            mock_base_model = Mock()
            mock_base_model.config.hidden_size = 768
            mock_auto_model.return_value = mock_base_model

            # Mock tokenizer
            mock_tok = Mock()
            mock_tok.pad_token = None
            mock_tok.eos_token = "</s>"
            mock_tok.__len__ = Mock(return_value=1000)
            mock_tokenizer.return_value = mock_tok

            # Mock causal model for epistemic estimation
            mock_causal_model.return_value = Mock()

            # Mock sentence transformer
            mock_sentence_transformer.return_value = Mock()

            model = MetacognitiveUncertaintyModel(
                base_model_name="distilbert-base-uncased",
                num_choices=4,
                uncertainty_weight=0.3,
                explanation_weight=0.2,
                use_epistemic_estimation=True,
                freeze_base_model=False
            )

            return model

    def test_initialization(self, mock_model):
        """Test model initialization."""
        assert mock_model.num_choices == 4
        assert mock_model.uncertainty_weight == 0.3
        assert mock_model.explanation_weight == 0.2
        assert mock_model.use_epistemic_estimation is True

        assert hasattr(mock_model, 'base_model')
        assert hasattr(mock_model, 'answer_head')
        assert hasattr(mock_model, 'uncertainty_head')
        assert hasattr(mock_model, 'epistemic_estimator')

    def test_forward_basic(self, mock_model, mock_tokenized_batch):
        """Test basic forward pass."""
        # Mock base model outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(2, 64, 768)
        mock_outputs.hidden_states = None
        mock_model.base_model.return_value = mock_outputs

        # Run forward pass
        outputs = mock_model(
            input_ids=mock_tokenized_batch["input_ids"],
            attention_mask=mock_tokenized_batch["attention_mask"],
            answer_labels=mock_tokenized_batch["answer_labels"],
            uncertainty_labels=mock_tokenized_batch["uncertainty_labels"]
        )

        assert isinstance(outputs, MetacognitiveOutput)
        assert outputs.answer_logits.shape == (2, 4)
        assert outputs.uncertainty_logits.shape == (2, 3)
        assert isinstance(outputs.answer_prediction, list)
        assert isinstance(outputs.uncertainty_type, list)

    def test_generate_explanations(self, mock_model):
        """Test explanation generation."""
        uncertainty_types = ["knowledge_gap", "ambiguous", "reasoning_error"]
        answer_confidences = [0.8, 0.6, 0.9]
        domain_contexts = ["mathematics", "physics", "chemistry"]

        explanations = mock_model._generate_explanations(
            uncertainty_types, answer_confidences, domain_contexts
        )

        assert len(explanations) == 3
        assert "mathematics" in explanations[0]
        assert "0.80" in explanations[0]
        assert all(isinstance(exp, str) for exp in explanations)

    def test_estimate_aleatoric_uncertainty(self, mock_model):
        """Test aleatoric uncertainty estimation."""
        logits = torch.randn(2, 4)
        uncertainty = mock_model._estimate_aleatoric_uncertainty(logits)

        assert isinstance(uncertainty, torch.Tensor)
        assert uncertainty.shape == torch.Size([])  # Scalar
        assert uncertainty.item() >= 0  # Entropy should be non-negative

    def test_compute_loss(self, mock_model):
        """Test loss computation."""
        # Create mock outputs
        outputs = MetacognitiveOutput(
            answer_logits=torch.randn(2, 4),
            answer_prediction=[1, 2],
            answer_confidence=0.8,
            uncertainty_logits=torch.randn(2, 3),
            uncertainty_type=["knowledge_gap", "ambiguous"],
            uncertainty_confidence=0.6,
            explanation="Test explanation",
            aleatoric_uncertainty=0.5
        )

        answer_labels = torch.tensor([1, 2])
        uncertainty_labels = torch.tensor([0, 1])

        losses = mock_model.compute_loss(
            outputs, answer_labels, uncertainty_labels
        )

        assert "answer_loss" in losses
        assert "uncertainty_loss" in losses
        assert "explanation_loss" in losses
        assert "calibration_loss" in losses
        assert "total_loss" in losses

        # Check that total loss combines all components
        expected_total = (
            losses["answer_loss"] +
            mock_model.uncertainty_weight * losses["uncertainty_loss"] +
            mock_model.explanation_weight * losses["explanation_loss"] +
            0.1 * losses["calibration_loss"]
        )
        assert torch.allclose(losses["total_loss"], expected_total)

    def test_predict_with_uncertainty(self, mock_model):
        """Test single prediction with uncertainty."""
        # Mock tokenizer
        mock_model.tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 50)),
            "attention_mask": torch.ones(1, 50)
        }

        # Mock forward pass
        with patch.object(mock_model, 'forward') as mock_forward:
            mock_output = MetacognitiveOutput(
                answer_logits=torch.randn(1, 4),
                answer_prediction=2,
                answer_confidence=0.85,
                uncertainty_logits=torch.randn(1, 3),
                uncertainty_type="knowledge_gap",
                uncertainty_confidence=0.7,
                explanation="I lack knowledge in this domain",
                epistemic_uncertainty=0.3,
                aleatoric_uncertainty=0.4
            )
            mock_forward.return_value = mock_output

            result = mock_model.predict_with_uncertainty(
                question="What is the capital of France?",
                choices=["London", "Berlin", "Paris", "Madrid"],
                domain="geography",
                return_explanations=True
            )

            assert "predicted_answer" in result
            assert "answer_confidence" in result
            assert "uncertainty_type" in result
            assert "epistemic_uncertainty" in result
            assert "aleatoric_uncertainty" in result
            assert "explanation" in result

    def test_calibrate_temperature(self, mock_model):
        """Test temperature calibration."""
        # Create mock validation loader
        mock_batch = {
            "input_ids": torch.randint(0, 1000, (2, 50)),
            "attention_mask": torch.ones(2, 50),
            "answer_labels": torch.tensor([1, 2])
        }

        mock_loader = [mock_batch]

        # Mock forward pass
        with patch.object(mock_model, 'forward') as mock_forward:
            mock_output = MetacognitiveOutput(
                answer_logits=torch.randn(2, 4),
                answer_prediction=[1, 2],
                answer_confidence=0.8,
                uncertainty_logits=torch.randn(2, 3),
                uncertainty_type=["knowledge_gap", "ambiguous"],
                uncertainty_confidence=0.6,
                explanation="Test explanation"
            )
            mock_forward.return_value = mock_output

            initial_temp = mock_model.uncertainty_head.temperature.item()

            # Run calibration
            mock_model.calibrate_temperature(mock_loader)

            # Temperature should be optimized (might be different)
            final_temp = mock_model.uncertainty_head.temperature.item()
            assert isinstance(final_temp, float)

    def test_get_model_info(self, mock_model):
        """Test model info retrieval."""
        info = mock_model.get_model_info()

        required_keys = [
            "base_model", "total_parameters", "trainable_parameters",
            "uncertainty_types", "num_choices", "use_epistemic_estimation",
            "uncertainty_weight", "explanation_weight"
        ]

        for key in required_keys:
            assert key in info

        assert info["uncertainty_types"] == mock_model.UNCERTAINTY_TYPES
        assert info["num_choices"] == 4
        assert isinstance(info["total_parameters"], int)
        assert isinstance(info["trainable_parameters"], int)

    def test_uncertainty_types(self, mock_model):
        """Test uncertainty type constants."""
        assert len(mock_model.UNCERTAINTY_TYPES) == 3
        assert "knowledge_gap" in mock_model.UNCERTAINTY_TYPES
        assert "ambiguous" in mock_model.UNCERTAINTY_TYPES
        assert "reasoning_error" in mock_model.UNCERTAINTY_TYPES

    def test_device_handling(self, mock_model):
        """Test model device handling."""
        # Test moving to different devices
        device = torch.device("cpu")
        mock_model.to(device)

        # Test that model parameters are on correct device
        for param in mock_model.parameters():
            assert param.device == device

    @pytest.mark.parametrize("freeze_base_model", [True, False])
    def test_freeze_base_model(self, freeze_base_model):
        """Test base model freezing functionality."""
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoModel.from_pretrained') as mock_auto_model, \
             patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.SentenceTransformer') as mock_sentence_transformer:

            # Mock setup
            mock_base_model = Mock()
            mock_base_model.config.hidden_size = 768
            mock_base_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
            mock_auto_model.return_value = mock_base_model

            mock_tok = Mock()
            mock_tok.pad_token = None
            mock_tok.eos_token = "</s>"
            mock_tok.__len__ = Mock(return_value=1000)
            mock_tokenizer.return_value = mock_tok

            mock_sentence_transformer.return_value = Mock()

            model = MetacognitiveUncertaintyModel(
                base_model_name="distilbert-base-uncased",
                freeze_base_model=freeze_base_model,
                use_epistemic_estimation=False
            )

            if freeze_base_model:
                # Check that parameters were frozen
                for param in mock_base_model.parameters():
                    assert not param.requires_grad
            else:
                # Check that parameters remain trainable
                for param in mock_base_model.parameters():
                    assert param.requires_grad

    def test_forward_with_domain_context(self, mock_model, mock_tokenized_batch):
        """Test forward pass with domain context."""
        # Mock base model outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(2, 64, 768)
        mock_model.base_model.return_value = mock_outputs

        # Run forward pass with domain context
        outputs = mock_model(
            input_ids=mock_tokenized_batch["input_ids"],
            attention_mask=mock_tokenized_batch["attention_mask"],
            domain_context=["mathematics", "physics"]
        )

        assert isinstance(outputs, MetacognitiveOutput)
        assert outputs.answer_logits.shape == (2, 4)

    def test_single_sample_forward(self, mock_model):
        """Test forward pass with single sample."""
        # Mock base model outputs for single sample
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 64, 768)
        mock_model.base_model.return_value = mock_outputs

        input_ids = torch.randint(0, 1000, (1, 64))
        attention_mask = torch.ones(1, 64)
        answer_labels = torch.tensor([2])

        outputs = mock_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            answer_labels=answer_labels
        )

        assert isinstance(outputs, MetacognitiveOutput)
        assert isinstance(outputs.answer_prediction, int)  # Single sample should return int
        assert isinstance(outputs.uncertainty_type, str)  # Single sample should return str