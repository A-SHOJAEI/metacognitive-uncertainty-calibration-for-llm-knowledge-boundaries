"""
Tests for error handling and input validation in the metacognitive uncertainty model.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model import (
    MetacognitiveUncertaintyModel,
    MetacognitiveOutput,
    UncertaintyHead,
    ModelInputError,
    ModelForwardError,
    UncertaintyEstimationError
)


class TestModelErrorHandling:
    """Test error handling and edge cases in the model."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoModel.from_pretrained') as mock_auto_model, \
             patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.models.model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
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

            mock_sentence_transformer.return_value = Mock()

            model = MetacognitiveUncertaintyModel(
                base_model_name="distilbert-base-uncased",
                num_choices=4,
                use_epistemic_estimation=False
            )

            return model

    def test_invalid_input_ids_none(self, mock_model):
        """Test error handling when input_ids is None."""
        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=None)
        assert "input_ids must be a torch.Tensor" in str(exc_info.value)

    def test_invalid_input_ids_wrong_type(self, mock_model):
        """Test error handling when input_ids is wrong type."""
        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids="not a tensor")
        assert "input_ids must be a torch.Tensor" in str(exc_info.value)

    def test_invalid_input_ids_wrong_dims(self, mock_model):
        """Test error handling when input_ids has wrong dimensions."""
        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=torch.randn(10))  # 1D instead of 2D
        assert "input_ids must be 2D tensor" in str(exc_info.value)

    def test_invalid_input_ids_empty_batch(self, mock_model):
        """Test error handling when batch size is 0."""
        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=torch.randn(0, 10))
        assert "invalid dimensions" in str(exc_info.value)

    def test_invalid_input_ids_empty_sequence(self, mock_model):
        """Test error handling when sequence length is 0."""
        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=torch.randn(2, 0))
        assert "invalid dimensions" in str(exc_info.value)

    def test_attention_mask_shape_mismatch(self, mock_model):
        """Test error handling when attention_mask doesn't match input_ids shape."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 15)  # Wrong shape

        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=input_ids, attention_mask=attention_mask)
        assert "attention_mask shape" in str(exc_info.value)

    def test_invalid_answer_labels_type(self, mock_model):
        """Test error handling for invalid answer_labels type."""
        input_ids = torch.randint(0, 1000, (2, 10))

        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=input_ids, answer_labels="invalid")
        assert "answer_labels must be a torch.Tensor" in str(exc_info.value)

    def test_invalid_answer_labels_shape(self, mock_model):
        """Test error handling for invalid answer_labels shape."""
        input_ids = torch.randint(0, 1000, (2, 10))
        answer_labels = torch.tensor([[1, 2]])  # Wrong shape

        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=input_ids, answer_labels=answer_labels)
        assert "answer_labels must be 1D tensor" in str(exc_info.value)

    def test_invalid_answer_labels_values(self, mock_model):
        """Test error handling for out-of-range answer_labels."""
        input_ids = torch.randint(0, 1000, (2, 10))
        answer_labels = torch.tensor([1, 5])  # Out of range (model has 4 choices)

        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=input_ids, answer_labels=answer_labels)
        assert "answer_labels must be in range" in str(exc_info.value)

    def test_invalid_uncertainty_labels_values(self, mock_model):
        """Test error handling for out-of-range uncertainty_labels."""
        input_ids = torch.randint(0, 1000, (2, 10))
        uncertainty_labels = torch.tensor([0, 5])  # Out of range (3 uncertainty types)

        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=input_ids, uncertainty_labels=uncertainty_labels)
        assert "uncertainty_labels must be in range" in str(exc_info.value)

    def test_invalid_domain_context_type(self, mock_model):
        """Test error handling for invalid domain_context type."""
        input_ids = torch.randint(0, 1000, (2, 10))

        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=input_ids, domain_context="not a list")
        assert "domain_context must be a list" in str(exc_info.value)

    def test_invalid_domain_context_length(self, mock_model):
        """Test error handling for domain_context length mismatch."""
        input_ids = torch.randint(0, 1000, (2, 10))
        domain_context = ["math"]  # Wrong length

        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=input_ids, domain_context=domain_context)
        assert "domain_context length" in str(exc_info.value)

    def test_invalid_domain_context_element_type(self, mock_model):
        """Test error handling for non-string elements in domain_context."""
        input_ids = torch.randint(0, 1000, (2, 10))
        domain_context = ["math", 123]  # Non-string element

        with pytest.raises(ModelInputError) as exc_info:
            mock_model.forward(input_ids=input_ids, domain_context=domain_context)
        assert "must be a string" in str(exc_info.value)

    def test_nan_in_hidden_states(self, mock_model):
        """Test handling of NaN values in hidden states."""
        input_ids = torch.randint(0, 1000, (2, 10))

        # Mock base model to return NaN values
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.full((2, 10, 768), float('nan'))
        mock_model.base_model.return_value = mock_outputs

        with pytest.raises(ModelForwardError) as exc_info:
            mock_model.forward(input_ids=input_ids)
        assert "NaN or Inf values" in str(exc_info.value)

    def test_base_model_missing_outputs(self, mock_model):
        """Test handling when base model doesn't return expected outputs."""
        input_ids = torch.randint(0, 1000, (2, 10))

        # Mock base model to return invalid outputs
        mock_outputs = Mock()
        del mock_outputs.last_hidden_state  # Missing attribute
        mock_model.base_model.return_value = mock_outputs

        with pytest.raises(ModelForwardError) as exc_info:
            mock_model.forward(input_ids=input_ids)
        assert "missing last_hidden_state" in str(exc_info.value)

    def test_all_zero_attention_mask(self, mock_model):
        """Test handling of all-zero attention masks."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.zeros(2, 10)  # All zeros

        # Mock base model outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.base_model.return_value = mock_outputs

        # Should not raise an error but log a warning
        outputs = mock_model.forward(input_ids=input_ids, attention_mask=attention_mask)
        assert isinstance(outputs, MetacognitiveOutput)

    def test_device_mismatch_handling(self, mock_model):
        """Test automatic device movement for mismatched tensors."""
        # Create inputs on CPU while model parameters might be on different device
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)

        # Mock base model outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.base_model.return_value = mock_outputs

        # Should automatically move tensors to model device
        outputs = mock_model.forward(input_ids=input_ids, attention_mask=attention_mask)
        assert isinstance(outputs, MetacognitiveOutput)

    def test_uncertainty_head_error_propagation(self, mock_model):
        """Test error handling when uncertainty head fails."""
        input_ids = torch.randint(0, 1000, (2, 10))

        # Mock base model outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.base_model.return_value = mock_outputs

        # Mock uncertainty head to raise an error
        with patch.object(mock_model.uncertainty_head, 'forward') as mock_uncertainty:
            mock_uncertainty.side_effect = RuntimeError("Uncertainty head failed")

            with pytest.raises(UncertaintyEstimationError) as exc_info:
                mock_model.forward(input_ids=input_ids)
            assert "Uncertainty estimation failed" in str(exc_info.value)

    def test_invalid_uncertainty_prediction_index(self, mock_model):
        """Test handling of invalid uncertainty prediction indices."""
        input_ids = torch.randint(0, 1000, (2, 10))

        # Mock base model outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.base_model.return_value = mock_outputs

        # Mock uncertainty head to return out-of-bounds predictions
        mock_uncertainty_outputs = {
            "uncertainty_logits": torch.tensor([[10, -5, -5], [-5, -5, 10]]),  # Will give indices 0, 2
            "uncertainty_probs": torch.softmax(torch.tensor([[10, -5, -5], [-5, -5, 10]]), dim=-1),
            "confidence": torch.tensor([0.9, 0.8]),
            "explanation_logits": torch.randn(2, 100),
            "pooled_representation": torch.randn(2, 768)
        }

        with patch.object(mock_model.uncertainty_head, 'forward') as mock_uncertainty:
            mock_uncertainty.return_value = mock_uncertainty_outputs

            # Should handle gracefully without error
            outputs = mock_model.forward(input_ids=input_ids)
            assert isinstance(outputs, MetacognitiveOutput)
            assert len(outputs.uncertainty_type) == 2

    def test_explanation_generation_failure_fallback(self, mock_model):
        """Test fallback when explanation generation fails."""
        input_ids = torch.randint(0, 1000, (2, 10))

        # Mock base model outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.base_model.return_value = mock_outputs

        # Mock _generate_explanations to fail
        with patch.object(mock_model, '_generate_explanations') as mock_explain:
            mock_explain.side_effect = RuntimeError("Explanation failed")

            # Should use fallback explanations
            outputs = mock_model.forward(input_ids=input_ids)
            assert isinstance(outputs, MetacognitiveOutput)
            assert outputs.explanation == ["Unable to generate explanation", "Unable to generate explanation"]

    def test_epistemic_uncertainty_nan_handling(self, mock_model):
        """Test handling of NaN values in epistemic uncertainty."""
        # Enable epistemic estimation
        mock_model.use_epistemic_estimation = True
        mock_model.training = False

        input_ids = torch.randint(0, 1000, (2, 10))

        # Mock base model outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.base_model.return_value = mock_outputs

        # Mock epistemic estimator to return NaN
        with patch.object(mock_model.epistemic_estimator, 'forward') as mock_epistemic:
            mock_epistemic.return_value = (
                torch.randn(2, 10, 1000),
                torch.full((2, 10), float('nan'))  # NaN variance
            )

            outputs = mock_model.forward(input_ids=input_ids)
            assert isinstance(outputs, MetacognitiveOutput)
            assert outputs.epistemic_uncertainty is None  # Should be set to None due to NaN

    def test_confident_value_conversion_edge_cases(self, mock_model):
        """Test handling of edge cases in confidence value conversion."""
        input_ids = torch.randint(0, 1000, (2, 10))

        # Mock base model outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.base_model.return_value = mock_outputs

        # Test with extreme confidence values
        extreme_confidence = torch.tensor([float('inf'), float('-inf')])

        with patch.object(mock_model, 'uncertainty_head') as mock_uncertainty_head:
            mock_uncertainty_outputs = {
                "uncertainty_logits": torch.randn(2, 3),
                "uncertainty_probs": torch.softmax(torch.randn(2, 3), dim=-1),
                "confidence": extreme_confidence,  # Contains inf values
                "explanation_logits": torch.randn(2, 100),
                "pooled_representation": torch.randn(2, 768)
            }
            mock_uncertainty_head.forward.return_value = mock_uncertainty_outputs

            outputs = mock_model.forward(input_ids=input_ids)
            assert isinstance(outputs, MetacognitiveOutput)
            assert isinstance(outputs.uncertainty_confidence, float)
            # Should fallback to 0.0 for invalid values
            assert outputs.uncertainty_confidence == 0.0