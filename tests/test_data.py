"""
Tests for data loading and preprocessing functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.loader import MMLUDataLoader, MMLUExample
from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing import MMLUPreprocessor


class TestMMLUDataLoader:
    """Test cases for MMLUDataLoader."""

    def test_init(self):
        """Test MMLUDataLoader initialization."""
        loader = MMLUDataLoader(
            cache_dir="/tmp/test",
            seed=42,
            uncertainty_augmentation=True
        )

        assert loader.cache_dir == "/tmp/test"
        assert loader.seed == 42
        assert loader.uncertainty_augmentation is True

    def test_domain_mapping(self):
        """Test domain mapping functionality."""
        loader = MMLUDataLoader()

        # Test known subjects
        assert loader.DOMAIN_MAPPING["abstract_algebra"] == "mathematics"
        assert loader.DOMAIN_MAPPING["college_biology"] == "biology"
        assert loader.DOMAIN_MAPPING["philosophy"] == "philosophy"

        # Test domain mapping completeness
        expected_domains = {
            "mathematics", "biology", "chemistry", "physics", "computer_science",
            "medicine", "engineering", "economics", "history", "geography",
            "politics", "psychology", "philosophy", "ethics", "law", "business",
            "religion", "social_science", "general"
        }

        actual_domains = set(loader.DOMAIN_MAPPING.values())
        assert actual_domains.issubset(expected_domains)

    def test_add_metadata(self):
        """Test metadata addition to examples."""
        loader = MMLUDataLoader(uncertainty_augmentation=True)

        example = {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
            "subject": "elementary_mathematics"
        }

        result = loader._add_metadata(example)

        assert "domain" in result
        assert result["domain"] == "mathematics"
        assert "uncertainty_type" in result
        assert result["uncertainty_type"] in ["knowledge_gap", "ambiguous", "reasoning_error"]

    def test_predict_uncertainty_type(self):
        """Test uncertainty type prediction heuristics."""
        loader = MMLUDataLoader()

        # Test ambiguous question
        example_ambiguous = {
            "question": "This is an unclear and ambiguous question",
            "choices": ["A", "B", "C", "D"],
            "subject": "test"
        }
        uncertainty_type = loader._predict_uncertainty_type(example_ambiguous)
        assert uncertainty_type == "ambiguous"

        # Test reasoning question
        example_reasoning = {
            "question": "Calculate the derivative of x^2",
            "choices": ["x", "2x", "x^2", "2"],
            "subject": "test"
        }
        uncertainty_type = loader._predict_uncertainty_type(example_reasoning)
        assert uncertainty_type == "reasoning_error"

        # Test knowledge gap question
        example_knowledge = {
            "question": "What is this fact?",
            "choices": ["A", "B", "C", "D"],
            "subject": "miscellaneous"
        }
        uncertainty_type = loader._predict_uncertainty_type(example_knowledge)
        assert uncertainty_type == "knowledge_gap"

    @patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.loader.load_dataset')
    def test_load_dataset_success(self, mock_load_dataset):
        """Test successful dataset loading."""
        # Mock dataset
        mock_data = [
            {
                "question": "Test question?",
                "choices": ["A", "B", "C", "D"],
                "answer": 1,
                "subject": "test_subject"
            }
        ]

        mock_dataset = Mock()
        mock_dataset.filter.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.__len__ = Mock(return_value=1)

        mock_load_dataset.return_value = mock_dataset

        loader = MMLUDataLoader()
        result = loader.load_dataset(split="test")

        mock_load_dataset.assert_called_once()
        assert result == mock_dataset

    def test_get_domain_splits(self, mock_mmlu_data):
        """Test domain splitting functionality."""
        from tests.conftest import MockMMLUDataset

        dataset = MockMMLUDataset(mock_mmlu_data)
        loader = MMLUDataLoader()

        domain_splits = loader.get_domain_splits(dataset)

        # Check that domains are correctly split
        assert "geography" in domain_splits
        assert "mathematics" in domain_splits
        assert "chemistry" in domain_splits

    def test_create_dataloader(self, mock_mmlu_data):
        """Test DataLoader creation."""
        from tests.conftest import MockMMLUDataset

        dataset = MockMMLUDataset(mock_mmlu_data)
        loader = MMLUDataLoader()

        dataloader = loader.create_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        assert dataloader.batch_size == 2
        assert dataloader.shuffle is False
        assert dataloader.num_workers == 0

        # Test collate function
        batch = list(dataloader)[0]
        assert "answer" in batch
        assert isinstance(batch["answer"], torch.Tensor)

    def test_get_uncertainty_distribution(self, mock_mmlu_data):
        """Test uncertainty distribution calculation."""
        from tests.conftest import MockMMLUDataset

        # Add uncertainty types to mock data
        for item in mock_mmlu_data:
            item["uncertainty_type"] = "knowledge_gap"
        mock_mmlu_data[0]["uncertainty_type"] = "ambiguous"

        dataset = MockMMLUDataset(mock_mmlu_data)
        loader = MMLUDataLoader(uncertainty_augmentation=True)

        distribution = loader.get_uncertainty_distribution(dataset)

        assert "knowledge_gap" in distribution
        assert "ambiguous" in distribution
        assert abs(sum(distribution.values()) - 1.0) < 1e-10  # Should sum to 1

    def test_sample_few_shot_examples(self, mock_mmlu_data):
        """Test few-shot example sampling."""
        from tests.conftest import MockMMLUDataset

        dataset = MockMMLUDataset(mock_mmlu_data)
        loader = MMLUDataLoader()

        # Test sampling
        examples = loader.sample_few_shot_examples(
            dataset,
            n_examples=2,
            per_domain=False
        )

        assert len(examples) == 2
        assert all(isinstance(ex, MMLUExample) for ex in examples)

        # Test per-domain sampling
        examples_per_domain = loader.sample_few_shot_examples(
            dataset,
            n_examples=3,
            per_domain=True
        )

        assert len(examples_per_domain) <= 3


class TestMMLUPreprocessor:
    """Test cases for MMLUPreprocessor."""

    def test_init(self):
        """Test MMLUPreprocessor initialization."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased",
            max_length=256,
            include_uncertainty_prompt=True
        )

        assert preprocessor.max_length == 256
        assert preprocessor.include_uncertainty_prompt is True
        assert preprocessor.tokenizer is not None

    def test_create_uncertainty_prompt(self):
        """Test uncertainty prompt creation."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased"
        )

        question = "What is the capital of France?"
        choices = ["London", "Berlin", "Paris", "Madrid"]

        prompt = preprocessor.create_uncertainty_prompt(
            question, choices, include_explanation=True
        )

        # Check prompt structure
        assert question in prompt
        assert "(A) London" in prompt
        assert "(B) Berlin" in prompt
        assert "(C) Paris" in prompt
        assert "(D) Madrid" in prompt
        assert "confidence" in prompt.lower()
        assert "answer:" in prompt.lower()
        assert "explanation:" in prompt.lower()

    def test_create_few_shot_prompt(self):
        """Test few-shot prompt creation."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased"
        )

        question = "What is 2 + 2?"
        choices = ["3", "4", "5", "6"]
        examples = [
            {
                "question": "What is 1 + 1?",
                "choices": ["1", "2", "3", "4"],
                "answer": 1,
                "uncertainty_type": "knowledge_gap"
            }
        ]

        prompt = preprocessor.create_few_shot_prompt(question, choices, examples)

        assert "Example 1:" in prompt
        assert "What is 1 + 1?" in prompt
        assert question in prompt

    def test_get_confidence_label(self):
        """Test confidence label mapping."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased"
        )

        assert preprocessor._get_confidence_label("knowledge_gap") == "UNCERTAIN"
        assert preprocessor._get_confidence_label("ambiguous") == "VERY_UNCERTAIN"
        assert preprocessor._get_confidence_label("reasoning_error") == "UNCERTAIN"
        assert preprocessor._get_confidence_label(None) == "CONFIDENT"

    def test_get_uncertainty_explanation(self):
        """Test uncertainty explanation generation."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased"
        )

        explanation = preprocessor._get_uncertainty_explanation("knowledge_gap")
        assert "KNOWLEDGE_GAP" in explanation

        explanation = preprocessor._get_uncertainty_explanation("ambiguous")
        assert "AMBIGUOUS" in explanation

        explanation = preprocessor._get_uncertainty_explanation("reasoning_error")
        assert "REASONING_ERROR" in explanation

        explanation = preprocessor._get_uncertainty_explanation(None)
        assert "confident" in explanation.lower()

    def test_tokenize_example(self):
        """Test example tokenization."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased",
            max_length=128
        )

        prompt = "What is the capital of France? (A) London (B) Paris"
        result = preprocessor.tokenize_example(
            prompt,
            answer=1,
            uncertainty_type="knowledge_gap"
        )

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "answer_labels" in result
        assert "uncertainty_labels" in result

        assert result["input_ids"].shape == (128,)
        assert result["attention_mask"].shape == (128,)
        assert result["answer_labels"].item() == 1
        assert result["uncertainty_labels"].item() == 0  # knowledge_gap maps to 0

    def test_batch_tokenize(self):
        """Test batch tokenization."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased",
            max_length=128
        )

        examples = [
            {
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
                "uncertainty_type": "reasoning_error"
            },
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": 2,
                "uncertainty_type": "knowledge_gap"
            }
        ]

        result = preprocessor.batch_tokenize(examples)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "answer_labels" in result
        assert "uncertainty_labels" in result

        assert result["input_ids"].shape == (2, 128)
        assert result["attention_mask"].shape == (2, 128)
        assert result["answer_labels"].shape == (2,)
        assert result["uncertainty_labels"].shape == (2,)

    def test_extract_answer_from_response(self):
        """Test answer extraction from model response."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased"
        )

        response = "The answer is B. CONFIDENT. I am confident in this answer."
        answer_idx, confidence, explanation = preprocessor.extract_answer_from_response(response)

        assert answer_idx == 1  # B = 1
        assert confidence == "CONFIDENT"

        # Test with uncertainty
        response = "A VERY_UNCERTAIN KNOWLEDGE_GAP: I lack sufficient knowledge in this domain."
        answer_idx, confidence, explanation = preprocessor.extract_answer_from_response(response)

        assert answer_idx == 0  # A = 0
        assert confidence == "VERY_UNCERTAIN"
        assert "I lack sufficient knowledge" in explanation

    def test_create_calibration_prompt(self):
        """Test calibration prompt creation."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased"
        )

        question = "What is 2 + 2?"
        choices = ["3", "4", "5", "6"]
        model_answer = "B"
        model_confidence = 0.85

        prompt = preprocessor.create_calibration_prompt(
            question, choices, model_answer, model_confidence
        )

        assert question in prompt
        assert model_answer in prompt
        assert "0.85" in prompt
        assert "calibrated" in prompt.lower()

    def test_augment_with_retrieval(self):
        """Test retrieval augmentation."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased"
        )

        question = "What is photosynthesis?"
        choices = ["A", "B", "C", "D"]
        contexts = [
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            "Plants use chlorophyll to capture light energy.",
            "The process produces oxygen as a byproduct."
        ]

        prompt = preprocessor.augment_with_retrieval(
            question, choices, contexts, max_contexts=2
        )

        assert question in prompt
        assert "Context 1:" in prompt
        assert "Context 2:" in prompt
        assert "Context 3:" not in prompt  # Should be limited to 2
        assert contexts[0] in prompt
        assert contexts[1] in prompt

    @pytest.mark.parametrize("uncertainty_type,expected_confidence", [
        ("knowledge_gap", "UNCERTAIN"),
        ("ambiguous", "VERY_UNCERTAIN"),
        ("reasoning_error", "UNCERTAIN"),
        (None, "CONFIDENT")
    ])
    def test_confidence_mapping(self, uncertainty_type, expected_confidence):
        """Test confidence level mapping for different uncertainty types."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased"
        )

        confidence = preprocessor._get_confidence_label(uncertainty_type)
        assert confidence == expected_confidence

    def test_tokenizer_special_tokens(self):
        """Test that special tokens are properly added."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased"
        )

        # Check that uncertainty-related tokens were added
        vocab = preprocessor.tokenizer.get_vocab()
        special_tokens = [
            "[UNCERTAIN]", "[CONFIDENT]", "[KNOWLEDGE_GAP]",
            "[AMBIGUOUS]", "[REASONING_ERROR]", "[EXPLANATION]"
        ]

        for token in special_tokens:
            assert token in vocab

    def test_max_length_truncation(self):
        """Test that sequences are properly truncated to max length."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased",
            max_length=50  # Very short for testing
        )

        # Create a very long prompt
        long_question = " ".join(["What is this very long question about something?"] * 20)
        choices = ["A", "B", "C", "D"]

        result = preprocessor.tokenize_example(long_question)

        assert result["input_ids"].shape == (50,)
        assert result["attention_mask"].shape == (50,)

    def test_batch_consistency(self):
        """Test that batch tokenization produces consistent results."""
        preprocessor = MMLUPreprocessor(
            tokenizer_name="distilbert-base-uncased",
            max_length=128
        )

        example = {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
            "uncertainty_type": "reasoning_error"
        }

        # Single tokenization
        single_result = preprocessor.tokenize_example(
            preprocessor.create_uncertainty_prompt(example["question"], example["choices"]),
            answer=example["answer"],
            uncertainty_type=example["uncertainty_type"]
        )

        # Batch tokenization with single example
        batch_result = preprocessor.batch_tokenize([example])

        # Results should be similar (may differ due to batching effects)
        assert single_result["input_ids"].shape == batch_result["input_ids"][0].shape
        assert single_result["answer_labels"] == batch_result["answer_labels"][0]
        assert single_result["uncertainty_labels"] == batch_result["uncertainty_labels"][0]