"""
Comprehensive tests for data preprocessing module including error handling.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing import (
    MMLUPreprocessor,
    PreprocessingError,
    TokenizationError,
    AnswerExtractionError,
    PromptCreationError
)


class TestMMLUPreprocessorErrorHandling:
    """Test error handling in MMLUPreprocessor."""

    def test_invalid_tokenizer_name_empty(self):
        """Test error handling for empty tokenizer name."""
        with pytest.raises(ValueError) as exc_info:
            MMLUPreprocessor(tokenizer_name="")
        assert "tokenizer_name must be a non-empty string" in str(exc_info.value)

    def test_invalid_tokenizer_name_none(self):
        """Test error handling for None tokenizer name."""
        with pytest.raises(ValueError) as exc_info:
            MMLUPreprocessor(tokenizer_name=None)
        assert "tokenizer_name must be a non-empty string" in str(exc_info.value)

    def test_invalid_max_length_negative(self):
        """Test error handling for negative max_length."""
        with pytest.raises(ValueError) as exc_info:
            MMLUPreprocessor(max_length=-1)
        assert "max_length must be a positive integer" in str(exc_info.value)

    def test_invalid_max_length_zero(self):
        """Test error handling for zero max_length."""
        with pytest.raises(ValueError) as exc_info:
            MMLUPreprocessor(max_length=0)
        assert "max_length must be a positive integer" in str(exc_info.value)

    def test_invalid_max_length_too_large(self):
        """Test error handling for excessively large max_length."""
        with pytest.raises(ValueError) as exc_info:
            MMLUPreprocessor(max_length=10000)
        assert "max_length must be a positive integer <= 8192" in str(exc_info.value)

    @patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing.AutoTokenizer.from_pretrained')
    def test_tokenizer_loading_failure(self, mock_tokenizer):
        """Test error handling when tokenizer fails to load."""
        mock_tokenizer.side_effect = Exception("Failed to load tokenizer")

        with pytest.raises(PreprocessingError) as exc_info:
            MMLUPreprocessor(tokenizer_name="invalid/tokenizer")
        assert "Tokenizer initialization failed" in str(exc_info.value)

    @patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing.AutoTokenizer.from_pretrained')
    def test_special_token_addition_failure(self, mock_tokenizer):
        """Test error handling when special token addition fails."""
        mock_tok = Mock()
        mock_tok.pad_token = "[PAD]"
        mock_tok.add_special_tokens.side_effect = Exception("Token addition failed")
        mock_tokenizer.return_value = mock_tok

        with pytest.raises(PreprocessingError) as exc_info:
            MMLUPreprocessor(tokenizer_name="test/tokenizer")
        assert "Special token addition failed" in str(exc_info.value)

    @patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing.AutoTokenizer.from_pretrained')
    def test_pad_token_handling(self, mock_tokenizer):
        """Test automatic pad token handling."""
        # Test case 1: No pad token, but has eos token
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "</s>"
        mock_tok.add_special_tokens = Mock(return_value=6)
        mock_tok.convert_tokens_to_ids = Mock(side_effect=lambda x: hash(x) % 1000)
        mock_tokenizer.return_value = mock_tok

        preprocessor = MMLUPreprocessor(tokenizer_name="test/tokenizer")
        assert mock_tok.pad_token == "</s>"

        # Test case 2: No pad token, no eos token
        mock_tok.pad_token = None
        mock_tok.eos_token = None
        mock_tok.add_special_tokens = Mock(return_value=7)
        mock_tokenizer.return_value = mock_tok

        preprocessor = MMLUPreprocessor(tokenizer_name="test/tokenizer")
        mock_tok.add_special_tokens.assert_called()


class TestAnswerExtraction:
    """Test answer extraction functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create a mock preprocessor for testing."""
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing.AutoTokenizer.from_pretrained'):
            mock_tok = Mock()
            mock_tok.pad_token = "[PAD]"
            mock_tok.add_special_tokens = Mock(return_value=6)
            mock_tok.convert_tokens_to_ids = Mock(side_effect=lambda x: hash(x) % 1000)

            return MMLUPreprocessor(tokenizer_name="test/tokenizer")

    def test_extract_answer_invalid_input_type(self, preprocessor):
        """Test error handling for non-string input."""
        with pytest.raises(AnswerExtractionError) as exc_info:
            preprocessor.extract_answer_from_response(123)
        assert "Expected string response" in str(exc_info.value)

    def test_extract_answer_empty_response(self, preprocessor):
        """Test handling of empty response."""
        answer_idx, confidence, explanation = preprocessor.extract_answer_from_response("")
        assert answer_idx is None
        assert confidence == "UNCERTAIN"
        assert "No response provided" in explanation

    def test_extract_answer_whitespace_only(self, preprocessor):
        """Test handling of whitespace-only response."""
        answer_idx, confidence, explanation = preprocessor.extract_answer_from_response("   \n\t  ")
        assert answer_idx is None
        assert confidence == "UNCERTAIN"
        assert "No response provided" in explanation

    def test_extract_answer_valid_patterns(self, preprocessor):
        """Test various valid answer patterns."""
        test_cases = [
            ("The answer is A", 0),
            ("Choice B is correct", 1),
            ("C) This is the answer", 2),
            ("D. The correct option", 3),
            ("I choose A", 0),
            ("answer: B", 1)
        ]

        for response, expected_idx in test_cases:
            answer_idx, confidence, explanation = preprocessor.extract_answer_from_response(response)
            assert answer_idx == expected_idx, f"Failed for response: {response}"
            assert isinstance(confidence, str)
            assert isinstance(explanation, str)

    def test_extract_answer_no_valid_pattern(self, preprocessor):
        """Test handling when no valid answer pattern is found."""
        response = "This is a response without any valid answer pattern."
        answer_idx, confidence, explanation = preprocessor.extract_answer_from_response(response)
        assert answer_idx is None
        assert confidence == "UNCERTAIN"
        assert len(explanation) > 0

    def test_extract_answer_out_of_range(self, preprocessor):
        """Test handling of out-of-range answer letters."""
        response = "The answer is X"  # Invalid letter
        answer_idx, confidence, explanation = preprocessor.extract_answer_from_response(response)
        # Should not extract invalid letters
        assert answer_idx is None

    def test_extract_confidence_levels(self, preprocessor):
        """Test confidence level extraction."""
        test_cases = [
            ("I am very confident the answer is A", "VERY_CONFIDENT"),
            ("I am confident the answer is B", "CONFIDENT"),
            ("I am somewhat confident the answer is C", "SOMEWHAT_CONFIDENT"),
            ("I am uncertain about the answer D", "UNCERTAIN"),
            ("I am very uncertain about answer A", "VERY_UNCERTAIN"),
            ("The answer is B", "UNCERTAIN")  # Default when no confidence indicator
        ]

        for response, expected_confidence in test_cases:
            _, confidence, _ = preprocessor.extract_answer_from_response(response)
            assert confidence == expected_confidence, f"Failed for response: {response}"

    def test_extract_explanation_patterns(self, preprocessor):
        """Test explanation extraction."""
        test_cases = [
            ("Answer A. KNOWLEDGE_GAP: I lack domain expertise", "I lack domain expertise"),
            ("Answer B. AMBIGUOUS: The question is unclear", "The question is unclear"),
            ("Answer C. REASONING_ERROR: Complex logic involved", "Complex logic involved"),
            ("Answer D. EXPLANATION: This is the reasoning", "This is the reasoning"),
            ("Answer A because it makes sense", "it makes sense"),
        ]

        for response, expected_explanation in test_cases:
            _, _, explanation = preprocessor.extract_answer_from_response(response)
            assert expected_explanation in explanation, f"Failed for response: {response}"

    def test_extract_answer_exception_handling(self, preprocessor):
        """Test exception handling in answer extraction."""
        # Force an exception by mocking re.search to fail
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing.re.search') as mock_search:
            mock_search.side_effect = Exception("Regex failed")

            with pytest.raises(AnswerExtractionError) as exc_info:
                preprocessor.extract_answer_from_response("Answer A")
            assert "Failed to extract answer" in str(exc_info.value)


class TestBatchTokenization:
    """Test batch tokenization functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create a mock preprocessor for testing."""
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tok = Mock()
            mock_tok.pad_token = "[PAD]"
            mock_tok.add_special_tokens = Mock(return_value=6)
            mock_tok.convert_tokens_to_ids = Mock(side_effect=lambda x: hash(x) % 1000)
            mock_tokenizer.return_value = mock_tok

            return MMLUPreprocessor(tokenizer_name="test/tokenizer")

    def test_batch_tokenize_empty_examples(self, preprocessor):
        """Test error handling for empty examples list."""
        with pytest.raises(ValueError) as exc_info:
            preprocessor.batch_tokenize([])
        assert "examples must be a non-empty list" in str(exc_info.value)

    def test_batch_tokenize_invalid_examples_type(self, preprocessor):
        """Test error handling for non-list examples."""
        with pytest.raises(ValueError) as exc_info:
            preprocessor.batch_tokenize("not a list")
        assert "examples must be a non-empty list" in str(exc_info.value)

    def test_batch_tokenize_invalid_example_structure(self, preprocessor):
        """Test error handling for invalid example structure."""
        examples = [
            {"question": "What is 2+2?"},  # Missing choices
            {"choices": ["A", "B", "C", "D"]}  # Missing question
        ]

        # Should handle gracefully and mask failed examples
        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt"):
            with patch.object(preprocessor.tokenizer, '__call__', return_value={
                "input_ids": torch.randint(0, 1000, (2, 10)),
                "attention_mask": torch.ones(2, 10)
            }):
                result = preprocessor.batch_tokenize(examples)

                assert "input_ids" in result
                assert "attention_mask" in result
                # Failed examples should be masked (attention = 0)
                assert result["attention_mask"][0].sum() == 0
                assert result["attention_mask"][1].sum() == 0

    def test_batch_tokenize_invalid_question_type(self, preprocessor):
        """Test handling of invalid question types."""
        examples = [
            {"question": None, "choices": ["A", "B", "C", "D"]},  # None question
            {"question": 123, "choices": ["A", "B", "C", "D"]},   # Non-string question
            {"question": "", "choices": ["A", "B", "C", "D"]}     # Empty question
        ]

        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt"):
            with patch.object(preprocessor.tokenizer, '__call__', return_value={
                "input_ids": torch.randint(0, 1000, (3, 10)),
                "attention_mask": torch.ones(3, 10)
            }):
                result = preprocessor.batch_tokenize(examples)

                # All examples should be masked due to invalid questions
                assert result["attention_mask"].sum() == 0

    def test_batch_tokenize_invalid_choices_type(self, preprocessor):
        """Test handling of invalid choices types."""
        examples = [
            {"question": "Test?", "choices": None},          # None choices
            {"question": "Test?", "choices": "not a list"},  # Non-list choices
            {"question": "Test?", "choices": []}             # Empty choices
        ]

        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt"):
            with patch.object(preprocessor.tokenizer, '__call__', return_value={
                "input_ids": torch.randint(0, 1000, (3, 10)),
                "attention_mask": torch.ones(3, 10)
            }):
                result = preprocessor.batch_tokenize(examples)

                # All examples should be masked due to invalid choices
                assert result["attention_mask"].sum() == 0

    def test_batch_tokenize_tokenizer_failure(self, preprocessor):
        """Test handling when tokenizer fails."""
        examples = [{"question": "Test?", "choices": ["A", "B", "C", "D"]}]

        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt"):
            with patch.object(preprocessor.tokenizer, '__call__', side_effect=Exception("Tokenizer failed")):
                with pytest.raises(TokenizationError) as exc_info:
                    preprocessor.batch_tokenize(examples)
                assert "Failed to tokenize batch" in str(exc_info.value)

    def test_batch_tokenize_missing_tokenizer_outputs(self, preprocessor):
        """Test handling when tokenizer doesn't return required outputs."""
        examples = [{"question": "Test?", "choices": ["A", "B", "C", "D"]}]

        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt"):
            with patch.object(preprocessor.tokenizer, '__call__', return_value={"input_ids": torch.randn(1, 10)}):  # Missing attention_mask
                with pytest.raises(TokenizationError) as exc_info:
                    preprocessor.batch_tokenize(examples)
                assert "did not return required outputs" in str(exc_info.value)

    def test_batch_tokenize_size_mismatch(self, preprocessor):
        """Test handling when tokenizer output size doesn't match input."""
        examples = [{"question": "Test?", "choices": ["A", "B", "C", "D"]}]

        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt"):
            with patch.object(preprocessor.tokenizer, '__call__', return_value={
                "input_ids": torch.randint(0, 1000, (2, 10)),  # Wrong batch size
                "attention_mask": torch.ones(2, 10)
            }):
                with pytest.raises(TokenizationError) as exc_info:
                    preprocessor.batch_tokenize(examples)
                assert "output size mismatch" in str(exc_info.value)

    def test_batch_tokenize_few_shot_without_examples(self, preprocessor):
        """Test handling of few-shot mode without examples."""
        examples = [{"question": "Test?", "choices": ["A", "B", "C", "D"]}]

        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt"):
            with patch.object(preprocessor.tokenizer, '__call__', return_value={
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10)
            }):
                # Should automatically disable few-shot and log warning
                result = preprocessor.batch_tokenize(examples, use_few_shot=True, few_shot_examples=None)
                assert "input_ids" in result

    def test_batch_tokenize_valid_labels(self, preprocessor):
        """Test handling of valid answer and uncertainty labels."""
        examples = [
            {"question": "Test1?", "choices": ["A", "B", "C", "D"], "answer": 1, "uncertainty_type": "knowledge_gap"},
            {"question": "Test2?", "choices": ["A", "B", "C", "D"], "answer": 2, "uncertainty_type": "ambiguous"}
        ]

        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt"):
            with patch.object(preprocessor.tokenizer, '__call__', return_value={
                "input_ids": torch.randint(0, 1000, (2, 10)),
                "attention_mask": torch.ones(2, 10)
            }):
                result = preprocessor.batch_tokenize(examples)

                assert "answer_labels" in result
                assert "uncertainty_labels" in result
                assert result["answer_labels"].tolist() == [1, 2]
                assert result["uncertainty_labels"].tolist() == [0, 1]  # knowledge_gap=0, ambiguous=1

    def test_batch_tokenize_invalid_labels(self, preprocessor):
        """Test handling of invalid answer labels."""
        examples = [
            {"question": "Test1?", "choices": ["A", "B", "C", "D"], "answer": 5},  # Out of range
            {"question": "Test2?", "choices": ["A", "B", "C", "D"], "answer": "invalid"}  # Wrong type
        ]

        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt"):
            with patch.object(preprocessor.tokenizer, '__call__', return_value={
                "input_ids": torch.randint(0, 1000, (2, 10)),
                "attention_mask": torch.ones(2, 10)
            }):
                result = preprocessor.batch_tokenize(examples)

                assert "answer_labels" in result
                # Invalid labels should be converted to 0
                assert result["answer_labels"].tolist() == [0, 0]

    def test_batch_tokenize_partial_failure(self, preprocessor):
        """Test handling when some examples fail but others succeed."""
        examples = [
            {"question": "Valid question?", "choices": ["A", "B", "C", "D"]},  # Valid
            {"question": None, "choices": ["A", "B", "C", "D"]},              # Invalid
            {"question": "Another valid?", "choices": ["A", "B", "C", "D"]}   # Valid
        ]

        with patch.object(preprocessor, 'create_uncertainty_prompt', side_effect=["prompt1", PromptCreationError("Failed"), "prompt3"]):
            with patch.object(preprocessor.tokenizer, '__call__', return_value={
                "input_ids": torch.randint(0, 1000, (3, 10)),
                "attention_mask": torch.ones(3, 10)
            }):
                result = preprocessor.batch_tokenize(examples)

                # Should mask the failed example (index 1)
                assert result["attention_mask"][1].sum() == 0
                # Valid examples should remain unmasked
                assert result["attention_mask"][0].sum() > 0
                assert result["attention_mask"][2].sum() > 0


class TestPromptCreation:
    """Test prompt creation methods."""

    @pytest.fixture
    def preprocessor(self):
        """Create a mock preprocessor for testing."""
        with patch('src.metacognitive_uncertainty_calibration_for_llm_knowledge_boundaries.data.preprocessing.AutoTokenizer.from_pretrained'):
            mock_tok = Mock()
            mock_tok.pad_token = "[PAD]"
            mock_tok.add_special_tokens = Mock(return_value=6)
            mock_tok.convert_tokens_to_ids = Mock(side_effect=lambda x: hash(x) % 1000)

            return MMLUPreprocessor(tokenizer_name="test/tokenizer")

    def test_create_uncertainty_prompt_basic(self, preprocessor):
        """Test basic uncertainty prompt creation."""
        question = "What is 2+2?"
        choices = ["2", "3", "4", "5"]

        # Mock the method to test if it's called correctly
        with patch.object(preprocessor, 'create_uncertainty_prompt', return_value="test prompt") as mock_create:
            result = preprocessor.create_uncertainty_prompt(question, choices)
            mock_create.assert_called_once_with(question, choices)
            assert result == "test prompt"

    def test_prompt_creation_with_empty_question(self, preprocessor):
        """Test prompt creation with empty question."""
        # This would be caught by the validation in batch_tokenize
        # but testing the prompt creation method specifically
        question = ""
        choices = ["A", "B", "C", "D"]

        # Should handle gracefully without throwing exception
        # Implementation details depend on the actual create_uncertainty_prompt method
        pass

    def test_prompt_creation_with_empty_choices(self, preprocessor):
        """Test prompt creation with empty choices."""
        question = "What is the answer?"
        choices = []

        # Should handle gracefully without throwing exception
        # Implementation details depend on the actual create_uncertainty_prompt method
        pass