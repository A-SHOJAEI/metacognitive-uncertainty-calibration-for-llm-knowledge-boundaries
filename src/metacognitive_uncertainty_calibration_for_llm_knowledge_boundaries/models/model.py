"""
Metacognitive Uncertainty Model for LLM Knowledge Boundary Detection.

This module implements a novel architecture that combines answer prediction
with uncertainty type classification and natural language explanation generation.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
from sentence_transformers import SentenceTransformer
import numpy as np

from ..utils.constants import (
    DEFAULT_NUM_UNCERTAINTY_TYPES,
    DEFAULT_EXPLANATION_VOCAB_SIZE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EPISTEMIC_SAMPLES,
    DEFAULT_HIDDEN_SIZE_REDUCTION_FACTOR,
    EPSILON,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_TEMPERATURE_LR,
    DEFAULT_TEMPERATURE_MAX_ITER,
    UNCERTAINTY_TYPES,
    ANSWER_CHOICE_ASCII_OFFSET
)

logger = logging.getLogger(__name__)


# Custom exceptions for model operations
class ModelError(Exception):
    """Base class for model-related errors."""
    pass


class ModelInputError(ModelError):
    """Raised when invalid inputs are provided to model."""
    pass


class ModelForwardError(ModelError):
    """Raised when forward pass fails."""
    pass


class UncertaintyEstimationError(ModelError):
    """Raised when uncertainty estimation fails."""
    pass


@dataclass
class MetacognitiveOutput:
    """Output from metacognitive uncertainty model."""

    # Answer prediction
    answer_logits: torch.Tensor
    answer_prediction: Union[int, List[int]]
    answer_confidence: float

    # Uncertainty estimation
    uncertainty_logits: torch.Tensor
    uncertainty_type: Union[str, List[str]]
    uncertainty_confidence: float

    # Natural language explanation
    explanation: Union[str, List[str]]
    explanation_logits: Optional[torch.Tensor] = None

    # Internal representations
    question_embedding: Optional[torch.Tensor] = None
    uncertainty_embedding: Optional[torch.Tensor] = None

    # Calibration metrics
    epistemic_uncertainty: Optional[float] = None
    aleatoric_uncertainty: Optional[float] = None


class UncertaintyHead(nn.Module):
    """
    Multi-head uncertainty estimation module that predicts both uncertainty
    type and generates natural language explanations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_uncertainty_types: int = DEFAULT_NUM_UNCERTAINTY_TYPES,
        explanation_vocab_size: int = DEFAULT_EXPLANATION_VOCAB_SIZE,
        dropout: float = DEFAULT_DROPOUT_RATE
    ) -> None:
        """
        Initialize uncertainty head.

        Args:
            hidden_size: Size of input hidden representations
            num_uncertainty_types: Number of uncertainty categories
            explanation_vocab_size: Vocabulary size for explanation generation
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_uncertainty_types = num_uncertainty_types

        # Uncertainty type classifier
        self.uncertainty_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_uncertainty_types)
        )

        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # Explanation generator (simple projection for now)
        self.explanation_projector = nn.Linear(hidden_size, explanation_vocab_size)

        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through uncertainty head.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask for sequence

        Returns:
            Dictionary with uncertainty predictions
        """
        # Pool hidden states with numerical stability
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        pooled = torch.clamp(pooled, min=-1e4, max=1e4)

        # Predict uncertainty type
        uncertainty_logits = self.uncertainty_classifier(pooled)
        uncertainty_probs = F.softmax(uncertainty_logits / self.temperature, dim=-1)

        # Estimate confidence
        confidence = self.confidence_estimator(pooled).squeeze(-1)

        # Generate explanation logits
        explanation_logits = self.explanation_projector(pooled)

        return {
            "uncertainty_logits": uncertainty_logits,
            "uncertainty_probs": uncertainty_probs,
            "confidence": confidence,
            "explanation_logits": explanation_logits,
            "pooled_representation": pooled
        }


class EpistemicUncertaintyEstimator(nn.Module):
    """
    Estimates epistemic (knowledge) uncertainty using Monte Carlo Dropout
    and ensemble-like techniques.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_samples: int = DEFAULT_EPISTEMIC_SAMPLES,
        dropout_rate: float = DEFAULT_DROPOUT_RATE
    ) -> None:
        """
        Initialize epistemic uncertainty estimator.

        Args:
            base_model: Base model to wrap
            num_samples: Number of MC samples
            dropout_rate: Dropout rate for MC sampling
        """
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate epistemic uncertainty via Monte Carlo sampling.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            training: Whether in training mode

        Returns:
            Tuple of (mean_logits, uncertainty_estimate)
        """
        if not training:
            # During inference, use MC dropout
            self.base_model.train()  # Enable dropout

            samples = []
            for _ in range(self.num_samples):
                with torch.no_grad():
                    logits = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    ).logits
                    samples.append(logits)

            samples = torch.stack(samples)  # [num_samples, batch_size, seq_len, vocab_size]

            # Calculate mean and variance
            mean_logits = samples.mean(dim=0)
            variance = samples.var(dim=0).mean(dim=-1)  # Average over vocabulary

            self.base_model.eval()  # Return to eval mode
            return mean_logits, variance
        else:
            logits = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits
            return logits, torch.zeros_like(logits[:, :, 0])


class MetacognitiveUncertaintyModel(nn.Module):
    """
    Complete metacognitive uncertainty model that combines answer prediction,
    uncertainty type classification, and natural language explanation generation.
    """

    UNCERTAINTY_TYPES = UNCERTAINTY_TYPES

    def __init__(
        self,
        base_model_name: str = "gpt2-medium",
        num_choices: int = 4,
        uncertainty_weight: float = 0.3,
        explanation_weight: float = 0.2,
        use_epistemic_estimation: bool = True,
        freeze_base_model: bool = False
    ) -> None:
        """
        Initialize metacognitive uncertainty model.

        Args:
            base_model_name: Name of base language model
            num_choices: Number of answer choices (typically 4 for MMLU)
            uncertainty_weight: Weight for uncertainty loss
            explanation_weight: Weight for explanation loss
            use_epistemic_estimation: Whether to use epistemic uncertainty estimation
            freeze_base_model: Whether to freeze base model parameters
        """
        super().__init__()

        self.base_model_name = base_model_name
        self.num_choices = num_choices
        self.uncertainty_weight = uncertainty_weight
        self.explanation_weight = explanation_weight
        self.use_epistemic_estimation = use_epistemic_estimation

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size

        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Answer prediction head
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_choices)
        )

        # Uncertainty estimation head
        self.uncertainty_head = UncertaintyHead(
            hidden_size=hidden_size,
            num_uncertainty_types=len(self.UNCERTAINTY_TYPES),
            explanation_vocab_size=len(self.tokenizer)
        )

        # Epistemic uncertainty estimator
        if use_epistemic_estimation:
            # Create a copy of base model for epistemic estimation
            self.epistemic_estimator = EpistemicUncertaintyEstimator(
                base_model=AutoModelForCausalLM.from_pretrained(base_model_name),
                num_samples=DEFAULT_EPISTEMIC_SAMPLES
            )

        # Domain embedding is used at inference time only (loaded on demand)

        logger.info(f"Initialized MetacognitiveUncertaintyModel with {base_model_name}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        answer_labels: Optional[torch.Tensor] = None,
        uncertainty_labels: Optional[torch.Tensor] = None,
        explanation_labels: Optional[torch.Tensor] = None,
        domain_context: Optional[List[str]] = None
    ) -> MetacognitiveOutput:
        """Forward pass through the complete model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            answer_labels: Ground truth answer labels [batch_size]
            uncertainty_labels: Ground truth uncertainty type labels [batch_size]
            explanation_labels: Ground truth explanation labels [batch_size, seq_len]
            domain_context: List of domain strings for each example

        Returns:
            MetacognitiveOutput with all predictions and uncertainties

        Raises:
            ModelInputError: If inputs are invalid or incompatible
            ModelForwardError: If forward pass fails
        """
        # Input validation
        try:
            # Validate input_ids
            if input_ids is None or not isinstance(input_ids, torch.Tensor):
                raise ModelInputError("input_ids must be a torch.Tensor")

            if input_ids.dim() != 2:
                raise ModelInputError(
                    f"input_ids must be 2D tensor (batch_size, seq_len), "
                    f"got shape {input_ids.shape}"
                )

            batch_size, seq_len = input_ids.shape
            if batch_size == 0 or seq_len == 0:
                raise ModelInputError(
                    f"input_ids has invalid dimensions: batch_size={batch_size}, seq_len={seq_len}"
                )

            # Validate attention mask if provided
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    raise ModelInputError("attention_mask must be a torch.Tensor or None")

                if attention_mask.shape != input_ids.shape:
                    raise ModelInputError(
                        f"attention_mask shape {attention_mask.shape} doesn't match "
                        f"input_ids shape {input_ids.shape}"
                    )

            # Validate answer_labels if provided
            if answer_labels is not None:
                if not isinstance(answer_labels, torch.Tensor):
                    raise ModelInputError("answer_labels must be a torch.Tensor or None")

                if answer_labels.dim() != 1 or answer_labels.size(0) != batch_size:
                    raise ModelInputError(
                        f"answer_labels must be 1D tensor with batch_size elements, "
                        f"got shape {answer_labels.shape}"
                    )

                if torch.any(answer_labels < 0) or torch.any(answer_labels >= self.num_choices):
                    raise ModelInputError(
                        f"answer_labels must be in range [0, {self.num_choices-1}], "
                        f"got range [{answer_labels.min()}, {answer_labels.max()}]"
                    )

            # Validate uncertainty_labels if provided
            if uncertainty_labels is not None:
                if not isinstance(uncertainty_labels, torch.Tensor):
                    raise ModelInputError("uncertainty_labels must be a torch.Tensor or None")

                if uncertainty_labels.dim() != 1 or uncertainty_labels.size(0) != batch_size:
                    raise ModelInputError(
                        f"uncertainty_labels must be 1D tensor with batch_size elements, "
                        f"got shape {uncertainty_labels.shape}"
                    )

                num_uncertainty_types = len(self.UNCERTAINTY_TYPES)
                if torch.any(uncertainty_labels < 0) or torch.any(uncertainty_labels >= num_uncertainty_types):
                    raise ModelInputError(
                        f"uncertainty_labels must be in range [0, {num_uncertainty_types-1}], "
                        f"got range [{uncertainty_labels.min()}, {uncertainty_labels.max()}]"
                    )

            # Validate domain_context if provided
            if domain_context is not None:
                if not isinstance(domain_context, list):
                    raise ModelInputError("domain_context must be a list or None")

                if len(domain_context) != batch_size:
                    raise ModelInputError(
                        f"domain_context length {len(domain_context)} doesn't match "
                        f"batch_size {batch_size}"
                    )

                for i, context in enumerate(domain_context):
                    if not isinstance(context, str):
                        raise ModelInputError(f"domain_context[{i}] must be a string")

            # Ensure tensors are on the same device as the model
            device = next(self.parameters()).device
            if input_ids.device != device:
                logger.debug(f"Moving input_ids from {input_ids.device} to {device}")
                input_ids = input_ids.to(device)

            if attention_mask is not None and attention_mask.device != device:
                logger.debug(f"Moving attention_mask from {attention_mask.device} to {device}")
                attention_mask = attention_mask.to(device)

            logger.debug(f"Forward pass: batch_size={batch_size}, seq_len={seq_len}, device={device}")

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ModelInputError(f"Input validation failed: {e}") from e

        try:
            # Get base model outputs
            logger.debug("Running base model forward pass")
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            if not hasattr(outputs, 'last_hidden_state'):
                raise ModelForwardError("Base model outputs missing last_hidden_state")

            hidden_states = outputs.last_hidden_state
            if hidden_states is None:
                raise ModelForwardError("Base model returned None for last_hidden_state")

            if hidden_states.shape[:2] != (batch_size, seq_len):
                raise ModelForwardError(
                    f"Hidden states shape {hidden_states.shape} doesn't match "
                    f"expected ({batch_size}, {seq_len}, hidden_size)"
                )

            # Pooling with numerical stability
            logger.debug("Computing pooled output")
            if attention_mask is not None:
                # Masked average pooling with epsilon for numerical stability
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled_output = (hidden_states * mask_expanded).sum(dim=1) / sum_mask
            else:
                pooled_output = hidden_states.mean(dim=1)

            # Clamp to prevent extreme values
            pooled_output = torch.clamp(pooled_output, min=-1e4, max=1e4)

        except Exception as e:
            logger.error(f"Base model forward pass failed: {e}", exc_info=True)
            raise ModelForwardError(f"Base model forward pass failed: {e}") from e

        try:
            # Answer prediction
            logger.debug("Computing answer predictions")
            answer_logits = self.answer_head(pooled_output)

            if answer_logits.shape != (batch_size, self.num_choices):
                raise ModelForwardError(
                    f"Answer logits shape {answer_logits.shape} doesn't match "
                    f"expected ({batch_size}, {self.num_choices})"
                )

            # Clamp answer logits for numerical stability
            answer_logits = torch.clamp(answer_logits, min=-100, max=100)

            answer_probs = F.softmax(answer_logits, dim=-1)
            answer_predictions = answer_logits.argmax(dim=-1)
            answer_confidence = answer_probs.max(dim=-1)[0]

        except Exception as e:
            logger.error(f"Answer prediction failed: {e}", exc_info=True)
            raise ModelForwardError(f"Answer prediction failed: {e}") from e

        try:
            # Uncertainty estimation
            logger.debug("Computing uncertainty estimates")
            uncertainty_outputs = self.uncertainty_head(hidden_states, attention_mask)

            required_keys = ["uncertainty_logits", "uncertainty_probs", "confidence", "explanation_logits"]
            missing_keys = [key for key in required_keys if key not in uncertainty_outputs]
            if missing_keys:
                raise ModelForwardError(f"Uncertainty head outputs missing keys: {missing_keys}")

            uncertainty_logits = uncertainty_outputs["uncertainty_logits"]
            uncertainty_probs = uncertainty_outputs["uncertainty_probs"]
            uncertainty_confidence = uncertainty_outputs["confidence"]

            # Clamp uncertainty logits for numerical stability
            uncertainty_logits = torch.clamp(uncertainty_logits, min=-100, max=100)

            # Get uncertainty type predictions with bounds checking
            uncertainty_predictions = uncertainty_logits.argmax(dim=-1)
            uncertainty_types = []

            for pred in uncertainty_predictions:
                pred_idx = pred.item() if torch.is_tensor(pred) else pred
                if 0 <= pred_idx < len(self.UNCERTAINTY_TYPES):
                    uncertainty_types.append(self.UNCERTAINTY_TYPES[pred_idx])
                else:
                    logger.warning(f"Invalid uncertainty prediction index {pred_idx}, using default")
                    uncertainty_types.append(self.UNCERTAINTY_TYPES[0])

        except Exception as e:
            logger.error(f"Uncertainty estimation failed: {e}", exc_info=True)
            raise UncertaintyEstimationError(f"Uncertainty estimation failed: {e}") from e

        try:
            # Generate explanations
            logger.debug("Generating explanations")
            explanation_logits = uncertainty_outputs["explanation_logits"]

            # Convert tensors to lists safely
            answer_confidence_list = []
            for conf in answer_confidence:
                conf_val = conf.item() if torch.is_tensor(conf) else conf
                if not isinstance(conf_val, (int, float)) or math.isnan(conf_val):
                    logger.warning(f"Invalid confidence value {conf_val}, using 0.0")
                    conf_val = 0.0
                answer_confidence_list.append(float(conf_val))

            explanations = self._generate_explanations(
                uncertainty_types,
                answer_confidence_list,
                domain_context
            )

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}", exc_info=True)
            # Use fallback explanations
            explanations = ["Unable to generate explanation"] * batch_size

        try:
            # Estimate epistemic uncertainty if enabled
            epistemic_uncertainty = None
            if self.use_epistemic_estimation and not self.training:
                logger.debug("Computing epistemic uncertainty")
                _, epistemic_var = self.epistemic_estimator(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    training=False
                )

                if epistemic_var is not None:
                    if torch.isnan(epistemic_var).any() or torch.isinf(epistemic_var).any():
                        logger.warning("Epistemic variance contains NaN/Inf values, setting to None")
                        epistemic_uncertainty = None
                    else:
                        epistemic_uncertainty = epistemic_var.mean(dim=-1)
                        if epistemic_uncertainty.numel() == 0:
                            epistemic_uncertainty = None

        except Exception as e:
            logger.error(f"Epistemic uncertainty estimation failed: {e}")
            epistemic_uncertainty = None

        # Safe conversion and output creation
        try:
            logger.debug("Creating model output")

            # Safe conversions with fallbacks
            answer_pred = (answer_predictions[0].item() if batch_size == 1
                          else answer_predictions.tolist())

            answer_conf = answer_confidence.mean().item() if answer_confidence.numel() > 0 else 0.0
            if math.isnan(answer_conf) or math.isinf(answer_conf):
                logger.warning("Invalid answer confidence, using 0.0")
                answer_conf = 0.0

            uncertainty_conf = uncertainty_confidence.mean().item() if uncertainty_confidence.numel() > 0 else 0.0
            if math.isnan(uncertainty_conf) or math.isinf(uncertainty_conf):
                logger.warning("Invalid uncertainty confidence, using 0.0")
                uncertainty_conf = 0.0

            uncertainty_type_out = (uncertainty_types[0] if batch_size == 1
                                  else uncertainty_types)

            explanation_out = (explanations[0] if batch_size == 1
                             else explanations)

            # Safe epistemic uncertainty conversion
            epistemic_unc = None
            if epistemic_uncertainty is not None and epistemic_uncertainty.numel() > 0:
                epistemic_val = epistemic_uncertainty.mean().item()
                if not (math.isnan(epistemic_val) or math.isinf(epistemic_val)):
                    epistemic_unc = epistemic_val

            # Safe aleatoric uncertainty estimation
            try:
                aleatoric_unc = self._estimate_aleatoric_uncertainty(answer_logits).item()
                if math.isnan(aleatoric_unc) or math.isinf(aleatoric_unc):
                    logger.warning("Invalid aleatoric uncertainty, using 0.0")
                    aleatoric_unc = 0.0
            except Exception as e:
                logger.error(f"Aleatoric uncertainty estimation failed: {e}")
                aleatoric_unc = 0.0

            # Safe uncertainty embedding extraction
            uncertainty_emb = uncertainty_outputs.get("pooled_representation")
            if uncertainty_emb is not None:
                if torch.isnan(uncertainty_emb).any() or torch.isinf(uncertainty_emb).any():
                    logger.warning("Uncertainty embedding contains NaN/Inf, setting to None")
                    uncertainty_emb = None

            output = MetacognitiveOutput(
                answer_logits=answer_logits,
                answer_prediction=answer_pred,
                answer_confidence=answer_conf,
                uncertainty_logits=uncertainty_logits,
                uncertainty_type=uncertainty_type_out,
                uncertainty_confidence=uncertainty_conf,
                explanation=explanation_out,
                explanation_logits=explanation_logits,
                question_embedding=pooled_output,
                uncertainty_embedding=uncertainty_emb,
                epistemic_uncertainty=epistemic_unc,
                aleatoric_uncertainty=aleatoric_unc
            )

            logger.debug("Model forward pass completed successfully")
            return output

        except Exception as e:
            logger.error(f"Failed to create model output: {e}", exc_info=True)
            raise ModelForwardError(f"Failed to create model output: {e}") from e

    def _generate_explanations(
        self,
        uncertainty_types: List[str],
        answer_confidences: List[float],
        domain_contexts: Optional[List[str]] = None
    ) -> List[str]:
        """Generate natural language explanations for uncertainty predictions.

        This method creates human-readable explanations based on the predicted
        uncertainty type and confidence scores. Explanations are tailored to
        different uncertainty categories to provide interpretable feedback.

        Args:
            uncertainty_types: List of predicted uncertainty types. Must be one of:
                - 'knowledge_gap': Insufficient domain knowledge
                - 'ambiguous': Question contains ambiguous elements
                - 'reasoning_error': Risk of logical reasoning errors
                - 'confident': High confidence in answer
            answer_confidences: List of confidence scores in range [0.0, 1.0]
                for each prediction.
            domain_contexts: Optional list of domain names/descriptions for
                contextualized explanations. If None, generic domain references
                are used.

        Returns:
            List of explanation strings, one for each uncertainty prediction.
            Each explanation includes the uncertainty type, confidence score,
            and domain-specific reasoning.

        Example:
            >>> uncertainty_types = ["knowledge_gap", "confident"]
            >>> confidences = [0.3, 0.9]
            >>> domains = ["quantum physics", "basic math"]
            >>> explanations = model._generate_explanations(
            ...     uncertainty_types, confidences, domains
            ... )
            >>> print(explanations[0])
            "I lack sufficient knowledge in quantum physics to answer
             confidently (confidence: 0.30). This question requires..."
        """
        explanations = []

        for i, (utype, confidence) in enumerate(zip(uncertainty_types, answer_confidences)):
            domain = domain_contexts[i] if domain_contexts and i < len(domain_contexts) else "this domain"

            if utype == "knowledge_gap":
                explanation = f"I lack sufficient knowledge in {domain} to answer confidently (confidence: {confidence:.2f}). " \
                             f"This question requires domain-specific expertise that may be outside my training data."

            elif utype == "ambiguous":
                explanation = f"This question contains ambiguous elements that make it difficult to determine a definitive answer " \
                             f"(confidence: {confidence:.2f}). The phrasing or answer choices may have multiple valid interpretations."

            elif utype == "reasoning_error":
                explanation = f"This question requires complex reasoning where I might make logical errors " \
                             f"(confidence: {confidence:.2f}). The multi-step reasoning process increases uncertainty."

            else:
                explanation = f"I am confident in this answer (confidence: {confidence:.2f})."

            explanations.append(explanation)

        return explanations

    def _estimate_aleatoric_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """Estimate aleatoric (data-dependent) uncertainty from prediction entropy.

        Aleatoric uncertainty represents the inherent uncertainty in the data
        that cannot be reduced by collecting more data. This is computed using
        the entropy of the predicted probability distribution over answer choices.
        Higher entropy indicates greater uncertainty in the prediction.

        The calculation follows the standard information-theoretic entropy:
        H(p) = -Σ p(x) * log(p(x))

        Args:
            logits: Raw model output logits of shape (batch_size, num_classes).
                These are converted to probabilities via softmax before
                entropy calculation.

        Returns:
            torch.Tensor: Mean aleatoric uncertainty score across the batch.
                Higher values indicate greater uncertainty. Range is [0, log(C)]
                where C is the number of classes.

        Note:
            - Small epsilon (1e-10) added to prevent log(0) numerical issues
            - Returns mean uncertainty across batch for scalar loss computation
            - This captures irreducible uncertainty from noisy/ambiguous data

        Example:
            >>> logits = torch.randn(32, 4)  # Batch of 32, 4 answer choices
            >>> uncertainty = model._estimate_aleatoric_uncertainty(logits)
            >>> print(f"Aleatoric uncertainty: {uncertainty:.4f}")
            Aleatoric uncertainty: 1.2456
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + EPSILON)).sum(dim=-1)
        return entropy.mean()

    def compute_loss(
        self,
        outputs: MetacognitiveOutput,
        answer_labels: torch.Tensor,
        uncertainty_labels: Optional[torch.Tensor] = None,
        explanation_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss for training.

        Args:
            outputs: Model outputs
            answer_labels: Ground truth answer labels
            uncertainty_labels: Ground truth uncertainty labels
            explanation_labels: Ground truth explanation labels

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # Answer prediction loss
        answer_loss = F.cross_entropy(outputs.answer_logits, answer_labels)
        losses["answer_loss"] = answer_loss

        # Uncertainty classification loss
        if uncertainty_labels is not None:
            uncertainty_loss = F.cross_entropy(outputs.uncertainty_logits, uncertainty_labels)
            losses["uncertainty_loss"] = uncertainty_loss
        else:
            losses["uncertainty_loss"] = torch.tensor(0.0, device=outputs.answer_logits.device)

        # Explanation generation loss (simplified)
        if explanation_labels is not None:
            # For simplicity, we'll use a basic loss here
            # In practice, this would be more sophisticated
            batch_size = outputs.explanation_logits.size(0)
            vocab_size = outputs.explanation_logits.size(1)
            explanation_loss = F.cross_entropy(
                outputs.explanation_logits,  # [batch_size, vocab_size]
                explanation_labels[:, 0] if explanation_labels.dim() > 1 else explanation_labels  # [batch_size]
            )
            losses["explanation_loss"] = explanation_loss
        else:
            losses["explanation_loss"] = torch.tensor(0.0, device=outputs.answer_logits.device)

        # Calibration loss (encourages well-calibrated confidence)
        predicted_confidence = torch.sigmoid(outputs.answer_logits.max(dim=-1)[0])
        actual_accuracy = (outputs.answer_logits.argmax(dim=-1) == answer_labels).float()
        calibration_loss = F.mse_loss(predicted_confidence, actual_accuracy)
        losses["calibration_loss"] = calibration_loss

        # Total loss
        total_loss = (
            answer_loss +
            self.uncertainty_weight * losses["uncertainty_loss"] +
            self.explanation_weight * losses["explanation_loss"] +
            0.1 * calibration_loss
        )
        losses["total_loss"] = total_loss

        return losses

    def predict_with_uncertainty(
        self,
        question: str,
        choices: List[str],
        domain: Optional[str] = None,
        return_explanations: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions with uncertainty estimation for a single question.

        Args:
            question: Question text
            choices: List of answer choices
            domain: Domain context
            return_explanations: Whether to return explanations

        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        self.eval()

        # Format input
        choices_text = "\n".join([f"({chr(ANSWER_CHOICE_ASCII_OFFSET + i)}) {choice}" for i, choice in enumerate(choices)])
        input_text = f"{question}\n\n{choices_text}\n\nAnswer:"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_SEQUENCE_LENGTH,
            padding=True
        )

        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                domain_context=[domain] if domain else None
            )

        result = {
            "predicted_answer": outputs.answer_prediction,
            "answer_confidence": outputs.answer_confidence,
            "uncertainty_type": outputs.uncertainty_type,
            "uncertainty_confidence": outputs.uncertainty_confidence,
            "epistemic_uncertainty": outputs.epistemic_uncertainty,
            "aleatoric_uncertainty": outputs.aleatoric_uncertainty
        }

        if return_explanations:
            result["explanation"] = outputs.explanation

        return result

    def calibrate_temperature(
        self,
        validation_loader: torch.utils.data.DataLoader,
        method: str = "temperature_scaling"
    ) -> None:
        """
        Calibrate model confidence using temperature scaling.

        Args:
            validation_loader: Validation data loader
            method: Calibration method to use
        """
        self.eval()

        if method == "temperature_scaling":
            # Collect predictions and labels
            logits_list = []
            labels_list = []

            device = next(self.parameters()).device
            with torch.no_grad():
                for batch in validation_loader:
                    outputs = self.forward(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device)
                    )
                    logits_list.append(outputs.answer_logits)
                    labels_list.append(batch["answer_labels"].to(device))

            logits = torch.cat(logits_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            # Optimize temperature parameter
            optimizer = torch.optim.LBFGS([self.uncertainty_head.temperature], lr=DEFAULT_TEMPERATURE_LR, max_iter=DEFAULT_TEMPERATURE_MAX_ITER)

            def closure():
                optimizer.zero_grad()
                loss = F.cross_entropy(logits / self.uncertainty_head.temperature, labels)
                loss.backward()
                return loss

            optimizer.step(closure)

            logger.info(f"Calibrated temperature: {self.uncertainty_head.temperature.item():.3f}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and parameter information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "base_model": self.base_model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "uncertainty_types": self.UNCERTAINTY_TYPES,
            "num_choices": self.num_choices,
            "use_epistemic_estimation": self.use_epistemic_estimation,
            "uncertainty_weight": self.uncertainty_weight,
            "explanation_weight": self.explanation_weight
        }