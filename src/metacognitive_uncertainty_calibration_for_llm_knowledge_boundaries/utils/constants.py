"""
Constants and magic numbers for metacognitive uncertainty calibration.

This module centralizes all magic numbers and configuration constants
used throughout the codebase for better maintainability.
"""

# Model Architecture Constants
DEFAULT_HIDDEN_SIZE_REDUCTION_FACTOR = 0.5  # For hidden layer size reduction
DEFAULT_EXPLANATION_VOCAB_SIZE = 50257       # GPT-2 vocabulary size
DEFAULT_NUM_UNCERTAINTY_TYPES = 3            # knowledge_gap, ambiguous, reasoning_error
DEFAULT_NUM_CHOICES = 4                      # Multiple choice options (A, B, C, D)
DEFAULT_EPISTEMIC_SAMPLES = 10               # MC Dropout samples for epistemic uncertainty

# Training Constants
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS = 500
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE = 3
DEFAULT_TEMPERATURE_LR = 0.01
DEFAULT_TEMPERATURE_MAX_ITER = 100

# Data Processing Constants
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_FEW_SHOT_EXAMPLES = 5

# Evaluation Constants
DEFAULT_CALIBRATION_BINS = 15
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_SELECTIVE_PREDICTION_COVERAGES = [0.5, 0.6, 0.7, 0.8, 0.9]

# Loss Weights
DEFAULT_UNCERTAINTY_WEIGHT = 0.3
DEFAULT_EXPLANATION_WEIGHT = 0.2
DEFAULT_CALIBRATION_WEIGHT = 0.1
DEFAULT_TEMPERATURE_REGULARIZATION = 0.01

# Numerical Stability Constants
EPSILON = 1e-10  # Small value to prevent log(0) and division by zero
LOG_MIN_VALUE = 1e-8  # Minimum value for logarithms
SOFTMAX_TEMPERATURE = 1.0

# Logging Constants
DEFAULT_LOG_EVERY_N_STEPS = 100
DEFAULT_EVAL_EVERY_N_STEPS = 500
DEFAULT_SAVE_EVERY_N_STEPS = 1000
DEFAULT_MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_LOG_BACKUP_COUNT = 5

# Memory and Performance Constants
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_GPU_MEMORY_FRACTION = 0.9
MAX_SEQUENCE_LENGTH_HARD_LIMIT = 8192
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 256

# Validation Ranges
MIN_LEARNING_RATE = 1e-8
MAX_LEARNING_RATE = 1e-1
MIN_DROPOUT_RATE = 0.0
MAX_DROPOUT_RATE = 0.9
MIN_WEIGHT_DECAY = 0.0
MAX_WEIGHT_DECAY = 1.0
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 10.0

# String Constants
UNCERTAINTY_TYPES = ["knowledge_gap", "ambiguous", "reasoning_error"]
CONFIDENCE_LEVELS = ["VERY_UNCERTAIN", "UNCERTAIN", "SOMEWHAT_CONFIDENT", "CONFIDENT", "VERY_CONFIDENT"]
DEVICE_OPTIONS = ["auto", "cpu", "cuda", "mps"]
LR_SCHEDULER_OPTIONS = ["cosine", "linear", "reduce_on_plateau", "exponential"]
DOMAIN_SPLIT_STRATEGIES = ["balanced", "random", "stratified"]

# Special Tokens
UNCERTAINTY_TOKENS = [
    "[UNCERTAIN]", "[CONFIDENT]", "[KNOWLEDGE_GAP]",
    "[AMBIGUOUS]", "[REASONING_ERROR]", "[EXPLANATION]"
]

# File Extensions and Formats
SUPPORTED_CONFIG_FORMATS = [".yaml", ".yml", ".json"]
LOG_FILE_EXTENSION = ".log"
CHECKPOINT_FILE_EXTENSION = ".pt"
METRICS_FILE_EXTENSION = ".json"

# MLflow Constants
DEFAULT_ARTIFACT_PATH = "model"
DEFAULT_REGISTERED_MODEL_NAME = "metacognitive-uncertainty-model"

# Answer Choice Labels
ANSWER_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]  # Support up to 6 choices
ANSWER_CHOICE_ASCII_OFFSET = 65  # ASCII value of 'A'

# Prompt Templates
UNCERTAINTY_PROMPT_PREFIX = "Please answer the following question and indicate your uncertainty:"
EXPLANATION_PROMPT_SUFFIX = "Please explain your reasoning and uncertainty:"
FEW_SHOT_SEPARATOR = "\n---\n"

# Metric Thresholds
ECE_GOOD_THRESHOLD = 0.1      # Expected Calibration Error below this is considered good
ACCURACY_BASELINE = 0.25      # Random chance for 4-choice questions
UNCERTAINTY_CORRELATION_THRESHOLD = 0.5  # Minimum correlation for good calibration

# Hyperparameter Search Ranges
LEARNING_RATE_SEARCH_RANGE = (1e-6, 1e-3)
UNCERTAINTY_WEIGHT_SEARCH_RANGE = (0.1, 0.8)
EXPLANATION_WEIGHT_SEARCH_RANGE = (0.1, 0.5)
DROPOUT_SEARCH_RANGE = (0.0, 0.3)
BATCH_SIZE_SEARCH_OPTIONS = [8, 16, 32, 64]

# Memory Management
CUDA_MEMORY_CLEANUP_THRESHOLD = 0.8  # Cleanup when memory usage exceeds this fraction
CPU_MEMORY_WARNING_THRESHOLD = 0.9   # Warn when CPU memory exceeds this fraction

# Progress Reporting
PROGRESS_UPDATE_FREQUENCY = 10  # Update progress every N% of completion
TQDM_UPDATE_FREQUENCY = 1      # Update tqdm progress bar every N batches

# Data Validation
MAX_QUESTION_LENGTH = 2000     # Maximum characters in a question
MAX_CHOICE_LENGTH = 500        # Maximum characters in an answer choice
MIN_QUESTION_LENGTH = 10       # Minimum characters in a question
MIN_CHOICES_PER_QUESTION = 2   # Minimum number of answer choices
MAX_CHOICES_PER_QUESTION = 6   # Maximum number of answer choices

# Export Constants
ONNX_OPSET_VERSION = 11
TORCHSCRIPT_STRICT = False
MODEL_EXPORT_BATCH_SIZE = 1

# Regex Patterns
ANSWER_EXTRACTION_PATTERNS = [
    r'(?:answer|choice)\s*(?:is\s*)?([ABCDEF])\b',
    r'\b([ABCDEF])\)\s',
    r'\b([ABCDEF])\.\s',
    r'\b([ABCDEF])\b'
]

CONFIDENCE_EXTRACTION_PATTERNS = {
    "VERY_CONFIDENT": r"very\s+confident|extremely\s+confident|completely\s+sure",
    "CONFIDENT": r"confident|sure|certain",
    "SOMEWHAT_CONFIDENT": r"somewhat\s+confident|moderately\s+confident|fairly\s+sure",
    "UNCERTAIN": r"uncertain|unsure|not\s+sure|don't\s+know",
    "VERY_UNCERTAIN": r"very\s+uncertain|extremely\s+uncertain|no\s+idea"
}

EXPLANATION_EXTRACTION_PATTERNS = [
    r'(?:KNOWLEDGE_GAP|knowledge\s+gap):\s*(.+?)(?:\n|$)',
    r'(?:AMBIGUOUS|ambiguous):\s*(.+?)(?:\n|$)',
    r'(?:REASONING_ERROR|reasoning\s+error):\s*(.+?)(?:\n|$)',
    r'(?:EXPLANATION|explanation):\s*(.+?)(?:\n|$)',
    r'(?:because|since|reason):\s*(.+?)(?:\n|$)'
]

# Documentation Constants
DOCSTRING_MAX_LINE_LENGTH = 88
DOCUMENTATION_URL = "https://github.com/A-SHOJAEI/metacognitive-uncertainty-calibration-for-llm-knowledge-boundaries"
CITATION_DOI = ""

# Version Information
API_VERSION = "1.0.0"
CONFIG_VERSION = "1.0.0"
MINIMUM_PYTHON_VERSION = "3.8"
MINIMUM_TORCH_VERSION = "1.12.0"
MINIMUM_TRANSFORMERS_VERSION = "4.20.0"