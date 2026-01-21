"""
Utilities package for the Transformer Hackathon.

This package contains helper modules:
    - metrics: Evaluation metrics (perplexity, accuracy, etc.)
    - checkpoint: Model saving and loading
    - huggingface_upload: Leaderboard integration
"""

from utils.metrics import (
    compute_perplexity,
    compute_accuracy,
    compute_distinct_n,
    TrainingMetrics
)
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    cleanup_old_checkpoints
)
from utils.huggingface_upload import (
    upload_to_leaderboard,
    fetch_leaderboard,
    display_leaderboard,
    HuggingFaceUploader
)

__all__ = [
    # Metrics
    "compute_perplexity",
    "compute_accuracy",
    "compute_distinct_n",
    "TrainingMetrics",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "get_latest_checkpoint",
    "cleanup_old_checkpoints",
    # HuggingFace
    "upload_to_leaderboard",
    "fetch_leaderboard",
    "display_leaderboard",
    "HuggingFaceUploader",
]
