"""
Evaluation metrics for the Transformer Hackathon.

This module provides:
    - compute_perplexity: Calculate perplexity from loss
    - compute_accuracy: Calculate token-level accuracy
    - compute_distinct_n: Calculate generation diversity
    - TrainingMetrics: Track and aggregate training statistics
"""

import math
import time
from typing import List, Dict, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    
    Lower perplexity indicates better model performance.
    A perplexity of N means the model is "as confused as if it had to
    choose uniformly among N options."
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
        
    Example:
        >>> loss = 4.0  # Cross-entropy loss
        >>> ppl = compute_perplexity(loss)
        >>> print(f"Perplexity: {ppl:.2f}")  # ~54.60
    """
    return math.exp(loss)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        logits: Model output logits of shape (batch, seq, vocab)
        targets: Target token IDs of shape (batch, seq)
        ignore_index: Token ID to ignore in accuracy computation
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    predictions = logits.argmax(dim=-1)
    
    # Create mask for valid positions
    mask = targets != ignore_index
    
    # Count correct predictions
    correct = (predictions == targets) & mask
    
    # Calculate accuracy
    if mask.sum() == 0:
        return 0.0
    
    return (correct.sum() / mask.sum()).item()


def compute_distinct_n(
    tokens: List[int],
    n: int = 2
) -> float:
    """
    Compute Distinct-N metric for generation diversity.
    
    Distinct-N = (number of unique n-grams) / (total number of n-grams)
    
    Higher values indicate more diverse, less repetitive text.
    
    Args:
        tokens: List of generated token IDs
        n: Size of n-grams to consider
        
    Returns:
        Distinct-N score between 0 and 1
        
    Example:
        >>> tokens = [1, 2, 3, 4, 1, 2, 5, 6]
        >>> distinct_2 = compute_distinct_n(tokens, n=2)
        >>> print(f"Distinct-2: {distinct_2:.3f}")
    """
    if len(tokens) < n:
        return 0.0
    
    # Extract n-grams
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    if not ngrams:
        return 0.0
    
    # Count unique n-grams
    unique_ngrams = set(ngrams)
    
    return len(unique_ngrams) / len(ngrams)


def compute_repetition_rate(tokens: List[int], window: int = 50) -> float:
    """
    Compute repetition rate in generated text.
    
    Measures how often tokens repeat within a sliding window.
    Lower is better (less repetition).
    
    Args:
        tokens: List of generated token IDs
        window: Window size for checking repetition
        
    Returns:
        Repetition rate between 0 and 1
    """
    if len(tokens) < 2:
        return 0.0
    
    repeat_count = 0
    total_checks = 0
    
    for i in range(1, len(tokens)):
        window_start = max(0, i - window)
        if tokens[i] in tokens[window_start:i]:
            repeat_count += 1
        total_checks += 1
    
    return repeat_count / total_checks if total_checks > 0 else 0.0


@dataclass
class TrainingMetrics:
    """
    Track and aggregate training statistics.
    
    Tracks:
        - Loss values
        - Training speed (tokens/second)
        - GPU utilization (if available)
        - Time statistics
        
    Example:
        >>> metrics = TrainingMetrics()
        >>> # During training loop:
        >>> metrics.update(loss=2.5, batch_tokens=512, step_time=0.1)
        >>> print(metrics.get_summary())
    """
    
    # Accumulated values
    total_loss: float = 0.0
    total_tokens: int = 0
    total_steps: int = 0
    total_time: float = 0.0
    
    # Lists for tracking
    loss_history: List[float] = field(default_factory=list)
    tokens_per_sec_history: List[float] = field(default_factory=list)
    
    # Best values
    best_loss: float = float('inf')
    best_perplexity: float = float('inf')
    
    # Training start time
    start_time: Optional[float] = None
    
    def start(self):
        """Mark the start of training."""
        self.start_time = time.time()
    
    def update(
        self,
        loss: float,
        batch_tokens: int,
        step_time: float
    ):
        """
        Update metrics with new values from a training step.
        
        Args:
            loss: Loss value for this step
            batch_tokens: Number of tokens processed in this batch
            step_time: Time taken for this step in seconds
        """
        self.total_loss += loss
        self.total_tokens += batch_tokens
        self.total_steps += 1
        self.total_time += step_time
        
        # Update history
        self.loss_history.append(loss)
        tokens_per_sec = batch_tokens / step_time if step_time > 0 else 0
        self.tokens_per_sec_history.append(tokens_per_sec)
        
        # Update best values
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_perplexity = compute_perplexity(loss)
    
    def update_from_eval(self, eval_loss: float):
        """Update best values from evaluation."""
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.best_perplexity = compute_perplexity(eval_loss)
    
    @property
    def avg_loss(self) -> float:
        """Average loss over all steps."""
        return self.total_loss / max(1, self.total_steps)
    
    @property
    def avg_tokens_per_sec(self) -> float:
        """Average tokens per second."""
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time
    
    @property
    def perplexity(self) -> float:
        """Current perplexity based on average loss."""
        return compute_perplexity(self.avg_loss)
    
    @property
    def elapsed_time(self) -> float:
        """Time elapsed since training started."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_recent_avg(self, n: int = 100) -> Tuple[float, float]:
        """Get average loss and tokens/sec for recent steps."""
        recent_loss = self.loss_history[-n:] if self.loss_history else [0.0]
        recent_tps = self.tokens_per_sec_history[-n:] if self.tokens_per_sec_history else [0.0]
        
        avg_loss = sum(recent_loss) / len(recent_loss)
        avg_tps = sum(recent_tps) / len(recent_tps)
        
        return avg_loss, avg_tps
    
    def get_summary(self) -> Dict:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with all summary statistics
        """
        return {
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "total_time_min": self.total_time / 60,
            "avg_loss": self.avg_loss,
            "best_loss": self.best_loss,
            "perplexity": self.perplexity,
            "best_perplexity": self.best_perplexity,
            "avg_tokens_per_sec": self.avg_tokens_per_sec,
        }
    
    def __str__(self) -> str:
        """Pretty string representation."""
        summary = self.get_summary()
        lines = [
            "=" * 50,
            "Training Metrics Summary",
            "=" * 50,
            f"Steps: {summary['total_steps']:,}",
            f"Tokens: {summary['total_tokens']:,}",
            f"Time: {summary['total_time_min']:.1f} minutes",
            f"",
            f"Average Loss: {summary['avg_loss']:.4f}",
            f"Best Loss: {summary['best_loss']:.4f}",
            f"Perplexity: {summary['perplexity']:.2f}",
            f"Best Perplexity: {summary['best_perplexity']:.2f}",
            f"",
            f"Tokens/Second: {summary['avg_tokens_per_sec']:.1f}",
            "=" * 50,
        ]
        return "\n".join(lines)


def get_gpu_utilization() -> Optional[Dict]:
    """
    Get GPU utilization statistics.
    
    Returns:
        Dictionary with GPU stats, or None if not available
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(',')
            return {
                'gpu_utilization': float(gpu_util),
                'memory_used_gb': float(mem_used) / 1024,
                'memory_total_gb': float(mem_total) / 1024,
                'memory_percent': float(mem_used) / float(mem_total) * 100
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    
    # Fallback to PyTorch memory info
    try:
        return {
            'memory_used_gb': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        }
    except Exception:
        return None
