"""
Checkpoint utilities for the Transformer Hackathon.

This module provides:
    - save_checkpoint: Save model and training state
    - load_checkpoint: Load model and training state
    - get_latest_checkpoint: Find the most recent checkpoint
    - cleanup_old_checkpoints: Remove old checkpoints to save space
"""

import os
import glob
import shutil
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    loss: float,
    metrics: Dict[str, Any],
    config: Any,
    checkpoint_dir: str = "checkpoints",
    filename: Optional[str] = None,
    tokenizer: Optional[Any] = None
) -> str:
    """
    Save a training checkpoint.
    
    Saves:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict (if provided)
        - Training progress (epoch, step, loss)
        - Metrics and configuration
        - Tokenizer (if provided)
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        scheduler: Optional learning rate scheduler
        epoch: Current epoch number
        step: Current training step
        loss: Current loss value
        metrics: Dictionary of training metrics
        config: Configuration object
        checkpoint_dir: Directory to save checkpoints
        filename: Optional specific filename (otherwise auto-generated)
        tokenizer: Optional tokenizer to save
        
    Returns:
        Path to the saved checkpoint
        
    Example:
        >>> save_checkpoint(
        ...     model, optimizer, scheduler,
        ...     epoch=1, step=1000, loss=2.5,
        ...     metrics={"perplexity": 12.18},
        ...     config=config
        ... )
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
    
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Prepare checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'metrics': metrics,
        'config': config,
    }
    
    # Add scheduler if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add tokenizer if provided
    if tokenizer is not None:
        checkpoint['tokenizer'] = {
            'char_to_id': tokenizer.char_to_id if hasattr(tokenizer, 'char_to_id') else None,
            'word_to_id': tokenizer.word_to_id if hasattr(tokenizer, 'word_to_id') else None,
            'vocab_size': tokenizer.vocab_size,
        }
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    # Also save as 'latest.pt' for easy access
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    shutil.copy(filepath, latest_path)
    
    return filepath


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Tuple[int, int, float, Dict[str, Any]]:
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Tuple of (epoch, step, loss, metrics)
        
    Example:
        >>> epoch, step, loss, metrics = load_checkpoint(
        ...     "checkpoints/latest.pt",
        ...     model, optimizer, scheduler
        ... )
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state loaded")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")
    
    # Extract training progress
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    metrics = checkpoint.get('metrics', {})
    
    print(f"Resumed from epoch {epoch}, step {step}, loss {loss:.4f}")
    
    return epoch, step, loss, metrics


def get_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """
    Find the most recent checkpoint file.
    
    Looks for 'latest.pt' first, then finds the checkpoint with
    the highest step number.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoints exist
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Check for latest.pt first
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    
    # Find all checkpoint files
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Sort by modification time (most recent last)
    checkpoints.sort(key=os.path.getmtime)
    
    return checkpoints[-1]


def cleanup_old_checkpoints(
    checkpoint_dir: str = "checkpoints",
    keep_last_n: int = 3,
    keep_best: bool = True
) -> int:
    """
    Remove old checkpoints to save disk space.
    
    Keeps the N most recent checkpoints and optionally the best checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
        
    Returns:
        Number of checkpoints deleted
    """
    if not os.path.exists(checkpoint_dir):
        return 0
    
    # Find all checkpoint files (excluding latest.pt and best.pt)
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    checkpoints = glob.glob(pattern)
    
    if len(checkpoints) <= keep_last_n:
        return 0
    
    # Sort by modification time (oldest first)
    checkpoints.sort(key=os.path.getmtime)
    
    # Determine which to keep
    protected = set()
    
    # Keep the N most recent
    for path in checkpoints[-keep_last_n:]:
        protected.add(path)
    
    # Keep best.pt reference if it exists
    if keep_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(best_path):
            protected.add(best_path)
    
    # Delete old checkpoints
    deleted = 0
    for path in checkpoints:
        if path not in protected:
            try:
                os.remove(path)
                deleted += 1
                print(f"Deleted old checkpoint: {path}")
            except OSError as e:
                print(f"Error deleting {path}: {e}")
    
    return deleted


def save_best_checkpoint(
    current_loss: float,
    best_loss: float,
    model: nn.Module,
    checkpoint_dir: str = "checkpoints"
) -> Tuple[bool, float]:
    """
    Save checkpoint if current loss is the best so far.
    
    Args:
        current_loss: Current validation loss
        best_loss: Best loss so far
        model: Model to save
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Tuple of (is_best, new_best_loss)
    """
    if current_loss < best_loss:
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': current_loss
        }, best_path)
        print(f"New best model saved with loss {current_loss:.4f}")
        return True, current_loss
    
    return False, best_loss


def get_checkpoint_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.
    
    Args:
        filepath: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    info = {
        'filepath': filepath,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'step': checkpoint.get('step', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
        'metrics': checkpoint.get('metrics', {}),
    }
    
    # Add file size
    info['size_mb'] = os.path.getsize(filepath) / (1024 * 1024)
    
    return info
