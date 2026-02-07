#!/usr/bin/env python3
"""
Training script for the Transformer Hackathon.

This script trains a GPT-style transformer model on text data with:
    - 45-minute training timer with countdown
    - Rich progress bar (loss, tokens/sec, ETA)
    - Automatic checkpointing every 5 minutes
    - GPU utilization monitoring
    - Graceful CPU fallback

Usage:
    python train.py                    # Train with defaults
    python train.py --max-time 10      # Train for 10 minutes
    python train.py --resume           # Resume from checkpoint
    python train.py --batch-size 32    # Custom batch size

TODO: Experiment with these optimizations:
    - Mixed precision training (--use-amp)
    - Gradient accumulation (--grad-accum-steps)
    - Different learning rates (--lr)
"""

import argparse
import os
import sys
import time
import math
from typing import Optional, Tuple
from datetime import timedelta

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, default_config
from model import GPTModel
from data import create_dataloaders
from utils.metrics import TrainingMetrics, compute_perplexity, get_gpu_utilization
from utils.checkpoint import (
    save_checkpoint, 
    load_checkpoint, 
    get_latest_checkpoint,
    cleanup_old_checkpoints
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a GPT-style transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training settings
    parser.add_argument("--max-time", type=float, default=45.0,
                       help="Maximum training time in minutes")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--seq-len", type=int, default=128,
                       help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    
    # Model settings
    parser.add_argument("--d-model", type=int, default=512,
                       help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=8,
                       help="Number of attention heads")
    
    # Optimization
    parser.add_argument("--use-amp", action="store_true",
                       help="Use automatic mixed precision (FP16)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--checkpoint-interval", type=float, default=5.0,
                       help="Checkpoint interval in minutes")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from latest checkpoint")
    
    # Data
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to training data (downloads TinyShakespeare if not provided)")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Steps between logging")
    parser.add_argument("--eval-interval", type=int, default=500,
                       help="Steps between evaluation")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")
    
    return parser.parse_args()


def get_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Create learning rate scheduler with warmup."""
    
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "00:00"
    td = timedelta(seconds=int(seconds))
    if seconds >= 3600:
        return str(td)
    else:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def print_progress(
    step: int,
    loss: float,
    tokens_per_sec: float,
    lr: float,
    elapsed: float,
    remaining: float,
    gpu_util: Optional[float] = None
):
    """Print training progress."""
    elapsed_str = format_time(elapsed)
    remaining_str = format_time(remaining)
    ppl = compute_perplexity(loss)
    
    gpu_str = f" | GPU: {gpu_util:.0f}%" if gpu_util is not None else ""
    
    print(f"\rStep {step:,} | Loss: {loss:.4f} | PPL: {ppl:.2f} | "
          f"Tok/s: {tokens_per_sec:.0f} | LR: {lr:.2e} | "
          f"Elapsed: {elapsed_str} | ETA: {remaining_str}{gpu_str}  ", end="")


def evaluate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    use_amp: bool = False,
    max_batches: Optional[int] = 200
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.
    
    Args:
        max_batches: Limit number of batches for faster evaluation (None for all)
        
    Returns:
        Tuple of (average loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Progress indicator
    print(f"   Evaluating on {max_batches if max_batches else 'all'} batches...", end="")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break
                
            x, y = x.to(device), y.to(device)
            
            with autocast(enabled=use_amp):
                output = model(x, labels=y)
                loss = output.loss
            
            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            
            if i % 50 == 0:
                print(".", end="", flush=True)
    
    print() # Newline
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = compute_perplexity(avg_loss)
    
    model.train()
    return avg_loss, perplexity


def train(args):
    """Main training function."""
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU available, using CPU (training will be slow)")
    
    # Create dataloaders
    print("\n📚 Loading data...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    vocab_size = tokenizer.vocab_size
    print(f"   Vocabulary size: {vocab_size}")
    
    # Create model
    print("\n🏗️  Building model...")
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_len,
        dropout=0.1
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    # Estimate total steps
    steps_per_epoch = len(train_loader)
    estimated_epochs = (args.max_time * 60) / (steps_per_epoch * 0.1)  # Rough estimate
    total_steps = int(steps_per_epoch * estimated_epochs)
    warmup_steps = int(total_steps * 0.1)
    
    # Create scheduler
    scheduler = get_lr_scheduler(optimizer, total_steps, warmup_steps)
    
    # Mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    if args.use_amp:
        print("   Using mixed precision (FP16)")
    
    # Resume from checkpoint if requested
    start_step = 0
    start_epoch = 0
    if args.resume:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path:
            start_epoch, start_step, _, _ = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
        else:
            print("No checkpoint found, starting from scratch")
    
    # Training metrics
    metrics = TrainingMetrics()
    metrics.start()
    
    # Save tokenizer
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.checkpoint_dir, "tokenizer.json"))
    
    # Training loop
    print(f"\n🏋️  Starting training for {args.max_time} minutes...")
    print("=" * 70)
    
    max_time_seconds = args.max_time * 60
    checkpoint_interval_seconds = args.checkpoint_interval * 60
    last_checkpoint_time = time.time()
    
    model.train()
    step = start_step
    epoch = start_epoch
    best_val_loss = float('inf')
    
    training_start = time.time()
    
    try:
        while True:
            epoch += 1
            
            for batch_idx, (x, y) in enumerate(train_loader):
                step_start = time.time()
                
                # Check time limit
                elapsed = time.time() - training_start
                if elapsed >= max_time_seconds:
                    raise StopIteration("Time limit reached")
                
                step += 1
                
                # Move to device
                x, y = x.to(device), y.to(device)
                
                # Forward pass with mixed precision
                with autocast(enabled=args.use_amp):
                    output = model(x, labels=y)
                    loss = output.loss / args.grad_accum_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if step % args.grad_accum_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                # Update metrics
                step_time = time.time() - step_start
                batch_tokens = x.numel()
                metrics.update(loss.item() * args.grad_accum_steps, batch_tokens, step_time)
                
                # Logging
                if step % args.log_interval == 0:
                    remaining = max_time_seconds - elapsed
                    _, avg_tps = metrics.get_recent_avg(args.log_interval)
                    gpu_util = None
                    if device.type == 'cuda':
                        gpu_info = get_gpu_utilization()
                        if gpu_info:
                            gpu_util = gpu_info.get('gpu_utilization')
                    
                    print_progress(
                        step=step,
                        loss=metrics.avg_loss,
                        tokens_per_sec=avg_tps,
                        lr=scheduler.get_last_lr()[0],
                        elapsed=elapsed,
                        remaining=remaining,
                        gpu_util=gpu_util
                    )
                
                # Evaluation
                if step % args.eval_interval == 0:
                    print("\n")
                    print("📊 Evaluating...")
                    val_loss, val_ppl = evaluate(
                        model, val_loader, device, 
                        args.use_amp, max_batches=200
                    )
                    print(f"   Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
                    
                    metrics.update_from_eval(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, optimizer, scheduler,
                            epoch, step, val_loss,
                            metrics.get_summary(),
                            args.__dict__,
                            args.checkpoint_dir,
                            filename="best.pt",
                            tokenizer=tokenizer
                        )
                    print()
                
                # Periodic checkpointing
                current_time = time.time()
                if current_time - last_checkpoint_time >= checkpoint_interval_seconds:
                    print("\n💾 Saving checkpoint...")
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, step, metrics.avg_loss,
                        metrics.get_summary(),
                        args.__dict__,
                        args.checkpoint_dir,
                        tokenizer=tokenizer
                    )
                    cleanup_old_checkpoints(args.checkpoint_dir, keep_last_n=3)
                    last_checkpoint_time = current_time
                    print()
    
    except (StopIteration, KeyboardInterrupt) as e:
        print(f"\n\n⏱️  Training stopped: {e if isinstance(e, StopIteration) else 'Interrupted'}")
    
    # Final checkpoint
    print("\n💾 Saving final checkpoint...")
    save_checkpoint(
        model, optimizer, scheduler,
        epoch, step, metrics.avg_loss,
        metrics.get_summary(),
        args.__dict__,
        args.checkpoint_dir,
        filename="final.pt",
        tokenizer=tokenizer
    )
    
    # Final evaluation
    print("\n📊 Final evaluation...")
    val_loss, val_ppl = evaluate(model, val_loader, device, args.use_amp)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(metrics)
    print(f"\nFinal Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Perplexity: {val_ppl:.2f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 70)
    
    return model, tokenizer, metrics


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer, metrics = train(args)
