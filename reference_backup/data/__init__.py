"""
Data package for the Transformer Hackathon.

This package contains data loading and processing utilities:
    - tokenizer: Simple character or BPE tokenizer
    - dataset: PyTorch Dataset for language modeling
"""

from data.tokenizer import SimpleTokenizer, CharTokenizer
from data.dataset import TextDataset, create_dataloaders, download_tiny_shakespeare, download_tinystories

__all__ = [
    "SimpleTokenizer",
    "CharTokenizer",
    "TextDataset",
    "create_dataloaders",
    "download_tiny_shakespeare",
]
