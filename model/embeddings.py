"""
Embedding layers for the Transformer model.

This module implements:
    - TokenEmbedding: Learnable embeddings for input tokens
    - PositionalEncoding: Sinusoidal position encodings from "Attention is All You Need"
    - TransformerEmbedding: Combined token + positional embeddings with dropout

The embeddings convert discrete token IDs into continuous vector representations
that the transformer can process.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that maps token indices to dense vectors.
    
    This is a standard learnable embedding table where each token in the
    vocabulary gets its own d_model-dimensional vector.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the embedding vectors
        
    Example:
        >>> embed = TokenEmbedding(vocab_size=1000, d_model=512)
        >>> tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (batch=2, seq=3)
        >>> embeddings = embed(tokens)  # (batch=2, seq=3, d_model=512)
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to embeddings.
        
        Args:
            x: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) as in the original paper
        # This helps maintain variance during training
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".
    
    Since the transformer has no recurrence or convolution, we need to inject
    information about the position of tokens in the sequence. This uses
    fixed sinusoidal functions of different frequencies:
    
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Dimension of the model
        max_seq_len: Maximum sequence length to support
        dropout: Dropout probability (default: 0.1)
        
    TODO: You could try learnable positional embeddings instead!
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_seq_len: int = 512, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        # Shape: (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        
        # Position indices: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the divisor term: 10000^(2i/d_model)
        # Using log for numerical stability: exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings with positional encoding added, same shape as input
        """
        seq_len = x.size(1)
        # Add positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learnable positional embeddings (alternative to sinusoidal).
    
    Instead of fixed sinusoidal patterns, this learns a separate embedding
    vector for each position. This is what GPT-2 and many modern models use.
    
    Args:
        d_model: Dimension of the model
        max_seq_len: Maximum sequence length to support
        dropout: Dropout probability
        
    TODO: Try this instead of sinusoidal encoding!
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Register position indices as a buffer
        positions = torch.arange(max_seq_len)
        self.register_buffer('positions', positions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings with positional encoding added
        """
        seq_len = x.size(1)
        positions = self.positions[:seq_len]
        pos_embeddings = self.position_embedding(positions)
        x = x + pos_embeddings
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Combined embedding layer for transformers.
    
    Combines token embeddings with positional encoding and applies dropout.
    This is the complete input processing for the transformer.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        use_learned_pos: Whether to use learned positional embeddings
        
    Example:
        >>> embed = TransformerEmbedding(vocab_size=1000, d_model=512, max_seq_len=128)
        >>> tokens = torch.tensor([[1, 2, 3, 4, 5]])  # (batch=1, seq=5)
        >>> embeddings = embed(tokens)  # (batch=1, seq=5, d_model=512)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_learned_pos: bool = False
    ):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        if use_learned_pos:
            self.positional_encoding = LearnedPositionalEncoding(
                d_model, max_seq_len, dropout
            )
        else:
            self.positional_encoding = PositionalEncoding(
                d_model, max_seq_len, dropout
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to full embeddings with position information.
        
        Args:
            x: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Complete embeddings of shape (batch_size, seq_len, d_model)
        """
        # Get token embeddings (already scaled by sqrt(d_model))
        tok_emb = self.token_embedding(x)
        
        # Add positional encoding
        return self.positional_encoding(tok_emb)
