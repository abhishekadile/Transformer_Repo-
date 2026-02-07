"""
Full Encoder Stack for the Transformer model.

The encoder consists of:
    1. Input embeddings (token + positional)
    2. Stack of N encoder blocks
    3. Optional final layer normalization

This module is used in encoder-decoder architectures (like the original Transformer).
For GPT-style decoder-only models, you would use the Decoder instead.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from model.embeddings import TransformerEmbedding
from model.encoder_block import EncoderBlock


class Encoder(nn.Module):
    """
    Transformer Encoder Stack.
    
    Combines embedding layer with multiple encoder blocks to create
    the complete encoder for processing input sequences.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model
        n_heads: Number of attention heads
        n_layers: Number of encoder blocks
        d_ff: Dimension of the feed-forward hidden layer
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        activation: Activation function for FFN
        pre_norm: Whether to use pre-layer normalization
        
    Example:
        >>> encoder = Encoder(vocab_size=10000, d_model=512, n_heads=8, n_layers=6, d_ff=2048)
        >>> tokens = torch.randint(0, 10000, (2, 50))  # (batch, seq)
        >>> output, attentions = encoder(tokens, return_attention=True)
        >>> print(output.shape)  # (2, 50, 512)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embedding layer (token + positional)
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Stack of encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization (only for pre-norm)
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if pre_norm else None
        
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Encode input sequence.
        
        Args:
            src: Source token indices of shape (batch, seq)
            mask: Optional padding mask of shape (batch, 1, 1, seq)
                 True values indicate positions to mask (not attend to)
            return_attention: Whether to return attention weights from all layers
            
        Returns:
            Tuple of:
                - Encoded representation of shape (batch, seq, d_model)
                - List of attention weights from each layer (if return_attention=True)
        """
        attention_weights = [] if return_attention else None
        
        # Apply embeddings
        x = self.embedding(src)
        
        # Pass through encoder blocks
        for layer in self.layers:
            x, attn = layer(x, mask=mask, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_weights.append(attn)
        
        # Apply final layer normalization
        if self.final_norm is not None:
            x = self.final_norm(x)
            
        return x, attention_weights
    
    def get_embedding(self, src: torch.Tensor) -> torch.Tensor:
        """Get just the embeddings without passing through layers."""
        return self.embedding(src)
