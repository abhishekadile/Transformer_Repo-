"""
Single Encoder Block for the Transformer model.

An encoder block consists of:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual connection + layer normalization)
    3. Position-wise Feed-Forward Network
    4. Add & Norm (residual connection + layer normalization)

This module is used to build the full encoder stack.
Note: For GPT-style (decoder-only) models, you mainly need the decoder block,
but the encoder is included for encoder-decoder architectures.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from model.attention import MultiHeadAttention
from model.feedforward import PositionwiseFeedForward


class EncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.
    
    Implements the standard encoder layer with:
    - Multi-head self-attention with residual connection
    - Feed-forward network with residual connection
    - Layer normalization (pre-norm or post-norm)
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of the feed-forward hidden layer
        dropout: Dropout probability
        activation: Activation function for FFN ("gelu" or "relu")
        pre_norm: Whether to use pre-layer normalization (more stable)
        
    Example:
        >>> block = EncoderBlock(d_model=512, n_heads=8, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq, d_model)
        >>> output, attn = block(x)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        self.pre_norm = pre_norm
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the encoder block.
        
        Args:
            x: Input tensor of shape (batch, seq, d_model)
            mask: Optional attention mask for padding
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq, d_model)
                - Attention weights if return_attention=True, else None
        """
        if self.pre_norm:
            # Pre-Layer Normalization (more stable training)
            # LN -> Attention -> Residual
            attn_output, attn_weights = self._self_attention_block(
                self.norm1(x), mask, return_attention
            )
            x = x + attn_output
            
            # LN -> FFN -> Residual
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-Layer Normalization (original Transformer)
            # Attention -> Residual -> LN
            attn_output, attn_weights = self._self_attention_block(
                x, mask, return_attention
            )
            x = self.norm1(x + attn_output)
            
            # FFN -> Residual -> LN
            x = self.norm2(x + self._ff_block(x))
            
        return x, attn_weights
    
    def _self_attention_block(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_attention: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply self-attention with dropout."""
        attn_output, attn_weights = self.self_attention(
            x, x, x, mask=mask, return_attention=return_attention
        )
        return self.dropout(attn_output), attn_weights
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network with dropout."""
        return self.dropout(self.feed_forward(x))
