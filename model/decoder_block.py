"""
Single Decoder Block for the Transformer model.

A decoder block consists of:
    1. Masked Multi-Head Self-Attention (causal masking for autoregressive generation)
    2. Add & Norm
    3. (Optional) Cross-Attention to encoder output
    4. Add & Norm
    5. Position-wise Feed-Forward Network
    6. Add & Norm

For GPT-style decoder-only models, we skip the cross-attention step.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from model.attention import MultiHeadAttention
from model.feedforward import PositionwiseFeedForward


class DecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block.
    
    Implements the decoder layer with:
    - Masked self-attention (causal, for autoregressive generation)
    - Optional cross-attention to encoder output (for encoder-decoder models)
    - Feed-forward network
    - Residual connections and layer normalization
    
    For GPT-style models, set use_cross_attention=False.
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of the feed-forward hidden layer
        dropout: Dropout probability
        activation: Activation function for FFN
        pre_norm: Whether to use pre-layer normalization
        use_cross_attention: Whether to include cross-attention layer
        
    Example:
        >>> # GPT-style (decoder-only)
        >>> block = DecoderBlock(d_model=512, n_heads=8, d_ff=2048, use_cross_attention=False)
        >>> x = torch.randn(2, 10, 512)
        >>> output, self_attn, _ = block(x)
        
        >>> # Encoder-decoder style
        >>> block = DecoderBlock(d_model=512, n_heads=8, d_ff=2048, use_cross_attention=True)
        >>> enc_output = torch.randn(2, 20, 512)
        >>> output, self_attn, cross_attn = block(x, encoder_output=enc_output)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
        layer_norm_eps: float = 1e-6,
        use_cross_attention: bool = False
    ):
        super().__init__()
        
        self.pre_norm = pre_norm
        self.use_cross_attention = use_cross_attention
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention (optional, for encoder-decoder models)
        if use_cross_attention:
            self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
            self.norm_cross = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
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
        encoder_output: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the decoder block.
        
        Args:
            x: Input tensor of shape (batch, seq, d_model)
            encoder_output: Encoder output for cross-attention (batch, enc_seq, d_model)
            self_attn_mask: Mask for self-attention (causal mask)
            cross_attn_mask: Mask for cross-attention (padding mask)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq, d_model)
                - Self-attention weights if return_attention=True
                - Cross-attention weights if return_attention=True and use_cross_attention=True
        """
        self_attn_weights = None
        cross_attn_weights = None
        
        if self.pre_norm:
            # Pre-Layer Normalization
            
            # 1. Masked Self-Attention
            attn_out, self_attn_weights = self._self_attention_block(
                self.norm1(x), self_attn_mask, return_attention
            )
            x = x + attn_out
            
            # 2. Cross-Attention (optional)
            if self.use_cross_attention and encoder_output is not None:
                cross_out, cross_attn_weights = self._cross_attention_block(
                    self.norm_cross(x), encoder_output, cross_attn_mask, return_attention
                )
                x = x + cross_out
            
            # 3. Feed-Forward
            x = x + self._ff_block(self.norm2(x))
            
        else:
            # Post-Layer Normalization
            
            # 1. Masked Self-Attention
            attn_out, self_attn_weights = self._self_attention_block(
                x, self_attn_mask, return_attention
            )
            x = self.norm1(x + attn_out)
            
            # 2. Cross-Attention (optional)
            if self.use_cross_attention and encoder_output is not None:
                cross_out, cross_attn_weights = self._cross_attention_block(
                    x, encoder_output, cross_attn_mask, return_attention
                )
                x = self.norm_cross(x + cross_out)
            
            # 3. Feed-Forward
            x = self.norm2(x + self._ff_block(x))
            
        return x, self_attn_weights, cross_attn_weights
    
    def _self_attention_block(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_attention: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply masked self-attention with dropout."""
        attn_output, attn_weights = self.self_attention(
            x, x, x, mask=mask, return_attention=return_attention
        )
        return self.dropout(attn_output), attn_weights
    
    def _cross_attention_block(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_attention: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply cross-attention to encoder output with dropout."""
        attn_output, attn_weights = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=mask,
            return_attention=return_attention
        )
        return self.dropout(attn_output), attn_weights
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network with dropout."""
        return self.dropout(self.feed_forward(x))
