"""
Multi-Head Self-Attention mechanism for the Transformer model.

This module implements:
    - ScaledDotProductAttention: Core attention computation
    - MultiHeadAttention: Multi-head attention with projections

Attention allows the model to focus on different parts of the input sequence
when computing representations. Multi-head attention does this in parallel
across multiple "heads", each learning different attention patterns.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    Computes attention weights and applies them to values:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    The scaling by sqrt(d_k) prevents the dot products from becoming too large,
    which would push softmax into regions with very small gradients.
    
    Args:
        dropout: Dropout probability for attention weights
        
    Example:
        >>> attn = ScaledDotProductAttention(dropout=0.1)
        >>> q = k = v = torch.randn(2, 8, 10, 64)  # (batch, heads, seq, d_k)
        >>> output, weights = attn(q, k, v)
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Uses PyTorch's optimized scaled_dot_product_attention (Flash Attention)
        when available for 2-3x speedup and lower memory usage.
        
        Args:
            query: Query tensor of shape (batch, heads, seq_q, d_k)
            key: Key tensor of shape (batch, heads, seq_k, d_k)
            value: Value tensor of shape (batch, heads, seq_k, d_v)
            mask: Optional attention mask. True/1 values are MASKED (not attended to).
                  Shape can be (batch, 1, 1, seq_k) or (batch, 1, seq_q, seq_k)
                  
        Returns:
            Tuple of:
                - Output tensor of shape (batch, heads, seq_q, d_v)
                - Attention weights of shape (batch, heads, seq_q, seq_k)
        """
        # Try to use PyTorch's optimized Flash Attention (available in PyTorch 2.0+)
        # This is 2-3x faster and uses less memory!
        if hasattr(F, 'scaled_dot_product_attention') and self.training:
            # Convert mask format: True -> mask out, need attn_mask where True = attend
            attn_mask = None
            if mask is not None:
                # Invert mask: True (masked) -> False (don't attend)
                attn_mask = ~mask if mask.dtype == torch.bool else (mask == 0)
            
            # Use optimized kernel (Flash Attention 2 if available)
            output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False  # We handle causality via mask
            )
            
            # For compatibility, return dummy attention weights during training
            # (Flash Attention doesn't return weights for efficiency)
            attention_weights = None
            
            return output, attention_weights
        
        else:
            # Fallback to manual implementation (for inference or older PyTorch)
            d_k = query.size(-1)
            
            # Compute attention scores: QK^T / sqrt(d_k)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            
            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == True, float('-inf'))
            
            # Convert to probabilities
            attention_weights = F.softmax(scores, dim=-1)
            
            # Apply dropout
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            output = torch.matmul(attention_weights, value)
            
            return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Instead of performing a single attention function, we project the queries,
    keys and values h times with different learned linear projections.
    We then perform attention in parallel on each of these projected versions,
    concatenate, and project once more.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        dropout: Dropout probability
        
    TODO: For better performance, try using torch's built-in scaled_dot_product_attention!
          It can use Flash Attention under the hood.
    
    Example:
        >>> mha = MultiHeadAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq, d_model)
        >>> output, weights = mha(x, x, x)  # Self-attention
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        # We use a single matrix for efficiency, then split
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        # For storing attention weights (useful for visualization)
        self.attn_weights: Optional[torch.Tensor] = None
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor of shape (batch, seq_q, d_model)
            key: Key tensor of shape (batch, seq_k, d_model)
            value: Value tensor of shape (batch, seq_k, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_q, d_model)
                - Attention weights if return_attention=True, else None
        """
        batch_size = query.size(0)
        
        # 1. Linear projections
        # (batch, seq, d_model) -> (batch, seq, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # 2. Reshape to multiple heads
        # (batch, seq, d_model) -> (batch, seq, n_heads, d_k) -> (batch, n_heads, seq, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply attention
        # Output: (batch, n_heads, seq_q, d_k)
        attn_output, attn_weights = self.attention(q, k, v, mask)
        
        # Store attention weights for visualization
        if return_attention:
            self.attn_weights = attn_weights
        
        # 4. Concatenate heads
        # (batch, n_heads, seq_q, d_k) -> (batch, seq_q, n_heads, d_k) -> (batch, seq_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 5. Final linear projection
        output = self.w_o(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive models.
    
    This is a convenience wrapper that automatically applies a causal mask,
    ensuring each position can only attend to previous positions (and itself).
    This is essential for GPT-style language models.
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length (for pre-computing mask)
        dropout: Dropout probability
        
    Example:
        >>> attn = CausalSelfAttention(d_model=512, n_heads=8, max_seq_len=128)
        >>> x = torch.randn(2, 10, 512)
        >>> output, _ = attn(x)  # Mask automatically applied
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Pre-compute causal mask
        # Upper triangular matrix of True values (positions to mask)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        # Shape: (1, 1, max_seq_len, max_seq_len) for broadcasting
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply causal self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq, d_model)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of output and optional attention weights
        """
        seq_len = x.size(1)
        # Get the appropriate slice of the pre-computed mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        return self.mha(x, x, x, mask=mask, return_attention=return_attention)
