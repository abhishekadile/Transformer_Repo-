"""
Position-wise Feed-Forward Network for the Transformer model.

This module implements the FFN sublayer that appears in each transformer block.
It applies two linear transformations with a non-linearity in between:

    FFN(x) = W2 * activation(W1 * x + b1) + b2

The FFN is applied to each position independently (hence "position-wise"),
which helps the model learn complex feature transformations.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Consists of two linear layers with an activation function in between.
    The inner dimension (d_ff) is typically 4x the model dimension.
    
    Args:
        d_model: Dimension of the model (input and output)
        d_ff: Dimension of the inner layer (typically 4 * d_model)
        dropout: Dropout probability
        activation: Activation function ("gelu" or "relu")
        
    TODO: Try different activation functions! 
          - GELU (default, smoother)
          - ReLU (classic)
          - SwiGLU (used in LLaMA, more parameters but better)
    
    Example:
        >>> ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq, d_model)
        >>> output = ffn(x)  # (batch, seq, d_model)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu"
    ):
        super().__init__()
        
        # First linear layer: d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch, seq, d_model)
            
        Returns:
            Output tensor of shape (batch, seq, d_model)
        """
        # Apply first linear + activation
        hidden = self.activation(self.linear1(x))
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Apply second linear
        output = self.linear2(hidden)
        
        return output


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network (used in LLaMA and other modern models).
    
    SwiGLU is a gated linear unit variant that often performs better than
    standard FFN, though it has ~33% more parameters:
    
        SwiGLU(x) = (x W1 * Swish(x W_gate)) W2
        
    where Swish(x) = x * sigmoid(x)
    
    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the inner layer
        dropout: Dropout probability
        
    TODO: This is an advanced optimization - try it if you have extra time!
    
    Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # For SwiGLU, we typically use 2/3 of d_ff for each of the two projections
        # to keep parameter count similar to standard FFN
        hidden_dim = int(2 * d_ff / 3)
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward.
        
        Args:
            x: Input tensor of shape (batch, seq, d_model)
            
        Returns:
            Output tensor of shape (batch, seq, d_model)
        """
        # Compute gated linear unit
        gate = F.silu(self.w_gate(x))  # Swish activation
        hidden = self.w1(x) * gate
        
        # Apply dropout and output projection
        hidden = self.dropout(hidden)
        output = self.w2(hidden)
        
        return output
