"""
Full Decoder Stack for the Transformer model.

The decoder consists of:
    1. Input embeddings (token + positional)
    2. Stack of N decoder blocks with causal masking
    3. Final layer normalization
    4. Output projection to vocabulary

This is the core component for GPT-style decoder-only language models.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from model.embeddings import TransformerEmbedding
from model.decoder_block import DecoderBlock


class Decoder(nn.Module):
    """
    Transformer Decoder Stack.
    
    Combines embedding layer with multiple decoder blocks to create
    the complete decoder for autoregressive text generation.
    
    For GPT-style models, this is the entire model (no encoder).
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model
        n_heads: Number of attention heads
        n_layers: Number of decoder blocks
        d_ff: Dimension of the feed-forward hidden layer
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        activation: Activation function for FFN
        pre_norm: Whether to use pre-layer normalization
        use_cross_attention: Whether decoder blocks include cross-attention
        
    Example:
        >>> # GPT-style decoder (no encoder)
        >>> decoder = Decoder(vocab_size=10000, d_model=512, n_heads=8, n_layers=6, d_ff=2048)
        >>> tokens = torch.randint(0, 10000, (2, 50))
        >>> logits = decoder(tokens)
        >>> print(logits.shape)  # (2, 50, 10000)
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
        layer_norm_eps: float = 1e-6,
        use_cross_attention: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Embedding layer (token + positional)
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm,
                layer_norm_eps=layer_norm_eps,
                use_cross_attention=use_cross_attention
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization (for pre-norm)
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if pre_norm else None
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Pre-compute causal mask
        self._register_causal_mask(max_seq_len)
        
        # Initialize weights
        self._init_weights()
        
    def _register_causal_mask(self, max_seq_len: int):
        """Pre-compute and register the causal attention mask."""
        # Create upper triangular mask (True = masked)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        # Shape: (1, 1, max_seq_len, max_seq_len) for broadcasting
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))
        
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """
        Decode input sequence autoregressively.
        
        Args:
            tgt: Target token indices of shape (batch, seq)
            encoder_output: Optional encoder output for cross-attention
            cross_attn_mask: Optional mask for cross-attention
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Logits of shape (batch, seq, vocab_size)
                - List of (self_attn, cross_attn) tuples if return_attention=True
        """
        seq_len = tgt.size(1)
        attention_weights = [] if return_attention else None
        
        # Get the causal mask for the current sequence length
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Apply embeddings
        x = self.embedding(tgt)
        
        # Pass through decoder blocks
        for layer in self.layers:
            x, self_attn, cross_attn = layer(
                x,
                encoder_output=encoder_output,
                self_attn_mask=causal_mask,
                cross_attn_mask=cross_attn_mask,
                return_attention=return_attention
            )
            if return_attention:
                attention_weights.append((self_attn, cross_attn))
        
        # Apply final layer normalization
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits, attention_weights
    
    def get_embedding(self, tgt: torch.Tensor) -> torch.Tensor:
        """Get just the embeddings without passing through layers."""
        return self.embedding(tgt)
