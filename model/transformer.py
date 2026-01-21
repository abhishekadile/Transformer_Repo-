"""
Complete GPT-style Transformer Model.

This module brings together all the components to create a complete
decoder-only transformer model for text generation.

The GPTModel class:
    - Wraps the decoder stack
    - Provides convenient methods for generation
    - Handles sampling strategies (greedy, top-k, top-p)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from dataclasses import dataclass

from model.decoder import Decoder


@dataclass
class GPTOutput:
    """Output container for GPT model."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    attention_weights: Optional[list] = None


class GPTModel(nn.Module):
    """
    GPT-style Decoder-Only Transformer for Text Generation.
    
    A complete language model that can:
    - Compute next-token probabilities given a context
    - Generate text autoregressively with various sampling strategies
    - Be trained with cross-entropy loss on sequences
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model (default: 512)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of decoder blocks (default: 6)
        d_ff: Dimension of feed-forward layer (default: 2048)
        max_seq_len: Maximum sequence length (default: 128)
        dropout: Dropout probability (default: 0.1)
        activation: Activation function ("gelu" or "relu")
        pre_norm: Whether to use pre-layer normalization
        
    Example:
        >>> model = GPTModel(vocab_size=5000, d_model=512, n_heads=8, n_layers=6)
        >>> tokens = torch.randint(0, 5000, (2, 50))
        >>> output = model(tokens)
        >>> print(output.logits.shape)  # (2, 50, 5000)
        
        >>> # Generate text
        >>> prompt = torch.tensor([[1, 2, 3]])  # Start tokens
        >>> generated = model.generate(prompt, max_new_tokens=50)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # The decoder is the complete model for GPT-style architecture
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            activation=activation,
            pre_norm=pre_norm,
            layer_norm_eps=layer_norm_eps,
            use_cross_attention=False  # GPT-style, no encoder
        )
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> GPTOutput:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token indices of shape (batch, seq)
            labels: Optional target labels for computing loss (batch, seq)
                   Typically this is input_ids shifted by one position
            return_attention: Whether to return attention weights
            
        Returns:
            GPTOutput containing logits, optional loss, and optional attention weights
        """
        # Get logits from decoder
        logits, attention_weights = self.decoder(
            input_ids, return_attention=return_attention
        )
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            # logits: (batch, seq, vocab) -> (batch * seq, vocab)
            # labels: (batch, seq) -> (batch * seq,)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100  # Ignore padding tokens
            )
        
        return GPTOutput(
            logits=logits,
            loss=loss,
            attention_weights=attention_weights if return_attention else None
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token indices of shape (batch, seq)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If > 0, only sample from top k tokens
            top_p: If < 1.0, use nucleus sampling with this threshold
            repetition_penalty: Penalty for repeating tokens (> 1.0 = avoid repetition)
            eos_token_id: Stop generation when this token is generated
            pad_token_id: Padding token ID for batched generation
            
        Returns:
            Generated token indices of shape (batch, seq + max_new_tokens)
            
        TODO: Implement KV caching for faster generation!
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Track which sequences have finished (hit EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Truncate if sequence is too long
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else \
                       input_ids[:, -self.max_seq_len:]
            
            # Get logits for next token
            output = self(idx_cond)
            logits = output.logits[:, -1, :]  # (batch, vocab)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, input_ids, repetition_penalty
                )
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                logits = self._top_k_filtering(logits, top_k)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                logits = self._top_p_filtering(logits, top_p)
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Handle EOS token
            if eos_token_id is not None:
                eos_mask = next_token.squeeze(-1) == eos_token_id
                finished = finished | eos_mask
                
                # Replace with pad token for finished sequences
                if pad_token_id is not None and finished.any():
                    next_token[finished] = pad_token_id
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if all sequences finished
            if eos_token_id is not None and finished.all():
                break
        
        return input_ids
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        for i in range(input_ids.size(0)):
            for token_id in set(input_ids[i].tolist()):
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty
        return logits
    
    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Filter logits to only keep top-k tokens."""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find where cumulative probability exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
    
    @classmethod
    def from_config(cls, config) -> 'GPTModel':
        """Create model from a Config object."""
        return cls(
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            d_ff=config.model.d_ff,
            max_seq_len=config.model.max_seq_len,
            dropout=config.model.dropout,
            activation=config.model.activation,
            pre_norm=config.model.pre_norm,
            layer_norm_eps=config.model.layer_norm_eps
        )
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters
            
        Returns:
            Number of parameters
        """
        n_params = self.n_params
        if non_embedding:
            n_params -= self.decoder.embedding.token_embedding.embedding.weight.numel()
        return n_params
    
    def estimate_flops(self, seq_len: int) -> int:
        """
        Estimate FLOPs for a forward pass.
        
        This is a rough estimate useful for efficiency comparisons.
        """
        # Embeddings
        flops = 2 * seq_len * self.d_model
        
        # Per layer
        n_layers = self.decoder.n_layers
        d_model = self.d_model
        d_ff = self.decoder.layers[0].feed_forward.linear1.out_features
        
        # Self-attention: 4 * d_model^2 (Q, K, V, O projections)
        # + 2 * seq_len^2 * d_model (attention computation)
        attn_flops = 4 * seq_len * d_model * d_model
        attn_flops += 2 * seq_len * seq_len * d_model
        
        # FFN: 2 * d_model * d_ff * 2 (two linear layers)
        ffn_flops = 4 * seq_len * d_model * d_ff
        
        flops += n_layers * (attn_flops + ffn_flops)
        
        # Output projection
        flops += seq_len * d_model * self.vocab_size
        
        return flops
