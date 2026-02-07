"""
Model package for the Transformer Hackathon.

This package contains all the modular components needed to build a
GPT-style decoder-only transformer model.

Components:
    - embeddings: Token and positional embeddings
    - attention: Multi-head self-attention mechanism
    - feedforward: Position-wise feed-forward network
    - encoder_block: Single encoder layer
    - decoder_block: Single decoder layer
    - encoder: Full encoder stack
    - decoder: Full decoder stack
    - transformer: Complete GPT model
"""

from model.embeddings import TokenEmbedding, PositionalEncoding, TransformerEmbedding
from model.attention import ScaledDotProductAttention, MultiHeadAttention
from model.feedforward import PositionwiseFeedForward
from model.encoder_block import EncoderBlock
from model.decoder_block import DecoderBlock
from model.encoder import Encoder
from model.decoder import Decoder
from model.transformer import GPTModel

__all__ = [
    # Embeddings
    "TokenEmbedding",
    "PositionalEncoding", 
    "TransformerEmbedding",
    # Attention
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    # Feed-forward
    "PositionwiseFeedForward",
    # Blocks
    "EncoderBlock",
    "DecoderBlock",
    # Stacks
    "Encoder",
    "Decoder",
    # Complete model
    "GPTModel",
]
