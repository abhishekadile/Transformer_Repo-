"""
Simple Tokenizer for the Transformer Hackathon.

This module provides:
    - CharTokenizer: Character-level tokenizer (simple, works for any text)
    - SimpleTokenizer: Word-level tokenizer with vocabulary building

For hackathon purposes, character-level tokenization is recommended because:
    1. No need to worry about unknown words
    2. Smaller vocabulary size
    3. Works immediately on any text

TODO: For better performance, try implementing BPE tokenization!
"""

import json
import os
import re
from typing import List, Dict, Optional, Union
from collections import Counter


class CharTokenizer:
    """
    Character-level tokenizer.
    
    Simply maps each character to a unique ID. This is the simplest
    form of tokenization and works well for small datasets.
    
    Args:
        text: Optional text to build vocabulary from
        
    Example:
        >>> tokenizer = CharTokenizer("hello world")
        >>> tokens = tokenizer.encode("hello")
        >>> print(tokens)  # [0, 1, 2, 2, 3]
        >>> print(tokenizer.decode(tokens))  # "hello"
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    def __init__(self, text: Optional[str] = None):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # Add special tokens first
        self._add_special_tokens()
        
        if text is not None:
            self.build_vocab(text)
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for token in special_tokens:
            idx = len(self.char_to_id)
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token
    
    @property
    def pad_token_id(self) -> int:
        return self.char_to_id[self.PAD_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.char_to_id[self.UNK_TOKEN]
    
    @property
    def bos_token_id(self) -> int:
        return self.char_to_id[self.BOS_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        return self.char_to_id[self.EOS_TOKEN]
    
    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)
    
    def build_vocab(self, text: str):
        """
        Build vocabulary from text.
        
        Args:
            text: Text to extract characters from
        """
        # Get unique characters
        chars = sorted(set(text))
        
        # Add to vocabulary (skip if already exists)
        for char in chars:
            if char not in self.char_to_id:
                idx = len(self.char_to_id)
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
    
    def encode(
        self, 
        text: str, 
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Text to encode
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_bos:
            tokens.append(self.bos_token_id)
        
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_token_id))
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            tokens: List of token IDs
            skip_special: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        special_ids = {self.pad_token_id, self.unk_token_id, 
                      self.bos_token_id, self.eos_token_id}
        
        chars = []
        for token_id in tokens:
            if skip_special and token_id in special_ids:
                continue
            chars.append(self.id_to_char.get(token_id, self.UNK_TOKEN))
        
        return "".join(chars)
    
    def save(self, path: str):
        """Save tokenizer vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_id': self.char_to_id,
                'type': 'char'
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = {int(v): k for k, v in data['char_to_id'].items()}
        return tokenizer


class SimpleTokenizer:
    """
    Simple word-level tokenizer.
    
    Splits text into words and maps each word to an ID.
    Uses a vocabulary with frequency-based filtering.
    
    Args:
        min_freq: Minimum frequency for a word to be included in vocabulary
        max_vocab_size: Maximum vocabulary size
        
    TODO: For better tokenization, try implementing BPE!
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    def __init__(
        self,
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None
    ):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for token in special_tokens:
            idx = len(self.word_to_id)
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
    
    @property
    def pad_token_id(self) -> int:
        return self.word_to_id[self.PAD_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.word_to_id[self.UNK_TOKEN]
    
    @property
    def bos_token_id(self) -> int:
        return self.word_to_id[self.BOS_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        return self.word_to_id[self.EOS_TOKEN]
    
    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into words."""
        # Simple word tokenization: split on whitespace and punctuation
        # Keep punctuation as separate tokens
        text = text.lower()
        tokens = re.findall(r"[\w]+|[^\s\w]", text)
        return tokens
    
    def build_vocab(self, text: str):
        """
        Build vocabulary from text.
        
        Args:
            text: Text to build vocabulary from
        """
        # Tokenize and count frequencies
        tokens = self._tokenize(text)
        self.word_freq = Counter(tokens)
        
        # Filter by frequency and limit size
        vocab_items = [
            (word, freq) for word, freq in self.word_freq.items()
            if freq >= self.min_freq
        ]
        vocab_items.sort(key=lambda x: (-x[1], x[0]))  # Sort by frequency, then alphabetically
        
        if self.max_vocab_size:
            vocab_items = vocab_items[:self.max_vocab_size - len(self.word_to_id)]
        
        # Add to vocabulary
        for word, _ in vocab_items:
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Text to encode
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_bos:
            tokens.append(self.bos_token_id)
        
        for word in self._tokenize(text):
            tokens.append(self.word_to_id.get(word, self.unk_token_id))
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            tokens: List of token IDs
            skip_special: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        special_ids = {self.pad_token_id, self.unk_token_id,
                      self.bos_token_id, self.eos_token_id}
        
        words = []
        for token_id in tokens:
            if skip_special and token_id in special_ids:
                continue
            words.append(self.id_to_word.get(token_id, self.UNK_TOKEN))
        
        # Simple reconstruction: add spaces between words
        return " ".join(words)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word_to_id': self.word_to_id,
                'min_freq': self.min_freq,
                'max_vocab_size': self.max_vocab_size,
                'type': 'word'
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            min_freq=data.get('min_freq', 1),
            max_vocab_size=data.get('max_vocab_size')
        )
        tokenizer.word_to_id = data['word_to_id']
        tokenizer.id_to_word = {int(v): k for k, v in data['word_to_id'].items()}
        return tokenizer
