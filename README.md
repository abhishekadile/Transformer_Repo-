# Transformer Hackathon 🚀

Build your own GPT-style transformer model from scratch and compete on the leaderboard!

**🏆 Leaderboard:** [https://huggingface.co/datasets/abhisu30/transformer-hackathon-leaderboard](https://huggingface.co/datasets/abhisu30/transformer-hackathon-leaderboard)

## Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the hackathon pipeline
python run_hackathon.py

# 3. Follow the prompts!
```

That's it! The script will automatically:
- Download TinyStories dataset (~50K stories)
- Train your model for 45 minutes
- Evaluate performance
- Upload to the leaderboard

---

## 📁 Repository Structure

```
transformer-hackathon/
├── README.md                 # This file
├── COLAB_GUIDE.md           # Google Colab instructions
├── requirements.txt          # Python dependencies
├── config.py                 # Hyperparameters (MODIFY THIS!)
│
├── model/                    # 🧠 Transformer components
│   ├── embeddings.py         # Token + positional embeddings
│   ├── attention.py          # Multi-head self-attention
│   ├── feedforward.py        # Feed-forward network
│   ├── encoder_block.py      # Encoder layer
│   ├── decoder_block.py      # Decoder layer
│   ├── encoder.py            # Full encoder stack
│   ├── decoder.py            # Full decoder stack
│   └── transformer.py        # Complete GPT model
│
├── data/                     # 📚 Data handling
│   ├── tokenizer.py          # Character/word tokenizer
│   └── dataset.py            # Dataset loading (TinyStories default)
│
├── utils/                    # 🔧 Utilities
│   ├── metrics.py            # Evaluation metrics
│   ├── checkpoint.py         # Save/load models
│   └── huggingface_upload.py # Leaderboard integration
│
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── generate.py               # Text generation
└── run_hackathon.py          # 🏆 Main hackathon script
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPT-Style Transformer                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: "The cat sat on the"                                    │
│              ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  TOKEN EMBEDDING + POSITIONAL ENCODING                   │   │
│  │  • Convert tokens to vectors                             │   │
│  │  • Add position information                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│              ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              DECODER BLOCK (×6)                          │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  MASKED MULTI-HEAD SELF-ATTENTION                  │  │   │
│  │  │  • Q, K, V projections                             │  │   │
│  │  │  • 8 attention heads                               │  │   │
│  │  │  • Causal masking (can only see past)              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │              ↓ + Residual + LayerNorm                    │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  FEED-FORWARD NETWORK                              │  │   │
│  │  │  • Linear(512 → 2048)                              │  │   │
│  │  │  • GELU activation                                 │  │   │
│  │  │  • Linear(2048 → 512)                              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │              ↓ + Residual + LayerNorm                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│              ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  OUTPUT PROJECTION                                        │   │
│  │  • Linear(512 → vocab_size)                              │   │
│  │  • Softmax → probability distribution                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│              ↓                                                   │
│  Output: "mat" (predicted next token)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Default Configuration:
• d_model = 512 (embedding dimension)
• n_heads = 8 (attention heads)
• n_layers = 6 (decoder blocks)
• d_ff = 2048 (feed-forward dimension)
• max_seq_len = 128 (context window)
• vocab_size = ~65 (character-level)
• Parameters = ~10M
```

---

## 🏆 Hackathon Rules

### Competition Categories

1. **🥇 Best Model Performance**
   - Lowest perplexity wins
   - Primary ranking metric

2. **⚡ Most Efficient Training**
   - Highest tokens/second
   - Speed matters!

3. **📝 Best Generation Quality**
   - Highest Distinct-2 score
   - Text should be diverse and coherent

4. **🎨 Most Creative Optimization**
   - Judged by organizers
   - Document your changes!

### Rules

- Training time: **Exactly 45 minutes**
- Hardware: Use whatever you have (GPU recommended)
- Code: Modify anything except timing enforcement
- Collaboration: Team up to 4 people

---

## 🔧 Optimization Ideas

Here are proven techniques to improve your model:

### Easy Wins 🟢

```python
# config.py
# 1. Enable mixed precision (2x speedup on modern GPUs!)
config.training.use_mixed_precision = True

# 2. Increase batch size if memory allows
config.training.batch_size = 32

# 3. Try different learning rates
config.training.learning_rate = 1e-3  # or 5e-4
```

### Medium Difficulty 🟡

```python
# 1. Gradient accumulation for larger effective batch size
config.training.gradient_accumulation_steps = 4

# 2. Bigger model (if GPU memory allows)
config.model.d_model = 768
config.model.n_layers = 8

# 3. Better learning rate schedule
config.training.lr_scheduler = "cosine"
config.training.warmup_ratio = 0.1
```

### Advanced 🔴

1. **Flash Attention** (in `model/attention.py`)
   ```python
   # Replace manual attention with PyTorch's optimized version
   from torch.nn.functional import scaled_dot_product_attention
   ```

2. **SwiGLU Activation** (in `model/feedforward.py`)
   ```python
   # Use SwiGLUFeedForward instead of PositionwiseFeedForward
   from model.feedforward import SwiGLUFeedForward
   ```

3. **Gradient Checkpointing** (save memory for bigger models)
   ```python
   from torch.utils.checkpoint import checkpoint
   ```

4. **Custom Optimizer** (Lion, AdaFactor, etc.)

---

## 📊 Understanding Your Metrics

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| **Perplexity** | Model uncertainty (lower = better) | < 20 |
| **Loss** | Cross-entropy loss | < 3.0 |
| **Tokens/sec** | Training speed | > 2000 |
| **Distinct-2** | Generation diversity | > 0.5 |

---

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```python
# Reduce batch size
config.training.batch_size = 8

# Or enable gradient checkpointing
# (advanced, requires code changes)
```

**"Training is too slow"**
```python
# Enable mixed precision
config.training.use_mixed_precision = True

# Reduce model size
config.model.d_model = 256
config.model.n_layers = 4
```

**"Loss is not decreasing"**
```python
# Try lower learning rate
config.training.learning_rate = 1e-4

# Check for NaN (enable gradient clipping)
config.training.max_grad_norm = 0.5
```

**"Text generation is repetitive"**
```python
# Use higher temperature and repetition penalty
config.generation.temperature = 1.0
config.generation.repetition_penalty = 1.2
```

---

## 💻 Running Individual Scripts

```bash
# Train only
python train.py --max-time 10  # 10 minute test run

# Train with custom settings
python train.py --batch-size 32 --lr 1e-3 --use-amp

# Evaluate a checkpoint
python evaluate.py --checkpoint checkpoints/best.pt --generate

# Generate text interactively
python generate.py

# Generate with custom settings
python generate.py --prompt "To be or not" --temperature 1.2 --max-tokens 200
```

---

## 📈 Leaderboard

Results are uploaded to a shared Hugging Face dataset. View the leaderboard:

```python
from utils import display_leaderboard
display_leaderboard()
```

Or check online at: [Hugging Face Leaderboard](https://huggingface.co/datasets/transformer-hackathon/leaderboard)

---

## 🎓 Learning Resources

- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Hugging Face Transformers Course](https://huggingface.co/course)

---

## 📄 License

MIT License - feel free to use, modify, and share!

---

**Good luck and have fun! 🚀**
