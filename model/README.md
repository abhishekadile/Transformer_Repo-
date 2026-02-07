# Model Directory

This directory contains both **Jupyter notebooks** for learning and **Python implementations** for training.

## 📓 Notebooks (For Practice & Learning)

Use these interactive notebooks to learn how transformers work:

- **Attention.ipynb** - Multi-head self-attention mechanism
- **Embeddings.ipynb** - Token and positional embeddings
- **FeedForward.ipynb** - Position-wise feed-forward networks
- **Decoder.ipynb** - Decoder blocks and stack
- **Transformer.ipynb** - Complete GPT model

Each notebook has:
- ✅ Step-by-step explanations
- ✅ Individual function cells
- ✅ Practice cells for your own implementation
- ✅ Test cells to verify your code

## 🐍 Python Files (Used by Training Scripts)

These files are imported by `train.py`, `run_hackathon.py`, etc.:

- `attention.py` - Attention mechanisms
- `embeddings.py` - Embedding layers
- `feedforward.py` - Feed-forward networks
- `decoder_block.py` - Single decoder layer
- `decoder.py` - Full decoder stack
- `encoder_block.py` - Single encoder layer (not used in GPT)
- `encoder.py` - Full encoder stack (not used in GPT)
- `transformer.py` - Complete GPT model

## 🎯 How to Use

### For Learning:
1. Open the notebooks in Jupyter or Google Colab
2. Run cells sequentially
3. Try implementing functions in the practice cells
4. Compare with the reference implementation

### For Training:
```bash
# The training scripts automatically import from the .py files
python run_hackathon.py --time 60
```

The notebooks are for **practice**, the `.py` files are what actually runs!
