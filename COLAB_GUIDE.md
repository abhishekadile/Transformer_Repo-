# Google Colab Guide 🚀

Complete guide to running the Transformer Hackathon in Google Colab.

## Table of Contents

1. [Quick Setup](#quick-setup)
2. [Enable GPU](#enable-gpu)
3. [Clone Repository](#clone-repository)
4. [Install Dependencies](#install-dependencies)
5. [Run Training](#run-training)
6. [Modify Hyperparameters](#modify-hyperparameters)
7. [Save Checkpoints](#save-checkpoints)
8. [Common Errors](#common-errors)

---

## Quick Setup

Copy and paste this into a Colab cell for instant setup:

```python
# 🚀 ONE-CLICK SETUP
# Run this cell to set everything up!

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ NO GPU! Go to Runtime > Change runtime type > GPU")
    raise RuntimeError("GPU required for reasonable training speed")

# Clone repository
!git clone https://github.com/YOUR_USERNAME/transformer-hackathon.git
%cd transformer-hackathon

# Install dependencies
!pip install -q torch numpy tqdm huggingface_hub

print("\n✅ Setup complete! Run the next cell to start training.")
```

---

## Enable GPU

**This is the most important step!**

1. Click **Runtime** in the top menu
2. Click **Change runtime type**
3. Under **Hardware accelerator**, select **GPU**
4. Click **Save**

Verify GPU is enabled:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

**Expected output:**
```
GPU available: True
GPU name: Tesla T4
```

### GPU Memory

| GPU Type | Memory | Recommended Batch Size |
|----------|--------|----------------------|
| Tesla T4 | 16 GB | 16-32 |
| Tesla P100 | 16 GB | 16-32 |
| Tesla V100 | 16 GB | 32-64 |
| A100 | 40 GB | 64-128 |

---

## Clone Repository

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/transformer-hackathon.git

# Change to the repository directory
%cd transformer-hackathon

# Verify files
!ls -la
```

---

## Install Dependencies

```python
# Install required packages
!pip install -q torch numpy tqdm huggingface_hub

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Run Training

### Option 1: Full Hackathon Pipeline (Recommended)

```python
# Run the complete hackathon pipeline
# This will:
# 1. Ask for your name
# 2. Train for 45 minutes
# 3. Evaluate your model
# 4. Upload results to leaderboard

!python run_hackathon.py
```

### Option 2: Quick Test Run

```python
# Short training run to test everything works
!python train.py --max-time 2  # 2 minutes only
```

### Option 3: Custom Training

```python
# Custom training with specific settings
!python train.py \
    --max-time 45 \
    --batch-size 32 \
    --lr 3e-4 \
    --use-amp \
    --d-model 512 \
    --n-layers 6
```

---

## Modify Hyperparameters

### Method 1: Command Line Arguments

```python
# Change batch size and learning rate
!python train.py --batch-size 32 --lr 1e-3

# Enable mixed precision (faster training!)
!python train.py --use-amp

# Bigger model
!python train.py --d-model 768 --n-layers 8
```

### Method 2: Edit config.py

```python
# View current config
!cat config.py

# Edit config using Python
config_code = '''
# config.py modifications
from config import Config

config = Config()

# Change these values:
config.model.d_model = 768
config.model.n_layers = 8
config.model.n_heads = 12

config.training.batch_size = 32
config.training.learning_rate = 1e-3
config.training.use_mixed_precision = True

print(config)
'''

# Write to temporary file and run
with open('check_config.py', 'w') as f:
    f.write(config_code)
!python check_config.py
```

### Method 3: Inline Code Changes

```python
# Modify and run training directly
import sys
sys.path.insert(0, '/content/transformer-hackathon')

from config import Config
from train import train

class Args:
    max_time = 45           # Training time in minutes
    batch_size = 32         # Increase if GPU memory allows
    seq_len = 128           # Sequence length
    lr = 3e-4               # Learning rate
    d_model = 512           # Model dimension
    n_layers = 6            # Number of layers
    n_heads = 8             # Attention heads
    use_amp = True          # Mixed precision (faster!)
    grad_accum_steps = 1    # Gradient accumulation
    max_grad_norm = 1.0     # Gradient clipping
    checkpoint_dir = "checkpoints"
    checkpoint_interval = 5.0
    resume = False
    data_path = None
    log_interval = 10
    eval_interval = 500
    seed = 42
    quiet = False

args = Args()
model, tokenizer, metrics = train(args)
```

---

## Save Checkpoints

### Save to Google Drive (Recommended)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints to Drive after training
!cp -r checkpoints /content/drive/MyDrive/transformer-hackathon-checkpoints

print("✅ Checkpoints saved to Google Drive!")
```

### Download Checkpoints Locally

```python
from google.colab import files

# Download the final checkpoint
files.download('checkpoints/final.pt')

# Or download all checkpoints as a zip
!zip -r checkpoints.zip checkpoints
files.download('checkpoints.zip')
```

### Resume Training

```python
# If you need to resume from a checkpoint:
!python train.py --resume --checkpoint-dir checkpoints
```

---

## Common Errors

### ❌ "CUDA out of memory"

**Solution:** Reduce batch size

```python
# Try smaller batch size
!python train.py --batch-size 8

# Or enable gradient accumulation for effective larger batch
!python train.py --batch-size 8 --grad-accum-steps 4
```

### ❌ "RuntimeError: CUDA error: device-side assert triggered"

**Solution:** This usually means an index out of bounds. Check your vocabulary size.

```python
# Verify tokenizer
from data import create_dataloaders
train_loader, val_loader, tokenizer = create_dataloaders()
print(f"Vocabulary size: {tokenizer.vocab_size}")
```

### ❌ "Connection reset" / "Session crashed"

**Solution:** Colab may disconnect for long training runs.

```python
# Prevent Colab from disconnecting (run in browser console)
# Press F12, go to Console tab, paste:
# function KeepClicking(){ console.log("clicking"); document.querySelector("colab-connect-button").click() }
# setInterval(KeepClicking,60000)

# Better solution: Use checkpoints and resume
!python train.py --checkpoint-interval 2  # Save every 2 minutes
```

### ❌ "No module named 'xxx'"

**Solution:** Install missing package

```python
# Install common missing packages
!pip install torch numpy tqdm huggingface_hub rich

# Restart runtime if needed
import os
os.kill(os.getpid(), 9)  # This will restart the runtime
```

### ❌ "Training is very slow"

**Solutions:**

```python
# 1. Make sure GPU is enabled (see above)

# 2. Enable mixed precision
!python train.py --use-amp

# 3. Check GPU utilization
!nvidia-smi

# 4. Reduce model size for faster iteration
!python train.py --d-model 256 --n-layers 4
```

---

## Evaluation and Generation

### Evaluate Your Model

```python
# Evaluate the final checkpoint
!python evaluate.py --checkpoint checkpoints/final.pt --generate

# Get detailed metrics
!python evaluate.py --output results.json
```

### Generate Text

```python
# Interactive generation
!python generate.py

# Generate from a specific prompt
!python generate.py --prompt "To be or not to be" --max-tokens 200

# Multiple samples with different temperatures
!python generate.py --prompt "The king" --n 3 --temperature 1.2
```

---

## Complete Colab Notebook Template

Copy this entire cell for a complete notebook:

```python
#@title 🚀 Transformer Hackathon - Complete Setup {display-mode: "form"}

#@markdown ## Step 1: Check GPU
import torch
assert torch.cuda.is_available(), "⚠️ Enable GPU: Runtime > Change runtime type > GPU"
print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

#@markdown ## Step 2: Clone and Setup
!git clone https://github.com/YOUR_USERNAME/transformer-hackathon.git 2>/dev/null || echo "Already cloned"
%cd transformer-hackathon
!pip install -q torch numpy tqdm huggingface_hub

#@markdown ## Step 3: Configuration
BATCH_SIZE = 16 #@param {type:"integer"}
LEARNING_RATE = 0.0003 #@param {type:"number"}
TRAINING_MINUTES = 45 #@param {type:"integer"}
USE_MIXED_PRECISION = True #@param {type:"boolean"}

#@markdown ## Step 4: Run Training
import subprocess
cmd = f"python train.py --max-time {TRAINING_MINUTES} --batch-size {BATCH_SIZE} --lr {LEARNING_RATE}"
if USE_MIXED_PRECISION:
    cmd += " --use-amp"
print(f"Running: {cmd}")
!{cmd}

#@markdown ## Step 5: Evaluate
!python evaluate.py --generate

#@markdown ## Step 6: Save to Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
!cp -r checkpoints /content/drive/MyDrive/transformer-checkpoints/
print("✅ Saved to Google Drive!")
```

---

## Tips for Best Performance

1. **Use GPU** - Always enable GPU in Colab settings
2. **Mixed Precision** - Use `--use-amp` for 2x speedup
3. **Monitor Progress** - Watch loss decrease, should see improvement within 5 minutes
4. **Save Early** - Use short checkpoint intervals in case of disconnection
5. **Experiment** - Try different hyperparameters, the defaults are just a starting point!

---

**Good luck with the hackathon! 🏆**
