import ast
import json
import os
import re

# Logic to extract code from files
def extract_code(file_path, node_names):
    """
    Extracts source code for specific nodes (classes/functions) from a file.
    Returns a list of tuples: (name, source_code, docstring).
    """
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Error parsing {file_path}: {e}")
        return []

    extracted = []
    lines = source.splitlines(keepends=True)

    # Helper to find valid nodes
    target_nodes = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if node.name in node_names:
                target_nodes[node.name] = node

    # Extract in order of request
    for name in node_names:
        if name in target_nodes:
            node = target_nodes[name]
            
            # Get start and end lines
            # -1 because ast lines are 1-indexed
            start_line = node.lineno - 1
            end_line = node.end_lineno
            
            # Adjust to include decorators if any
            if hasattr(node, 'decorator_list') and node.decorator_list:
                start_line = node.decorator_list[0].lineno - 1

            code_segment = "".join(lines[start_line:end_line])
            
            # Extract docstring
            docstring = ast.get_docstring(node) or ""
            
            extracted.append({
                "name": name,
                "code": code_segment,
                "docstring": docstring
            })
        else:
            print(f"Warning: Node '{name}' not found in {file_path}")
            # Fallback manual extraction if AST fails (unlikely but good safety)
            # logic could go here, but deciding to keep it simple for now

    return extracted

def  extract_imports(file_path):
    """Simple extraction of import statements from the top of the file."""
    if not os.path.exists(file_path):
        return ""
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    imports = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            # naive check, checks if it's a standard library or external import
            # We want to exclude local imports like 'from model.x import y' usually, 
            # unless we are keeping them. 
            # For notebooks, we usually want torch imports, but local imports might break 
            # if the file structure isn't preserved or if we are defining the classes in previous cells.
            
            if "model." in stripped or "data." in stripped or "utils." in stripped:
                # Comment out local imports
                imports.append(f"# {line.rstrip()}  # Local import disabled for notebook")
            else:
                imports.append(line.rstrip())
        elif not stripped and imports:
             # Keep empty lines between imports
             imports.append("")
        elif stripped and imports and not (stripped.startswith("import") or stripped.startswith("from")):
            # Stop at first non-import code (ignoring comments/docstrings briefly)
             if not stripped.startswith("#") and not stripped.startswith('"""'):
                 break
                 
    return "\n".join(imports).strip()

# Notebook Builders
class NotebookBuilder:
    def __init__(self, filename):
        self.filename = filename
        self.cells = []
        
    def add_markdown(self, source):
        self.cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.splitlines()]
        })
        
    def add_code(self, source):
        self.cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()]
        })
        
    def save(self):
        notebook = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"Generated {self.filename}")

# Main Generation Logic
def generate_notebooks():
    root_dir = os.getcwd()
    src_dir = os.path.join(root_dir, "reference") # Moved source files here
    
    # ---------------------------------------------------------
    # 01_Data_Processing.ipynb
    # ---------------------------------------------------------
    nb = NotebookBuilder("01_Data_Processing.ipynb")
    nb.add_markdown("# Data Processing\n\nIn this notebook, we will implement the tokenizer and dataset classes to prepare our text for the model.")
    
    # Imports
    nb.add_markdown("## Imports")
    nb.add_code("import torch\nfrom torch.utils.data import Dataset, DataLoader, random_split\nimport json\nimport os\nimport re\nfrom collections import Counter\nfrom typing import List, Dict, Optional, Tuple")
    
    # Tokenizer
    nb.add_markdown("## Character Tokenizer\n\nFirst, we build a simple character-level tokenizer.")
    items = extract_code(os.path.join(src_dir, "data", "tokenizer.py"), ["CharTokenizer"])
    for item in items:
        nb.add_code(item["code"])
        
    # Dataset
    nb.add_markdown("## Text Dataset\n\nNext, we define the PyTorch Dataset.")
    items = extract_code(os.path.join(src_dir, "data", "dataset.py"), ["TextDataset"])
    for item in items:
        nb.add_code(item["code"])

    # DataLoader
    nb.add_markdown("## Create Dataloaders\n\nFunction to download data and create dataloaders.")
    items = extract_code(os.path.join(src_dir, "data", "dataset.py"), ["download_tinystories", "download_tiny_shakespeare", "create_dataloaders"])
    for item in items:
        nb.add_code(item["code"])

    # Test
    nb.add_markdown("## Test Data Pipeline")
    nb.add_code("""# Quick test
if __name__ == "__main__":
    print("Testing pipeline...")
    # NOTE: Set data_path to None to download, or point to a local file
    try:
        train_loader, val_loader, tokenizer = create_dataloaders(
            batch_size=4, seq_len=32, max_stories=100
        )
        x, y = next(iter(train_loader))
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Decoded: {tokenizer.decode(x[0].tolist())[:50]}...")
    except Exception as e:
        print(f"Could not run test (missing internet?): {e}")
""")
    nb.save()
    
    # ---------------------------------------------------------
    # 02_Embeddings.ipynb
    # ---------------------------------------------------------
    nb = NotebookBuilder("02_Embeddings.ipynb")
    nb.add_markdown("# Embeddings\n\nWe convert token IDs into dense vectors and add positional information.")
    nb.add_code("import torch\nimport torch.nn as nn\nimport math\nfrom typing import Optional")
    
    items = extract_code(os.path.join(src_dir, "model", "embeddings.py"), ["TokenEmbedding", "PositionalEncoding", "TransformerEmbedding"])
    for item in items:
        # nb.add_markdown(f"### {item['name']}") # Optional: Header for each class
        nb.add_code(item["code"])
        
    nb.add_markdown("## Test Embeddings")
    nb.add_code("""d_model = 512
vocab_size = 1000
embed = TransformerEmbedding(vocab_size, d_model)
x = torch.randint(0, vocab_size, (2, 32)) # Batch 2, Seq 32
y = embed(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}") # Should be (2, 32, 512)
""")
    nb.save()

    # ---------------------------------------------------------
    # 03_Attention.ipynb
    # ---------------------------------------------------------
    nb = NotebookBuilder("03_Attention.ipynb")
    nb.add_markdown("# Attention Mechanisms\n\nImplementation of Scaled Dot-Product Attention and Multi-Head Attention.")
    nb.add_code("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\nfrom typing import Optional, Tuple")
    
    items = extract_code(os.path.join(src_dir, "model", "attention.py"), ["ScaledDotProductAttention", "MultiHeadAttention"])
    for item in items:
        nb.add_code(item["code"])
        
    nb.add_markdown("## Test Attention")
    nb.add_code("""mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 32, 512)
out, attn = mha(x, x, x, return_attention=True)
print(f"Output shape: {out.shape}")
print(f"Attention shape: {attn.shape}")
""")
    nb.save()

    # ---------------------------------------------------------
    # 04_FeedForward.ipynb
    # ---------------------------------------------------------
    nb = NotebookBuilder("04_FeedForward.ipynb")
    nb.add_markdown("# Feed Forward Network\n\nThe position-wise feed-forward network.")
    nb.add_code("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom typing import Literal")
    
    items = extract_code(os.path.join(src_dir, "model", "feedforward.py"), ["PositionwiseFeedForward"])
    for item in items:
        nb.add_code(item["code"])
        
    nb.add_markdown("## Test FFN")
    nb.add_code("""ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)
x = torch.randn(2, 32, 512)
out = ffn(x)
print(f"Output shape: {out.shape}")
""")
    nb.save()

    # ---------------------------------------------------------
    # 05_Encoder.ipynb
    # ---------------------------------------------------------
    nb = NotebookBuilder("05_Encoder.ipynb")
    nb.add_markdown("# Transformer Encoder\n\nBuilding the Encoder Block and the full Encoder Stack.")
    nb.add_code("""import torch
import torch.nn as nn
from typing import Optional, List, Tuple

# Note: Imports are simplified for standalone generation
""")
    
    items = extract_code(os.path.join(src_dir, "model", "encoder_block.py"), ["EncoderBlock"])
    for item in items:
        nb.add_code(item["code"])
        
    items = extract_code(os.path.join(src_dir, "model", "encoder.py"), ["Encoder"])
    for item in items:
        nb.add_code(item["code"])
        
    nb.save()

    # ---------------------------------------------------------
    # 06_Decoder.ipynb
    # ---------------------------------------------------------
    nb = NotebookBuilder("06_Decoder.ipynb")
    nb.add_markdown("# Transformer Decoder\n\nBuilding the Decoder Block and the full Decoder Stack (GPT Style).")
    nb.add_code("""import torch
import torch.nn as nn
from typing import Optional, List, Tuple
""")

    items = extract_code(os.path.join(src_dir, "model", "decoder_block.py"), ["DecoderBlock"])
    for item in items:
        nb.add_code(item["code"])
        
    items = extract_code(os.path.join(src_dir, "model", "decoder.py"), ["Decoder"])
    for item in items:
        nb.add_code(item["code"])    

    nb.save()
    
    # ---------------------------------------------------------
    # 07_Transformer.ipynb
    # ---------------------------------------------------------
    nb = NotebookBuilder("07_Transformer.ipynb")
    nb.add_markdown("# GPT Model\n\nThe full GPT Model.")
    nb.add_code("""import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any
""")

    # Helper classes first
    items = extract_code(os.path.join(src_dir, "model", "transformer.py"), ["GPTOutput", "GPTModel"])
    for item in items:
        nb.add_code(item["code"])

    nb.save()

    # ---------------------------------------------------------
    # 08_Training.ipynb
    # ---------------------------------------------------------
    nb = NotebookBuilder("Training_and_Generation.ipynb") # Renamed Target
    nb.add_markdown("# Training & Generation\n\nPutting it all together to train and generate text.")
    nb.add_code("""import torch
import torch.nn as nn
from torch.optim import AdamW
import math
import time
import sys
import os

# Add reference folder to path to allow imports from reference implementation
current_dir = os.getcwd()
reference_dir = os.path.join(current_dir, 'reference')
if reference_dir not in sys.path:
    sys.path.append(reference_dir)

# Import everything from reference
try:
    from config import Config
    from model import GPTModel
    from data import create_dataloaders
    from utils.metrics import compute_perplexity
    from torch.cuda.amp import GradScaler, autocast
except ImportError as e:
    print(f"Could not import modules: {e}")
""")

    # We will copy the train loop snippets rather than the whole file, or simpler functions
    # extracting specific functions from train.py
    
    nb.add_markdown("## Training Functions")
    items = extract_code(os.path.join(src_dir, "train.py"), ["get_lr_scheduler", "evaluate", "train", "print_progress"])
    for item in items:
        nb.add_code(item["code"])

    nb.add_markdown("## Generation")
    items = extract_code(os.path.join(src_dir, "generate.py"), ["generate_text"])
    for item in items:
        nb.add_code(item["code"])
        
    nb.save()

if __name__ == "__main__":
    generate_notebooks()
