# Transformer Language Model (Work in Progress)

A **from-scratch** Transformer language model implemented in PyTorch.  
Currently experimenting with **Rotary Positional Embeddings (RoPE)** and other post-Transformer-paper techniques to see how they affect performance.

---

## Overview

- **Core Idea**: Character-level language modeling using self-attention and causal masking.  
- **Why?**: To deeply understand how Transformers work and explore the impact of newer methods like RoPE.
- **Disclaimer**: **Work in progress**—expect changes, experiments, and incomplete features.

---

## Repository Structure
```
transformer-lm/ 
    ├── src/ │ 
    ├── data_utils.py # Data loading, tokenization │ 
    ├── model.py # Transformer model & submodules │ 
    ├── train.py # Training loop │ 
    └── generate.py # Generate text from a checkpoint 
├── notebooks/ │ 
└── experiments.ipynb # Colab/Jupyter experiments 
├── README.md 
├── requirements.txt 
```

---

## Installation

1. **Clone**:
   ```bash
   git clone https://github.com/<YOUR-USERNAME>/transformer-lm.git
   cd transformer-lm

2. **Create & Activate** (optional):
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate (Windows)
```

3. **Install**
```bash
pip install -r requirements.txt
```

## Usage

1. **Train Exemple**
```bash
python src/train.py \
  --data_file data/shakespeare_dataset.txt \
  --seq_length 64 \
  --batch_size 64 \
  --epochs 10 \
  --lr 3e-4 \
  --save_path checkpoints/model.pt
```
Use --use_wandb if you have Weights & Biases installed (optional).
Future updates will include perplexity calculations on validation sets to better gauge performance.


2. **Generate**

```bash
python src/generate.py \
  --model_checkpoint checkpoints/model.pt \
  --data_file data/my_corpus.txt \
  --prompt "KING: Oh God !" \
  --max_new_tokens 200
```

Note: Make sure your --model_checkpoint matches the model saved during training.
Current Progress

Baseline: Basic Transformer with standard learned positional embeddings
Ongoing: Integrating RoPE for better handling of positional information
Experiments: Testing different initializations, optimizers (AdamW, Adafactor), etc.


## Next steps

- RoPE finalization: Complete integration of Rotary Positional Embeddings
- Validation metrics: Incorporate perplexity (and possibly other validation stats) for performance tracking
- Alternative embedding methods for positionnal encoding : Compare RoPE vs. learned embeddings vs. fixed sinusoidal positional encodings
- Scaling: Larger models, longer sequence lengths, and bigger datasets
- Optimizers: Testing AdamW vs. Adafactor for potential performance gains


## References

1. [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)  
2. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)  
3. [minGPT by Andrej Karpathy](https://github.com/karpathy/minGPT) (many parts code and ideas inspired by this project and his the youtube series on transformers)


