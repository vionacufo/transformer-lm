# src/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim, "model_dim must be divisible by num_heads"

        # Combined QKV projection
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.output = nn.Linear(model_dim, model_dim)

        # Causal mask: large maximum sequence length for safety
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 2048, 2048)))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).split(D, dim=2)  # [B, T, 3*D] -> 3 tensors [B, T, D] each
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]  # shape: [B, heads, T, head_dim]

        # Scaled Dot-Product
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # Combine heads
        return self.output(out)

class FeedForward(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = SelfAttention(model_dim, num_heads)
        self.ffn = FeedForward(model_dim, dropout)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention -> Residual -> LayerNorm
        x = x + self.dropout(self.attn(self.ln1(x)))
        # FFN -> Residual -> LayerNorm
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, model_dim=256, num_layers=4, num_heads=4, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(model_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        tok_emb = self.token_emb(idx)            # [B, T, D]
        pos_emb = self.pos_emb(pos)             # [T, D]
        x = self.dropout(tok_emb + pos_emb)      # [B, T, D]

        x = self.blocks(x)
        x = self.ln(x)

        logits = self.head(x)
        return logits
