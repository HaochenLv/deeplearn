import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def alibi_biases(n_heads: int, seq_len: int, device):
    """返回 [1, n_heads, seq_len, seq_len] 的 ALiBi 偏置（含因果掩码）。"""
    start = 2.0 ** (-8.0 / n_heads)
    slopes = torch.tensor([start ** (i + 1) for i in range(n_heads)], device=device)
    pos = torch.arange(seq_len, device=device)
    rel = pos[None, :] - pos[:, None]            # rel[q,k] = k - q
    bias = slopes.view(n_heads, 1, 1) * rel.view(1, seq_len, seq_len)  # -> [H, L, L]
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    bias = bias.masked_fill(~causal, float("-inf"))
    return bias.unsqueeze(0)  # [H, L, L] -> [1, H, L, L]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, bias, pad_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)               # 各 [B,T,H,dh]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B,H,T,dh]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att + bias                           # [B,H,T,T] + [1,H,T,T]
        if pad_mask is not None:
            att = att.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, bias, pad_mask=None):
        x = x + self.attn(self.ln1(x), bias, pad_mask)
        x = x + self.ff(self.ln2(x))
        return x


class AdditionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_heads = n_heads
        self.max_len = max_len

    def forward(self, idx, pad_mask=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        bias = alibi_biases(self.n_heads, T, idx.device)
        for blk in self.blocks:
            x = blk(x, bias, pad_mask)
        x = self.ln_f(x)
        return self.head(x)
