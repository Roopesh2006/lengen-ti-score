"""
lengen.py — minimal decoder-only transformer for the length-generalization study
on integer addition. Positional scheme is selectable (nope / ape / alibi / rope)
and attention maps are exposed for mechanistic analysis. CPU-only, tiny models.
"""
import math, argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- vocabulary -----------------------------------
# 0-9 digits, '+', '=', PAD, EOS
DIGITS = list("0123456789")
VOCAB = DIGITS + ["+", "=", "P", "E"]            # P=pad, E=eos/sep
STOI = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)
PAD = STOI["P"]


def make_example(na, nb, rng):
    """a (na digits) + b (nb digits); output = digits of (a+b) reversed (LSB first)."""
    a = rng.integers(10 ** (na - 1), 10 ** na) if na > 1 else rng.integers(0, 10)
    b = rng.integers(10 ** (nb - 1), 10 ** nb) if nb > 1 else rng.integers(0, 10)
    s = a + b
    a_s, b_s = str(a), str(b)
    out = str(s)[::-1]                            # reversed sum
    prompt = list(a_s) + ["+"] + list(b_s) + ["="]
    target = list(out) + ["E"]
    return prompt, target


def make_task(task, n, rng):
    """Length-generalization tasks on digit strings of length n.
    copy: output the string;  reverse: output reversed string."""
    s = [DIGITS[rng.integers(0, 10)] for _ in range(n)]
    prompt = s + ["="]
    if task == "copy":
        target = s + ["E"]
    elif task == "reverse":
        target = s[::-1] + ["E"]
    else:
        raise ValueError(task)
    return prompt, target


def encode_batch(items, Lmax):
    """items: list of (prompt, target). Returns x, y(targets shifted), loss_mask."""
    B = len(items)
    x = np.full((B, Lmax), PAD, dtype=np.int64)
    tgt = np.full((B, Lmax), PAD, dtype=np.int64)
    mask = np.zeros((B, Lmax), dtype=np.float32)
    for i, (p, t) in enumerate(items):
        seq = p + t
        ids = [STOI[c] for c in seq]
        n = len(ids)
        x[i, :n] = ids
        # predict token j+1 from position j; loss only on target region
        for j in range(n - 1):
            tgt[i, j] = ids[j + 1]
            if j >= len(p) - 1:                   # positions that output target chars
                mask[i, j] = 1.0
    return torch.from_numpy(x), torch.from_numpy(tgt), torch.from_numpy(mask)


# ----------------------------- model -----------------------------------------
class Head(nn.Module):
    def __init__(self, d, dh, pe, Lmax):
        super().__init__()
        self.q = nn.Linear(d, dh, bias=False)
        self.k = nn.Linear(d, dh, bias=False)
        self.v = nn.Linear(d, dh, bias=False)
        self.dh, self.pe = dh, pe
        self.Lmax = Lmax

    def _rope(self, x):
        # x: (B,L,dh) apply rotary to pairs
        B, L, dh = x.shape
        half = dh // 2
        pos = torch.arange(L, device=x.device).float()
        inv = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
        ang = pos[:, None] * inv[None, :]         # (L, half)
        cos, sin = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, alibi_slope=None, need_attn=False):
        B, L, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.pe == "rope":
            q, k = self._rope(q), self._rope(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)   # (B,L,L)
        if self.pe == "alibi":
            dist = torch.arange(L)[None, :] - torch.arange(L)[:, None]   # j-i
            att = att + alibi_slope * (-torch.abs(dist).float()).to(x.device)
        causal = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
        att = att.masked_fill(causal[None], float("-inf"))
        a = F.softmax(att, dim=-1)
        out = a @ v
        return (out, a) if need_attn else (out, None)


class Block(nn.Module):
    def __init__(self, d, nh, pe, Lmax, ff):
        super().__init__()
        self.heads = nn.ModuleList([Head(d, d // nh, pe, Lmax) for _ in range(nh)])
        self.proj = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        # ALiBi per-head slopes (geometric) as in Press et al.
        self.register_buffer("slopes", torch.tensor(
            [2 ** (-8 * (i + 1) / nh) for i in range(nh)]))
        self.pe = pe

    def forward(self, x, need_attn=False):
        h = self.ln1(x); outs = []; attns = []
        for i, head in enumerate(self.heads):
            o, a = head(h, alibi_slope=self.slopes[i], need_attn=need_attn)
            outs.append(o); attns.append(a)
        x = x + self.proj(torch.cat(outs, dim=-1))
        x = x + self.mlp(self.ln2(x))
        return x, attns


@torch.no_grad()
def translation_invariance(model, items, Lmax):
    """R^2 of an offset-only model of the averaged attention: how much each head's
    attention depends ONLY on relative offset (i-j) vs absolute position.
    1.0 = perfectly translation-invariant (relative); low = absolute-position dependent.
    Returned averaged over heads and layers."""
    from lengen import encode_batch
    x, _, _ = encode_batch(items, Lmax)
    L0 = max(len(p) + len(t) for p, t in items)
    model.eval()
    _, attn_all = model(x, need_attn=True)
    scores = []
    for layer in attn_all:
        for A in layer:                           # A: (B,L,L)
            Abar = A[:, :L0, :L0].mean(0).cpu().numpy()   # (L0,L0) averaged over batch
            ii, jj = np.tril_indices(L0, 0)               # causal entries i>=j
            vals = Abar[ii, jj]
            off = ii - jj
            m = np.zeros_like(vals)
            for d in np.unique(off):                      # offset-only prediction
                m[off == d] = vals[off == d].mean()
            ss_res = ((vals - m) ** 2).sum()
            ss_tot = ((vals - vals.mean()) ** 2).sum() + 1e-12
            scores.append(1.0 - ss_res / ss_tot)
    model.train()
    return float(np.mean(scores))


class TinyGPT(nn.Module):
    def __init__(self, d=64, nh=4, nl=2, pe="nope", Lmax=48, ff=128):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pe = pe
        if pe == "ape":
            self.pos = nn.Embedding(Lmax, d)
        self.blocks = nn.ModuleList([Block(d, nh, pe, Lmax, ff) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.Lmax = Lmax

    def forward(self, x, need_attn=False):
        h = self.tok(x)
        if self.pe == "ape":
            L = x.shape[1]
            h = h + self.pos(torch.arange(L, device=x.device))[None]
        attn_all = []
        for b in self.blocks:
            h, a = b(h, need_attn=need_attn)
            attn_all.append(a)
        return self.head(self.lnf(h)), attn_all
