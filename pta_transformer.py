from typing import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataclasses
from tqdm.auto import tqdm

import torchvision
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

# Use GPU if available, otherwise CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from datasets import load_dataset
import pandas as pd
import numpy as np
import torch

ds = load_dataset("paperswithbacktest/Stocks-Daily-Price", split="train")  # -> Dataset
ds_aapl = ds.filter(lambda x: x["symbol"] == "AAPL")

df = ds_aapl.to_pandas()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

n = len(df)
cut = int(0.80 * n)
df_train = df.iloc[:cut]
df_test  = df.iloc[cut:]

print("rows:", len(df_train), len(df_test))

# Make sliding windows (past W days -> next day's target)

def make_sequences_frame(frame, window=90,
                         feat_cols=("open","high","low","close","volume","adj_close"),
                         target_col="adj_close"):
    arr = frame[list(feat_cols)].to_numpy(dtype=np.float32)  # (T, F)
    X, y = [], []
    tgt_idx = list(feat_cols).index(target_col)
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])                 # (window, F)
        y.append(arr[i+window, tgt_idx])          # next-step target
    if len(X) == 0:
        return np.empty((0, window, len(feat_cols)), np.float32), np.empty((0,), np.float32)
    return np.stack(X), np.array(y, dtype=np.float32)

feat_cols = ("open","high","low","close","volume","adj_close")
W = 30

X_tr, y_tr = make_sequences_frame(df_train, window=W, feat_cols=feat_cols)
X_te, y_te = make_sequences_frame(df_test,  window=W, feat_cols=feat_cols)

X_tr = torch.tensor(X_tr); y_tr = torch.tensor(y_tr)
X_te = torch.tensor(X_te); y_te = torch.tensor(y_te)

# Normalize features using TRAIN stats only

mean = X_tr.mean(dim=(0,1), keepdim=True)
std  = X_tr.std(dim=(0,1), keepdim=True) + 1e-6
X_tr = (X_tr - mean) / std
X_te = (X_te - mean) / std

from torch.utils.data import TensorDataset, DataLoader
import torch

BATCH = 64

train_ds = TensorDataset(X_tr, y_tr)   # shapes: (N_tr, T, F), (N_tr,)
test_ds  = TensorDataset(X_te, y_te)   # shapes: (N_te, T, F), (N_te,)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, drop_last=False)

import math
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_model(model, train_loader, test_loader, epochs=60, lr=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)                 # (B,)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)

        train_mse = running / len(train_loader.dataset)

        # monitor progress
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            mse, rmse, dir_acc = evaluate_model(model, test_loader)
            print(f"Epoch {ep:03d} | Train MSE {train_mse:.4e} | "
                  f"Test RMSE {rmse:,.2f} | DirAcc {dir_acc:.3f}")

    return evaluate_model(model, test_loader)


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        p = model(xb).cpu()
        preds.append(p)
        trues.append(yb)
    y = torch.cat(trues)        # true next-day price (or return)
    p = torch.cat(preds)

    mse  = F.mse_loss(p, y).item()
    rmse = math.sqrt(mse)

    # Directional accuracy (sign of change)
    dy_true = y[1:] - y[:-1]
    dy_pred = p[1:] - p[:-1]
    dir_acc = (torch.sign(dy_true) == torch.sign(dy_pred)).float().mean().item()

    return mse, rmse, dir_acc



from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        super().__init__()
        self.W_K = nn.Linear(dim, n_hidden)
        self.W_Q = nn.Linear(dim, n_hidden)
        self.W_V = nn.Linear(dim, n_hidden)
        self.n_hidden = n_hidden

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, dim), attn_mask: (B, T, T) with True=keep, False=mask or None
        Q = self.W_Q(x)                                    # (B, T, n_hidden)
        K = self.W_K(x)                                    # (B, T, n_hidden)
        V = self.W_V(x)                                    # (B, T, n_hidden)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.n_hidden)  # (B, T, T)
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~attn_mask, float('-inf'))

        alpha = F.softmax(scores, dim=-1)                  # (B, T, T)
        out = alpha @ V                                    # (B, T, n_hidden)
        return out, alpha

class MultiHeadedAttention(nn.Module):
    def __init__(self, dim: int, n_hidden: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([AttentionHead(dim, n_hidden) for _ in range(num_heads)])
        self.project_concatenate = nn.Linear(num_heads * n_hidden, dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        alloutputs: List[torch.Tensor] = []
        allalphas: List[torch.Tensor] = []
        for head in self.heads:
            out_h, alpha_h = head(x, attn_mask)           # out_h: (B, T, n_hidden), alpha_h: (B, T, T)
            alloutputs.append(out_h)
            allalphas.append(alpha_h)
        concat_out = torch.cat(alloutputs, dim=-1)         # (B, T, H*n_hidden)
        attn_output = self.project_concatenate(concat_out) # (B, T, dim)
        attn_alphas = torch.stack(allalphas, dim=1)        # (B, H, T, T)
        return attn_output, attn_alphas


class FFN(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionResidual(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(dim, attn_dim, num_heads)
        self.ffn  = FFN(dim, mlp_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, alphas = self.attn(x=x, attn_mask=attn_mask)
        x = attn_out + x
        x = self.ffn(x) + x
        return x, alphas


class Transformer(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionResidual(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], return_attn: bool=False)\
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        allattentions: List[torch.Tensor] = []
        for layer in self.layers:
            x, alphas = layer(x, attn_mask)
            if return_attn:
                allattentions.append(alphas)
        collected_attns = torch.stack(allattentions, dim=1) if return_attn else None  # (B, L, H, T, T) or None
        return x, collected_attns


class ProjectThenAttend(nn.Module):
    """
    First project (time-compress early chunks) then attend across [S-1] projected tokens + last K tokens.
    - Input:  x  (B, T, D)  with T = S * K
    - Output: (B, T, D)  (same length; projected tokens broadcast back to their chunks)
    """
    def __init__(self, dim: int, attn_dim: int, num_heads: int, chunk_size: int, keep_last_n: int = 1):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.keep_last_n = keep_last_n

        # MHA over the aggregated sequence of length (S-keep_last_n) + keep_last_n*K
        self.mha = MultiHeadedAttention(dim=dim, n_hidden=attn_dim, num_heads=num_heads)

        # Time-compression (shared across chunks): Linear over the K time steps -> 1
        # We'll apply it on (B,S-keep_last_n,D,K) after permuting, so it learns temporal weights.
        self.time_projector = nn.Linear(chunk_size, 1)

        # Gating + projection for fusing attended context back to tokens
        self.fuse_proj = nn.Linear(dim, dim)
        self.fuse_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        K = self.chunk_size
        assert T % K == 0, f"seq_len {T} must be a multiple of chunk_size {K}"
        S = T // K
        keep_last_n = self.keep_last_n
        assert 1 <= keep_last_n <= S, "keep_last_n must be in [1, S]"

        # reshape into chunks
        x_chunks = x.view(B, S, K, D)  # (B, S, K, D)

        # indices
        S_proj = S - keep_last_n
        proj_part = x_chunks[:, :S_proj, :, :]      # (B, S_proj, K, D)
        tail_part = x_chunks[:, S_proj:, :, :]      # (B, keep_last_n, K, D)

        # ---- First: PROJECT the first S-keep_last_n chunks along time K -> 1
        # Move time to last for Linear(K->1): (B, S_proj, D, K)
        proj_part_for_linear = proj_part.permute(0, 1, 3, 2)
        # Apply shared temporal projector
        proj_tokens = self.time_projector(proj_part_for_linear)  # (B, S_proj, D, 1)
        proj_tokens = proj_tokens.squeeze(-1)                    # (B, S_proj, D)

        # Keep last keep_last_n*K tokens as-is (flatten to tokens)
        tail_tokens = tail_part.reshape(B, keep_last_n * K, D)   # (B, keep_last_n*K, D)

        # Concatenate: [S_proj projected tokens] + [keep_last_n*K tail tokens]
        agg_seq = torch.cat([proj_tokens, tail_tokens], dim=1)   # (B, S_proj + keep_last_n*K, D)

        # ---- Then: ATTEND over the aggregated  sequence
        # (no mask by default, but you could add a causal mask here if you want)
        attended, alphas = self.mha(agg_seq, attn_mask=None)     # attended: (B, S_proj + keep_last_n*K, D)

        # Split outputs back: first S_proj correspond to projected chunks; rest to tail tokens
        attended_proj = attended[:, :S_proj, :]                  # (B, S_proj, D)
        attended_tail = attended[:, S_proj:, :]                  # (B, keep_last_n*K, D)

        # Broadcast projected tokens back over their original K positions
        attended_proj_broadcast = attended_proj.unsqueeze(2).expand(B, S_proj, K, D)  # (B, S_proj, K, D)

        # Reshape tail back to (B, keep_last_n, K, D)
        attended_tail_chunks = attended_tail.view(B, keep_last_n, K, D)               # (B, keep_last_n, K, D)

        # Fuse with original tokens (residual + gated projection)
        fused_proj = proj_part + torch.tanh(self.fuse_gate) * self.fuse_proj(attended_proj_broadcast)
        fused_tail = tail_part + torch.tanh(self.fuse_gate) * self.fuse_proj(attended_tail_chunks)

        # Concatenate back to (B, S, K, D) -> (B, T, D)
        fused = torch.cat([fused_proj, fused_tail], dim=1).contiguous().view(B, T, D)
        return fused, alphas  # alphas: (B, H, L, L) on aggregated L=(S_proj + keep_last_n*K)


class AttentionResidualPTA(nn.Module):
    """Residual block that uses Project-Then-Attend inside, then FFN."""
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, chunk_size: int, keep_last_n: int = 1):
        super().__init__()
        self.pta = ProjectThenAttend(dim=dim, attn_dim=attn_dim, num_heads=num_heads,
                                     chunk_size=chunk_size, keep_last_n=keep_last_n)
        self.ffn = FFN(dim, mlp_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        out, A = self.pta(x, attn_mask)
        x = out + x
        x = self.ffn(x) + x
        return x, A


class TransformerPTA(nn.Module):
    """Stack of Project-Then-Attend residual blocks."""
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int,
                 chunk_size: int, keep_last_n: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionResidualPTA(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                 num_heads=num_heads, chunk_size=chunk_size, keep_last_n=keep_last_n)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], return_attn: bool=False):
        all_as = []
        for layer in self.layers:
            x, A = layer(x, attn_mask)
            if return_attn:
                all_as.append(A)
        return (x, all_as) if return_attn else (x, None)


# Placeholder for TransformerHier - use vanilla Transformer for now
class TransformerHier(nn.Module):
    """Placeholder for hierarchical transformer - uses vanilla transformer internally."""
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int, chunk_size: int):
        super().__init__()
        # For now, just use vanilla transformer
        self.transformer = Transformer(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                       num_heads=num_heads, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], return_attn: bool=False):
        return self.transformer(x, attn_mask, return_attn)


class TimeSeriesTransformer(nn.Module):
    """
    Supports: vanilla / hierarchical / project-then-attend (PTA).
    """
    def __init__(self, num_features, seq_len, dim=64, attn_dim=16, mlp_dim=128, num_heads=4, num_layers=1,
                 mode: str = "pta",    # "vanilla" | "hier" | "pta"
                 chunk_size: int = 30, keep_last_n: int = 1):
        super().__init__()
        assert mode in {"vanilla", "hier", "pta"}
        if mode in {"hier", "pta"}:
            assert seq_len % chunk_size == 0

        self.seq_len = seq_len
        self.in_proj = nn.Linear(num_features, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, dim))

        if mode == "vanilla":
            self.backbone = Transformer(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                        num_heads=num_heads, num_layers=num_layers)
        elif mode == "hier":
            self.backbone = TransformerHier(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                            num_heads=num_heads, num_layers=num_layers, chunk_size=chunk_size)
        else:  # "pta"
            self.backbone = TransformerPTA(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                           num_heads=num_heads, num_layers=num_layers,
                                           chunk_size=chunk_size, keep_last_n=keep_last_n)

        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.in_proj(x)                         # (B, T, D)
        h = h + self.pos_emb[:, :h.size(1)]         # (B, T, D)
        h, _ = self.backbone(h, attn_mask=None, return_attn=False)
        last = h[:, -1, :]
        y = self.head(last).squeeze(-1)
        return y


# Create and train the model
if __name__ == "__main__":
    print("Creating PTA Transformer model...")
    input_dim = len(feat_cols)  # 6 features
    seq_len = W  # 30
    
    # Set chunk_size to be a divisor of seq_len
    chunk_size = 10  # Should divide seq_len (30)
    assert seq_len % chunk_size == 0, f"chunk_size {chunk_size} must divide seq_len {seq_len}"
    
    model = TimeSeriesTransformer(
        num_features=input_dim,
        seq_len=seq_len,
        dim=128,
        attn_dim=64,
        mlp_dim=256,
        num_heads=4,
        num_layers=3,
        mode="pta",  # Use Project-Then-Attend mode
        chunk_size=chunk_size,
        keep_last_n=1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nStarting training...")
    
    # Train the model
    train_one_model(model, train_loader, test_loader, epochs=60, lr=1e-3)
    
    print("\nTraining completed!")

