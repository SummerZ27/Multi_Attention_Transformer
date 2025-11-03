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


class StockPredictor(nn.Module):
    """Wrapper model that uses Transformer for stock price prediction."""
    def __init__(self, input_dim: int, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        # Project input features to transformer dimension
        self.input_proj = nn.Linear(input_dim, dim)
        self.transformer = Transformer(dim, attn_dim, mlp_dim, num_heads, num_layers)
        # Pooling: use mean of all time steps, then predict scalar
        self.output_proj = nn.Linear(dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) where F is number of features
        x = self.input_proj(x)  # (B, T, dim)
        x, _ = self.transformer(x, attn_mask=None, return_attn=False)  # (B, T, dim)
        # Mean pooling over time dimension
        x = x.mean(dim=1)  # (B, dim)
        # Predict scalar value
        x = self.output_proj(x)  # (B, 1)
        return x.squeeze(-1)  # (B,)


# Create and train the model
if __name__ == "__main__":
    print("Creating Transformer model...")
    input_dim = len(feat_cols)  # 6 features
    model = StockPredictor(
        input_dim=input_dim,
        dim=128,
        attn_dim=64,
        mlp_dim=256,
        num_heads=4,
        num_layers=3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nStarting training...")
    
    # Train the model
    train_one_model(model, train_loader, test_loader, epochs=60, lr=1e-3)
    
    print("\nTraining completed!")

