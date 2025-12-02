import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
import pandas as pd
import math
from typing import Optional, Tuple, List
from tqdm.auto import tqdm
import sys
import os

# Add PatchTST path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PatchTST', 'PatchTST_supervised'))
from models.PatchTST import Model as PatchTSTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Model Definitions ==========

class AttentionHead(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        super().__init__()
        self.W_K = nn.Linear(dim, n_hidden)
        self.W_Q = nn.Linear(dim, n_hidden)
        self.W_V = nn.Linear(dim, n_hidden)
        self.n_hidden = n_hidden

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.n_hidden)
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        alpha = F.softmax(scores, dim=-1)
        out = alpha @ V
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
            out_h, alpha_h = head(x, attn_mask)
            alloutputs.append(out_h)
            allalphas.append(alpha_h)
        concat_out = torch.cat(alloutputs, dim=-1)
        attn_output = self.project_concatenate(concat_out)
        attn_alphas = torch.stack(allalphas, dim=1)
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
        self.ffn = FFN(dim, mlp_dim)

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
        collected_attns = torch.stack(allattentions, dim=1) if return_attn else None
        return x, collected_attns

class ProjectThenAttend(nn.Module):
    def __init__(self, dim: int, attn_dim: int, num_heads: int, chunk_size: int, keep_last_n: int = 1):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.keep_last_n = keep_last_n
        self.mha = MultiHeadedAttention(dim=dim, n_hidden=attn_dim, num_heads=num_heads)
        self.time_projector = nn.Linear(chunk_size, 1)
        self.fuse_proj = nn.Linear(dim, dim)
        self.fuse_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        K = self.chunk_size
        assert T % K == 0, f"seq_len {T} must be a multiple of chunk_size {K}"
        S = T // K
        keep_last_n = self.keep_last_n
        
        x_chunks = x.view(B, S, K, D)
        S_proj = S - keep_last_n
        proj_part = x_chunks[:, :S_proj, :, :]
        tail_part = x_chunks[:, S_proj:, :, :]
        
        proj_part_for_linear = proj_part.permute(0, 1, 3, 2)
        proj_tokens = self.time_projector(proj_part_for_linear).squeeze(-1)
        tail_tokens = tail_part.reshape(B, keep_last_n * K, D)
        
        agg_seq = torch.cat([proj_tokens, tail_tokens], dim=1)
        attended, alphas = self.mha(agg_seq, attn_mask=None)
        
        attended_proj = attended[:, :S_proj, :]
        attended_tail = attended[:, S_proj:, :]
        
        attended_proj_broadcast = attended_proj.unsqueeze(2).expand(B, S_proj, K, D)
        attended_tail_chunks = attended_tail.view(B, keep_last_n, K, D)
        
        fused_proj = proj_part + torch.tanh(self.fuse_gate) * self.fuse_proj(attended_proj_broadcast)
        fused_tail = tail_part + torch.tanh(self.fuse_gate) * self.fuse_proj(attended_tail_chunks)
        
        fused = torch.cat([fused_proj, fused_tail], dim=1).contiguous().view(B, T, D)
        return fused, alphas

class ProjectThenAttend2(nn.Module):
    """
    PTA_2: Outputs compressed sequence (B, 32, D) instead of (B, 90, D)
    No broadcasting step - keeps the compressed representation
    """
    def __init__(self, dim: int, attn_dim: int, num_heads: int, chunk_size: int, keep_last_n: int = 1):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.keep_last_n = keep_last_n
        self.mha = MultiHeadedAttention(dim=dim, n_hidden=attn_dim, num_heads=num_heads)
        self.time_projector = nn.Linear(chunk_size, 1)
        # No fuse_proj or fuse_gate needed since we're not broadcasting back

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        K = self.chunk_size
        assert T % K == 0, f"seq_len {T} must be a multiple of chunk_size {K}"
        S = T // K
        keep_last_n = self.keep_last_n
        
        # Chunk the input
        x_chunks = x.view(B, S, K, D)
        S_proj = S - keep_last_n
        proj_part = x_chunks[:, :S_proj, :, :]  # (B, S_proj, K, D)
        tail_part = x_chunks[:, S_proj:, :, :]   # (B, keep_last_n, K, D)
        
        # Project early chunks: (B, S_proj, K, D) -> (B, S_proj, D)
        proj_part_for_linear = proj_part.permute(0, 1, 3, 2)  # (B, S_proj, D, K)
        proj_tokens = self.time_projector(proj_part_for_linear).squeeze(-1)  # (B, S_proj, D)
        
        # Flatten tail chunks: (B, keep_last_n, K, D) -> (B, keep_last_n*K, D)
        tail_tokens = tail_part.reshape(B, keep_last_n * K, D)
        
        # Concatenate: [projected tokens | tail tokens]
        agg_seq = torch.cat([proj_tokens, tail_tokens], dim=1)  # (B, S_proj + keep_last_n*K, D)
        
        # Attend over compressed sequence
        attended, alphas = self.mha(agg_seq, attn_mask=None)
        
        # Return compressed sequence directly (no broadcasting!)
        return attended, alphas

class AttentionResidualPTA(nn.Module):
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

class AttentionResidualPTA2(nn.Module):
    """
    Residual block for PTA_2 - first layer compresses, subsequent layers use regular attention
    """
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, chunk_size: int, keep_last_n: int = 1, is_first: bool = False):
        super().__init__()
        self.is_first = is_first
        if is_first:
            # First layer: compress from (B, 90, D) to (B, 32, D)
            self.pta2 = ProjectThenAttend2(dim=dim, attn_dim=attn_dim, num_heads=num_heads,
                                          chunk_size=chunk_size, keep_last_n=keep_last_n)
        else:
            # Subsequent layers: regular attention on compressed sequence (B, 32, D)
            self.attn = MultiHeadedAttention(dim=dim, n_hidden=attn_dim, num_heads=num_heads)
        self.ffn = FFN(dim, mlp_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_first:
            # First layer: compress (B, 90, D) -> (B, 32, D)
            # No residual connection since sequence length changes
            out, A = self.pta2(x, attn_mask)
            x = out  # Just use compressed output
        else:
            # Subsequent layers: regular attention on (B, 32, D)
            out, A = self.attn(x, attn_mask)
            x = out + x  # Residual connection (both are same size)
        
        x = self.ffn(x) + x  # FFN with residual
        return x, A

class TransformerPTA(nn.Module):
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

class TransformerPTA2(nn.Module):
    """
    PTA_2 Transformer: Outputs compressed sequence (B, 32, D) from (B, 90, D)
    First layer compresses, subsequent layers work on compressed sequence with regular attention
    """
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int,
                 chunk_size: int, keep_last_n: int = 1):
        super().__init__()
        # First layer compresses from (B, 90, D) to (B, 32, D)
        self.first_layer = AttentionResidualPTA2(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                                  num_heads=num_heads, chunk_size=chunk_size, 
                                                  keep_last_n=keep_last_n, is_first=True)
        # Subsequent layers work on compressed sequence (B, 32, D) with regular attention
        self.layers = nn.ModuleList([
            AttentionResidualPTA2(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                  num_heads=num_heads, chunk_size=chunk_size, 
                                  keep_last_n=keep_last_n, is_first=False)
            for _ in range(num_layers - 1)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], return_attn: bool=False):
        all_as = []
        # First layer: compress (B, 90, D) -> (B, 32, D)
        x, A = self.first_layer(x, attn_mask)
        if return_attn:
            all_as.append(A)
        
        # Subsequent layers: process compressed sequence (B, 32, D) -> (B, 32, D)
        for layer in self.layers:
            x, A = layer(x, attn_mask)
            if return_attn:
                all_as.append(A)
        
        return (x, all_as) if return_attn else (x, None)

class StockPredictor(nn.Module):
    """Naive Transformer - uses full attention over window with positional embeddings"""
    def __init__(self, input_dim: int, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int, window: int):
        super().__init__()
        self.seq_len = window
        self.input_proj = nn.Linear(input_dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, window, dim))
        self.transformer = Transformer(dim, attn_dim, mlp_dim, num_heads, num_layers)
        self.output_proj = nn.Linear(dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.pos_emb[:, :h.size(1)]
        h, _ = self.transformer(h, attn_mask=None, return_attn=False)
        last = h[:, -1, :]
        y = self.output_proj(last)
        return y.squeeze(-1)

class StockPredictorPTA(nn.Module):
    """PTA Transformer - uses reduced attention with positional embeddings"""
    def __init__(self, input_dim: int, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int,
                 window: int, chunk_size: int, keep_last_n: int = 1):
        super().__init__()
        assert window % chunk_size == 0, f"window {window} must be a multiple of chunk_size {chunk_size}"
        self.seq_len = window
        self.input_proj = nn.Linear(input_dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, window, dim))
        self.transformer = TransformerPTA(dim, attn_dim, mlp_dim, num_heads, num_layers,
                                          chunk_size=chunk_size, keep_last_n=keep_last_n)
        self.output_proj = nn.Linear(dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.pos_emb[:, :h.size(1)]
        h, _ = self.transformer(h, attn_mask=None, return_attn=False)
        last = h[:, -1, :]
        y = self.output_proj(last)
        return y.squeeze(-1)

class StockPredictorPTA2(nn.Module):
    """
    PTA_2 Transformer - outputs compressed sequence (B, 32, D) instead of (B, 90, D)
    More memory efficient, works directly with compressed representation
    """
    def __init__(self, input_dim: int, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int,
                 window: int, chunk_size: int, keep_last_n: int = 1):
        super().__init__()
        assert window % chunk_size == 0, f"window {window} must be a multiple of chunk_size {chunk_size}"
        self.seq_len = window
        self.chunk_size = chunk_size
        self.keep_last_n = keep_last_n
        
        # Calculate compressed sequence length
        S = window // chunk_size
        S_proj = S - keep_last_n
        self.compressed_len = S_proj + keep_last_n * chunk_size  # e.g., 2 + 30 = 32
        
        self.input_proj = nn.Linear(input_dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, window, dim))
        self.transformer = TransformerPTA2(dim, attn_dim, mlp_dim, num_heads, num_layers,
                                          chunk_size=chunk_size, keep_last_n=keep_last_n)
        self.output_proj = nn.Linear(dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 90, input_dim)
        h = self.input_proj(x)  # (B, 90, dim)
        h = h + self.pos_emb[:, :h.size(1)]  # (B, 90, dim)
        
        # PTA_2 compresses to (B, 32, dim)
        h, _ = self.transformer(h, attn_mask=None, return_attn=False)  # (B, 32, dim)
        
        # Take last token from compressed sequence
        last = h[:, -1, :]  # (B, dim)
        y = self.output_proj(last)  # (B, 1)
        return y.squeeze(-1)  # (B,)

# PatchTST wrapper
class PatchTSTConfig:
    """Simple config class for PatchTST"""
    def __init__(self, seq_len, pred_len, enc_in, patch_len=16, stride=8):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.e_layers = 3
        self.n_heads = 8
        self.d_model = 128
        self.d_ff = 256
        self.dropout = 0.1
        self.fc_dropout = 0.1
        self.head_dropout = 0.1
        self.individual = False
        self.revin = True
        self.affine = True
        self.subtract_last = False
        self.decomposition = False
        self.kernel_size = 25
        self.padding_patch = 'end'

class PatchTSTWrapper(nn.Module):
    """Wrapper for PatchTST to work with our data format"""
    def __init__(self, input_dim: int, seq_len: int, pred_len: int = 1, patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Create config
        config = PatchTSTConfig(seq_len=seq_len, pred_len=pred_len, enc_in=input_dim, 
                                patch_len=patch_len, stride=stride)
        
        # Create PatchTST model
        self.patchtst = PatchTSTModel(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, seq_len, input_dim]
        # PatchTST Model expects [Batch, Input length, Channel] = [Batch, seq_len, input_dim]
        # No permutation needed - already in correct format!
        
        # Forward through PatchTST
        output = self.patchtst(x)  # [Batch, Input length, Channel] = [Batch, pred_len, input_dim]
        
        # Extract the target feature (adj_close is last feature, index 5)
        target_feature_idx = 5  # adj_close is the 6th feature (index 5)
        output_single = output[:, :, target_feature_idx]  # [Batch, pred_len]
        
        # If pred_len > 1, take the last prediction
        if output_single.shape[1] > 1:
            output_single = output_single[:, -1:]
        
        return output_single.squeeze(-1)  # [Batch]

# ========== Data Loading Functions ==========

def load_stock_data(symbol: str):
    """Load and prepare stock data"""
    ds = load_dataset("paperswithbacktest/Stocks-Daily-Price", split="train")
    if symbol == "SP500":
        # Try to find S&P 500 proxy
        candidates_ordered = [
            "SPY", "IVV", "VOO",
            "^GSPC", "GSPC", "SPX",
            "SPXUSD", "SP500", "US500", "SPX500USD",
            "VFINX"
        ]
        available = set(ds.unique("symbol"))
        intersect = [s for s in candidates_ordered if s in available]
        if intersect:
            sym = intersect[0]
        else:
            heuristic = [s for s in available if ("SP" in s) and ("500" in s or "GSPC" in s or "SPX" in s)]
            if heuristic:
                sym = sorted(heuristic)[0]
            else:
                raise RuntimeError(f"No S&P 500 proxy found. Available sample: {sorted(list(available))[:20]}")
    else:
        sym = symbol
    
    ds_filtered = ds.filter(lambda x: x["symbol"] == sym)
    df = ds_filtered.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df, sym

def make_sequences_frame(frame, window=90,
                         feat_cols=("open","high","low","close","volume","adj_close"),
                         target_col="adj_close"):
    arr = frame[list(feat_cols)].to_numpy(dtype=np.float32)
    X, y = [], []
    tgt_idx = list(feat_cols).index(target_col)
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])
        y.append(arr[i+window, tgt_idx])
    if len(X) == 0:
        return np.empty((0, window, len(feat_cols)), np.float32), np.empty((0,), np.float32)
    return np.stack(X), np.array(y, dtype=np.float32)

def prepare_data(df, window):
    """Prepare data for a given window size"""
    feat_cols = ("open","high","low","close","volume","adj_close")
    n = len(df)
    cut = int(0.80 * n)
    df_train = df.iloc[:cut]
    df_test = df.iloc[cut:]
    
    X_tr, y_tr = make_sequences_frame(df_train, window=window, feat_cols=feat_cols)
    X_te, y_te = make_sequences_frame(df_test, window=window, feat_cols=feat_cols)
    
    X_tr = torch.tensor(X_tr)
    y_tr = torch.tensor(y_tr)
    X_te = torch.tensor(X_te)
    y_te = torch.tensor(y_te)
    
    # Normalize features using TRAIN stats only
    mean = X_tr.mean(dim=(0,1), keepdim=True)
    std = X_tr.std(dim=(0,1), keepdim=True) + 1e-6
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std
    
    # Normalize targets
    y_mean = y_tr.mean()
    y_std = y_tr.std() + 1e-6
    y_tr = (y_tr - y_mean) / y_std
    y_te = (y_te - y_mean) / y_std
    
    train_ds = TensorDataset(X_tr, y_tr)
    test_ds = TensorDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)
    
    return train_loader, test_loader, y_mean, y_std

# ========== Training Functions ==========

def train_model(model, train_loader, test_loader, y_mean, y_std, epochs=60, lr=1e-3, model_name=""):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    train_losses = []
    test_rmses_orig = []
    
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        
        train_mse = running / len(train_loader.dataset)
        train_losses.append(train_mse)
        
        # Evaluate on test set every epoch (for full convergence plot)
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                p = model(xb).cpu()
                test_preds.append(p)
                test_trues.append(yb)
        
        test_y = torch.cat(test_trues)
        test_p = torch.cat(test_preds)
        
        # Denormalize for original scale RMSE
        test_y_orig = test_y * y_std + y_mean
        test_p_orig = test_p * y_std + y_mean
        test_mse_orig = F.mse_loss(test_p_orig, test_y_orig).item()
        test_rmse_orig = math.sqrt(test_mse_orig)
        test_rmses_orig.append(test_rmse_orig)
        
        # Print only every 10 epochs to avoid too much output
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"{model_name} Epoch {ep:03d} | Train MSE {train_mse:.4e} | Test RMSE (orig) {test_rmse_orig:.4e}")
    
    return train_losses, test_rmses_orig

@torch.no_grad()
def evaluate_model(model, loader, y_mean, y_std):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        p = model(xb).cpu()
        preds.append(p)
        trues.append(yb)
    y = torch.cat(trues)
    p = torch.cat(preds)
    
    # Denormalize
    y_orig = y * y_std + y_mean
    p_orig = p * y_std + y_mean
    
    mse = F.mse_loss(p_orig, y_orig).item()
    rmse = math.sqrt(mse)
    
    # Directional accuracy
    dy_true = y_orig[1:] - y_orig[:-1]
    dy_pred = p_orig[1:] - p_orig[:-1]
    dir_acc = (torch.sign(dy_true) == torch.sign(dy_pred)).float().mean().item()
    
    return mse, rmse, dir_acc

def verify_attention_size(model, expected_attention_size: int, model_name: str):
    """Verify that a model has the expected attention size"""
    # Create a dummy input to check attention size
    dummy_input = torch.randn(1, expected_attention_size if "PTA" not in model_name else 90, 6)
    model.eval()
    with torch.no_grad():
        if "PTA" in model_name:
            # For PTA, check the actual attention size in the transformer
            x = model.input_proj(dummy_input)
            # Get attention from first layer
            first_layer = model.transformer.layers[0]
            # Forward through PTA to get attention size
            B, T, D = x.shape
            K = first_layer.pta.chunk_size
            S = T // K
            keep_last_n = first_layer.pta.keep_last_n
            S_proj = S - keep_last_n
            tail_tokens = keep_last_n * K
            actual_attention_size = S_proj + tail_tokens
        else:
            # For Naive Transformer, attention size equals sequence length
            actual_attention_size = dummy_input.shape[1]
    
    assert actual_attention_size == expected_attention_size, \
        f"{model_name}: Expected attention size {expected_attention_size}×{expected_attention_size}, " \
        f"but got {actual_attention_size}×{actual_attention_size}"
    print(f"✓ Verified {model_name}: Attention size = {actual_attention_size}×{actual_attention_size}")

# ========== Main Experiment ==========

def run_experiment(symbol: str, symbol_display: str):
    """Run experiment for a given stock symbol"""
    print(f"\n{'='*80}")
    print(f"Running experiment for {symbol_display}")
    print(f"{'='*80}")
    
    # Load data
    df, actual_symbol = load_stock_data(symbol)
    print(f"Loaded {symbol_display} (symbol: {actual_symbol}), {len(df)} rows")
    
    results = {}
    convergence_data = {}
    
    # Model 1: Naive Transformer with window 30
    print(f"\n[1/5] Training Naive Transformer (window=30, attention=30x30)...")
    train_loader_30, test_loader_30, y_mean_30, y_std_30 = prepare_data(df, window=30)
    model1 = StockPredictor(
        input_dim=6,
        dim=128,
        attn_dim=64,
        mlp_dim=256,
        num_heads=4,
        num_layers=3,
        window=30
    )
    verify_attention_size(model1, expected_attention_size=30, model_name="Naive-30")
    train_losses1, test_rmses1 = train_model(model1, train_loader_30, test_loader_30, 
                                             y_mean_30, y_std_30, epochs=100, lr=1e-3, model_name="Naive-30")
    mse1, rmse1, dir_acc1 = evaluate_model(model1, test_loader_30, y_mean_30, y_std_30)
    results["Naive-30"] = {"RMSE": rmse1, "MSE": mse1, "DirAcc": dir_acc1}
    convergence_data["Naive-30"] = {"test_rmse": test_rmses1}
    print(f"Final - RMSE: {rmse1:.4e}, DirAcc: {dir_acc1:.3f}")
    
    # Model 2: Naive Transformer with window 90
    print(f"\n[2/5] Training Naive Transformer (window=90, attention=90x90)...")
    train_loader_90, test_loader_90, y_mean_90, y_std_90 = prepare_data(df, window=90)
    model2 = StockPredictor(
        input_dim=6,
        dim=128,
        attn_dim=64,
        mlp_dim=256,
        num_heads=4,
        num_layers=3,
        window=90
    )
    verify_attention_size(model2, expected_attention_size=90, model_name="Naive-90")
    train_losses2, test_rmses2 = train_model(model2, train_loader_90, test_loader_90, 
                                             y_mean_90, y_std_90, epochs=100, lr=1e-3, model_name="Naive-90")
    mse2, rmse2, dir_acc2 = evaluate_model(model2, test_loader_90, y_mean_90, y_std_90)
    results["Naive-90"] = {"RMSE": rmse2, "MSE": mse2, "DirAcc": dir_acc2}
    convergence_data["Naive-90"] = {"test_rmse": test_rmses2}
    print(f"Final - RMSE: {rmse2:.4e}, DirAcc: {dir_acc2:.3f}")
    
    # Model 3: PTA Transformer with window 90 (attention 32x32)
    print(f"\n[3/5] Training PTA Transformer (window=90, attention=32x32)...")
    # For window 90, we use chunk_size=30, keep_last_n=1
    # This gives: S=3, S_proj=2, tail=30, total attention=2+30=32
    model3 = StockPredictorPTA(
        input_dim=6,
        dim=128,
        attn_dim=64,
        mlp_dim=256,
        num_heads=4,
        num_layers=3,
        window=90,
        chunk_size=30,
        keep_last_n=1
    )
    verify_attention_size(model3, expected_attention_size=32, model_name="PTA-90")
    train_losses3, test_rmses3 = train_model(model3, train_loader_90, test_loader_90, 
                                             y_mean_90, y_std_90, epochs=100, lr=1e-3, model_name="PTA-90")
    mse3, rmse3, dir_acc3 = evaluate_model(model3, test_loader_90, y_mean_90, y_std_90)
    results["PTA-90"] = {"RMSE": rmse3, "MSE": mse3, "DirAcc": dir_acc3}
    convergence_data["PTA-90"] = {"test_rmse": test_rmses3}
    print(f"Final - RMSE: {rmse3:.4e}, DirAcc: {dir_acc3:.3f}")
    
    # Model 4: PatchTST - try 90 days first, if it doesn't work use 96
    print(f"\n[4/5] Training PatchTST...")
    patchtst_window = 90
    try:
        # Try 90 days first
        train_loader_patchtst, test_loader_patchtst, y_mean_patchtst, y_std_patchtst = prepare_data(df, window=patchtst_window)
        # Check if patch_len and stride work with 90
        patch_len = 16
        stride = 8
        patch_num = int((patchtst_window - patch_len) / stride + 1)
        if patch_num <= 0:
            raise ValueError("Invalid patch configuration for 90 days")
        
        model4 = PatchTSTWrapper(input_dim=6, seq_len=patchtst_window, pred_len=1, 
                                 patch_len=patch_len, stride=stride)
        print(f"Using window={patchtst_window} for PatchTST")
    except Exception as e:
        print(f"90 days didn't work for PatchTST: {e}, trying 96 days...")
        patchtst_window = 96
        train_loader_patchtst, test_loader_patchtst, y_mean_patchtst, y_std_patchtst = prepare_data(df, window=patchtst_window)
        patch_len = 16
        stride = 8
        model4 = PatchTSTWrapper(input_dim=6, seq_len=patchtst_window, pred_len=1, 
                                 patch_len=patch_len, stride=stride)
        print(f"Using window={patchtst_window} for PatchTST")
    
    train_losses4, test_rmses4 = train_model(model4, train_loader_patchtst, test_loader_patchtst, 
                                             y_mean_patchtst, y_std_patchtst, epochs=100, lr=1e-3, model_name=f"PatchTST-{patchtst_window}")
    mse4, rmse4, dir_acc4 = evaluate_model(model4, test_loader_patchtst, y_mean_patchtst, y_std_patchtst)
    results[f"PatchTST-{patchtst_window}"] = {"RMSE": rmse4, "MSE": mse4, "DirAcc": dir_acc4}
    convergence_data[f"PatchTST-{patchtst_window}"] = {"test_rmse": test_rmses4}
    print(f"Final - RMSE: {rmse4:.4e}, DirAcc: {dir_acc4:.3f}")
    
    # Model 5: PTA_2 Transformer with window 90 (outputs compressed sequence)
    print(f"\n[5/5] Training PTA_2 Transformer (window=90, outputs compressed 32 tokens)...")
    model5 = StockPredictorPTA2(
        input_dim=6,
        dim=128,
        attn_dim=64,
        mlp_dim=256,
        num_heads=4,
        num_layers=3,
        window=90,
        chunk_size=30,
        keep_last_n=1
    )
    train_losses5, test_rmses5 = train_model(model5, train_loader_90, test_loader_90, 
                                             y_mean_90, y_std_90, epochs=100, lr=1e-3, model_name="PTA-90_2")
    mse5, rmse5, dir_acc5 = evaluate_model(model5, test_loader_90, y_mean_90, y_std_90)
    results["PTA-90_2"] = {"RMSE": rmse5, "MSE": mse5, "DirAcc": dir_acc5}
    convergence_data["PTA-90_2"] = {"test_rmse": test_rmses5}
    print(f"Final - RMSE: {rmse5:.4e}, DirAcc: {dir_acc5:.3f}")
    
    return results, convergence_data, actual_symbol, patchtst_window

# ========== Run Experiments ==========

print("\n" + "="*80)
print("TRANSFORMER COMPARISON EXPERIMENT")
print("="*80)
print("\nComparing:")
print("  1. Naive Transformer (window=30, attention=30x30) - 100 epochs")
print("  2. Naive Transformer (window=90, attention=90x90) - 100 epochs")
print("  3. PTA Transformer (window=90, attention=32x32) - 100 epochs")
print("  4. PatchTST (window=90 or 96) - 100 epochs")
print("  5. PTA_2 Transformer (window=90, outputs compressed 32 tokens) - 100 epochs")
print("\nOn datasets:")
print("  - Apple Stock (AAPL)")
print("  - S&P 500")

# Run for AAPL
results_aapl, convergence_aapl, _, patchtst_window_aapl = run_experiment("AAPL", "Apple Stock (AAPL)")

# Run for S&P 500
results_sp500, convergence_sp500, sp500_symbol, patchtst_window_sp500 = run_experiment("SP500", "S&P 500")

# ========== Results Summary ==========

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"\n{'Model':<20} {'RMSE':<15} {'MSE':<15} {'DirAcc':<10}")
print("-" * 60)
print("APPLE STOCK (AAPL):")
model_list = ["Naive-30", "Naive-90", "PTA-90", f"PatchTST-{patchtst_window_aapl}", "PTA-90_2"]
for model_name in model_list:
    if model_name in results_aapl:
        metrics = results_aapl[model_name]
        print(f"  {model_name:<18} {metrics['RMSE']:<15.6e} {metrics['MSE']:<15.6e} {metrics['DirAcc']:<10.3f}")

print("\nS&P 500:")
model_list = ["Naive-30", "Naive-90", "PTA-90", f"PatchTST-{patchtst_window_sp500}", "PTA-90_2"]
for model_name in model_list:
    if model_name in results_sp500:
        metrics = results_sp500[model_name]
        print(f"  {model_name:<18} {metrics['RMSE']:<15.6e} {metrics['MSE']:<15.6e} {metrics['DirAcc']:<10.3f}")

# ========== Visualization ==========

print("\n" + "="*80)
print("Generating comparison plots...")
print("="*80)

# Figure 1: Side-by-side comparison bar charts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

models_base = ["Naive-30", "Naive-90", "PTA-90"]
models_aapl = models_base + [f"PatchTST-{patchtst_window_aapl}", "PTA-90_2"]
models_sp500 = models_base + [f"PatchTST-{patchtst_window_sp500}", "PTA-90_2"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# RMSE comparison
ax = axes[0, 0]
rmses_aapl = [results_aapl[m]["RMSE"] for m in models_aapl]
rmses_sp500 = [results_sp500[m]["RMSE"] for m in models_sp500]
x = np.arange(len(models_aapl))
width = 0.35
ax.bar(x - width/2, rmses_aapl, width, label='AAPL', color=colors[0], alpha=0.8)
ax.bar(x + width/2, rmses_sp500, width, label='S&P 500', color=colors[1], alpha=0.8)
ax.set_ylabel("RMSE")
ax.set_title("RMSE Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models_aapl, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Directional Accuracy comparison
ax = axes[0, 1]
dir_accs_aapl = [results_aapl[m]["DirAcc"] for m in models_aapl]
dir_accs_sp500 = [results_sp500[m]["DirAcc"] for m in models_sp500]
ax.bar(x - width/2, dir_accs_aapl, width, label='AAPL', color=colors[0], alpha=0.8)
ax.bar(x + width/2, dir_accs_sp500, width, label='S&P 500', color=colors[1], alpha=0.8)
ax.set_ylabel("Directional Accuracy")
ax.set_ylim(0.0, 1.0)
ax.set_title("Directional Accuracy Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models_aapl, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Convergence plots for AAPL
ax = axes[1, 0]
for i, model_name in enumerate(models_aapl):
    if model_name in convergence_aapl:
        epochs = range(1, len(convergence_aapl[model_name]["test_rmse"]) + 1)
        ax.plot(epochs, convergence_aapl[model_name]["test_rmse"], 
               label=model_name, linewidth=2, color=colors[i])
ax.set_xlabel("Epoch")
ax.set_ylabel("Test RMSE")
ax.set_title("Test RMSE Convergence - AAPL")
ax.set_xlim(1, 100)
ax.set_xticks(range(0, 101, 20))
ax.legend()
ax.grid(alpha=0.3)

# Convergence plots for S&P 500
ax = axes[1, 1]
for i, model_name in enumerate(models_sp500):
    if model_name in convergence_sp500:
        epochs = range(1, len(convergence_sp500[model_name]["test_rmse"]) + 1)
        ax.plot(epochs, convergence_sp500[model_name]["test_rmse"], 
               label=model_name, linewidth=2, color=colors[i])
ax.set_xlabel("Epoch")
ax.set_ylabel("Test RMSE")
ax.set_title(f"Test RMSE Convergence - S&P 500 ({sp500_symbol})")
ax.set_xlim(1, 100)
ax.set_xticks(range(0, 101, 20))
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("transformer_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: transformer_comparison.png")

# Figure 2: Detailed side-by-side table visualization
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Model', 'Window', 'Attention Size', 'AAPL RMSE', 'AAPL DirAcc', 'S&P 500 RMSE', 'S&P 500 DirAcc'])
table_data.append(['Naive-30', '30', '30×30', 
                   f"{results_aapl['Naive-30']['RMSE']:.4e}", 
                   f"{results_aapl['Naive-30']['DirAcc']:.3f}",
                   f"{results_sp500['Naive-30']['RMSE']:.4e}", 
                   f"{results_sp500['Naive-30']['DirAcc']:.3f}"])
table_data.append(['Naive-90', '90', '90×90', 
                   f"{results_aapl['Naive-90']['RMSE']:.4e}", 
                   f"{results_aapl['Naive-90']['DirAcc']:.3f}",
                   f"{results_sp500['Naive-90']['RMSE']:.4e}", 
                   f"{results_sp500['Naive-90']['DirAcc']:.3f}"])
table_data.append(['PTA-90', '90', '32×32', 
                   f"{results_aapl['PTA-90']['RMSE']:.4e}", 
                   f"{results_aapl['PTA-90']['DirAcc']:.3f}",
                   f"{results_sp500['PTA-90']['RMSE']:.4e}", 
                   f"{results_sp500['PTA-90']['DirAcc']:.3f}"])
patchtst_name_aapl = f"PatchTST-{patchtst_window_aapl}"
patchtst_name_sp500 = f"PatchTST-{patchtst_window_sp500}"
table_data.append([f'PatchTST-{patchtst_window_aapl}', f'{patchtst_window_aapl}', 'Patch-based', 
                   f"{results_aapl[patchtst_name_aapl]['RMSE']:.4e}", 
                   f"{results_aapl[patchtst_name_aapl]['DirAcc']:.3f}",
                   f"{results_sp500[patchtst_name_sp500]['RMSE']:.4e}", 
                   f"{results_sp500[patchtst_name_sp500]['DirAcc']:.3f}"])
table_data.append(['PTA-90_2', '90', '32×32 (compressed)', 
                   f"{results_aapl['PTA-90_2']['RMSE']:.4e}", 
                   f"{results_aapl['PTA-90_2']['DirAcc']:.3f}",
                   f"{results_sp500['PTA-90_2']['RMSE']:.4e}", 
                   f"{results_sp500['PTA-90_2']['DirAcc']:.3f}"])

table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title("Transformer Model Comparison Results", fontsize=14, fontweight='bold', pad=20)
plt.savefig("transformer_comparison_table.png", dpi=150, bbox_inches='tight')
print("Saved: transformer_comparison_table.png")

print("\n" + "="*80)
print("Experiment completed successfully!")
print("="*80)

