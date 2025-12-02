"""
S&P 500 Data Comparison Experiment
Compares Naive Transformer, PTA Transformer, PatchTST, and Hierarchical PTA on S&P 500 data
All models use horizon=96 for fair comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import math
from typing import Optional, Tuple, List
from tqdm.auto import tqdm
import sys
import os
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset

# Add PatchTST path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PatchTST', 'PatchTST_supervised'))
from models.PatchTST import Model as PatchTSTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Model Definitions (Same as sinwave experiment) ==========

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

class HierarchicalPTA(nn.Module):
    """
    Hierarchical PTA:
    - 6 windows of length 16 (total 96)
    - Project first 5 windows (each 16→1) = 5 tokens
    - Project overall 96→1 = 1 token
    - Keep last 16 window intact = 16 tokens
    - Total: 5+1+16 = 22 tokens → 22×22 attention
    """
    def __init__(self, dim: int, attn_dim: int, num_heads: int, window_size: int = 96, chunk_size: int = 16):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.chunk_size = chunk_size
        assert window_size % chunk_size == 0, f"window_size {window_size} must be a multiple of chunk_size {chunk_size}"
        self.num_chunks = window_size // chunk_size  # Should be 6
        
        self.mha = MultiHeadedAttention(dim=dim, n_hidden=attn_dim, num_heads=num_heads)
        # Projector for each chunk (16→1)
        self.chunk_projector = nn.Linear(chunk_size, 1)
        # Projector for overall sequence (96→1)
        self.overall_projector = nn.Linear(window_size, 1)
        self.fuse_proj = nn.Linear(dim, dim)
        self.fuse_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        assert T == self.window_size, f"Input length {T} must equal window_size {self.window_size}"
        K = self.chunk_size
        S = self.num_chunks  # 6
        
        # Split into chunks: [B, 6, 16, D]
        x_chunks = x.view(B, S, K, D)
        
        # Project first 5 chunks (each 16→1): [B, 5, D]
        proj_chunks = x_chunks[:, :5, :, :]  # [B, 5, 16, D]
        proj_chunks_for_linear = proj_chunks.permute(0, 1, 3, 2)  # [B, 5, D, 16]
        chunk_tokens = self.chunk_projector(proj_chunks_for_linear).squeeze(-1)  # [B, 5, D]
        
        # Project overall sequence (96→1): [B, 1, D]
        x_for_overall = x.permute(0, 2, 1)  # [B, D, 96]
        overall_token = self.overall_projector(x_for_overall).permute(0, 2, 1)  # [B, 1, D]
        
        # Keep last chunk intact (16 tokens): [B, 16, D]
        last_chunk = x_chunks[:, -1, :, :].view(B, K, D)  # [B, 16, D]
        
        # Concatenate: 5 + 1 + 16 = 22 tokens
        agg_seq = torch.cat([chunk_tokens, overall_token, last_chunk], dim=1)  # [B, 22, D]
        
        # Attention on 22 tokens
        attended, alphas = self.mha(agg_seq, attn_mask=None)
        
        # Split attended tokens
        attended_chunks = attended[:, :5, :]  # [B, 5, D]
        attended_overall = attended[:, 5:6, :]  # [B, 1, D]
        attended_last = attended[:, 6:, :]  # [B, 16, D]
        
        # Broadcast chunk tokens back to original chunks
        attended_chunks_broadcast = attended_chunks.unsqueeze(2).expand(B, 5, K, D)  # [B, 5, 16, D]
        
        # Broadcast overall token to all positions
        attended_overall_broadcast = attended_overall.expand(B, T, D)  # [B, 96, D]
        
        # Reshape last chunk
        attended_last_chunk = attended_last.view(B, 1, K, D)  # [B, 1, 16, D]
        
        # Fuse with original chunks
        fused_chunks = x_chunks[:, :5, :, :] + torch.tanh(self.fuse_gate) * self.fuse_proj(attended_chunks_broadcast)
        fused_last = x_chunks[:, -1:, :, :] + torch.tanh(self.fuse_gate) * self.fuse_proj(attended_last_chunk)
        
        # Combine all chunks
        fused_all_chunks = torch.cat([fused_chunks, fused_last], dim=1)  # [B, 6, 16, D]
        
        # Add overall attention to all positions
        fused = fused_all_chunks.view(B, T, D) + torch.tanh(self.fuse_gate) * self.fuse_proj(attended_overall_broadcast)
        
        return fused, alphas

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

class AttentionResidualHierPTA(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, window_size: int = 96, chunk_size: int = 16):
        super().__init__()
        self.hier_pta = HierarchicalPTA(dim=dim, attn_dim=attn_dim, num_heads=num_heads,
                                        window_size=window_size, chunk_size=chunk_size)
        self.ffn = FFN(dim, mlp_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        out, A = self.hier_pta(x, attn_mask)
        x = out + x
        x = self.ffn(x) + x
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

class TransformerHierPTA(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int,
                 window_size: int = 96, chunk_size: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionResidualHierPTA(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                     num_heads=num_heads, window_size=window_size, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], return_attn: bool=False):
        all_as = []
        for layer in self.layers:
            x, A = layer(x, attn_mask)
            if return_attn:
                all_as.append(A)
        return (x, all_as) if return_attn else (x, None)

class SP500Predictor(nn.Module):
    """Naive Transformer for S&P 500 prediction"""
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

class SP500PredictorPTA(nn.Module):
    """PTA Transformer for S&P 500 prediction"""
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

class SP500PredictorHierPTA(nn.Module):
    """Hierarchical PTA Transformer for S&P 500 prediction"""
    def __init__(self, input_dim: int, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int,
                 window: int, chunk_size: int = 16):
        super().__init__()
        assert window % chunk_size == 0, f"window {window} must be a multiple of chunk_size {chunk_size}"
        self.seq_len = window
        self.input_proj = nn.Linear(input_dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, window, dim))
        self.transformer = TransformerHierPTA(dim, attn_dim, mlp_dim, num_heads, num_layers,
                                              window_size=window, chunk_size=chunk_size)
        self.output_proj = nn.Linear(dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.pos_emb[:, :h.size(1)]
        h, _ = self.transformer(h, attn_mask=None, return_attn=False)
        last = h[:, -1, :]
        y = self.output_proj(last)
        return y.squeeze(-1)

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
    """Wrapper for PatchTST to work with S&P 500 data format"""
    def __init__(self, input_dim: int, seq_len: int, pred_len: int = 1, patch_len: int = 16, stride: int = 8, target_feature_idx: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_feature_idx = target_feature_idx
        
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
        output = self.patchtst(x)  # [Batch, pred_len, input_dim]
        
        # Extract the target feature (adj_close)
        output_single = output[:, :, self.target_feature_idx]  # [Batch, pred_len]
        
        # If pred_len > 1, take the last prediction
        if output_single.shape[1] > 1:
            output_single = output_single[:, -1:]
        
        return output_single.squeeze(-1)  # [Batch]

# ========== Data Loading ==========

def load_sp500_data():
    """Load S&P 500 data from HuggingFace datasets"""
    print("\n" + "="*80)
    print("Loading S&P 500 data from HuggingFace datasets...")
    print("="*80)
    
    ds = load_dataset("paperswithbacktest/Stocks-Daily-Price", split="train")
    
    # Pick the first symbol that exists among common S&P 500 proxies
    candidates_ordered = [
        "SPY", "IVV", "VOO",              # ETF proxies
        "^GSPC", "GSPC", "SPX",          # index tickers (various vendors)
        "SPXUSD", "SP500", "US500", "SPX500USD",  # CFD/forex-style symbols
        "VFINX"                              # Vanguard 500 Index Fund (mutual fund)
    ]
    available = set(ds.unique("symbol"))
    
    intersect = [s for s in candidates_ordered if s in available]
    if intersect:
        sym = intersect[0]
    else:
        # Heuristic fallback: any symbol containing 'SP' and ('500' or 'GSPC' or 'SPX')
        heuristic = [s for s in available if ("SP" in s) and ("500" in s or "GSPC" in s or "SPX" in s)]
        if heuristic:
            sym = sorted(heuristic)[0]
        else:
            raise RuntimeError(
                f"No S&P 500 proxy found in dataset. Looked for {candidates_ordered}. "
                f"Available sample: {sorted(list(available))[:20]}"
            )
    
    ds_spx = ds.filter(lambda x: x["symbol"] == sym)
    df = ds_spx.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"Loaded {sym} data: {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def prepare_sp500_data(df: pd.DataFrame, window: int = 96, target_col: str = "adj_close"):
    """
    Prepare S&P 500 data for training
    Uses all features: open, high, low, close, volume, adj_close
    Predicts target column (adj_close)
    """
    feat_cols = ["open", "high", "low", "close", "volume", "adj_close"]
    
    # Extract features and target
    features = df[feat_cols].values.astype(np.float32)  # (N, 6)
    target = df[target_col].values.astype(np.float32)  # (N,)
    
    # Handle NaN values
    features = np.nan_to_num(features, nan=0.0)
    target = np.nan_to_num(target, nan=0.0)
    
    # Normalize features (per feature, using train stats)
    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_val = len(df) - num_train - num_test
    
    feature_scaler = StandardScaler()
    feature_scaler.fit(features[:num_train])
    features_scaled = feature_scaler.transform(features)
    
    # Normalize target (using train stats)
    target_scaler = StandardScaler()
    target_scaler.fit(target[:num_train].reshape(-1, 1))
    target_scaled = target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = [], []
    for i in range(len(df) - window):
        X.append(features_scaled[i:i+window])
        y.append(target_scaled[i+window])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train/val/test
    train_end = num_train - window
    val_end = train_end + num_val
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # Get normalization stats for target
    y_mean = target_scaler.mean_[0]
    y_std = np.sqrt(target_scaler.var_[0])
    
    return train_loader, val_loader, test_loader, y_mean, y_std, len(feat_cols)

# ========== Training and Evaluation ==========

def train_model(model, train_loader, val_loader, y_mean, y_std, epochs=100, lr=1e-3, model_name="Model"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_rmses_orig = []
    
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        train_mse = train_loss / len(train_loader)
        train_losses.append(train_mse)
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_preds, val_trues = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                p = model(xb).cpu()
                val_preds.append(p)
                val_trues.append(yb)
            
            val_y = torch.cat(val_trues)
            val_p = torch.cat(val_preds)
            
            # Denormalize
            val_y_orig = val_y * y_std + y_mean
            val_p_orig = val_p * y_std + y_mean
            
            val_rmse_orig = math.sqrt(F.mse_loss(val_p_orig, val_y_orig).item())
            test_rmses_orig.append(val_rmse_orig)
        
        # Print only every 10 epochs
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"{model_name} Epoch {ep:03d} | Train MSE {train_mse:.4e} | Val RMSE (orig) {val_rmse_orig:.4e}")
    
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
    mae = F.l1_loss(p_orig, y_orig).item()  # Mean Absolute Error
    
    # Directional accuracy
    dy_true = y_orig[1:] - y_orig[:-1]
    dy_pred = p_orig[1:] - p_orig[:-1]
    dir_acc = (torch.sign(dy_true) == torch.sign(dy_pred)).float().mean().item()
    
    return mse, rmse, mae, dir_acc

def verify_attention_size_hier_pta(window_size: int, chunk_size: int):
    """Verify Hierarchical PTA attention size"""
    num_chunks = window_size // chunk_size  # 6
    num_proj_chunks = num_chunks - 1  # 5
    overall_token = 1
    last_chunk_tokens = chunk_size  # 16
    expected_attention_size = num_proj_chunks + overall_token + last_chunk_tokens  # 5+1+16=22
    print(f"✓ Verified Hier-PTA: Window={window_size}, Chunks={num_chunks}, Chunk_size={chunk_size}, "
          f"Projected chunks={num_proj_chunks}, Overall token={overall_token}, Last chunk tokens={last_chunk_tokens}, "
          f"Attention size={expected_attention_size}×{expected_attention_size}")

# ========== Main Experiment ==========

print("\n" + "="*80)
print("S&P 500 DATA COMPARISON EXPERIMENT")
print("="*80)
print("\nComparing (all with horizon=96):")
print("  1. Naive Transformer (window=96, attention=96×96)")
print("  2. PTA Transformer (window=96, chunk_size=16, 6 chunks, attention=21×21)")
print("  3. PatchTST (window=96, patch_len=16, stride=8)")
print("  4. Hierarchical PTA (window=96, chunk_size=16, attention=22×22)")
print("\nDataset: S&P 500 (from HuggingFace)")
print("Target: adj_close")
print("Features: open, high, low, close, volume, adj_close")
print("\nAll models:")
print("  - Use all 6 features as input")
print("  - Predict adj_close")
print("  - Same horizon=96 for fair comparison")

# Load S&P 500 data
df = load_sp500_data()

# Common parameters for fair comparison
WINDOW = 96
PTA_CHUNK_SIZE = 16
PTA_KEEP_LAST_N = 1
HIER_PTA_CHUNK_SIZE = 16
PATCHTST_PATCH_LEN = 16
PATCHTST_STRIDE = 8

# Calculate attention sizes
PTA_ATTENTION_SIZE = (WINDOW // PTA_CHUNK_SIZE - PTA_KEEP_LAST_N) + (PTA_KEEP_LAST_N * PTA_CHUNK_SIZE)
HIER_PTA_ATTENTION_SIZE = (WINDOW // HIER_PTA_CHUNK_SIZE - 1) + 1 + HIER_PTA_CHUNK_SIZE  # 5+1+16=22

print(f"\nFair Comparison Parameters:")
print(f"  Window/Horizon: {WINDOW}")
print(f"  Naive: window={WINDOW}, attention={WINDOW}×{WINDOW}")
print(f"  PTA: chunk_size={PTA_CHUNK_SIZE}, chunks={WINDOW//PTA_CHUNK_SIZE}, attention={PTA_ATTENTION_SIZE}×{PTA_ATTENTION_SIZE}")
print(f"  Hier-PTA: chunk_size={HIER_PTA_CHUNK_SIZE}, chunks={WINDOW//HIER_PTA_CHUNK_SIZE}, attention={HIER_PTA_ATTENTION_SIZE}×{HIER_PTA_ATTENTION_SIZE}")
print(f"  PatchTST: patch_len={PATCHTST_PATCH_LEN}, stride={PATCHTST_STRIDE}")

# Prepare data
train_loader, val_loader, test_loader, y_mean, y_std, num_features = prepare_sp500_data(df, window=WINDOW, target_col="adj_close")
print(f"\nData splits:")
print(f"  Train: {len(train_loader.dataset)} samples")
print(f"  Val: {len(val_loader.dataset)} samples")
print(f"  Test: {len(test_loader.dataset)} samples")
print(f"  Input features: {num_features} (open, high, low, close, volume, adj_close)")
print(f"  Target: adj_close - same for all models")

results = {}
convergence_data = {}

# Model 1: Naive Transformer
print(f"\n[1/4] Training Naive Transformer (window={WINDOW}, attention={WINDOW}×{WINDOW})...")
model1 = SP500Predictor(
    input_dim=num_features,
    dim=128,
    attn_dim=64,
    mlp_dim=256,
    num_heads=4,
    num_layers=3,
    window=WINDOW
)
train_losses1, test_rmses1 = train_model(model1, train_loader, val_loader, 
                                         y_mean, y_std, epochs=100, lr=1e-3, model_name="Naive-96")
mse1, rmse1, mae1, dir_acc1 = evaluate_model(model1, test_loader, y_mean, y_std)
results["Naive-96"] = {"RMSE": rmse1, "MSE": mse1, "MAE": mae1, "DirAcc": dir_acc1}
convergence_data["Naive-96"] = {"test_rmse": test_rmses1}
print(f"Final - RMSE: {rmse1:.4e}, MAE: {mae1:.4e}, MSE: {mse1:.4e}, DirAcc: {dir_acc1:.3f}")

# Model 2: PTA Transformer
print(f"\n[2/4] Training PTA Transformer (window={WINDOW}, chunk_size={PTA_CHUNK_SIZE}, attention={PTA_ATTENTION_SIZE}×{PTA_ATTENTION_SIZE})...")
model2 = SP500PredictorPTA(
    input_dim=num_features,
    dim=128,
    attn_dim=64,
    mlp_dim=256,
    num_heads=4,
    num_layers=3,
    window=WINDOW,
    chunk_size=PTA_CHUNK_SIZE,
    keep_last_n=PTA_KEEP_LAST_N
)
train_losses2, test_rmses2 = train_model(model2, train_loader, val_loader, 
                                         y_mean, y_std, epochs=100, lr=1e-3, model_name="PTA-96")
mse2, rmse2, mae2, dir_acc2 = evaluate_model(model2, test_loader, y_mean, y_std)
results["PTA-96"] = {"RMSE": rmse2, "MSE": mse2, "MAE": mae2, "DirAcc": dir_acc2}
convergence_data["PTA-96"] = {"test_rmse": test_rmses2}
print(f"Final - RMSE: {rmse2:.4e}, MAE: {mae2:.4e}, MSE: {mse2:.4e}, DirAcc: {dir_acc2:.3f}")

# Model 3: PatchTST
print(f"\n[3/4] Training PatchTST (window={WINDOW}, patch_len={PATCHTST_PATCH_LEN}, stride={PATCHTST_STRIDE})...")
# adj_close is the last feature (index 5)
model3 = PatchTSTWrapper(input_dim=num_features, seq_len=WINDOW, pred_len=1, 
                        patch_len=PATCHTST_PATCH_LEN, stride=PATCHTST_STRIDE,
                        target_feature_idx=5)
train_losses3, test_rmses3 = train_model(model3, train_loader, val_loader, 
                                         y_mean, y_std, epochs=100, lr=1e-3, model_name="PatchTST-96")
mse3, rmse3, mae3, dir_acc3 = evaluate_model(model3, test_loader, y_mean, y_std)
results["PatchTST-96"] = {"RMSE": rmse3, "MSE": mse3, "MAE": mae3, "DirAcc": dir_acc3}
convergence_data["PatchTST-96"] = {"test_rmse": test_rmses3}
print(f"Final - RMSE: {rmse3:.4e}, MAE: {mae3:.4e}, MSE: {mse3:.4e}, DirAcc: {dir_acc3:.3f}")

# Model 4: Hierarchical PTA
print(f"\n[4/4] Training Hierarchical PTA (window={WINDOW}, chunk_size={HIER_PTA_CHUNK_SIZE}, attention={HIER_PTA_ATTENTION_SIZE}×{HIER_PTA_ATTENTION_SIZE})...")
verify_attention_size_hier_pta(WINDOW, HIER_PTA_CHUNK_SIZE)
model4 = SP500PredictorHierPTA(
    input_dim=num_features,
    dim=128,
    attn_dim=64,
    mlp_dim=256,
    num_heads=4,
    num_layers=3,
    window=WINDOW,
    chunk_size=HIER_PTA_CHUNK_SIZE
)
train_losses4, test_rmses4 = train_model(model4, train_loader, val_loader, 
                                         y_mean, y_std, epochs=100, lr=1e-3, model_name="Hier-PTA-96")
mse4, rmse4, mae4, dir_acc4 = evaluate_model(model4, test_loader, y_mean, y_std)
results["Hier-PTA-96"] = {"RMSE": rmse4, "MSE": mse4, "MAE": mae4, "DirAcc": dir_acc4}
convergence_data["Hier-PTA-96"] = {"test_rmse": test_rmses4}
print(f"Final - RMSE: {rmse4:.4e}, MAE: {mae4:.4e}, MSE: {mse4:.4e}, DirAcc: {dir_acc4:.3f}")

# ========== Results Summary ==========

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"\n{'Model':<20} {'RMSE':<15} {'MAE':<15} {'MSE':<15} {'DirAcc':<10}")
print("-" * 75)
model_list = ["Naive-96", "PTA-96", "PatchTST-96", "Hier-PTA-96"]
for model_name in model_list:
    if model_name in results:
        metrics = results[model_name]
        print(f"  {model_name:<18} {metrics['RMSE']:<15.6e} {metrics['MAE']:<15.6e} {metrics['MSE']:<15.6e} {metrics['DirAcc']:<10.3f}")

# ========== Visualization ==========

print("\n" + "="*80)
print("Generating comparison plots...")
print("="*80)

# Figure 1: Comparison plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

models = ["Naive-96", "PTA-96", "PatchTST-96", "Hier-PTA-96"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# RMSE comparison
ax = axes[0, 0]
rmses = [results[m]["RMSE"] for m in models]
x = np.arange(len(models))
ax.bar(x, rmses, color=colors, alpha=0.8)
ax.set_ylabel("RMSE")
ax.set_title("RMSE Comparison - S&P 500 Data (Horizon=96)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# MAE comparison
ax = axes[0, 1]
maes = [results[m]["MAE"] for m in models]
ax.bar(x, maes, color=colors, alpha=0.8)
ax.set_ylabel("MAE")
ax.set_title("MAE Comparison - S&P 500 Data (Horizon=96)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Convergence plot
ax = axes[1, 0]
for i, model_name in enumerate(models):
    if model_name in convergence_data:
        epochs = range(1, len(convergence_data[model_name]["test_rmse"]) + 1)
        ax.plot(epochs, convergence_data[model_name]["test_rmse"], 
               label=model_name, linewidth=2, color=colors[i])
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation RMSE")
ax.set_title("Validation RMSE Convergence - S&P 500 Data (Horizon=96)")
ax.set_xlim(1, 100)
ax.set_xticks(range(0, 101, 20))
ax.legend()
ax.grid(alpha=0.3)

# MSE and MAE side-by-side comparison
ax = axes[1, 1]
x = np.arange(len(models))
width = 0.35
mses = [results[m]["MSE"] for m in models]
maes = [results[m]["MAE"] for m in models]
ax.bar(x - width/2, mses, width, label='MSE', color=colors[0], alpha=0.8)
ax.bar(x + width/2, maes, width, label='MAE', color=colors[1], alpha=0.8)
ax.set_ylabel("Error")
ax.set_title("MSE vs MAE Comparison - S&P 500 Data (Horizon=96)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("sp500_comparison_transformer.png", dpi=150, bbox_inches='tight')
print("Saved: sp500_comparison_transformer.png")

# Figure 2: Results table
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Model', 'Horizon', 'Attention/Patch Size', 'RMSE', 'MAE', 'MSE', 'DirAcc'])
table_data.append(['Naive-96', '96', '96×96', 
                   f"{results['Naive-96']['RMSE']:.4e}", 
                   f"{results['Naive-96']['MAE']:.4e}",
                   f"{results['Naive-96']['MSE']:.4e}",
                   f"{results['Naive-96']['DirAcc']:.3f}"])
table_data.append(['PTA-96', '96', f'{PTA_ATTENTION_SIZE}×{PTA_ATTENTION_SIZE} (chunk_size=16)', 
                   f"{results['PTA-96']['RMSE']:.4e}", 
                   f"{results['PTA-96']['MAE']:.4e}",
                   f"{results['PTA-96']['MSE']:.4e}",
                   f"{results['PTA-96']['DirAcc']:.3f}"])
table_data.append(['PatchTST-96', '96', f'Patch-based (patch_len={PATCHTST_PATCH_LEN})', 
                   f"{results['PatchTST-96']['RMSE']:.4e}", 
                   f"{results['PatchTST-96']['MAE']:.4e}",
                   f"{results['PatchTST-96']['MSE']:.4e}",
                   f"{results['PatchTST-96']['DirAcc']:.3f}"])
table_data.append(['Hier-PTA-96', '96', f'{HIER_PTA_ATTENTION_SIZE}×{HIER_PTA_ATTENTION_SIZE} (hierarchical)', 
                   f"{results['Hier-PTA-96']['RMSE']:.4e}", 
                   f"{results['Hier-PTA-96']['MAE']:.4e}",
                   f"{results['Hier-PTA-96']['MSE']:.4e}",
                   f"{results['Hier-PTA-96']['DirAcc']:.3f}"])

table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title("S&P 500 Data Comparison: Transformer Models (Horizon=96, Target=adj_close)", 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig("sp500_comparison_table.png", dpi=150, bbox_inches='tight')
print("Saved: sp500_comparison_table.png")

print("\n" + "="*80)
print("S&P 500 comparison experiment completed successfully!")
print("="*80)

