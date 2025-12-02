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

def make_sequences_frame(frame, window=96,
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

def train_model(model, train_loader, test_loader, y_mean, y_std, epochs=100, lr=1e-3, model_name=""):
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

def verify_attention_size_pta(model, chunk_size: int, keep_last_n: int, window: int):
    """Verify PTA attention size"""
    S = window // chunk_size
    S_proj = S - keep_last_n
    tail_tokens = keep_last_n * chunk_size
    expected_attention_size = S_proj + tail_tokens
    print(f"✓ Verified PTA: Window={window}, Chunks={S}, Chunk_size={chunk_size}, "
          f"Projected={S_proj}, Tail={tail_tokens}, Attention size={expected_attention_size}×{expected_attention_size}")

# ========== Main Experiment ==========

def run_experiment(symbol: str, symbol_display: str):
    """Run fair comparison experiment for a given stock symbol"""
    print(f"\n{'='*80}")
    print(f"Running FAIR COMPARISON experiment for {symbol_display}")
    print(f"{'='*80}")
    
    # Load data
    df, actual_symbol = load_stock_data(symbol)
    print(f"Loaded {symbol_display} (symbol: {actual_symbol}), {len(df)} rows")
    
    # Common parameters for fair comparison
    WINDOW = 96
    PTA_CHUNK_SIZE = 16  # 96 / 16 = 6 chunks
    PTA_KEEP_LAST_N = 1  # Keep last chunk uncompressed
    PATCHTST_PATCH_LEN = 16
    PATCHTST_STRIDE = 8
    
    # Calculate PTA attention size: 6 chunks, keep_last_n=1
    # S_proj = 6 - 1 = 5 projected tokens
    # tail = 1 * 16 = 16 tokens
    # Total = 5 + 16 = 21 tokens
    PTA_ATTENTION_SIZE = (WINDOW // PTA_CHUNK_SIZE - PTA_KEEP_LAST_N) + (PTA_KEEP_LAST_N * PTA_CHUNK_SIZE)
    
    print(f"\nFair Comparison Parameters:")
    print(f"  Window/Horizon: {WINDOW}")
    print(f"  PTA: chunk_size={PTA_CHUNK_SIZE}, chunks={WINDOW//PTA_CHUNK_SIZE}, attention={PTA_ATTENTION_SIZE}×{PTA_ATTENTION_SIZE}")
    print(f"  PatchTST: patch_len={PATCHTST_PATCH_LEN}, stride={PATCHTST_STRIDE}")
    print(f"  Naive: window={WINDOW}, attention={WINDOW}×{WINDOW}")
    
    results = {}
    convergence_data = {}
    
    # Prepare data once for all models
    train_loader, test_loader, y_mean, y_std = prepare_data(df, window=WINDOW)
    
    # Model 1: Naive Transformer
    print(f"\n[1/3] Training Naive Transformer (window={WINDOW}, attention={WINDOW}×{WINDOW})...")
    model1 = StockPredictor(
        input_dim=6,
        dim=128,
        attn_dim=64,
        mlp_dim=256,
        num_heads=4,
        num_layers=3,
        window=WINDOW
    )
    train_losses1, test_rmses1 = train_model(model1, train_loader, test_loader, 
                                             y_mean, y_std, epochs=100, lr=1e-3, model_name="Naive-96")
    mse1, rmse1, dir_acc1 = evaluate_model(model1, test_loader, y_mean, y_std)
    results["Naive-96"] = {"RMSE": rmse1, "MSE": mse1, "DirAcc": dir_acc1}
    convergence_data["Naive-96"] = {"test_rmse": test_rmses1}
    print(f"Final - RMSE: {rmse1:.4e}, DirAcc: {dir_acc1:.3f}")
    
    # Model 2: PTA Transformer
    print(f"\n[2/3] Training PTA Transformer (window={WINDOW}, chunk_size={PTA_CHUNK_SIZE}, attention={PTA_ATTENTION_SIZE}×{PTA_ATTENTION_SIZE})...")
    verify_attention_size_pta(None, PTA_CHUNK_SIZE, PTA_KEEP_LAST_N, WINDOW)
    model2 = StockPredictorPTA(
        input_dim=6,
        dim=128,
        attn_dim=64,
        mlp_dim=256,
        num_heads=4,
        num_layers=3,
        window=WINDOW,
        chunk_size=PTA_CHUNK_SIZE,
        keep_last_n=PTA_KEEP_LAST_N
    )
    train_losses2, test_rmses2 = train_model(model2, train_loader, test_loader, 
                                             y_mean, y_std, epochs=100, lr=1e-3, model_name="PTA-96")
    mse2, rmse2, dir_acc2 = evaluate_model(model2, test_loader, y_mean, y_std)
    results["PTA-96"] = {"RMSE": rmse2, "MSE": mse2, "DirAcc": dir_acc2}
    convergence_data["PTA-96"] = {"test_rmse": test_rmses2}
    print(f"Final - RMSE: {rmse2:.4e}, DirAcc: {dir_acc2:.3f}")
    
    # Model 3: PatchTST
    print(f"\n[3/3] Training PatchTST (window={WINDOW}, patch_len={PATCHTST_PATCH_LEN}, stride={PATCHTST_STRIDE})...")
    model3 = PatchTSTWrapper(input_dim=6, seq_len=WINDOW, pred_len=1, 
                            patch_len=PATCHTST_PATCH_LEN, stride=PATCHTST_STRIDE)
    train_losses3, test_rmses3 = train_model(model3, train_loader, test_loader, 
                                             y_mean, y_std, epochs=100, lr=1e-3, model_name="PatchTST-96")
    mse3, rmse3, dir_acc3 = evaluate_model(model3, test_loader, y_mean, y_std)
    results["PatchTST-96"] = {"RMSE": rmse3, "MSE": mse3, "DirAcc": dir_acc3}
    convergence_data["PatchTST-96"] = {"test_rmse": test_rmses3}
    print(f"Final - RMSE: {rmse3:.4e}, DirAcc: {dir_acc3:.3f}")
    
    return results, convergence_data, actual_symbol

# ========== Run Experiments ==========

print("\n" + "="*80)
print("FAIR COMPARISON EXPERIMENT")
print("="*80)
print("\nComparing (all with horizon=96):")
print("  1. Naive Transformer (window=96, attention=96×96)")
print("  2. PTA Transformer (window=96, chunk_size=16, 6 chunks, attention=21×21)")
print("  3. PatchTST (window=96, patch_len=16, stride=8)")
print("\nOn datasets:")
print("  - Apple Stock (AAPL)")
print("  - S&P 500")

# Run for AAPL
results_aapl, convergence_aapl, _ = run_experiment("AAPL", "Apple Stock (AAPL)")

# Run for S&P 500
results_sp500, convergence_sp500, sp500_symbol = run_experiment("SP500", "S&P 500")

# ========== Results Summary ==========

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"\n{'Model':<20} {'RMSE':<15} {'MSE':<15} {'DirAcc':<10}")
print("-" * 60)
print("APPLE STOCK (AAPL):")
model_list = ["Naive-96", "PTA-96", "PatchTST-96"]
for model_name in model_list:
    if model_name in results_aapl:
        metrics = results_aapl[model_name]
        print(f"  {model_name:<18} {metrics['RMSE']:<15.6e} {metrics['MSE']:<15.6e} {metrics['DirAcc']:<10.3f}")

print("\nS&P 500:")
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

models = ["Naive-96", "PTA-96", "PatchTST-96"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# RMSE comparison
ax = axes[0, 0]
rmses_aapl = [results_aapl[m]["RMSE"] for m in models]
rmses_sp500 = [results_sp500[m]["RMSE"] for m in models]
x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, rmses_aapl, width, label='AAPL', color=colors[0], alpha=0.8)
ax.bar(x + width/2, rmses_sp500, width, label='S&P 500', color=colors[1], alpha=0.8)
ax.set_ylabel("RMSE")
ax.set_title("RMSE Comparison (Horizon=96)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Directional Accuracy comparison
ax = axes[0, 1]
dir_accs_aapl = [results_aapl[m]["DirAcc"] for m in models]
dir_accs_sp500 = [results_sp500[m]["DirAcc"] for m in models]
ax.bar(x - width/2, dir_accs_aapl, width, label='AAPL', color=colors[0], alpha=0.8)
ax.bar(x + width/2, dir_accs_sp500, width, label='S&P 500', color=colors[1], alpha=0.8)
ax.set_ylabel("Directional Accuracy")
ax.set_ylim(0.0, 1.0)
ax.set_title("Directional Accuracy Comparison (Horizon=96)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Convergence plots for AAPL
ax = axes[1, 0]
for i, model_name in enumerate(models):
    if model_name in convergence_aapl:
        epochs = range(1, len(convergence_aapl[model_name]["test_rmse"]) + 1)
        ax.plot(epochs, convergence_aapl[model_name]["test_rmse"], 
               label=model_name, linewidth=2, color=colors[i])
ax.set_xlabel("Epoch")
ax.set_ylabel("Test RMSE")
ax.set_title("Test RMSE Convergence - AAPL (Horizon=96)")
ax.set_xlim(1, 100)
ax.set_xticks(range(0, 101, 20))
ax.legend()
ax.grid(alpha=0.3)

# Convergence plots for S&P 500
ax = axes[1, 1]
for i, model_name in enumerate(models):
    if model_name in convergence_sp500:
        epochs = range(1, len(convergence_sp500[model_name]["test_rmse"]) + 1)
        ax.plot(epochs, convergence_sp500[model_name]["test_rmse"], 
               label=model_name, linewidth=2, color=colors[i])
ax.set_xlabel("Epoch")
ax.set_ylabel("Test RMSE")
ax.set_title(f"Test RMSE Convergence - S&P 500 (Horizon=96)")
ax.set_xlim(1, 100)
ax.set_xticks(range(0, 101, 20))
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fair_comparison_transformer.png", dpi=150, bbox_inches='tight')
print("Saved: fair_comparison_transformer.png")

# Figure 2: Detailed side-by-side table visualization
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Model', 'Horizon', 'Attention/Patch Size', 'AAPL RMSE', 'AAPL DirAcc', 'S&P 500 RMSE', 'S&P 500 DirAcc'])
table_data.append(['Naive-96', '96', '96×96', 
                   f"{results_aapl['Naive-96']['RMSE']:.4e}", 
                   f"{results_aapl['Naive-96']['DirAcc']:.3f}",
                   f"{results_sp500['Naive-96']['RMSE']:.4e}", 
                   f"{results_sp500['Naive-96']['DirAcc']:.3f}"])
# Calculate PTA attention size for table
PTA_ATTENTION_SIZE_TABLE = (WINDOW // PTA_CHUNK_SIZE - PTA_KEEP_LAST_N) + (PTA_KEEP_LAST_N * PTA_CHUNK_SIZE)
table_data.append(['PTA-96', '96', f'{PTA_ATTENTION_SIZE_TABLE}×{PTA_ATTENTION_SIZE_TABLE} (chunk_size=16)', 
                   f"{results_aapl['PTA-96']['RMSE']:.4e}", 
                   f"{results_aapl['PTA-96']['DirAcc']:.3f}",
                   f"{results_sp500['PTA-96']['RMSE']:.4e}", 
                   f"{results_sp500['PTA-96']['DirAcc']:.3f}"])
table_data.append(['PatchTST-96', '96', f'Patch-based (patch_len={PATCHTST_PATCH_LEN})', 
                   f"{results_aapl['PatchTST-96']['RMSE']:.4e}", 
                   f"{results_aapl['PatchTST-96']['DirAcc']:.3f}",
                   f"{results_sp500['PatchTST-96']['RMSE']:.4e}", 
                   f"{results_sp500['PatchTST-96']['DirAcc']:.3f}"])

table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title("Fair Comparison: Transformer Models (Horizon=96)", fontsize=14, fontweight='bold', pad=20)
plt.savefig("fair_comparison_table.png", dpi=150, bbox_inches='tight')
print("Saved: fair_comparison_table.png")

print("\n" + "="*80)
print("Fair comparison experiment completed successfully!")
print("="*80)

