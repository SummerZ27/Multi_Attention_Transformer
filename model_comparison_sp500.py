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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Data Loading and Preparation (S&P 500 proxy) ==========
print("\n" + "="*60)
print("Loading and preparing S&P 500 (proxy) data...")
print("="*60)

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

n = len(df)
cut = int(0.80 * n)
df_train = df.iloc[:cut]
df_test = df.iloc[cut:]

print(f"{sym} rows: {len(df_train)} train, {len(df_test)} test")

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

feat_cols = ("open","high","low","close","volume","adj_close")
W = 90  # keep the same window as the original experiment

X_tr, y_tr = make_sequences_frame(df_train, window=W, feat_cols=feat_cols)
X_te, y_te = make_sequences_frame(df_test, window=W, feat_cols=feat_cols)

# Store unnormalized targets for metrics on original scale
y_tr_unnorm = y_tr.copy()
y_te_unnorm = y_te.copy()

X_tr_torch = torch.tensor(X_tr)
y_tr_torch = torch.tensor(y_tr)
X_te_torch = torch.tensor(X_te)
y_te_torch = torch.tensor(y_te)

# Normalize features using TRAIN stats only
mean = X_tr_torch.mean(dim=(0,1), keepdim=True)
std = X_tr_torch.std(dim=(0,1), keepdim=True) + 1e-6
X_tr_torch = (X_tr_torch - mean) / std
X_te_torch = (X_te_torch - mean) / std

# Normalize targets
y_mean = y_tr_torch.mean()
y_std = y_tr_torch.std() + 1e-6
y_tr_torch = (y_tr_torch - y_mean) / y_std
y_te_torch = (y_te_torch - y_mean) / y_std

# Prepare normalized NumPy arrays for sklearn models (use the exact same scaling)
X_tr_norm_np = X_tr_torch.numpy()
X_te_norm_np = X_te_torch.numpy()
y_tr_norm_np = y_tr_torch.numpy()
y_te_norm_np = y_te_torch.numpy()

BATCH = 64
train_ds = TensorDataset(X_tr_torch, y_tr_torch)
test_ds = TensorDataset(X_te_torch, y_te_torch)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, drop_last=False)

# ========== Model Definitions (same as original) ==========

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

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, seq_len, dim=64, attn_dim=16, mlp_dim=128, num_heads=4, num_layers=1,
                 mode: str = "pta", chunk_size: int = 30, keep_last_n: int = 1):
        super().__init__()
        assert mode in {"vanilla", "pta"}
        if mode == "pta":
            assert seq_len % chunk_size == 0

        self.seq_len = seq_len
        self.in_proj = nn.Linear(num_features, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, dim))

        if mode == "vanilla":
            self.backbone = Transformer(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                        num_heads=num_heads, num_layers=num_layers)
        else:  # "pta"
            self.backbone = TransformerPTA(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim,
                                           num_heads=num_heads, num_layers=num_layers,
                                           chunk_size=chunk_size, keep_last_n=keep_last_n)

        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.in_proj(x)
        h = h + self.pos_emb[:, :h.size(1)]
        h, _ = self.backbone(h, attn_mask=None, return_attn=False)
        last = h[:, -1, :]
        y = self.head(last).squeeze(-1)
        return y

class StockPredictor(nn.Module):
    def __init__(self, input_dim: int, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.transformer = Transformer(dim, attn_dim, mlp_dim, num_heads, num_layers)
        self.output_proj = nn.Linear(dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x, _ = self.transformer(x, attn_mask=None, return_attn=False)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return x.squeeze(-1)

class LSTM_Model(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)

# ========== Training Functions ==========

def train_neural_model(model, train_loader, test_loader, epochs=60, lr=1e-3, model_name=""):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    test_rmses = []
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
        
        # Evaluate on test set
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
        test_mse = F.mse_loss(test_p, test_y).item()
        test_rmse = math.sqrt(test_mse)
        test_losses.append(test_mse)
        test_rmses.append(test_rmse)

        # Also track RMSE on original (denormalized) target scale for fair comparison
        test_y_orig = test_y * y_std + y_mean
        test_p_orig = test_p * y_std + y_mean
        test_mse_orig = F.mse_loss(test_p_orig, test_y_orig).item()
        test_rmse_orig = math.sqrt(test_mse_orig)
        test_rmses_orig.append(test_rmse_orig)
        
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"{model_name} Epoch {ep:03d} | Train MSE {train_mse:.4e} | Test RMSE(norm) {test_rmse:.4e} | Test RMSE(orig) {test_rmse_orig:.4e}")
    
    return train_losses, test_losses, test_rmses, test_rmses_orig

@torch.no_grad()
def evaluate_neural_model(model, loader):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        p = model(xb).cpu()
        preds.append(p)
        trues.append(yb)
    y = torch.cat(trues)
    p = torch.cat(preds)
    
    # Compute metrics on original (denormalized) scale
    y_orig = y * y_std + y_mean
    p_orig = p * y_std + y_mean
    mse = F.mse_loss(p_orig, y_orig).item()
    rmse = math.sqrt(mse)
    
    # Directional accuracy
    dy_true = y_orig[1:] - y_orig[:-1]
    dy_pred = p_orig[1:] - p_orig[:-1]
    dir_acc = (torch.sign(dy_true) == torch.sign(dy_pred)).float().mean().item()
    
    return mse, rmse, dir_acc

# ========== Train All Models (same settings as original) ==========

results = {}
convergence_data = {}

print("\n" + "="*60)
print("Training Models (S&P 500 proxy)")
print("="*60)

# 1. PTA Transformer (3*30=90)
print("\n[1/6] Training PTA Transformer (window=90, chunks=3*30)...")
pta_model = TimeSeriesTransformer(
    num_features=len(feat_cols),
    seq_len=90,
    dim=128,
    attn_dim=64,
    mlp_dim=256,
    num_heads=4,
    num_layers=3,
    mode="pta",
    chunk_size=30,  # Split 90 into 3*30
    keep_last_n=1
)
train_losses_pta, test_losses_pta, test_rmses_pta, test_rmses_pta_orig = train_neural_model(
    pta_model, train_loader, test_loader, epochs=60, lr=1e-3, model_name="PTA Transformer"
)
convergence_data["PTA Transformer"] = {
    "train_loss": train_losses_pta,
    "test_loss": test_losses_pta,
    "test_rmse": test_rmses_pta,
    "test_rmse_orig": test_rmses_pta_orig
}
mse, rmse, dir_acc = evaluate_neural_model(pta_model, test_loader)
results["PTA Transformer"] = {"RMSE": rmse, "MSE": mse, "DirAcc": dir_acc}
print(f"Final - RMSE: {rmse:.4e}, DirAcc: {dir_acc:.3f}")

# 2. Naive Transformer
print("\n[2/6] Training Naive Transformer (window=90)...")
naive_model = StockPredictor(
    input_dim=len(feat_cols),
    dim=128,
    attn_dim=64,
    mlp_dim=256,
    num_heads=4,
    num_layers=3
)
train_losses_naive, test_losses_naive, test_rmses_naive, test_rmses_naive_orig = train_neural_model(
    naive_model, train_loader, test_loader, epochs=60, lr=1e-3, model_name="Naive Transformer"
)
convergence_data["Naive Transformer"] = {
    "train_loss": train_losses_naive,
    "test_loss": test_losses_naive,
    "test_rmse": test_rmses_naive,
    "test_rmse_orig": test_rmses_naive_orig
}
mse, rmse, dir_acc = evaluate_neural_model(naive_model, test_loader)
results["Naive Transformer"] = {"RMSE": rmse, "MSE": mse, "DirAcc": dir_acc}
print(f"Final - RMSE: {rmse:.4e}, DirAcc: {dir_acc:.3f}")

# 3. LSTM
print("\n[3/6] Training LSTM (window=90)...")
lstm_model = LSTM_Model(num_features=len(feat_cols), hidden_dim=128, num_layers=2)
train_losses_lstm, test_losses_lstm, test_rmses_lstm, test_rmses_lstm_orig = train_neural_model(
    lstm_model, train_loader, test_loader, epochs=60, lr=1e-3, model_name="LSTM"
)
convergence_data["LSTM"] = {
    "train_loss": train_losses_lstm,
    "test_loss": test_losses_lstm,
    "test_rmse": test_rmses_lstm,
    "test_rmse_orig": test_rmses_lstm_orig
}
mse, rmse, dir_acc = evaluate_neural_model(lstm_model, test_loader)
results["LSTM"] = {"RMSE": rmse, "MSE": mse, "DirAcc": dir_acc}
print(f"Final - RMSE: {rmse:.4e}, DirAcc: {dir_acc:.3f}")

# 4. Linear Regression (flatten window, normalized scale)
print("\n[4/6] Training Linear Regression...")
X_tr_flat = X_tr_norm_np.reshape(X_tr_norm_np.shape[0], -1)  # (N, window*features)
X_te_flat = X_te_norm_np.reshape(X_te_norm_np.shape[0], -1)
lr_model = LinearRegression()
lr_model.fit(X_tr_flat, y_tr_norm_np)
lr_pred_te_norm = lr_model.predict(X_te_flat)
# Inverse transform to original scale for fair comparison
y_mean_np = float(y_mean.item())
y_std_np = float(y_std.item())
lr_pred_te = lr_pred_te_norm * y_std_np + y_mean_np
lr_rmse = np.sqrt(mean_squared_error(y_te_unnorm, lr_pred_te))
lr_mse = mean_squared_error(y_te_unnorm, lr_pred_te)
# Directional accuracy on original scale
lr_dir_acc = np.mean(np.sign(np.diff(y_te_unnorm)) == np.sign(np.diff(lr_pred_te)))
results["Linear Regression"] = {"RMSE": lr_rmse, "MSE": lr_mse, "DirAcc": lr_dir_acc}
print(f"Linear Regression - RMSE: {lr_rmse:.4e}, DirAcc: {lr_dir_acc:.3f}")

# 5. Random Forest (normalized scale)
print("\n[5/6] Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_tr_flat, y_tr_norm_np)
rf_pred_te_norm = rf_model.predict(X_te_flat)
rf_pred_te = rf_pred_te_norm * y_std_np + y_mean_np
rf_rmse = np.sqrt(mean_squared_error(y_te_unnorm, rf_pred_te))
rf_mse = mean_squared_error(y_te_unnorm, rf_pred_te)
rf_dir_acc = np.mean(np.sign(np.diff(y_te_unnorm)) == np.sign(np.diff(rf_pred_te)))
results["Random Forest"] = {"RMSE": rf_rmse, "MSE": rf_mse, "DirAcc": rf_dir_acc}
print(f"Random Forest - RMSE: {rf_rmse:.4e}, DirAcc: {rf_dir_acc:.3f}")

# 6. XGBoost (normalized scale)
print("\n[6/6] Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_tr_flat, y_tr_norm_np)
xgb_pred_te_norm = xgb_model.predict(X_te_flat)
xgb_pred_te = xgb_pred_te_norm * y_std_np + y_mean_np
xgb_rmse = np.sqrt(mean_squared_error(y_te_unnorm, xgb_pred_te))
xgb_mse = mean_squared_error(y_te_unnorm, xgb_pred_te)
xgb_dir_acc = np.mean(np.sign(np.diff(y_te_unnorm)) == np.sign(np.diff(xgb_pred_te)))
results["XGBoost"] = {"RMSE": xgb_rmse, "MSE": xgb_mse, "DirAcc": xgb_dir_acc}
print(f"XGBoost - RMSE: {xgb_rmse:.4e}, DirAcc: {xgb_dir_acc:.3f}")

# ========== Results Summary ==========
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY (S&P 500 proxy)")
print("="*60)
print(f"{'Model':<20} {'RMSE':<15} {'MSE':<15} {'DirAcc':<10}")
print("-" * 60)
for model_name, metrics in sorted(results.items(), key=lambda x: x[1]["RMSE"]):
    print(f"{model_name:<20} {metrics['RMSE']:<15.6e} {metrics['MSE']:<15.6e} {metrics['DirAcc']:<10.3f}")

print("\n" + "="*60)
print("Generating simplified comparison plots (S&P 500 proxy)...")
print("="*60)

# Figure 1: Model Performance (original price scale)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models = list(results.keys())
rmses = [results[m]["RMSE"] for m in models]
dir_accs = [results[m]["DirAcc"] for m in models]

axes[0].bar(models, rmses, color=sns.color_palette("husl", len(models)))
axes[0].set_ylabel("RMSE (original scale)")
axes[0].set_title("Model RMSE (Original Scale) - S&P proxy")
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(models, dir_accs, color=sns.color_palette("husl", len(models)))
axes[1].set_ylabel("Directional Accuracy")
axes[1].set_ylim(0.0, 1.0)
axes[1].set_title("Directional Accuracy - S&P proxy")
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("model_performance_sp500.png", dpi=150, bbox_inches='tight')
print("Saved: model_performance_sp500.png")

# Figure 2: Neural Models Test RMSE Convergence (original scale)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
neural_models = ["PTA Transformer", "Naive Transformer", "LSTM"]
for model_name in neural_models:
    if model_name in convergence_data and "test_rmse_orig" in convergence_data[model_name]:
        epochs = range(1, len(convergence_data[model_name]["test_rmse_orig"]) + 1)
        ax.plot(epochs, convergence_data[model_name]["test_rmse_orig"], label=model_name, linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Test RMSE (original scale)")
ax.set_title("Neural Models Test RMSE Convergence (Original Scale) - S&P proxy")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("neural_convergence_sp500.png", dpi=150, bbox_inches='tight')
print("Saved: neural_convergence_sp500.png")

print("\n" + "="*60)
print("All simplified plots saved successfully (S&P 500 proxy)!")
print("="*60)


