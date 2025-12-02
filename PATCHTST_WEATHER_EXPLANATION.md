# How PatchTST Works with Weather.csv

## Weather Data Structure

Your `weather.csv` has:
- **22 columns total**: 1 date column + 21 weather features
- **Features**: pressure, temperature, humidity, wind, rain, solar radiation, CO2, etc.
- **Format**: Multivariate time series (multiple features measured over time)

## How PatchTST Processes Weather Data

### 1. **Data Format**
PatchTST expects: `[Batch, seq_len, num_features]`
- **seq_len**: Input sequence length (e.g., 96 timesteps)
- **num_features**: Number of weather features (21 for weather.csv)
- **Batch**: Number of samples

### 2. **Channel-Independent Processing**
PatchTST processes each feature **independently**:
- Each of the 21 weather features is treated as a separate channel
- Patches are created **along the time dimension** for each channel
- This is called "channel-independent" processing

### 3. **Patching Mechanism**

For weather data with `seq_len=96`, `patch_len=16`, `stride=8`:

```
For EACH of the 21 features:
  Input: [96 timesteps] of that feature
         ↓
  Create patches: [patch1: t0-t15, patch2: t8-t23, patch3: t16-t31, ...]
         ↓
  Number of patches: (96 - 16) / 8 + 1 = 11 patches
         ↓
  Each patch: [16 timesteps] → projected to [d_model dimensions]
         ↓
  Transformer processes: [11 patch tokens]
```

**Visual Example for ONE feature:**
```
Feature: Temperature
Timesteps: [t0, t1, ..., t95] (96 values)
         ↓
Patches (patch_len=16, stride=8):
  Patch 1: [t0-t15]   → Token 1
  Patch 2: [t8-t23]   → Token 2
  Patch 3: [t16-t31]  → Token 3
  ...
  Patch 11: [t88-t95] → Token 11
         ↓
Attention over 11 patch tokens
         ↓
Output: [11 patch tokens] → Head → [pred_len predictions]
```

### 4. **RevIN (Reversible Instance Normalization)**
- **Applied per channel**: Each of the 21 features is normalized independently
- **Before processing**: Normalize each feature's time series
- **After processing**: Denormalize predictions back to original scale
- **Purpose**: Handles different scales across features (pressure vs temperature vs humidity)

### 5. **Multivariate Prediction**

PatchTST can predict in two modes:

**Mode M (Multivariate → Multivariate):**
- Input: All 21 features for 96 timesteps
- Output: All 21 features for `pred_len` timesteps
- Each feature predicted independently

**Mode MS (Multivariate → Single):**
- Input: All 21 features for 96 timesteps  
- Output: One target feature (e.g., temperature) for `pred_len` timesteps
- Uses all features to predict one target

### 6. **Complete Flow**

```
Input: (Batch, 96, 21)
    │
    ├─→ For each of 21 features:
    │   │
    │   ├─→ RevIN Normalize: (Batch, 96)
    │   │
    │   ├─→ Create Patches: (Batch, 11, 16)
    │   │   - patch_len=16, stride=8
    │   │   - 11 patches per feature
    │   │
    │   ├─→ Project Patches: (Batch, 11, d_model)
    │   │   - Each patch → d_model dimensions
    │   │
    │   ├─→ Transformer: (Batch, 11, d_model)
    │   │   - Attention over 11 patch tokens
    │   │
    │   ├─→ Head: (Batch, 11, d_model) → (Batch, pred_len)
    │   │   - Predict future timesteps
    │   │
    │   └─→ RevIN Denormalize: (Batch, pred_len)
    │
    └─→ Stack all features: (Batch, pred_len, 21)
```

### 7. **Key Differences from Stock Data**

| Aspect | Stock Data | Weather Data |
|--------|------------|--------------|
| **Features** | 6 (OHLCV + adj_close) | 21 (pressure, temp, humidity, etc.) |
| **Scales** | Similar (all prices) | Very different (pressure vs temp vs wind) |
| **RevIN** | Less critical | **Critical** (handles scale differences) |
| **Prediction** | Single value (next day) | Multiple timesteps (pred_len) |

### 8. **Why PatchTST Works Well for Weather**

1. **Handles Multiple Scales**: RevIN normalizes each feature independently
2. **Efficient**: Patches reduce attention complexity
3. **Channel-Independent**: Each feature processed separately (good for diverse features)
4. **Long Sequences**: Can handle long input sequences efficiently

### 9. **Configuration for Weather.csv**

From the PatchTST weather script:
```python
seq_len = 336        # Input: 336 timesteps
pred_len = 96        # Predict: 96 timesteps ahead
enc_in = 21          # 21 weather features
patch_len = 16       # Each patch = 16 timesteps
stride = 8           # Overlap between patches
d_model = 128        # Model dimension
n_heads = 16         # Attention heads
e_layers = 3         # Encoder layers
features = 'M'       # Multivariate → Multivariate
```

### 10. **Data Loading Process**

```python
# 1. Load CSV
df = pd.read_csv('weather.csv')  # Shape: (N, 22)

# 2. Extract features (skip date column)
features = df.columns[1:]  # 21 features
data = df[features].values  # Shape: (N, 21)

# 3. Normalize (per feature, using train stats)
scaler.fit(train_data)  # Fit on training data only
data_scaled = scaler.transform(data)  # Shape: (N, 21)

# 4. Create sequences
# For each sample i:
#   Input: data_scaled[i:i+seq_len]  # (seq_len, 21)
#   Target: data_scaled[i+seq_len:i+seq_len+pred_len]  # (pred_len, 21)
```

### 11. **PatchTST Architecture for Weather**

```
Input: (B, 336, 21)  [336 timesteps, 21 features]
    │
    ├─→ RevIN (per channel): (B, 336, 21)
    │
    ├─→ For each feature (channel-independent):
    │   │
    │   ├─→ Extract feature: (B, 336)
    │   │
    │   ├─→ Create patches: (B, 41, 16)
    │   │   - (336-16)/8 + 1 = 41 patches
    │   │
    │   ├─→ Project: (B, 41, 128)
    │   │
    │   ├─→ Transformer: (B, 41, 128)
    │   │   - Attention over 41 patch tokens
    │   │
    │   └─→ Head: (B, 41, 128) → (B, 96)
    │       - Predict 96 future timesteps
    │
    ├─→ Stack: (B, 96, 21)
    │
    └─→ RevIN Denormalize: (B, 96, 21)
```

### 12. **Comparison with Our Stock Prediction**

**Stock Prediction (Our Use Case):**
- Input: (B, 96, 6) - 96 days, 6 features
- Output: (B, 1) - Predict next day's price
- Single-step prediction

**Weather Prediction (Standard PatchTST):**
- Input: (B, 336, 21) - 336 timesteps, 21 features
- Output: (B, 96, 21) - Predict 96 timesteps ahead for all features
- Multi-step prediction

### 13. **Adapting PatchTST for Single-Step Stock Prediction**

We adapted PatchTST to:
1. Use `pred_len=1` (predict 1 step ahead)
2. Extract only the target feature (adj_close) from output
3. Work with our data format `[Batch, seq_len, features]`

## Summary

PatchTST on weather data:
- **Processes each feature independently** (channel-independent)
- **Uses patches** to reduce attention complexity
- **Uses RevIN** to handle different feature scales
- **Can predict multiple timesteps** for all features
- **Efficient** for long sequences with many features

For our comparison, we'll adapt it to:
- Predict single timestep (like stock prediction)
- Use same horizon (96) for fair comparison
- Extract target feature from multivariate output



