# PTA (Project-Then-Attend) Architecture Explanation

## Overview

PTA is a **memory-efficient transformer architecture** designed to handle long sequences by reducing the quadratic attention complexity from O(T²) to O(L²) where L << T.

## Key Idea

Instead of attending over all T tokens, PTA:
1. **Chunks** the sequence into smaller pieces
2. **Projects** (compresses) early chunks into single tokens
3. **Attends** over the compressed sequence + recent tokens
4. **Broadcasts** the attended information back to original positions

## Architecture Breakdown

### Input Format
- Input: `x` with shape `(Batch, T, D)` where:
  - `T` = sequence length (e.g., 90 days)
  - `D` = feature dimension (e.g., 128)
  - `K` = chunk size (e.g., 30)
  - `S` = number of chunks = `T / K` (e.g., 90/30 = 3)

### Step-by-Step Process

#### Step 1: Chunking
```
Input: (B, T=90, D)
       ↓
Reshape into chunks: (B, S=3, K=30, D)
```

The sequence is divided into `S` chunks, each of size `K`.

**Example with T=90, K=30:**
```
Original: [t1, t2, ..., t30, t31, ..., t60, t61, ..., t90]
Chunks:   [Chunk1: t1-t30,  Chunk2: t31-t60,  Chunk3: t61-t90]
```

#### Step 2: Split into Projected and Tail Parts
```
Chunks: (B, S=3, K=30, D)
         ↓
Split:
- proj_part: (B, S_proj=2, K=30, D)  ← First S-keep_last_n chunks
- tail_part:  (B, keep_last_n=1, K=30, D)  ← Last keep_last_n chunks
```

**With keep_last_n=1:**
- **Projected part**: Chunks 1-2 (tokens 1-60) → will be compressed
- **Tail part**: Chunk 3 (tokens 61-90) → kept as-is for fine-grained attention

#### Step 3: Projection (Time Compression)
```
proj_part: (B, S_proj=2, K=30, D)
           ↓ permute(0,1,3,2)
           (B, S_proj=2, D, K=30)
           ↓ time_projector (Linear(K→1))
proj_tokens: (B, S_proj=2, D)
```

Each chunk of `K` timesteps is compressed into a single token using a learned linear projection.

**Visual Example:**
```
Chunk 1: [t1, t2, ..., t30] → Projected Token 1
Chunk 2: [t31, t32, ..., t60] → Projected Token 2
```

The `time_projector` learns to aggregate temporal information within each chunk.

#### Step 4: Prepare Tail Tokens
```
tail_part: (B, keep_last_n=1, K=30, D)
           ↓ reshape
tail_tokens: (B, keep_last_n*K=30, D)
```

The last chunk(s) are kept as individual tokens (not compressed).

**Visual Example:**
```
Tail: [t61, t62, ..., t90] → kept as 30 individual tokens
```

#### Step 5: Concatenate and Attend
```
agg_seq = [proj_tokens (2 tokens) | tail_tokens (30 tokens)]
         = (B, 32, D)
         ↓ Multi-Head Attention
attended: (B, 32, D)
```

**Attention Complexity:**
- **Naive Transformer**: O(90²) = 8,100 attention operations
- **PTA**: O(32²) = 1,024 attention operations
- **Reduction**: ~87% fewer operations!

#### Step 6: Split and Broadcast Back
```
attended: (B, 32, D)
          ↓ split
- attended_proj: (B, S_proj=2, D)
- attended_tail: (B, keep_last_n*K=30, D)
```

**Broadcast projected tokens:**
```
attended_proj: (B, 2, D)
               ↓ unsqueeze + expand
attended_proj_broadcast: (B, 2, K=30, D)
```

Each projected token is broadcast back to fill its original chunk.

**Reshape tail tokens:**
```
attended_tail: (B, 30, D)
               ↓ reshape
attended_tail_chunks: (B, 1, K=30, D)
```

#### Step 7: Fusion (Residual Connection)
```
fused_proj = proj_part + tanh(fuse_gate) * fuse_proj(attended_proj_broadcast)
fused_tail = tail_part + tanh(fuse_gate) * fuse_proj(attended_tail_chunks)
```

The attended information is fused back with the original tokens using:
- **Residual connection**: `original + attended`
- **Gated fusion**: `tanh(fuse_gate)` controls the strength of fusion
- **Projection**: `fuse_proj` transforms the attended features

#### Step 8: Reconstruct Full Sequence
```
fused = [fused_proj | fused_tail]
       = (B, S=3, K=30, D)
       ↓ view
       (B, T=90, D)
```

## Complete Flow Diagram

```
Input: (B, 90, D)
    │
    ├─→ Chunk: (B, 3, 30, D)
    │
    ├─→ Split:
    │   ├─→ proj_part: (B, 2, 30, D) ──┐
    │   └─→ tail_part: (B, 1, 30, D)   │
    │                                    │
    ├─→ Project: (B, 2, D) ─────────────┤
    │                                    │
    ├─→ Flatten tail: (B, 30, D) ───────┤
    │                                    │
    └─→ Concat: (B, 32, D) ──────────────┘
            │
            ├─→ Attention: (B, 32, D)
            │
            ├─→ Split:
            │   ├─→ attended_proj: (B, 2, D) ──→ Broadcast: (B, 2, 30, D)
            │   └─→ attended_tail: (B, 30, D) ─→ Reshape: (B, 1, 30, D)
            │
            └─→ Fuse & Reconstruct: (B, 90, D)
```

## Key Components

### 1. Time Projector (`time_projector`)
- **Purpose**: Compress K timesteps into 1 token
- **Type**: `nn.Linear(chunk_size, 1)`
- **Learns**: How to aggregate temporal information within chunks

### 2. Multi-Head Attention (`mha`)
- **Purpose**: Attend over compressed sequence
- **Input size**: `S_proj + keep_last_n*K` (much smaller than T)
- **Output**: Contextualized representations

### 3. Fusion Gate (`fuse_gate`)
- **Purpose**: Control how much attended information to fuse
- **Type**: Learnable parameter (initialized to 0)
- **Activation**: `tanh(fuse_gate)` ensures values in [-1, 1]

### 4. Fusion Projection (`fuse_proj`)
- **Purpose**: Transform attended features before fusion
- **Type**: `nn.Linear(dim, dim)`

## Attention Size Comparison

For **T=90, K=30, keep_last_n=1**:

| Model | Attention Size | Complexity |
|-------|---------------|------------|
| Naive Transformer | 90×90 | O(8,100) |
| PTA | 32×32 | O(1,024) |
| **Reduction** | **87%** | **87%** |

**Calculation:**
- `S = 90/30 = 3`
- `S_proj = 3 - 1 = 2`
- `tail_tokens = 1 × 30 = 30`
- **Total attention size = 2 + 30 = 32**

## Advantages

1. **Memory Efficient**: Reduces attention memory from O(T²) to O(L²)
2. **Scalable**: Can handle much longer sequences
3. **Preserves Recent Information**: Last chunk(s) kept at full resolution
4. **Learnable Compression**: Time projector learns optimal aggregation
5. **Residual Connections**: Preserves original information

## Trade-offs

1. **Information Loss**: Early chunks compressed (may lose fine-grained details)
2. **Chunk Size Sensitivity**: Performance depends on choosing good chunk size
3. **Boundary Effects**: Information at chunk boundaries may be less well-attended

## Hyperparameters

- **`chunk_size` (K)**: Size of each chunk (e.g., 30)
- **`keep_last_n`**: Number of chunks to keep uncompressed (e.g., 1)
- **`dim`**: Feature dimension (e.g., 128)
- **`attn_dim`**: Attention hidden dimension (e.g., 64)
- **`num_heads`**: Number of attention heads (e.g., 4)

## Example Configuration

For window=90:
```python
chunk_size = 30      # Split 90 into 3 chunks of 30
keep_last_n = 1      # Keep last chunk uncompressed
# Results in: 2 projected tokens + 30 tail tokens = 32 attention size
```

## Code Flow Summary

```python
# 1. Chunk
x_chunks = x.view(B, S, K, D)

# 2. Split
proj_part = x_chunks[:, :S_proj, :, :]
tail_part = x_chunks[:, S_proj:, :, :]

# 3. Project
proj_tokens = time_projector(proj_part.permute(0,1,3,2)).squeeze(-1)

# 4. Prepare tail
tail_tokens = tail_part.reshape(B, keep_last_n*K, D)

# 5. Attend
agg_seq = torch.cat([proj_tokens, tail_tokens], dim=1)
attended, alphas = mha(agg_seq)

# 6. Broadcast back
attended_proj_broadcast = attended[:, :S_proj].unsqueeze(2).expand(B, S_proj, K, D)
attended_tail_chunks = attended[:, S_proj:].view(B, keep_last_n, K, D)

# 7. Fuse
fused_proj = proj_part + tanh(fuse_gate) * fuse_proj(attended_proj_broadcast)
fused_tail = tail_part + tanh(fuse_gate) * fuse_proj(attended_tail_chunks)

# 8. Reconstruct
fused = torch.cat([fused_proj, fused_tail], dim=1).view(B, T, D)
```

## Comparison with Other Architectures

| Architecture | Attention Complexity | Memory | Recent Info |
|--------------|---------------------|--------|-------------|
| **Naive Transformer** | O(T²) | High | Full |
| **PTA** | O(L²) where L<<T | Low | Full (last chunk) |
| **Sparse Attention** | O(T√T) | Medium | Full |
| **Linformer** | O(T) | Low | Compressed |

PTA provides a good balance: significant memory reduction while preserving fine-grained information for recent tokens.

