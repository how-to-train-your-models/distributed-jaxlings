# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/chapters/chapter_03_model_parallelism.ipynb)
#
# # Chapter 3: Model Parallelism
# ### Tensor, Pipeline, and Sequence Parallelism
#
# > **Course: Distributed Training — From Concepts to JAX**
#
# ---
#
# ## Learning Objectives
#
# - Explain when model parallelism is needed vs data parallelism
# - Implement column-parallel and row-parallel linear layers (tensor parallelism)
# - Calculate pipeline bubble overhead and explain micro-batching
# - Describe sequence parallelism and when it helps
# - Reason about communication patterns for each approach
#

# %% [markdown]
# ---
# ## 1. Why Model Parallelism?
#
# Data parallelism requires **each GPU to hold the entire model**. When a single layer's weights exceed GPU memory (e.g., a 50K×50K embedding table), data parallelism alone isn't enough.
#
# Model parallelism **splits the model itself** across GPUs. There are three main flavors:
#
# | Type | Split axis | Best for |
# |------|-----------|----------|
# | **Tensor parallelism** | Within a layer (weight matrices) | Wide layers (large d_model) |
# | **Pipeline parallelism** | Across layers | Deep models (many layers) |
# | **Sequence parallelism** | Sequence dimension | Long context (large seq_len) |
#
# Modern LLM training (Megatron-LM) uses all three simultaneously — **3D parallelism**.
#

# %% [markdown]
# ---
# ## 2. Tensor Parallelism
#
# Tensor parallelism (Megatron-LM, Shoeybi et al. 2019) splits individual weight matrices across GPUs.
#
# ### Column-parallel linear
# Split output dimension across GPUs. Each GPU computes part of `Y = X @ W`:
#
# ```
# W shape: [in, out]  →  GPU0: [in, out/N],  GPU1: [in, out/N]
#
# GPU0: Y0 = X @ W0   (shape: [batch, out/N])
# GPU1: Y1 = X @ W1   (shape: [batch, out/N])
#
# Concat: Y = [Y0 | Y1]  (shape: [batch, out])
# ```
# Communication: **AllGather** on output.
#
# ### Row-parallel linear
# Split input dimension. Each GPU receives a slice of the input:
#
# ```
# W shape: [in, out]  →  GPU0: [in/N, out],  GPU1: [in/N, out]
# Input split:            GPU0: X0=[batch,in/N], GPU1: X1=[batch,in/N]
#
# GPU0: P0 = X0 @ W0   (shape: [batch, out])
# GPU1: P1 = X1 @ W1   (shape: [batch, out])
#
# Sum: Y = P0 + P1      (shape: [batch, out])
# ```
# Communication: **AllReduce** on output.
#
# ### Megatron MLP pattern
# ```
#   Input X (replicated)          ← no comm needed
#       │
#   Column-parallel FC1           ← each GPU computes own slice
#       │  (no comm between layers if next is row-parallel)
#   Row-parallel FC2              ← each GPU gets its input slice
#       │
#   AllReduce → Output (replicated)
# ```
# This is a **fused** pattern: the AllGather from column-parallel cancels with the scatter needed for row-parallel!
#

# %% [markdown]
# ---
# ## 3. Pipeline Parallelism
#
# Split the model **by layers**: GPU 0 runs layers 0-7, GPU 1 runs layers 8-15, etc.
#
# ### Naive pipeline — the bubble problem
# ```
# Time →
# GPU0:  [F0][F1][F2][F3]              [B3][B2][B1][B0]
# GPU1:      [F0][F1][F2][F3]      [B3][B2][B1][B0]
# GPU2:          [F0][F1][F2][F3][B3][B2][B1][B0]
#               ↑──── bubble ────↑
# ```
# **Pipeline bubble fraction** = `(p-1)/(m+p-1)` where `p` = pipeline stages, `m` = micro-batches.
#
# As `m → ∞`, bubble → 0. Rule of thumb: use `m ≥ 4p` micro-batches.
#
# ### GPipe vs 1F1B schedule
# - **GPipe**: run all forward passes, then all backward passes. Simple but high memory.
# - **1F1B** (PipeDream): interleave forward and backward. Memory = O(p), not O(m×p).
#
# ```
# 1F1B schedule (4 stages, 8 micro-batches):
# GPU0: F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
# GPU1:    F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4    B5 B6 B7
# ```
#

# %% [markdown]
# ---
# ## 4. Sequence Parallelism
#
# For very long sequences (32K+ tokens), **attention activations** dominate memory:
# $$\text{Attention memory} \propto B \times S^2 \times H$$
#
# Sequence parallelism splits the **sequence dimension** across GPUs. Each GPU processes `S/N` tokens.
#
# Combined with tensor parallelism in Megatron:
# - **Sequence parallel** regions: LayerNorm, Dropout — split along sequence
# - **Tensor parallel** regions: Attention, MLP — split along head/hidden dim
# - **AllGather** / **Reduce-Scatter** at boundaries between regions
#

# %% [markdown]
# ---
# ## Judge Setup
#

# %%
import numpy as np

class Judge:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, got, expected, tol=1e-5):
        if isinstance(expected, np.ndarray):
            ok = np.allclose(got, expected, atol=tol)
        elif isinstance(expected, (int, float)):
            ok = abs(float(got) - float(expected)) / (abs(float(expected)) + 1e-9) < tol
        else:
            ok = bool(got) == bool(expected)
        if ok:
            self.passed += 1
            print(f"✅ {name}: PASSED")
        else:
            self.failed += 1
            print(f"❌ {name}: FAILED — got {got!r}, expected {expected!r}")
        return ok

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*40}")
        print(f"  Results: {self.passed}/{total} passed")
        print("  🎉 Chapter 3 complete!" if self.failed == 0
              else f"  {self.failed} remaining.")
        print('='*40)

judge = Judge()
print("Judge ready!")

# %% [markdown]
# ---
# ## Exercise 1: Column-Parallel Linear Layer
#
# Implement a column-parallel linear layer. The weight matrix is split along the output dimension. Each GPU computes its partition of the output.
#
# ```
# W: [in_features, out_features]  →  split into N shards along axis 1
# Shard i: W_i shape [in_features, out_features // N]
#
# Y_i = X @ W_i          (local matmul on each GPU)
# Y   = concat(Y_i, axis=-1)   (AllGather: reassemble full output)
# ```
#
# TODO: Implement `column_parallel_linear`
#

# %%
import numpy as np

def column_parallel_linear(
    X: np.ndarray,
    W: np.ndarray,
    n_gpus: int
) -> np.ndarray:
    """
    Simulate column-parallel linear layer across n_gpus.
    
    Args:
        X:      Input tensor, shape [batch, in_features]
        W:      Full weight matrix, shape [in_features, out_features]
        n_gpus: Number of GPUs to split across
    
    Returns:
        Y: Output tensor, shape [batch, out_features]
           (equivalent to X @ W)
    """
    in_features, out_features = W.shape
    assert out_features % n_gpus == 0
    
    # TODO: Split W along the output (column) dimension
    # W_shards = [W[:, i*shard_out:(i+1)*shard_out] for i in range(n_gpus)]
    W_shards = []  # TODO
    
    # TODO: Each GPU computes its local output shard: Y_i = X @ W_shards[i]
    Y_shards = []  # TODO
    
    # TODO: AllGather — concatenate all shards along the output dimension
    Y = None  # TODO: np.concatenate(Y_shards, axis=-1)
    
    return Y


# Test: result must match naive X @ W
np.random.seed(7)
X = np.random.randn(4, 8)   # batch=4, in=8
W = np.random.randn(8, 16)  # in=8, out=16

Y_ref  = X @ W
Y_tp   = column_parallel_linear(X, W, n_gpus=4)

print(f"Reference shape: {Y_ref.shape}, TP shape: {Y_tp.shape if Y_tp is not None else None}")
judge.check("Ex1: Column-parallel linear correctness", Y_tp, Y_ref)


# %% [markdown]
# <details>
# <summary>💡 Hint</summary>
#
# ```python
# shard_out = out_features // n_gpus
# W_shards  = [W[:, i*shard_out:(i+1)*shard_out] for i in range(n_gpus)]
# Y_shards  = [X @ Ws for Ws in W_shards]
# Y         = np.concatenate(Y_shards, axis=-1)
# ```
# </details>
#

# %% [markdown]
# ---
# ## Exercise 2: Row-Parallel Linear Layer
#
# Implement a row-parallel linear layer. The weight matrix is split along the **input** (row) dimension. The input `X` must also be pre-split.
#
# ```
# W: [in_features, out_features]  →  split into N shards along axis 0
# X: [batch, in_features]         →  split into N shards along axis 1
#
# P_i = X_i @ W_i        (local partial result, shape [batch, out_features])
# Y   = sum(P_i)          (AllReduce: sum all partial results)
# ```
#
# TODO: Implement `row_parallel_linear`
#

# %%
def row_parallel_linear(
    X: np.ndarray,
    W: np.ndarray,
    n_gpus: int
) -> np.ndarray:
    """
    Simulate row-parallel linear layer across n_gpus.
    
    Args:
        X:      Input, shape [batch, in_features]  (will be split across GPUs)
        W:      Full weight matrix, shape [in_features, out_features]
        n_gpus: Number of GPUs
    
    Returns:
        Y: shape [batch, out_features]  (equivalent to X @ W)
    """
    in_features, out_features = W.shape
    assert in_features % n_gpus == 0
    shard_in = in_features // n_gpus

    # TODO: Split X along the input (feature) dimension
    X_shards = []  # TODO: [X[:, i*shard_in:(i+1)*shard_in] for i in range(n_gpus)]

    # TODO: Split W along the row (input) dimension
    W_shards = []  # TODO: [W[i*shard_in:(i+1)*shard_in, :] for i in range(n_gpus)]

    # TODO: Each GPU computes partial result: P_i = X_shards[i] @ W_shards[i]
    partials = []  # TODO

    # TODO: AllReduce (sum) all partial results
    Y = None  # TODO: np.sum(partials, axis=0)  after stacking

    return Y


# Test
np.random.seed(3)
X2 = np.random.randn(4, 16)
W2 = np.random.randn(16, 8)

Y_ref2 = X2 @ W2
Y_rp   = row_parallel_linear(X2, W2, n_gpus=4)

judge.check("Ex2: Row-parallel linear correctness", Y_rp, Y_ref2)


# %% [markdown]
# ---
# ## Exercise 3: Fused Tensor-Parallel MLP
#
# Combine column-parallel and row-parallel into a **fused MLP** block (Megatron pattern). The key insight: the AllGather from the column-parallel layer feeds directly into the scatter for the row-parallel layer — **no extra communication needed** between the two layers!
#
# ```
# X (replicated) → Column-parallel FC1 (each GPU: shard of hidden) 
#                → GELU activation (local)
#                → Row-parallel FC2  (reduce across GPUs)
#                → Y (replicated)
# ```
#
# In the fused version, between FC1 and FC2 each GPU holds `X_mid_i = GELU(X @ W1_i)` — its own hidden shard. FC2 takes these shards as row-parallel input. No AllGather needed mid-block!
#
# TODO: Implement `tensor_parallel_mlp`
#

# %%
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def tensor_parallel_mlp(
    X: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    n_gpus: int
) -> np.ndarray:
    """
    Fused tensor-parallel MLP: Y = RowParallel(GELU(ColParallel(X, W1)), W2)
    
    Key: each GPU computes its own mid-shard without cross-GPU communication
    between W1 and W2. Only ONE AllReduce at the end (from row-parallel).
    
    Args:
        X:      [batch, d_model]
        W1:     [d_model, 4*d_model]  (expand to 4x hidden)
        W2:     [4*d_model, d_model]  (project back)
        n_gpus: number of GPUs
    
    Returns:
        Y: [batch, d_model]
    """
    d_model = X.shape[-1]
    d_ff = W1.shape[1]  # 4 * d_model
    assert d_ff % n_gpus == 0
    shard = d_ff // n_gpus

    partials = []
    for i in range(n_gpus):
        # TODO: Each GPU gets its column shard of W1
        W1_i = None  # TODO: W1[:, i*shard:(i+1)*shard]

        # TODO: Compute local hidden state and apply GELU
        H_i = None   # TODO: gelu(X @ W1_i)   shape: [batch, shard]

        # TODO: Each GPU gets its row shard of W2 (rows match its column shard of W1)
        W2_i = None  # TODO: W2[i*shard:(i+1)*shard, :]

        # TODO: Compute partial output
        P_i = None   # TODO: H_i @ W2_i   shape: [batch, d_model]

        partials.append(P_i)

    # TODO: AllReduce — sum partial outputs
    Y = None  # TODO: np.sum(np.stack(partials), axis=0)
    return Y


# Reference: sequential MLP
np.random.seed(42)
batch, d = 4, 8
X3  = np.random.randn(batch, d)
W1  = np.random.randn(d, 4*d)
W2  = np.random.randn(4*d, d)

Y_ref3 = gelu(X3 @ W1) @ W2
Y_tp3  = tensor_parallel_mlp(X3, W1, W2, n_gpus=4)

judge.check("Ex3: Tensor-parallel MLP correctness", Y_tp3, Y_ref3)


# %% [markdown]
# ---
# ## Exercise 4: Pipeline Bubble Efficiency
#
# Calculate the **pipeline bubble overhead** and determine the minimum number of micro-batches needed to achieve a target efficiency.
#
# Recall:
# $$\text{bubble fraction} = \frac{p - 1}{m + p - 1}$$
#
# where $p$ = pipeline stages, $m$ = micro-batches.
#
# TODO: Implement `pipeline_bubble_fraction` and `min_microbatches_for_efficiency`
#

# %%
def pipeline_bubble_fraction(p: int, m: int) -> float:
    """
    Compute the fraction of time wasted in pipeline bubbles.
    
    Args:
        p: Number of pipeline stages (GPUs)
        m: Number of micro-batches per step
    
    Returns:
        Bubble fraction in [0, 1)
    """
    # TODO
    pass


def min_microbatches_for_efficiency(p: int, target_efficiency: float) -> int:
    """
    Find the minimum number of micro-batches m such that
    pipeline efficiency (1 - bubble_fraction) >= target_efficiency.
    
    Args:
        p:                  Number of pipeline stages
        target_efficiency:  Minimum acceptable efficiency (e.g. 0.95 for 95%)
    
    Returns:
        Minimum m (integer >= 1)
    """
    # TODO: Iterate over m=1,2,3,... until efficiency threshold is met
    pass


# Tests
bub1 = pipeline_bubble_fraction(p=4, m=1)   # 3/4 = 75% bubble!
bub8 = pipeline_bubble_fraction(p=4, m=8)   # 3/11 ≈ 27%
bub32 = pipeline_bubble_fraction(p=4, m=32) # 3/35 ≈ 8.6%

print(f"p=4, m=1:  bubble={bub1:.1%}  efficiency={1-bub1:.1%}")
print(f"p=4, m=8:  bubble={bub8:.1%}  efficiency={1-bub8:.1%}")
print(f"p=4, m=32: bubble={bub32:.1%} efficiency={1-bub32:.1%}")

judge.check("Ex4a: bubble p=4 m=1",  bub1,  3/4,   tol=1e-4)
judge.check("Ex4b: bubble p=4 m=8",  bub8,  3/11,  tol=1e-4)
judge.check("Ex4c: bubble p=4 m=32", bub32, 3/35,  tol=1e-4)

# For 95% efficiency with 8 pipeline stages, need at least 57 micro-batches
m_min = min_microbatches_for_efficiency(p=8, target_efficiency=0.95)
print(f"\nFor p=8 stages and 95% efficiency: need m >= {m_min}")
judge.check("Ex4d: min microbatches p=8 eff=95%", m_min, 133, tol=0.01)


# %% [markdown]
# <details>
# <summary>💡 Hint</summary>
#
# ```python
# # bubble fraction:
# return (p - 1) / (m + p - 1)
#
# # min microbatches:
# # Solve: 1 - (p-1)/(m+p-1) >= target
# # => (p-1)/(m+p-1) <= 1 - target
# # => m+p-1 >= (p-1)/(1-target)
# # => m >= (p-1)/(1-target) - (p-1) = (p-1)*target/(1-target)
# import math
# return math.ceil((p-1) * target_efficiency / (1 - target_efficiency))
# ```
# </details>
#

# %% [markdown]
# ---
# ## Exercise 5: Sequence Parallel Split and Merge
#
# Sequence parallelism splits activation tensors along the sequence dimension. Implement the scatter (split) and gather (merge) operations.
#
# In Megatron-LM these correspond to **Reduce-Scatter** (before tensor-parallel region) and **AllGather** (after tensor-parallel region).
#
# TODO: Implement `sequence_scatter` and `sequence_gather`
#

# %%
def sequence_scatter(
    X: np.ndarray,
    n_gpus: int
) -> list:
    """
    Scatter a tensor along the sequence dimension across n_gpus.
    
    Args:
        X: shape [batch, seq_len, d_model]
        n_gpus: number of GPUs
    
    Returns:
        List of n_gpus tensors, each shape [batch, seq_len//n_gpus, d_model]
    """
    batch, seq_len, d = X.shape
    assert seq_len % n_gpus == 0
    # TODO: Split X along the sequence (axis=1) dimension
    pass


def sequence_gather(
    shards: list
) -> np.ndarray:
    """
    Gather sequence-parallel shards back into a full tensor.
    
    Args:
        shards: list of tensors, each [batch, seq_chunk, d_model]
    
    Returns:
        X: shape [batch, seq_len, d_model]
    """
    # TODO: Concatenate along the sequence (axis=1) dimension
    pass


# Test
X_full = np.random.randn(2, 16, 8)  # batch=2, seq=16, d=8
shards = sequence_scatter(X_full, n_gpus=4)

print(f"Full shape: {X_full.shape}")
print(f"Shard shapes: {[s.shape for s in shards]}")

X_recon = sequence_gather(shards)
judge.check("Ex5a: Scatter shard shape", shards[0].shape, (2, 4, 8))
judge.check("Ex5b: Gather reconstructs original", X_recon, X_full)

# %% [markdown]
# ---
# ## Summary
#

# %%
judge.summary()

# %% [markdown]
# ---
# ## Key Takeaways
#
# 1. **Tensor parallelism** splits weight matrices — column-parallel needs AllGather, row-parallel needs AllReduce. Fuse them to eliminate mid-block communication.
# 2. **Pipeline parallelism** splits layers. Bubble overhead = `(p-1)/(m+p-1)`. Use enough micro-batches to keep GPUs busy.
# 3. **Sequence parallelism** splits along the token dimension, enabling very long contexts without OOM.
# 4. **3D parallelism** (tensor × pipeline × data) is used in Megatron-LM, DeepSpeed, and JAX-based systems.
#
# ---
# **Next:** [Chapter 4 — Communication Primitives](./chapter_04_communication_primitives.ipynb)
#
