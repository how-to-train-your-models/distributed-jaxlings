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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/chapters/chapter_02_data_parallelism.ipynb)
#
# # Chapter 2: Data Parallelism
#
# > **Course: Distributed Training — From Concepts to JAX**
#
# ---
#
# ## Learning Objectives
#
# - Explain the data parallel training loop
# - Describe the Ring-AllReduce algorithm and its bandwidth efficiency
# - Distinguish synchronous vs asynchronous gradient updates
# - Implement gradient accumulation to simulate larger batch sizes
# - Explain ZeRO stages and the memory savings they provide
#

# %% [markdown]
# ---
# ## 1. The Data Parallel Idea
#
# Data parallelism is the simplest and most widely used form of distributed training.
#
# **Core idea:**
# 1. Each worker (GPU) holds a **full copy** of the model
# 2. The global mini-batch is **split** across workers (each sees a different micro-batch)
# 3. Each worker computes a forward + backward pass independently
# 4. Gradients are **averaged** across all workers (AllReduce)
# 5. Each worker applies the same gradient update → models stay in sync
#
# ```
# Global batch = [b0, b1, b2, b3]  (4 micro-batches)
#
# GPU 0: model_copy | b0 → grad_0 ─┐
# GPU 1: model_copy | b1 → grad_1 ─┤→ AllReduce → avg_grad → update all
# GPU 2: model_copy | b2 → grad_2 ─┤
# GPU 3: model_copy | b3 → grad_3 ─┘
# ```
#
# **Effective batch size** = micro_batch_size × n_workers. Data parallelism with N GPUs gives a theoretical **N× speedup** — bounded by communication cost.
#
# ### Real-world example
# PyTorch DDP (`DistributedDataParallel`) uses data parallelism. It overlaps gradient AllReduce with the backward pass, so communication is hidden behind compute:
# ```python
# model = DDP(model)  # wraps model, hooks AllReduce into .backward()
# loss.backward()     # computes grads AND synchronizes them
# optimizer.step()    # all workers apply identical update
# ```
#

# %% [markdown]
# ---
# ## 2. AllReduce: Synchronizing Gradients
#
# AllReduce is the collective operation that **reduces tensors across all workers and distributes the result back to all workers**.
#
# ### Naive AllReduce (parameter server)
# ```
# All workers → send grad to server → server sums → broadcast back
# ```
# - Server bandwidth: O(N × model_size) — bottleneck!
#
# ### Ring-AllReduce
# Each GPU is arranged in a ring. The algorithm runs in **two phases**:
#
# **Phase 1 — Reduce-Scatter** (N-1 steps):
# - Each GPU sends a chunk to its right neighbor and receives a chunk from left
# - After N-1 steps, each GPU holds the **fully reduced** version of one chunk
#
# **Phase 2 — AllGather** (N-1 steps):
# - Each GPU sends its complete chunk rightward
# - After N-1 steps, every GPU has all chunks → full result
#
# **Bandwidth efficiency:**
# - Each GPU sends/receives exactly `2(N-1)/N × S` bytes total
# - As N → ∞, bandwidth per GPU → **2S** (independent of N!)
# - Ring-AllReduce is **bandwidth-optimal**
#
# ```
# Ring with 4 GPUs, gradient split into 4 chunks [A, B, C, D]
#
# Initial:  GPU0=[A,B,C,D]  GPU1=[A,B,C,D]  GPU2=[A,B,C,D]  GPU3=[A,B,C,D]
#
# --- Reduce-Scatter (3 steps) ---
# Step 1:   GPU0 sends A→GPU1, GPU1 sends B→GPU2, GPU2 sends C→GPU3, GPU3 sends D→GPU0
# Step 2:   GPU0 sends D+A→GPU1, GPU1 sends A+B→GPU2, ...
# Step 3:   Each GPU owns one fully-reduced chunk
#
# --- AllGather (3 steps) ---
# Each GPU distributes its chunk around the ring
#
# Final:    All GPUs have [A+A+A+A, B+B+B+B, C+C+C+C, D+D+D+D] / 4
# ```
#

# %% [markdown]
# ---
# ## 3. Synchronous vs Asynchronous Training
#
# | | Synchronous (SSP) | Asynchronous (ASP) |
# |---|---|---|
# | **Gradient update** | All workers sync before step | Workers update independently |
# | **Convergence** | Identical to single-GPU math | Stale gradients → noise |
# | **Stragglers** | Slowest worker blocks all | No blocking |
# | **Used by** | PyTorch DDP, JAX pmap | Hogwild!, some RL systems |
#
# Modern LLM training is **always synchronous** — stale gradients hurt convergence too much at scale.
#

# %% [markdown]
# ---
# ## 4. Gradient Accumulation
#
# When GPU memory limits the batch size, you can **accumulate gradients** over multiple forward/backward passes before updating:
#
# ```python
# effective_batch = micro_batch * accumulation_steps * n_gpus
#
# for step, batch in enumerate(dataloader):
#     loss = model(batch) / accumulation_steps
#     loss.backward()                          # accumulates in .grad
#     if (step + 1) % accumulation_steps == 0:
#         optimizer.step()                     # update once
#         optimizer.zero_grad()
# ```
#
# With gradient accumulation, you can train with an **effective batch size** that doesn't fit in memory.
#

# %% [markdown]
# ---
# ## 5. ZeRO: Zero Redundancy Optimizer
#
# Data parallelism stores a **full model copy on every GPU** — wasteful! ZeRO (Rajbhandari et al., 2020) eliminates this redundancy progressively:
#
# ```
#               Memory per GPU (175B model, 64 GPUs)
# Base DDP  ──────────────────────────────────────── 2800 GB  (all on every GPU!)
# ZeRO-1    Shard optimizer states          ──────── 730 GB   (4× reduction)
# ZeRO-2    Shard optimizer states+gradients ──────  365 GB   (8× reduction)
# ZeRO-3    Shard everything (weights too)  ───────   43 GB   (64× = N GPUs)
# ```
#
# **ZeRO-1:** Each rank owns 1/N of the optimizer states. After AllReduce of gradients, each rank updates its shard, then AllGather parameters.
#
# **ZeRO-2:** Additionally shard gradients. After Reduce-Scatter, each rank owns only the gradient shard it needs.
#
# **ZeRO-3:** Shard weights too. Parameters are gathered on-demand during forward/backward pass.
#
# **Trade-off:** Communication volume increases (need extra AllGather for parameters), but memory savings often outweigh this.
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
            ok = abs(got - expected) / (abs(expected) + 1e-9) < tol
        else:
            ok = got == expected
        if ok:
            self.passed += 1
            print(f"✅ {name}: PASSED")
        else:
            self.failed += 1
            print(f"❌ {name}: FAILED")
            print(f"   got:      {got}")
            print(f"   expected: {expected}")
        return ok

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*40}")
        print(f"  Results: {self.passed}/{total} passed")
        if self.failed == 0:
            print("  🎉 Chapter 2 complete!")
        else:
            print(f"  {self.failed} exercise(s) remaining.")
        print('='*40)

judge = Judge()
print("Judge ready!")

# %% [markdown]
# ---
# ## Exercise 1: Data Parallel Gradient Averaging
#
# Simulate data parallelism. Each 'worker' independently computes gradients on its micro-batch. Implement the function that **averages gradients across all workers**.
#
# In real systems this is done by AllReduce. Here we simulate it in Python.
#
# ```
# TODO: Implement data_parallel_average
# ```
#

# %%
import numpy as np

def simulate_worker_gradients(n_workers: int, n_params: int, seed: int = 42) -> list:
    """Simulate each worker computing gradients on its micro-batch."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n_params) for _ in range(n_workers)]


def data_parallel_average(worker_grads: list) -> np.ndarray:
    """
    Given a list of gradient arrays (one per worker), return the
    element-wise average — the gradient every worker should apply.
    
    Args:
        worker_grads: List of np.ndarray, each shape (n_params,)
    
    Returns:
        np.ndarray of shape (n_params,) — the averaged gradient
    """
    # TODO: Stack all worker gradients and compute the mean across workers
    pass


# Test
grads = simulate_worker_gradients(n_workers=4, n_params=10)
avg = data_parallel_average(grads)

expected = np.mean(np.stack(grads), axis=0)
print(f"Averaged gradient shape: {avg.shape}")
print(f"First 3 values: {avg[:3]}")
judge.check("Ex1: Data parallel gradient average", avg, expected)


# %% [markdown]
# <details>
# <summary>💡 Hint</summary>
#
# ```python
# stacked = np.stack(worker_grads)   # shape: (n_workers, n_params)
# return np.mean(stacked, axis=0)    # shape: (n_params,)
# ```
# </details>
#

# %% [markdown]
# ---
# ## Exercise 2: Ring-AllReduce
#
# Implement a simplified Ring-AllReduce in two phases.
#
# **Setup:** N workers, each holding a full gradient vector. Split into N equal chunks.
#
# **Phase 1 — Reduce-Scatter:** For N-1 steps, each worker sends chunk `(rank - step) % N` to its right neighbor and **adds** (accumulates) the received chunk.
#
# **Phase 2 — AllGather:** For N-1 steps, each worker sends chunk `(rank - step + 1) % N` to its right neighbor, **overwriting** (not adding) the received chunk.
#
# ```
# TODO: Implement ring_allreduce
# ```
#

# %%
def ring_allreduce(worker_data: list) -> list:
    """
    Perform Ring-AllReduce across workers.
    
    Args:
        worker_data: list of np.ndarray (one per worker), each shape (n,)
                     Values represent local gradients before reduction.
    
    Returns:
        list of np.ndarray — each worker now holds the element-wise SUM
        (not average) of all inputs. Shape unchanged.
    
    Note: We return the SUM here (not average). The caller divides by N.
    """
    n = len(worker_data)
    size = len(worker_data[0])
    assert size % n == 0, "For simplicity, size must be divisible by n_workers"
    chunk_size = size // n

    # Split each worker's data into n chunks
    # chunks[rank][chunk_idx] = np.ndarray of shape (chunk_size,)
    chunks = [
        [worker_data[rank][i*chunk_size:(i+1)*chunk_size].copy() for i in range(n)]
        for rank in range(n)
    ]

    # ── Phase 1: Reduce-Scatter ──────────────────────────────────────────
    # For step in range(n-1):
    #   Each rank sends chunk index `(rank - step) % n` to rank `(rank + 1) % n`
    #   Receiving rank ADDS the received chunk to its own copy of that chunk
    for step in range(n - 1):
        new_chunks = [row[:] for row in chunks]  # shallow copy of lists
        for rank in range(n):
            # TODO: compute which chunk this rank sends in this step
            send_idx = 0      # TODO: (rank - step) % n
            recv_from = 0     # TODO: (rank - 1) % n  (left neighbor sends to us)
            recv_idx = 0      # TODO: (rank - 1 - step) % n
            
            # TODO: accumulate received chunk into our copy
            # new_chunks[rank][recv_idx] += chunks[recv_from][recv_idx]
            pass
        chunks = new_chunks

    # After reduce-scatter: chunks[rank][rank] holds the fully reduced chunk

    # ── Phase 2: AllGather ───────────────────────────────────────────────
    # For step in range(n-1):
    #   Each rank sends chunk index `(rank - step + 1) % n` to rank `(rank + 1) % n`
    #   Receiving rank OVERWRITES its copy of that chunk
    for step in range(n - 1):
        new_chunks = [row[:] for row in chunks]
        for rank in range(n):
            # TODO: compute send/receive indices
            recv_from = 0   # TODO: (rank - 1) % n
            recv_idx = 0    # TODO: (rank - step) % n  (note: +1-1 simplifies)
            
            # TODO: overwrite received chunk
            # new_chunks[rank][recv_idx] = chunks[recv_from][recv_idx].copy()
            pass
        chunks = new_chunks

    # Reassemble each worker's full array
    result = [np.concatenate(chunks[rank]) for rank in range(n)]
    return result


# Test: all workers should end up with the same summed gradient
np.random.seed(0)
data = [np.random.randn(8).astype(np.float64) for _ in range(4)]
true_sum = np.sum(data, axis=0)

results = ring_allreduce(data)
print(f"True sum:        {true_sum.round(3)}")
print(f"Worker 0 result: {results[0].round(3)}")
print(f"Worker 2 result: {results[2].round(3)}")

judge.check("Ex2a: Ring-AllReduce worker 0 equals sum", results[0], true_sum)
judge.check("Ex2b: All workers agree",
            np.allclose(results[0], results[1]) and np.allclose(results[1], results[3]),
            True)

# %% [markdown]
# <details>
# <summary>💡 Hint — Phase 1 (Reduce-Scatter)</summary>
#
# ```python
# for step in range(n - 1):
#     new_chunks = [row[:] for row in chunks]
#     for rank in range(n):
#         recv_from = (rank - 1) % n
#         recv_idx  = (rank - 1 - step) % n
#         new_chunks[rank][recv_idx] = (
#             chunks[rank][recv_idx] + chunks[recv_from][recv_idx]
#         )
#     chunks = new_chunks
# ```
# </details>
#
# <details>
# <summary>💡 Hint — Phase 2 (AllGather)</summary>
#
# ```python
# for step in range(n - 1):
#     new_chunks = [row[:] for row in chunks]
#     for rank in range(n):
#         recv_from = (rank - 1) % n
#         recv_idx  = (rank - step) % n
#         new_chunks[rank][recv_idx] = chunks[recv_from][recv_idx].copy()
#     chunks = new_chunks
# ```
# </details>
#

# %% [markdown]
# ---
# ## Exercise 3: Gradient Accumulation
#
# Implement a training loop with gradient accumulation. Gradients from multiple micro-batches should be accumulated before performing an optimizer step.
#
# We use a simple linear regression model to keep it self-contained.
#
# ```
# TODO: Implement the gradient accumulation loop
# ```
#

# %%
import numpy as np

class SimpleLinear:
    """Minimal linear regression: y = X @ w"""
    def __init__(self, in_features, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal(in_features) * 0.01
        self.grad = np.zeros_like(self.w)

    def forward(self, X):
        return X @ self.w

    def backward(self, X, residuals):
        """Accumulate gradients. residuals = (y_pred - y_true)."""
        self.grad += (2 / len(X)) * X.T @ residuals

    def step(self, lr):
        self.w -= lr * self.grad

    def zero_grad(self):
        self.grad = np.zeros_like(self.w)


def train_with_accumulation(
    model: SimpleLinear,
    X: np.ndarray,
    y: np.ndarray,
    n_steps: int,
    accumulation_steps: int,
    micro_batch_size: int,
    lr: float = 0.01
) -> list:
    """
    Train using gradient accumulation.
    
    Each optimizer step should see gradients accumulated over
    `accumulation_steps` micro-batches of size `micro_batch_size`.
    
    IMPORTANT: Scale the loss by 1/accumulation_steps before backprop
    so the effective gradient magnitude matches a single large batch.
    
    Args:
        model:              SimpleLinear instance
        X, y:               Full dataset
        n_steps:            Total number of optimizer steps
        accumulation_steps: Number of micro-batches per optimizer step
        micro_batch_size:   Samples per micro-batch
        lr:                 Learning rate
    
    Returns:
        List of loss values (one per optimizer step)
    """
    losses = []
    n = len(X)
    rng = np.random.default_rng(99)

    for step in range(n_steps):
        # TODO: zero gradients at the start of each optimizer step
        # model.zero_grad()
        
        step_loss = 0.0
        
        for acc_step in range(accumulation_steps):
            # Sample a random micro-batch
            idx = rng.integers(0, n, micro_batch_size)
            Xb, yb = X[idx], y[idx]
            
            # TODO: forward pass
            y_pred = None  # TODO: model.forward(Xb)
            
            # TODO: compute MSE loss (scaled by 1/accumulation_steps)
            residuals = None  # TODO: y_pred - yb
            loss = 0.0        # TODO: np.mean(residuals**2) / accumulation_steps
            step_loss += loss
            
            # TODO: backward pass (note: model.backward accumulates into .grad)
            # The residuals passed to backward should also be scaled by 1/accumulation_steps
            # model.backward(Xb, residuals / accumulation_steps)
        
        # TODO: optimizer step
        # model.step(lr)
        
        losses.append(step_loss)
    
    return losses


# Generate synthetic data: y = 2*x1 + 3*x2 + noise
rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 2))
y = X @ np.array([2.0, 3.0]) + 0.1 * rng.standard_normal(1000)

model = SimpleLinear(in_features=2)
losses = train_with_accumulation(model, X, y, n_steps=200, accumulation_steps=4,
                                  micro_batch_size=32, lr=0.05)

print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")
print(f"Learned weights: {model.w} (target: [2.0, 3.0])")

judge.check("Ex3a: Loss decreased", losses[-1] < losses[0], True)
judge.check("Ex3b: Weights converged to [2, 3]",
            np.allclose(model.w, [2.0, 3.0], atol=0.1), True)

# %% [markdown]
# ---
# ## Exercise 4: ZeRO-1 Optimizer State Sharding
#
# In ZeRO-1, each rank is responsible for updating **only its shard** of the parameters. After the update, parameters are gathered (AllGather) so all ranks have the full updated model.
#
# Implement the function that assigns parameter indices to ranks:
#
# ```
# TODO: Implement zero1_assign_shards and zero1_update_shard
# ```
#

# %%
import numpy as np
from typing import List, Tuple


def zero1_assign_shards(n_params: int, n_ranks: int) -> List[np.ndarray]:
    """
    Assign parameter indices to ranks for ZeRO-1 sharding.
    Distribute parameters as evenly as possible (round-robin or contiguous).
    
    Args:
        n_params: Total number of parameters
        n_ranks:  Number of workers/GPUs
    
    Returns:
        List of length n_ranks, each element is a np.ndarray of parameter
        indices assigned to that rank.
    """
    # TODO: Assign indices 0..n_params-1 to ranks as evenly as possible
    # Hint: np.array_split(np.arange(n_params), n_ranks)
    pass


def zero1_update_shard(
    params: np.ndarray,
    grads: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    shard_indices: np.ndarray,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    t: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply an Adam update to only the parameters in shard_indices.
    The optimizer states (m, v) are only maintained for shard_indices.
    
    Args:
        params:        Full parameter array (n_params,) — modified in-place at shard_indices
        grads:         Full gradient array (n_params,)
        m:             Momentum for this shard (len(shard_indices),)
        v:             Variance for this shard (len(shard_indices),)
        shard_indices: Indices this rank is responsible for
        lr, beta1, beta2, eps: Adam hyperparameters
        t:             Current timestep (for bias correction)
    
    Returns:
        (updated_params, new_m, new_v)  — params modified in-place
    """
    # TODO: Extract the gradient shard
    g = None  # TODO: grads[shard_indices]
    
    # TODO: Adam moment updates
    new_m = None  # TODO: beta1 * m + (1 - beta1) * g
    new_v = None  # TODO: beta2 * v + (1 - beta2) * g**2
    
    # TODO: Bias-corrected estimates
    m_hat = None  # TODO: new_m / (1 - beta1**t)
    v_hat = None  # TODO: new_v / (1 - beta2**t)
    
    # TODO: Update only the shard of params
    # params[shard_indices] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return params, new_m, new_v


# Test ZeRO-1 sharding
n_params, n_ranks = 10, 3
shards = zero1_assign_shards(n_params, n_ranks)

print("Shard assignments:")
for r, s in enumerate(shards):
    print(f"  Rank {r}: params {s.tolist()}")

all_idx = np.sort(np.concatenate(shards))
judge.check("Ex4a: All params assigned", all_idx, np.arange(n_params))
judge.check("Ex4b: Max shard size difference <= 1",
            max(len(s) for s in shards) - min(len(s) for s in shards) <= 1, True)

# Test the update
np.random.seed(1)
params = np.ones(10)
grads  = np.full(10, 0.1)
shard0 = shards[0]  # e.g., [0,1,2,3]
m0 = np.zeros(len(shard0))
v0 = np.zeros(len(shard0))

params, m0, v0 = zero1_update_shard(params, grads, m0, v0, shard0, lr=0.01, t=1)
print(f"\nAfter update, params[0]: {params[0]:.6f} (expected ~0.9)")
print(f"Params outside shard unchanged: {np.all(params[shard0[-1]+1:] == 1.0)}")
judge.check("Ex4c: Shard updated correctly",
            abs(params[0] - (1.0 - 0.01)) < 0.005, True)
judge.check("Ex4d: Non-shard params untouched",
            float(np.all(params[shards[1]] == 1.0)), 1.0)

# %% [markdown]
# <details>
# <summary>💡 Hint — zero1_assign_shards</summary>
#
# ```python
# return np.array_split(np.arange(n_params), n_ranks)
# ```
# </details>
#
# <details>
# <summary>💡 Hint — zero1_update_shard</summary>
#
# ```python
# g = grads[shard_indices]
# new_m = beta1 * m + (1 - beta1) * g
# new_v = beta2 * v + (1 - beta2) * g**2
# m_hat = new_m / (1 - beta1**t)
# v_hat = new_v / (1 - beta2**t)
# params[shard_indices] -= lr * m_hat / (np.sqrt(v_hat) + eps)
# return params, new_m, new_v
# ```
# </details>
#

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
# 1. **Data parallelism** replicates the model across GPUs and splits the data. Easy to implement, scales well to 100s of GPUs.
# 2. **Ring-AllReduce** is bandwidth-optimal: each GPU's communication cost is constant regardless of number of GPUs.
# 3. **Gradient accumulation** lets you simulate larger effective batch sizes without extra memory.
# 4. **ZeRO-1/2/3** progressively shard optimizer states, gradients, and weights — enabling training of models much larger than a single GPU's memory.
#
# ---
# **Next:** [Chapter 3 — Model Parallelism](./chapter_03_model_parallelism.ipynb)
#
