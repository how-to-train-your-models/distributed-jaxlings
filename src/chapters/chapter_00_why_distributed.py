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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/notebooks/chapter_00_why_distributed.ipynb)
#
# # Chapter 0: Why Distributed Training?
#
# > **Course: Distributed Training in JAX** — built around JAX + Equinox + Optax + Orbax.
#
# ---
#
# ## What we build
#
# A **model-sizing calculator**: given `(n_layers, hidden, n_heads, seq, batch)`, compute params,
# training memory, FLOPs/step, and verdict on whether it fits a single GPU. We then watch a
# transformer hit OOM as we scale it — the empirical "why" of this entire course.
#
# **Real-world hook:** sizing a Llama-3 70B run, or noting that Meta's DLRM has embedding tables
# >10 TB while an H100 has 80 GB. Without distribution, neither model exists on any single device.
#
# ## Learning Objectives
#
# By the end of this chapter you will be able to:
# - Explain *why* large models require distributed training
# - Calculate memory requirements for training a transformer model
# - Estimate compute (FLOPs) for a training run
# - Identify when a model fits on a single GPU vs requires multiple GPUs
# - Name the parallelism axes used across the rest of the course
# - Combine the above into a single sizing calculator
#

# %% [markdown]
# ---
# ## 1. The Scaling Hypothesis
#
# Modern deep learning is driven by a simple empirical observation: **more parameters + more data + more compute → better models**.
#
# The [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022) quantified this:
#
# $$L(N, D) \approx \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + E$$
#
# where $N$ = parameters, $D$ = training tokens. The optimal ratio is roughly **20 tokens per parameter**.
#
# ### Real-world examples
#
# | Model        | Parameters | Training Tokens | GPUs used          |
# |--------------|------------|-----------------|--------------------|
# | GPT-2        | 1.5B       | ~40B            | ~32 V100s          |
# | GPT-3        | 175B       | 300B            | ~10,000 V100s      |
# | LLaMA-3 70B  | 70B        | 15T             | ~2,000 A100s       |
# | GPT-4 (est.) | ~1.8T      | ~13T            | ~25,000 A100s      |
#
# **The core problem:** A single GPU has at most 80GB of memory (A100 SXM). GPT-3 at 175B parameters in FP32 requires **700 GB** just to store weights. That's the *why* of distributed training.
#

# %% [markdown]
# ---
# ## 2. Memory Math: What Goes on a GPU?
#
# Training a model requires storing much more than just the weights:
#
# | Component            | Size (FP32) | Notes                              |
# |----------------------|-------------|------------------------------------|
# | Weights              | 4N bytes    | N = number of parameters           |
# | Gradients            | 4N bytes    | Same shape as weights              |
# | Adam optimizer state | 8N bytes    | momentum (4N) + variance (4N)      |
# | Activations          | variable    | Depends on batch size & seq length |
#
# **Total (without activations): 16N bytes = 16 × parameters**
#
# ### Example: GPT-3 (175B params)
# ```
# Weights:    175B × 4 bytes = 700 GB
# Gradients:  175B × 4 bytes = 700 GB
# Adam:       175B × 8 bytes = 1,400 GB
# ─────────────────────────────────────
# Total:                       2,800 GB  ≈ 35 × A100 80GB GPUs (minimum!)
# ```
#
# With **mixed precision** (FP16 forward/backward, FP32 master weights):
# - FP16 weights + gradients: 2N + 2N = 4N bytes
# - FP32 master weights + Adam: 4N + 8N = 12N bytes
# - **Total: 16N bytes** (same! the savings come from activations)
#
# ### Activation memory
# For a transformer layer with sequence length $S$, batch $B$, hidden dim $H$:
# $$\text{Activation memory per layer} \approx 12 \cdot B \cdot S \cdot H \text{ bytes (FP16)}$$
#
# With **gradient checkpointing**, you trade compute for memory by recomputing activations during the backward pass, reducing activation memory to $O(\sqrt{L})$ instead of $O(L)$ for $L$ layers.
#

# %% [markdown]
# ---
# ## 3. Compute Math: How Many FLOPs?
#
# For a transformer model with $N$ parameters trained on $D$ tokens:
#
# $$\text{FLOPs} \approx 6ND$$
#
# The factor of 6 comes from: 2 (multiply-add) × 3 (forward + backward, backward ≈ 2× forward).
#
# ### Example: LLaMA-2 7B
# ```
# N = 7B params, D = 2T tokens
# FLOPs = 6 × 7×10⁹ × 2×10¹² = 8.4×10²² FLOPs
#
# A100 GPU peak: 312 TFLOPS (BF16)
# MFU (model FLOP utilization) ≈ 40%
# Effective throughput = 0.4 × 312×10¹² = 124.8 TFLOPS
#
# Time on 1 GPU = 8.4×10²² / 124.8×10¹² ≈ 673,000 seconds ≈ 7.8 years
# Time on 1024 GPUs ≈ 2.8 days ✓
# ```
#
# This is *why* we need distributed training — not just memory, but **time**.
#

# %% [markdown]
# ---
# ## 4. Hardware Overview
#
# ### GPU Memory Hierarchy
# ```
# GPU HBM (High Bandwidth Memory)  ← where tensors live (40-80GB on A100)
#     ↕  ~2 TB/s bandwidth
# L2 Cache                          ← 40MB on A100
#     ↕  ~10 TB/s bandwidth
# L1 Cache / Shared Memory          ← per SM, 192KB on A100
#     ↕  ~20 TB/s bandwidth
# Registers                         ← fastest, per thread
# ```
#
# ### Multi-GPU Interconnects
#
# | Interconnect      | Bandwidth        | Use case                      |
# |-------------------|------------------|-------------------------------|
# | NVLink 4.0        | 900 GB/s total   | Within a node (up to 8 GPUs)  |
# | NVSwitch          | 900 GB/s         | Full all-to-all within node   |
# | PCIe 5.0          | 128 GB/s         | CPU↔GPU, slower inter-GPU     |
# | InfiniBand HDR    | 200 Gb/s (~25GB/s)| Between nodes                |
# | InfiniBand NDR    | 400 Gb/s (~50GB/s)| Between nodes, A100 clusters |
#
# **Key insight:** NVLink is ~36× faster than InfiniBand. Communication *within* a node is cheap; *across* nodes is the bottleneck.
#

# %% [markdown]
# ---
# ## 5. Parallelism Axes (Preview)
#
# Each axis gets its own implementation chapter later in the course.
#
# | Axis                        | Idea                                                          | Built in |
# |-----------------------------|---------------------------------------------------------------|----------|
# | **Data Parallelism (DP)**   | Replicate model, shard the batch.                              | Ch 2     |
# | **FSDP / ZeRO**             | DP, but shard optimizer state, gradients, and params.          | Ch 4     |
# | **Tensor Parallelism (TP)** | Split individual weight matrices across devices.               | Ch 5     |
# | **Embedding Parallelism**   | Row-shard embedding tables (DLRM-style), all-to-all lookups.   | Ch 6     |
# | **Pipeline Parallelism**    | Split *layers* across devices; micro-batch through the stages. | Ch 7     |
# | **Sequence Parallelism**    | Shard along the sequence dim (long-context attention).         | Ch 8     |
# | **Expert Parallelism**      | Shard MoE experts across devices; route tokens via all-to-all. | Ch 12    |
#
# Frontier training stacks (MaxText, Megatron-LM, DeepSpeed) **combine** several of these
# in a single mesh — by Ch 10 we'll wire DP × TP × FSDP × bf16 into one trainer.
#

# %% [markdown]
# ---
# ## Judge Setup
#
# Run this cell first. The `Judge` class validates your exercise solutions.
#

# %%
import sys
import pathlib



from judge import Judge

judge = Judge("Chapter 0", default_tol=0.01)
print("Judge loaded. Let's go!")


# %% [markdown]
# ---
# ## Exercise 1: Count Transformer Parameters
#
# Given the architecture of a GPT-style transformer, calculate the **total number of parameters**.
#
# A GPT model consists of:
# - **Embedding layer:** `vocab_size × d_model`
# - **Position embedding:** `max_seq_len × d_model`
# - Per layer (× `n_layers`):
#   - **Attention:** Q, K, V projections (`d_model × d_model` each) + output projection (`d_model × d_model`)
#   - **MLP:** two linear layers (`d_model × 4*d_model` and `4*d_model × d_model`)
#   - **LayerNorms:** 2 × `2*d_model` (scale + bias each)
# - **Final LayerNorm:** `2 × d_model`
# - **LM Head:** `d_model × vocab_size` (often tied with embedding — set `tie_embeddings=True` to skip)
#
# ```
# TODO: Implement count_transformer_params
# ```
#

# %%
def count_transformer_params(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    max_seq_len: int,
    tie_embeddings: bool = True
) -> int:
    """
    Calculate the total number of parameters in a GPT-style transformer.

    Args:
        vocab_size:      Size of token vocabulary
        d_model:         Hidden dimension size
        n_layers:        Number of transformer layers
        max_seq_len:     Maximum sequence length (for position embeddings)
        tie_embeddings:  If True, LM head shares weights with token embedding

    Returns:
        Total parameter count
    """
    # TODO: Calculate each component and sum them

    # Embeddings
    token_emb = 0       # TODO
    pos_emb = 0         # TODO

    # Per-layer components
    attn_params = 0     # TODO: Q, K, V, O projections (each d_model × d_model)
    mlp_params = 0      # TODO: two linear layers
    ln_params = 0       # TODO: 2 layer norms, each has scale + bias of size d_model

    per_layer = attn_params + mlp_params + ln_params

    # Final layer norm
    final_ln = 0        # TODO

    # LM Head
    lm_head = 0         # TODO: 0 if tie_embeddings else vocab_size * d_model

    total = token_emb + pos_emb + n_layers * per_layer + final_ln + lm_head
    return total


# Test: GPT-2 Small — should be ~117M params
gpt2_small = count_transformer_params(
    vocab_size=50257, d_model=768, n_layers=12, max_seq_len=1024, tie_embeddings=True
)
print(f"GPT-2 Small params: {gpt2_small:,}")
judge.check("Ex1: GPT-2 Small param count", gpt2_small, 124_439_808, tol=0.01)


# %% [markdown]
# <details>
# <summary>💡 Hint (click to reveal)</summary>
#
# ```python
# token_emb = vocab_size * d_model
# pos_emb   = max_seq_len * d_model
# attn_params = 4 * (d_model * d_model)   # Q, K, V, O each d_model×d_model
# mlp_params  = 2 * (d_model * 4 * d_model)
# ln_params   = 2 * (2 * d_model)          # scale and bias for each LN
# final_ln    = 2 * d_model
# lm_head     = 0 if tie_embeddings else vocab_size * d_model
# ```
# </details>
#

# %% [markdown]
# ---
# ## Exercise 2: Training Memory Requirements
#
# Given a parameter count, calculate the **GPU memory required** to train with the Adam optimizer.
#
# Recall:
# - **FP32 training:** weights (4 bytes) + gradients (4 bytes) + Adam m (4 bytes) + Adam v (4 bytes) = **16 bytes/param**
# - **Mixed precision (BF16/FP16):**
#   - FP16 weights: 2 bytes/param
#   - FP16 gradients: 2 bytes/param
#   - FP32 master weights: 4 bytes/param
#   - Adam m (FP32): 4 bytes/param
#   - Adam v (FP32): 4 bytes/param
#   - **Total: 16 bytes/param**
#
# ```
# TODO: Implement training_memory_gb
# ```
#

# %%
def training_memory_gb(n_params: int, mixed_precision: bool = False) -> float:
    """
    Estimate GPU memory (in GB) needed for training with Adam,
    excluding activation memory.

    Args:
        n_params:         Number of model parameters
        mixed_precision:  If True, use mixed precision accounting
                          (note: total bytes/param is the same as FP32!)

    Returns:
        Memory in gigabytes (GB, not GiB — use 1e9 not 1<<30)
    """
    # TODO: Calculate bytes per parameter
    bytes_per_param = 0  # TODO: same for both fp32 and mixed precision!

    total_bytes = n_params * bytes_per_param
    return total_bytes / 1e9


# Test: GPT-3 (175B params) should need ~2800 GB
mem = training_memory_gb(175_000_000_000, mixed_precision=False)
print(f"GPT-3 training memory: {mem:.0f} GB")
judge.check("Ex2: GPT-3 training memory (GB)", mem, 2800.0, tol=0.01)

# How many A100 80GB GPUs minimum?
min_gpus = -(-mem // 80)  # ceiling division
print(f"Minimum A100 80GB GPUs needed: {min_gpus:.0f}")


# %% [markdown]
# ---
# ## Exercise 3: Does the Model Fit?
#
# Write a function that checks whether a model can be trained on a given number of GPUs, accounting for memory overhead.
#
# In practice, activation memory and other overhead consume roughly **20-30%** of GPU memory. Use 0.8 as the usable fraction (safety margin).
#
# ```
# TODO: Implement can_train_on_cluster
# ```
#

# %%
def can_train_on_cluster(
    n_params: int,
    n_gpus: int,
    gpu_memory_gb: float,
    safety_margin: float = 0.8
) -> tuple[bool, float]:
    """
    Check if a model can be trained on a cluster using model parallelism
    (assuming model weights are perfectly sharded across all GPUs).

    Args:
        n_params:        Number of model parameters
        n_gpus:          Total number of GPUs in the cluster
        gpu_memory_gb:   Memory per GPU in GB
        safety_margin:   Fraction of GPU memory usable (rest is for activations etc.)

    Returns:
        (fits: bool, utilization: float) where utilization is fraction of total memory used
    """
    # TODO: Calculate total available memory across all GPUs (with safety margin)
    total_available_gb = 0  # TODO

    # TODO: Calculate required memory
    required_gb = 0  # TODO: use training_memory_gb from Exercise 2

    # TODO: Determine if it fits and compute utilization
    fits = False        # TODO
    utilization = 0.0  # TODO: required / total_available

    return fits, utilization


# Test cases
fits, util = can_train_on_cluster(7_000_000_000, n_gpus=8, gpu_memory_gb=80)
print(f"LLaMA-7B on 8×A100: fits={fits}, utilization={util:.1%}")
judge.check("Ex3a: LLaMA-7B fits on 8xA100", fits, True)

fits2, util2 = can_train_on_cluster(70_000_000_000, n_gpus=8, gpu_memory_gb=80)
print(f"LLaMA-70B on 8×A100: fits={fits2}, utilization={util2:.1%}")
judge.check("Ex3b: LLaMA-70B does NOT fit on 8xA100", fits2, False)


# %% [markdown]
# ---
# ## Exercise 4: Training Time Estimation
#
# Estimate how long training will take given the compute budget.
#
# Key formula: $\text{FLOPs} = 6 \times N \times D$ where $N$ = parameters, $D$ = training tokens.
#
# **Model FLOP Utilization (MFU)** accounts for real-world inefficiencies (memory bandwidth, communication overhead). Typical values:
# - Single GPU (no communication): ~50-60% MFU
# - Multi-GPU with NVLink: ~45-55% MFU
# - Multi-node: ~35-45% MFU
#
# ```
# TODO: Implement estimate_training_time
# ```
#

# %%
def estimate_training_time(
    n_params: int,
    n_tokens: int,
    n_gpus: int,
    gpu_tflops: float,
    mfu: float = 0.45
) -> dict:
    """
    Estimate total training time.

    Args:
        n_params:    Number of model parameters
        n_tokens:    Number of training tokens
        n_gpus:      Number of GPUs
        gpu_tflops:  Peak TFLOPS per GPU (for the precision used)
        mfu:         Model FLOP Utilization (0 to 1)

    Returns:
        dict with keys: total_flops, effective_tflops, seconds, hours, days
    """
    # TODO: Calculate total FLOPs for the training run
    total_flops = 0  # TODO: 6 * n_params * n_tokens

    # TODO: Calculate effective cluster throughput (FLOPs/second)
    effective_flops_per_sec = 0  # TODO: n_gpus * gpu_tflops * 1e12 * mfu

    # TODO: Calculate time
    seconds = 0  # TODO
    hours = 0    # TODO
    days = 0     # TODO

    return {
        "total_flops": total_flops,
        "effective_tflops": effective_flops_per_sec / 1e12,
        "seconds": seconds,
        "hours": hours,
        "days": days,
    }


# LLaMA-2 7B training: 7B params, 2T tokens, 1024 A100s (312 TFLOPS BF16)
result = estimate_training_time(
    n_params=7_000_000_000,
    n_tokens=2_000_000_000_000,
    n_gpus=1024,
    gpu_tflops=312,
    mfu=0.45
)
print(f"Total FLOPs: {result['total_flops']:.2e}")
print(f"Effective throughput: {result['effective_tflops']:.1f} TFLOPS")
print(f"Estimated training time: {result['days']:.1f} days")

# Should be roughly 3.7 days
judge.check("Ex4: LLaMA-2 7B training days", result['days'], 3.74, tol=0.05)

# %% [markdown]
# <details>
# <summary>💡 Hint</summary>
#
# ```python
# total_flops = 6 * n_params * n_tokens
# effective_flops_per_sec = n_gpus * gpu_tflops * 1e12 * mfu
# seconds = total_flops / effective_flops_per_sec
# hours = seconds / 3600
# days = hours / 24
# ```
# </details>
#

# %% [markdown]
# ---
# ## Build: The Sizing Calculator
#
# Compose Exercises 1–4 into a single function that takes a model+cluster spec and returns
# a verdict. We'll reach for this in every later chapter to predict whether a configuration
# will fit before we run it.
#
# ```
# TODO: Implement size_run — compose count_transformer_params, training_memory_gb,
#       can_train_on_cluster, and estimate_training_time into one report.
# ```
#

# %%
def size_run(
    *,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    max_seq_len: int,
    n_tokens: int,
    n_gpus: int,
    gpu_memory_gb: float = 80.0,
    gpu_tflops: float = 312.0,
    mfu: float = 0.45,
) -> dict:
    """One-stop sizing: params, training memory, fit verdict, training time."""
    # TODO: call count_transformer_params, training_memory_gb,
    #       can_train_on_cluster, estimate_training_time and assemble a report dict.
    return {}


# Example: can we train Llama-3 70B on 256 H100s in BF16?
# (Llama-3 70B: vocab 128_256, d_model 8192, n_layers 80, max_seq 8192,
#  ~15T training tokens.)
report = size_run(
    vocab_size=128_256,
    d_model=8192,
    n_layers=80,
    max_seq_len=8192,
    n_tokens=15_000_000_000_000,
    n_gpus=256,
    gpu_memory_gb=80.0,
    gpu_tflops=989.0,  # H100 BF16 dense
    mfu=0.40,
)
print(report)


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
# 1. **Memory wall:** Training a large model requires ~16 bytes per parameter (weights + grads + Adam state).
#    GPT-3 needs ~2.8 TB — impossible on a single GPU.
# 2. **Time wall:** Even if memory weren't an issue, single-GPU training would take years; thousands of
#    GPUs cut it to days.
# 3. **Communication is the bottleneck:** NVLink (within node) is fast; InfiniBand (cross-node) is the
#    real constraint at scale.
# 4. **Multiple parallelism axes:** DP, FSDP, TP, embedding, pipeline, sequence, expert — each solves a
#    different bottleneck. The rest of the course implements them one by one in JAX + Equinox.
#
# ---
# **Next:** [Chapter 1 — JAX, Equinox & TinyGPT](./chapter_01_jax_equinox_intro.ipynb) — a hands-on
# tour of the JAX primitives and the TinyGPT model we'll use throughout the course.
#

# %%
