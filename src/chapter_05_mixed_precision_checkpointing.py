# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/chapters/chapter_05_mixed_precision_checkpointing.ipynb)
#
# # Chapter 5: Mixed Precision Training & Fault Tolerance
#
# > **Course: Distributed Training — From Concepts to JAX**
#
# ---
#
# ## Learning Objectives
#
# - Explain FP16 vs BF16 and when each overflows
# - Implement loss scaling to prevent FP16 underflow
# - Implement gradient checkpointing memory/compute trade-off
# - Design a checkpoint save/restore strategy for distributed training
#

# %% [markdown]
# ---
# ## 1. Floating Point Formats
#
# | Format | Bits | Exponent | Mantissa | Max value | Min normal |
# |--------|------|----------|----------|-----------|------------|
# | FP32   | 32   | 8        | 23       | ~3.4×10³⁸ | ~1.2×10⁻³⁸ |
# | FP16   | 16   | 5        | 10       | 65504     | ~6.1×10⁻⁵  |
# | BF16   | 16   | 8        | 7        | ~3.4×10³⁸ | ~1.2×10⁻³⁸ |
#
# **FP16** has the same number of mantissa bits as FP32 relatively, but a tiny exponent range:
# - Max value **65504** — typical gradients at init (~1e-3) are fine, but loss can overflow at start
# - Subnormals below **6×10⁻⁵** — small gradients vanish (underflow)
#
# **BF16** trades mantissa precision for range (same exponent as FP32):
# - Can represent the same magnitude as FP32 — no overflow risk
# - Only 7 mantissa bits (vs 10 for FP16) — less numerical precision per value
# - Used in all modern LLM training (A100, TPUs)
#
# ### Mixed precision recipe
# ```
# FP16/BF16 forward pass      ← fast tensor cores, 2× memory savings on activations
# FP16/BF16 backward pass     ← same
# FP32 gradient accumulation  ← prevent precision loss when summing small values  
# FP32 optimizer states       ← Adam m, v need precision; weights need exact updates
# FP32 master weights         ← high-precision copy for optimizer step
# Copy FP32 → FP16 weights    ← before next forward pass
# ```
#

# %% [markdown]
# ---
# ## 2. Loss Scaling
#
# With FP16, small gradients **underflow to zero** before the backward pass completes. Solution: **scale the loss** before backward, then unscale gradients before the optimizer step.
#
# ```
# loss_scaled = loss * scale_factor      # e.g., scale=1024
# loss_scaled.backward()                 # grads are scale_factor× larger → no underflow
# grads /= scale_factor                  # unscale before optimizer step
# if grads contain inf/nan: skip step    # overflow happened, reduce scale
# else: optimizer.step(); maybe increase scale
# ```
#
# **Dynamic loss scaling** adjusts the scale factor automatically:
# - If no overflow for `growth_interval` steps → multiply scale by `growth_factor` (e.g. ×2)
# - If overflow detected → divide scale by `backoff_factor` (e.g. ÷2)
#

# %% [markdown]
# ---
# ## 3. Gradient Checkpointing
#
# During the backward pass, PyTorch/JAX needs the **activations** computed in the forward pass to compute gradients.
#
# **Without checkpointing:** Store all activations — memory = O(L × B × S × H).
#
# **With checkpointing:** Discard activations after each layer; recompute them during backward.
# - Memory: O(√L × B × S × H) with optimal checkpoint placement
# - Compute: ~33% overhead (one extra forward pass for the recomputed segments)
#
# ```
# Forward:  L0 → ✓save → L1 → ✗discard → L2 → ✓save → L3 → ✗discard
# Backward: need L3 acts → recompute from checkpoint at L2
#           need L1 acts → recompute from checkpoint at L0
# ```
#
# For a 72-layer model with B=4, S=2048, H=4096 (BF16):
# - Without checkpointing: ~72 × 4 × 2048 × 4096 × 2 ≈ **4.8 GB** activations
# - With checkpointing: ≈ **0.7 GB** (checkpoint every 8 layers)
#

# %% [markdown]
# ---
# ## Judge Setup
#

# %%
import numpy as np
import struct
from judge import Judge

judge = Judge("Chapter 5", default_tol=1e-4)
print("Judge ready!")


# %% [markdown]
# ---
# ## Exercise 1: FP16 Overflow and Underflow Detection
#
# FP16 max is **65504**. Values above this become `inf`; values below `~6e-8` (subnormal threshold) become `0`.
#
# TODO: Implement `check_fp16_issues`
#

# %%
def check_fp16_issues(values: np.ndarray) -> dict:
    """
    Check an FP32 array for values that would overflow or underflow in FP16.
    
    FP16 range: max representable = 65504
                min normal = 6.104e-5  (below this, value becomes subnormal or 0)
    
    Args:
        values: FP32 numpy array
    
    Returns:
        dict with keys:
          'n_overflow':  count of |values| > 65504
          'n_underflow': count of 0 < |values| < 6.104e-5
          'n_safe':      count of safe values (including exact zero)
          'overflow_frac':  fraction of elements that overflow
    """
    FP16_MAX = 65504.0
    FP16_MIN_NORMAL = 6.104e-5
    
    abs_vals = np.abs(values)
    
    # TODO: count overflows (|x| > FP16_MAX)
    n_overflow = 0   # TODO
    
    # TODO: count underflows (0 < |x| < FP16_MIN_NORMAL)
    n_underflow = 0  # TODO
    
    # TODO: count safe values (exactly 0 or within FP16 normal range)
    n_safe = 0       # TODO
    
    overflow_frac = n_overflow / len(values)
    
    return {'n_overflow': n_overflow, 'n_underflow': n_underflow,
            'n_safe': n_safe, 'overflow_frac': overflow_frac}


test_vals = np.array([1e-6, 1e-4, 1.0, 100.0, 65505.0, -70000.0, 0.0, 3e-5])
result = check_fp16_issues(test_vals)
print(f"Overflow:  {result['n_overflow']}")   # expect 2
print(f"Underflow: {result['n_underflow']}")  # expect 2 (1e-6 and 3e-5)
print(f"Safe:      {result['n_safe']}")       # expect 4

judge.check("Ex1a: overflow count",  result['n_overflow'],  2)
judge.check("Ex1b: underflow count", result['n_underflow'], 2)
judge.check("Ex1c: safe count",      result['n_safe'],      4)


# %% [markdown]
# ---
# ## Exercise 2: Loss Scaling
#
# Implement the core loss scaling logic: scale before backward, unscale after, detect inf/nan.
#
# TODO: Implement `LossScaler`
#

# %%
class LossScaler:
    """
    Dynamic loss scaler for FP16 training.
    
    Usage:
        scaler = LossScaler(init_scale=1024)
        scaled_loss = scaler.scale(loss)
        # ... backward on scaled_loss ...
        ok = scaler.unscale_and_check(grads)
        if ok:
            optimizer.step()
        scaler.update(overflow=not ok)
    """
    def __init__(self, init_scale=1024.0, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._steps_since_overflow = 0

    def scale_loss(self, loss: float) -> float:
        """Return loss * current scale."""
        # TODO
        pass

    def unscale_grads(self, grads: np.ndarray) -> np.ndarray:
        """Divide gradients by the current scale factor."""
        # TODO
        pass

    def has_overflow(self, grads: np.ndarray) -> bool:
        """Return True if any gradient element is inf or nan."""
        # TODO: np.any(~np.isfinite(grads))
        pass

    def update(self, overflow: bool) -> None:
        """
        Update scale factor:
        - If overflow: multiply scale by backoff_factor, reset step counter
        - If no overflow: increment counter; if counter >= growth_interval,
          multiply scale by growth_factor and reset counter
        """
        # TODO
        pass


scaler = LossScaler(init_scale=1024.0, growth_factor=2.0,
                    backoff_factor=0.5, growth_interval=3)

# Test scaling
scaled = scaler.scale_loss(0.5)
judge.check("Ex2a: scale_loss", scaled, 512.0)

# Test unscaling
grads = np.array([1024.0, 2048.0, 512.0])
unscaled = scaler.unscale_grads(grads)
judge.check("Ex2b: unscale_grads", unscaled, np.array([1.0, 2.0, 0.5]))

# Test overflow detection
judge.check("Ex2c: has_overflow — clean grads", scaler.has_overflow(unscaled), False)
judge.check("Ex2d: has_overflow — inf grad",
            scaler.has_overflow(np.array([1.0, np.inf, 0.5])), True)

# Test dynamic scale update: no overflow for 3 steps → scale doubles
for _ in range(3):
    scaler.update(overflow=False)
judge.check("Ex2e: scale grew after growth_interval steps", scaler.scale, 2048.0)

# Overflow: scale halves
scaler.update(overflow=True)
judge.check("Ex2f: scale backed off after overflow", scaler.scale, 1024.0)


# %% [markdown]
# ---
# ## Exercise 3: Gradient Checkpointing Memory Savings
#
# Calculate the memory savings from gradient checkpointing given a checkpoint interval.
#
# With checkpointing every `k` layers:
# - Store activations for `k` layers at a time (recompute the rest)
# - Memory ≈ activation_per_layer × k  (plus checkpoint tensors)
#
# TODO: Implement `checkpoint_memory_analysis`
#

# %%
def activation_memory_per_layer_gb(
    batch_size: int, seq_len: int, d_model: int,
    n_heads: int, bytes_per_elem: int = 2
) -> float:
    """
    Estimate activation memory per transformer layer in GB.
    
    Components stored per layer (approximate, from Megatron analysis):
    - Attention QKV: 3 * B * S * H
    - Attention scores: B * n_heads * S * S  (quadratic in seq_len!)
    - MLP activations: 4 * B * S * H
    - LayerNorm inputs: 2 * B * S * H
    Total ≈ (9*H + n_heads*S) * B * S  elements
    """
    H, B, S = d_model, batch_size, seq_len
    
    # TODO: calculate total elements using the formula above
    total_elements = 0  # TODO: (9 * H + n_heads * S) * B * S
    
    return total_elements * bytes_per_elem / 1e9


def checkpoint_memory_analysis(
    n_layers: int,
    checkpoint_interval: int,
    activation_gb_per_layer: float
) -> dict:
    """
    Compare memory usage with vs without gradient checkpointing.
    
    Without checkpointing: store all n_layers activations.
    With checkpointing every k layers:
      - Keep checkpoint activations: ceil(n_layers/k) * activation_gb_per_layer
      - Plus recomputation buffer: k * activation_gb_per_layer (worst case)
    
    Returns dict with: no_checkpoint_gb, with_checkpoint_gb, memory_reduction_factor
    """
    import math
    
    # TODO: memory without checkpointing
    no_ckpt_gb = 0  # TODO: n_layers * activation_gb_per_layer
    
    # TODO: memory with checkpointing
    # checkpoints stored: ceil(n_layers / checkpoint_interval)
    # recomputation buffer: checkpoint_interval layers at a time
    n_checkpoints = 0  # TODO: math.ceil(n_layers / checkpoint_interval)
    with_ckpt_gb = 0   # TODO: (n_checkpoints + checkpoint_interval) * activation_gb_per_layer
    
    reduction = no_ckpt_gb / with_ckpt_gb if with_ckpt_gb > 0 else 0
    
    return {'no_checkpoint_gb': no_ckpt_gb,
            'with_checkpoint_gb': with_ckpt_gb,
            'memory_reduction_factor': reduction}


# LLaMA-2 7B: 32 layers, B=4, S=2048, H=4096, 32 heads
act_per_layer = activation_memory_per_layer_gb(4, 2048, 4096, 32, bytes_per_elem=2)
print(f"Activation memory per layer: {act_per_layer:.3f} GB")

analysis = checkpoint_memory_analysis(32, checkpoint_interval=4, 
                                       activation_gb_per_layer=act_per_layer)
print(f"Without checkpointing: {analysis['no_checkpoint_gb']:.2f} GB")
print(f"With checkpointing:    {analysis['with_checkpoint_gb']:.2f} GB")
print(f"Memory reduction:      {analysis['memory_reduction_factor']:.1f}×")

judge.check("Ex3a: activation per layer shape", act_per_layer > 0.01, True)
judge.check("Ex3b: checkpointing reduces memory",
            analysis['with_checkpoint_gb'] < analysis['no_checkpoint_gb'], True)
judge.check("Ex3c: memory reduction > 2×",
            analysis['memory_reduction_factor'] > 2.0, True)

# %% [markdown]
# ---
# ## Exercise 4: Distributed Checkpoint Design
#
# Design a checkpoint that saves model state across multiple ranks. Each rank saves its shard; a separate metadata file records how to reconstruct the full model.
#
# TODO: Implement `save_distributed_checkpoint` and `load_distributed_checkpoint`
#

# %%
import json
import os

def save_distributed_checkpoint(
    rank: int,
    params_shard: np.ndarray,
    opt_state_shard: dict,
    step: int,
    checkpoint_dir: str,
    n_ranks: int
) -> str:
    """
    Save a single rank's checkpoint shard.
    
    Each rank saves:
      {checkpoint_dir}/step_{step:06d}/rank_{rank:04d}.npz
    
    Rank 0 additionally saves a metadata JSON:
      {checkpoint_dir}/step_{step:06d}/metadata.json
    
    Args:
        rank:            This rank's index
        params_shard:    This rank's parameter shard
        opt_state_shard: Dict with 'm' and 'v' arrays for this rank
        step:            Training step number
        checkpoint_dir:  Base directory
        n_ranks:         Total ranks (for metadata)
    
    Returns:
        Path to the saved file
    """
    step_dir = os.path.join(checkpoint_dir, f"step_{step:06d}")
    os.makedirs(step_dir, exist_ok=True)
    
    # TODO: Save rank shard as .npz
    shard_path = os.path.join(step_dir, f"rank_{rank:04d}.npz")
    # np.savez(shard_path, params=params_shard, m=opt_state_shard['m'], v=opt_state_shard['v'])
    # TODO
    
    # TODO: Rank 0 saves metadata JSON
    if rank == 0:
        metadata = {
            'step': step,
            'n_ranks': n_ranks,
            'shard_size': len(params_shard),
            'format': 'npz'
        }
        meta_path = os.path.join(step_dir, 'metadata.json')
        # TODO: json.dump(metadata, open(meta_path, 'w'))
    
    return shard_path


def load_distributed_checkpoint(
    rank: int,
    checkpoint_dir: str,
    step: int
) -> tuple:
    """
    Load a rank's checkpoint shard.
    
    Returns:
        (params_shard, opt_state_shard, metadata)
        where opt_state_shard = {'m': ..., 'v': ...}
    """
    step_dir = os.path.join(checkpoint_dir, f"step_{step:06d}")
    shard_path = os.path.join(step_dir, f"rank_{rank:04d}.npz")
    meta_path  = os.path.join(step_dir, 'metadata.json')
    
    # TODO: Load .npz and metadata
    data = None      # TODO: np.load(shard_path)
    metadata = None  # TODO: json.load(open(meta_path))
    
    params_shard    = None  # TODO: data['params']
    opt_state_shard = None  # TODO: {'m': data['m'], 'v': data['v']}
    
    return params_shard, opt_state_shard, metadata


# Test with a simulated 4-rank setup
import tempfile
tmpdir = tempfile.mkdtemp()

n_ranks = 4
params_full = np.random.randn(100).astype(np.float32)
shards = np.array_split(params_full, n_ranks)

# Save all ranks
for r in range(n_ranks):
    opt = {'m': np.zeros_like(shards[r]), 'v': np.ones_like(shards[r])}
    save_distributed_checkpoint(r, shards[r], opt, step=100, 
                                  checkpoint_dir=tmpdir, n_ranks=n_ranks)

# Load rank 2 and verify
loaded_params, loaded_opt, loaded_meta = load_distributed_checkpoint(2, tmpdir, step=100)

judge.check("Ex4a: loaded params match saved", loaded_params, shards[2])
judge.check("Ex4b: metadata step", loaded_meta['step'], 100)
judge.check("Ex4c: metadata n_ranks", loaded_meta['n_ranks'], 4)
judge.check("Ex4d: opt state v all ones", np.all(loaded_opt['v'] == 1.0), True)

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
# 1. **Use BF16** for modern LLM training — same exponent range as FP32, no overflow risk unlike FP16.
# 2. **Loss scaling** prevents FP16 gradient underflow; dynamic scaling adapts automatically.
# 3. **Gradient checkpointing** trades ~33% extra compute for a large memory reduction — essential for long sequences.
# 4. **Distributed checkpoints** must save per-rank shards with metadata for robust recovery.
#
# ---
# **Next:** [Chapter 6 — JAX Fundamentals for Distributed Training](./chapter_06_jax_fundamentals.ipynb)
#
