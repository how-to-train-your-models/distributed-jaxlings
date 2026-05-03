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
# - Explain the data parallel training loop and why gradient averaging keeps model copies in sync
# - Describe the Ring-AllReduce algorithm and its bandwidth efficiency
# - Map Ring-AllReduce to JAX's `jax.lax.pmean` / `jax.pmap` APIs
# - Distinguish synchronous vs asynchronous gradient updates
# - Implement gradient accumulation to simulate larger batch sizes
# - Explain ZeRO stages and the memory savings they provide
#

# %% [markdown]
# ---
# ## Setup
#

# %%
import os
# Simulate 4 CPU devices so pmap examples run on any machine (including single-GPU Colab).
# Must be set before JAX is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax

from typing import Any, List, Tuple
from jaxtyping import Array, Float, Integer
from judge import Judge

judge = Judge("Chapter 2")
print(f"JAX devices: {jax.devices()}")
print("Judge ready!")

# %% [markdown]
# ---
# ## 1. The Data Parallel Idea
#
# Data parallelism is the simplest and most widely used form of distributed training.
#
# **Core idea:**
# 1. Each worker (GPU) holds a **full copy** of the model
# 2. The global mini-batch is **split** across workers — each sees a different micro-batch
# 3. Each worker computes a forward + backward pass independently
# 4. Gradients are **averaged** across all workers (AllReduce)
# 5. Each worker applies the same gradient update → models stay in sync
#
# ```
# Global batch = [b0, b1, b2, b3]  (4 micro-batches)
#
# GPU 0: model_copy | b0 → dW_0 ─┐
# GPU 1: model_copy | b1 → dW_1 ─┤→ AllReduce → dW_avg → update all
# GPU 2: model_copy | b2 → dW_2 ─┤
# GPU 3: model_copy | b3 → dW_3 ─┘
# ```
#
# **Effective batch size** = micro_batch_size × n_workers.
#
# ### The gradient is already a partial sum
#
# For a linear layer `Y_BxSxF = X_BxSxD @ W_DxF`, the gradient of the loss w.r.t. `W` is:
#
# ```
# dW_DxF = einsum("b s d, b s f -> d f", X_BxSxD, δ_BxSxF)
# ```
#
# This einsum sums over `b` and `s`, collapsing the entire batch into a single `[D, F]`
# tensor — **the same shape as W**. Each entry `dW[d, f]` tells you how much to nudge
# `W[d, f]`. The gradient is already a *partial sum* over the local micro-batch.
#
# ### Why AllReduce is necessary
#
# Without AllReduce, each worker updates its model copy with its own local gradient:
#
# ```
# ✗ Without AllReduce:
#   GPU 0: W -= lr * dW_0   ← only saw batch 0
#   GPU 1: W -= lr * dW_1   ← only saw batch 1
#   → models diverge after one step — the copies are no longer identical
# ```
#
# We need every worker to agree on a single `dW` before updating.
# That requires combining N gradient tensors into one — this is a **reduce** operation.
#
# **Reduce** means: combine many values into one using some operation.
# Here the operation is element-wise mean across workers:
#
# ```
# reduce(mean, [dW_0, dW_1, dW_2, dW_3])  →  dW_avg      (one [D,F] tensor)
#
# dW_avg[d, f] = mean over workers of dW_w[d, f]     ∀ (d, f)
# ```
#
# The "All" in AllReduce means the result is sent back to **all** workers, not just one.
#
# **Important:** AllReduce only produces `dW_avg` — it does not touch the model weights.
# The weight update is a separate local step that each worker performs independently
# after AllReduce completes:
#
# ```
# Step 1 — AllReduce (communication):
#   [dW_0, dW_1, dW_2, dW_3]  →  dW_avg     (every worker now holds this)
#
# Step 2 — Weight update (local, no communication):
#   GPU 0: W -= lr * dW_avg   ┐
#   GPU 1: W -= lr * dW_avg   ├  each worker updates its own copy identically
#   GPU 2: W -= lr * dW_avg   ┘
# ```
#
# Because every worker applies the same `dW_avg`, the copies stay in sync after the update.
# AllReduce is the **only communication step** in data parallelism.
#
# **Axis legend:** `B`=batch · `S`=seq · `D`=model dim · `F`=ff dim · `W`=workers · `P`=params · `Ps`=param shard
#

# %% [markdown]
# ---
# ### Exercise 1: Data Parallel Gradient Averaging
#
# Each worker computes `dW` over its micro-batch.
# Implement `worker_grad_W` and `data_parallel_average`.
#

# %%
def worker_grad_W(
    X_BxSxD:     Float[Array, "B S D"],
    delta_BxSxF: Float[Array, "B S F"],
) -> Float[Array, "D F"]:
    """
    Gradient of a linear layer's weight matrix on one micro-batch.

    dW_DxF = einsum("b s d, b s f -> d f", X_BxSxD, delta_BxSxF)

    Args:
        X_BxSxD:     activations,        shape [B, S, D]
        delta_BxSxF: upstream gradient,  shape [B, S, F]

    Returns:
        dW_DxF of shape [D, F]
    """
    raise NotImplementedError(
        "TODO: return jnp.einsum('bsd,bsf->df', X_BxSxD, delta_BxSxF)"
    )


def data_parallel_average(
    worker_grads_WxDxF: Float[Array, "W D F"],
) -> Float[Array, "D F"]:
    """
    Average gradients from all workers — simulates the AllReduce result.

    Reduces along the W axis element-wise; D and F are unchanged.
    Shape: [W, D, F] → [D, F]

    Args:
        worker_grads_WxDxF: stacked gradients, shape [W, D, F]

    Returns:
        dW_avg_DxF of shape [D, F]
    """
    raise NotImplementedError(
        "TODO: return jnp.mean(worker_grads_WxDxF, axis=0)\n"
        "      equivalently: jnp.einsum('wdf->df', worker_grads_WxDxF) / W"
    )


# ── Test ─────────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(0)
B, S, D, F, W = 4, 6, 8, 16, 3

Xs_WxBxSxD     = jax.random.normal(key, (W, B, S, D))
deltas_WxBxSxF = jax.random.normal(key, (W, B, S, F))

local_grads_WxDxF = jnp.stack([worker_grad_W(Xs_WxBxSxD[w], deltas_WxBxSxF[w]) for w in range(W)])
avg_grad_DxF      = data_parallel_average(local_grads_WxDxF)

expected_avg_DxF = jnp.mean(
    jnp.stack([jnp.einsum("bsd,bsf->df", Xs_WxBxSxD[w], deltas_WxBxSxF[w]) for w in range(W)]),
    axis=0,
)

judge.check("Ex1a: worker_grad_W shape",         local_grads_WxDxF.shape, (W, D, F))
judge.check("Ex1b: data_parallel_average shape", avg_grad_DxF.shape,      (D, F))
judge.check("Ex1c: averaged values correct",     avg_grad_DxF,            expected_avg_DxF)

# %% [markdown]
# <details>
# <summary>💡 Hint</summary>
#
# ```python
# # worker_grad_W
# return jnp.einsum("bsd,bsf->df", X_BxSxD, delta_BxSxF)
#
# # data_parallel_average
# return jnp.mean(worker_grads_WxDxF, axis=0)
# ```
# </details>
#

# %% [markdown]
# ---
# ## 2. AllReduce: Synchronizing Gradients
#
# AllReduce **reduces tensors across all workers and distributes the result back to everyone**.
# It is the only communication step in data parallelism.
#
# ### Three topologies — very different bandwidth costs
#
# **Parameter server** (not AllReduce — a different architecture entirely):
# ```
# All workers → send dW[P] to a dedicated server node → server sums → broadcasts back
# Server bandwidth: N × P incoming + N × P outgoing = 2NP   ← bottleneck scales with N
# ```
# A central node holds the parameters and acts as an aggregator.
# **JAX does not support this** — JAX is SPMD: every device runs the same program.
# There is no concept of a "server" role in JAX.
#
# **Naive AllReduce** (star/coordinator topology):
# ```
# All workers → send dW[P] to one coordinator worker → coordinator sums → broadcasts back
# Coordinator bandwidth: N × P in + N × P out = 2NP   ← same bottleneck, different node
# ```
#
# **Ring-AllReduce** — no bottleneck node, bandwidth-optimal:
#
# Workers are arranged in a ring. The gradient vector (shape `[P]`) is split into N chunks.
#
# **Phase 1 — Reduce-Scatter** (N-1 steps):
# Each worker sends one chunk rightward and **adds** the received chunk to its own.
# After N-1 steps, each worker owns the **fully reduced** version of exactly one chunk.
#
# **Phase 2 — AllGather** (N-1 steps):
# Each worker sends its complete (reduced) chunk rightward, **overwriting** the receiver's copy.
# After N-1 steps, every worker has all chunks → full averaged gradient.
#
# ```
# Ring with 4 workers, gradient split into 4 chunks [A, B, C, D]
#
# Initial:          W0=[A,B,C,D]  W1=[A,B,C,D]  W2=[A,B,C,D]  W3=[A,B,C,D]
# After Reduce-Scatter:  W0 owns ΣA    W1 owns ΣB    W2 owns ΣC    W3 owns ΣD
# After AllGather:       all workers hold [ΣA, ΣB, ΣC, ΣD]
# ```
#
# **Bandwidth per worker:** `2(N-1)/N × P` → converges to **2P** as N → ∞.
# Each worker always sends/receives ≈ 2P regardless of how many workers there are.
#
# **Memory:** each worker holds exactly one `P`-sized buffer throughout.
# Chunks are overwritten in-place — memory per worker stays `O(P)`.
#

# %% [markdown]
# ---
# ### Exercise 2: Ring-AllReduce
#
# Implement the two-phase Ring-AllReduce over a list of 1-D JAX arrays (one per worker).
#

# %%
def ring_allreduce(
    worker_data: List[Float[Array, "P"]],
) -> List[Float[Array, "P"]]:
    """
    Ring-AllReduce over a list of 1-D JAX arrays (one per worker).

    Args:
        worker_data: list of n arrays, each shape [P]  (P must be divisible by n)

    Returns:
        list of n arrays — each worker holds the element-wise SUM.
        (divide by n externally to get the mean)
    """
    n = len(worker_data)
    p = worker_data[0].shape[0]
    assert p % n == 0, "P must be divisible by n for this simplified version"
    chunk = p // n

    # chunks[w][i] = slice i of worker w's gradient vector
    chunks = [
        [worker_data[w][i * chunk:(i + 1) * chunk] for i in range(n)]
        for w in range(n)
    ]

    # ── Phase 1: Reduce-Scatter ──────────────────────────────────────────────
    # Goal: worker w ends up with the fully-reduced version of chunk w.
    # Each step: every worker receives one chunk from its left neighbour and adds it.
    for step in range(n - 1):
        new_chunks = [row[:] for row in chunks]
        for w in range(n):
            src = (w - 1) % n              # left neighbour
            idx = 0                        # TODO: (w - 1 - step) % n
            raise NotImplementedError(
                "TODO: idx = (w - 1 - step) % n\n"
                "      new_chunks[w][idx] = chunks[w][idx] + chunks[src][idx]"
            )
        chunks = new_chunks

    # ── Phase 2: AllGather ───────────────────────────────────────────────────
    # Goal: every worker has all fully-reduced chunks.
    # Each step: every worker receives a complete chunk from its left neighbour and overwrites.
    for step in range(n - 1):
        new_chunks = [row[:] for row in chunks]
        for w in range(n):
            src = (w - 1) % n
            idx = 0                        # TODO: (w - step) % n
            raise NotImplementedError(
                "TODO: idx = (w - step) % n\n"
                "      new_chunks[w][idx] = chunks[src][idx]"
            )
        chunks = new_chunks

    return [jnp.concatenate(chunks[w]) for w in range(n)]


# ── Test ─────────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(1)
W, P = 4, 8
data = [jax.random.normal(key, (P,)) for _ in range(W)]

true_sum_P = jnp.sum(jnp.stack(data), axis=0)
results    = ring_allreduce(data)

judge.check("Ex2a: worker 0 holds the sum",
            results[0], true_sum_P)
judge.check("Ex2b: all workers agree",
            jnp.allclose(results[0], results[1]) and jnp.allclose(results[1], results[3]),
            True)

# %% [markdown]
# <details>
# <summary>💡 Hint — Reduce-Scatter</summary>
#
# ```python
# for step in range(n - 1):
#     new_chunks = [row[:] for row in chunks]
#     for w in range(n):
#         src = (w - 1) % n
#         idx = (w - 1 - step) % n
#         new_chunks[w][idx] = chunks[w][idx] + chunks[src][idx]
#     chunks = new_chunks
# ```
# </details>
#
# <details>
# <summary>💡 Hint — AllGather</summary>
#
# ```python
# for step in range(n - 1):
#     new_chunks = [row[:] for row in chunks]
#     for w in range(n):
#         src = (w - 1) % n
#         idx = (w - step) % n
#         new_chunks[w][idx] = chunks[src][idx]
#     chunks = new_chunks
# ```
# </details>
#

# %% [markdown]
# ---
# ## 2b. JAX in Practice: `pmap` + `lax.pmean`
#
# You just implemented Ring-AllReduce by hand. In JAX you never write this —
# XLA compiles it for you. The full data-parallel training loop is:
#
# ```python
# @functools.partial(jax.pmap, axis_name="devices")
# def train_step(W_DxF, X_BxSxD, Y_BxSxF):
#     grads_DxF = jax.grad(loss_fn)(W_DxF, X_BxSxD, Y_BxSxF)
#     grads_DxF = jax.lax.pmean(grads_DxF, axis_name="devices")  # ← Ring-AllReduce here
#     updates, new_opt_state = optimizer.update(grads_DxF, opt_state)
#     return optax.apply_updates(W_DxF, updates)
# ```
#
# `jax.pmap` maps a function across devices (the implicit `W` axis).
# `jax.lax.pmean` averages the tensor element-wise across all devices in `axis_name`:
#
# ```
# Each device holds:  grads_DxF   shape [D, F]
# After pmean:        grads_DxF   shape [D, F]   (same shape — now averaged across W devices)
# ```
#
# The `W` axis is implicit inside `pmap` — `pmean` collapses it without you ever
# stacking tensors manually. XLA emits Ring-AllReduce automatically.
#
# ### JAX collective API
#
# | Operation | JAX API | Shape inside pmap |
# |---|---|---|
# | AllReduce mean | `jax.lax.pmean(x, axis_name)` | `[D, F] → [D, F]` (averaged) |
# | AllReduce sum  | `jax.lax.psum(x, axis_name)`  | `[D, F] → [D, F]` (summed) |
# | Reduce-Scatter | `jax.lax.reduce_scatter(x, axis_name, scatter_dimension=0)` | `[D, F] → [D//W, F]` |
# | AllGather      | `jax.lax.all_gather(x, axis_name)` | `[D, F] → [W, D, F]` |
# | Parallel map   | `jax.pmap(fn, axis_name)` | maps fn over leading device axis |
#

# %%
# ── Demo: pmap + lax.pmean (read-only, no exercise) ──────────────────────────

def mse_loss(
    W_DxF:   Float[Array, "D F"],
    X_BxSxD: Float[Array, "B S D"],
    Y_BxSxF: Float[Array, "B S F"],
) -> Float[Array, ""]:
    """MSE loss for a linear layer. Axes: B=batch, S=seq, D=in_dim, F=out_dim."""
    Y_hat_BxSxF = jnp.einsum("bsd,df->bsf", X_BxSxD, W_DxF)
    return jnp.mean((Y_hat_BxSxF - Y_BxSxF) ** 2)


n_devices = len(jax.devices())
key = jax.random.PRNGKey(42)
B_demo, S_demo, D_demo, F_demo = 2, 4, 6, 8

W_true_DxF      = jax.random.normal(key, (D_demo, F_demo))
X_ndevxBxSxD    = jax.random.normal(key, (n_devices, B_demo, S_demo, D_demo))
Y_ndevxBxSxF    = jnp.stack([
    jnp.einsum("bsd,df->bsf", X_ndevxBxSxD[i], W_true_DxF) for i in range(n_devices)
])

W_init_DxF           = jax.random.normal(jax.random.PRNGKey(99), (D_demo, F_demo)) * 0.01
W_replicated_ndevxDxF = jax.device_put_replicated(W_init_DxF, jax.devices())


@functools.partial(jax.pmap, axis_name="devices")
def pmap_grad_step(
    W_DxF:   Float[Array, "D F"],
    X_BxSxD: Float[Array, "B S D"],
    Y_BxSxF: Float[Array, "B S F"],
) -> Float[Array, "D F"]:
    """Compute gradient on each device's micro-batch, then AllReduce via pmean."""
    grads_DxF = jax.grad(mse_loss)(W_DxF, X_BxSxD, Y_BxSxF)
    return jax.lax.pmean(grads_DxF, axis_name="devices")   # Ring-AllReduce


avg_grads_ndevxDxF = pmap_grad_step(W_replicated_ndevxDxF, X_ndevxBxSxD, Y_ndevxBxSxF)

print(f"Devices: {n_devices}")
print(f"All devices agree on averaged gradient: "
      f"{jnp.allclose(avg_grads_ndevxDxF[0], avg_grads_ndevxDxF[-1])}")
print(f"Gradient shape per device: {avg_grads_ndevxDxF[0].shape}  "
      f"(D={D_demo}, F={F_demo}) — same as W, not [W,D,F]")

# %% [markdown]
# ---
# ## 3. Synchronous vs Asynchronous Training
#
# | | Synchronous (SSP) | Asynchronous (ASP) |
# |---|---|---|
# | **Gradient update** | All workers sync before step | Workers update independently |
# | **Convergence** | Identical to single-GPU math | Stale gradients → noise |
# | **Stragglers** | Slowest worker blocks all | No blocking |
# | **Used by** | JAX `pmap`, PyTorch DDP | Hogwild!, some RL systems |
#
# **What "stale" means:** a stale gradient was computed at model weights from a previous step.
# You descend the loss landscape using a slope measured at a different point —
# the further your current weights are from where the gradient was measured,
# the more likely you are to step in the wrong direction.
#
# ### Illustration: fresh vs stale gradient on a 1D quadratic
#

# %%
# ── Staleness illustration (read-only) ───────────────────────────────────────
# Loss: L(w) = (w - w*)²,  true gradient: dL/dw = 2(w - w*)
w_star = 3.0
lr_demo = 0.3

def grad_quadratic(w: float) -> float:
    return 2.0 * (w - w_star)

w_sync  = 0.0
w_async = 0.0

for step in range(8):
    # Synchronous: gradient at current w
    w_sync -= lr_demo * grad_quadratic(w_sync)

    # Asynchronous: gradient was computed 2 steps ago
    w_stale_ref = w_async - 2 * lr_demo * grad_quadratic(w_async)
    w_async    -= lr_demo * grad_quadratic(w_stale_ref)

print(f"After 8 steps:")
print(f"  Sync  → w = {w_sync:.4f}  (distance to w*={w_star}: {abs(w_sync - w_star):.4f})")
print(f"  Async → w = {w_async:.4f}  (distance to w*={w_star}: {abs(w_async - w_star):.4f})")
print(f"  Sync converges tighter: {abs(w_sync - w_star) < abs(w_async - w_star)}")

# %% [markdown]
# Modern LLM training is **always synchronous** — stale gradients hurt convergence too much
# at scale, and the compute-to-communication ratio is large enough that waiting for the
# slowest worker costs relatively little.
#

# %% [markdown]
# ---
# ## 4. Gradient Accumulation
#
# When GPU memory limits the per-step batch size, accumulate gradients over several
# micro-batches before updating:
#
# ```
# effective_batch = micro_batch × accumulation_steps × n_workers
# ```
#
# In JAX functional style this means summing `jax.value_and_grad` outputs before
# passing them to `optax`:
#
# ```python
# grads_acc_DxF = jnp.zeros_like(W_DxF)
# step_loss = 0.0
# for X_mb_BxSxD, Y_mb_BxSxF in zip(micro_Xs, micro_Ys):
#     loss, g_DxF   = jax.value_and_grad(mse_loss)(W_DxF, X_mb_BxSxD, Y_mb_BxSxF)
#     step_loss     += loss
#     grads_acc_DxF  = grads_acc_DxF + g_DxF
# avg_grad_DxF = grads_acc_DxF / n_acc
# updates, new_opt_state = optimizer.update(avg_grad_DxF, opt_state)
# W_DxF = optax.apply_updates(W_DxF, updates)
# ```
#
# Summing `dW` over micro-batches and dividing by `n_acc` is mathematically identical
# to computing the gradient on the full effective batch in a single forward+backward pass.
#

# %% [markdown]
# ---
# ### Exercise 3: Gradient Accumulation in JAX
#
# Train a linear model `Y_BxSxF = X_BxSxD @ W_DxF` with gradient accumulation.
# Use `jax.value_and_grad` for differentiation and `optax.adam` for the optimizer.
#

# %%
def accumulate_and_step(
    W_DxF:     Float[Array, "D F"],
    opt_state: Any,
    optimizer: Any,
    micro_Xs:  List[Float[Array, "B S D"]],
    micro_Ys:  List[Float[Array, "B S F"]],
) -> Tuple[Float[Array, "D F"], Any, Float[Array, ""]]:
    """
    Accumulate gradients over micro-batches, then apply one optimizer step.

    Args:
        W_DxF:     weight matrix [D, F]
        opt_state: current optax optimizer state
        optimizer: optax optimizer (e.g. optax.adam(...))
        micro_Xs:  list of micro-batch inputs,  each [B, S, D]
        micro_Ys:  list of micro-batch targets, each [B, S, F]

    Returns:
        (new_W_DxF, new_opt_state, avg_step_loss)
    """
    raise NotImplementedError(
        "TODO:\n"
        "  n_acc = len(micro_Xs)\n"
        "  grads_acc_DxF = jnp.zeros_like(W_DxF)\n"
        "  step_loss = 0.0\n"
        "  for X_mb_BxSxD, Y_mb_BxSxF in zip(micro_Xs, micro_Ys):\n"
        "      loss, g_DxF = jax.value_and_grad(mse_loss)(W_DxF, X_mb_BxSxD, Y_mb_BxSxF)\n"
        "      step_loss += loss\n"
        "      grads_acc_DxF = grads_acc_DxF + g_DxF\n"
        "  avg_grad_DxF = grads_acc_DxF / n_acc\n"
        "  updates, new_opt_state = optimizer.update(avg_grad_DxF, opt_state)\n"
        "  new_W_DxF = optax.apply_updates(W_DxF, updates)\n"
        "  return new_W_DxF, new_opt_state, step_loss / n_acc"
    )


# ── Test ─────────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(2)
B, S, D, F = 8, 4, 6, 3

W_true_DxF = jax.random.normal(key, (D, F))
W_init_DxF = jax.random.normal(jax.random.PRNGKey(99), (D, F)) * 0.01

optimizer = optax.adam(1e-2)
opt_state = optimizer.init(W_init_DxF)
W_cur_DxF = W_init_DxF

n_steps, n_acc = 300, 4
losses = []
for _ in range(n_steps):
    Xs = [jax.random.normal(key, (B, S, D)) for _ in range(n_acc)]
    Ys = [jnp.einsum("bsd,df->bsf", X, W_true_DxF) for X in Xs]
    W_cur_DxF, opt_state, l = accumulate_and_step(W_cur_DxF, opt_state, optimizer, Xs, Ys)
    losses.append(float(l))

judge.check("Ex3a: loss decreased",    losses[-1] < losses[0],                          True)
judge.check("Ex3b: weights converged", jnp.allclose(W_cur_DxF, W_true_DxF, atol=0.05), True)

# %% [markdown]
# <details>
# <summary>💡 Hint</summary>
#
# ```python
# n_acc = len(micro_Xs)
# grads_acc_DxF = jnp.zeros_like(W_DxF)
# step_loss = 0.0
# for X_mb_BxSxD, Y_mb_BxSxF in zip(micro_Xs, micro_Ys):
#     loss, g_DxF = jax.value_and_grad(mse_loss)(W_DxF, X_mb_BxSxD, Y_mb_BxSxF)
#     step_loss += loss
#     grads_acc_DxF = grads_acc_DxF + g_DxF
# avg_grad_DxF = grads_acc_DxF / n_acc
# updates, new_opt_state = optimizer.update(avg_grad_DxF, opt_state)
# new_W_DxF = optax.apply_updates(W_DxF, updates)
# return new_W_DxF, new_opt_state, step_loss / n_acc
# ```
# </details>
#

# %% [markdown]
# ---
# ## 5. ZeRO: Zero Redundancy Optimizer
#
# Data parallelism stores a **full model copy on every GPU** — wasteful!
# ZeRO (Rajbhandari et al., 2020) eliminates this redundancy progressively:
#
# ```
#               Memory per GPU (175B model, 64 GPUs)
# Base DDP  ──────────────────────────────────────── 2800 GB  (full copy on every GPU)
# ZeRO-1    Shard optimizer states          ──────── 730 GB   (4× reduction)
# ZeRO-2    Shard optimizer states+gradients ──────  365 GB   (8× reduction)
# ZeRO-3    Shard everything (weights too)  ───────   43 GB   (64× = N GPUs)
# ```
#
# **ZeRO-1:** Each rank owns 1/N of the optimizer states (e.g. Adam `m`, `v` moments).
# After a normal AllReduce of the full gradient, each rank updates **only its shard** of params,
# then AllGathers the updated shards so all ranks have the complete updated model.
#
# **ZeRO-2:** Additionally shard the gradients. After Reduce-Scatter, each rank receives
# only the gradient shard it needs — no rank ever holds the full `[P]` gradient.
#
# **ZeRO-3:** Shard the weights too. No rank holds a full copy of W at rest.
# Before each layer's forward pass, ranks AllGather that layer's weights, compute,
# then **discard** them immediately. Memory is `O(P/N)` but AllGather runs at every
# layer in both forward and backward — significantly more communication than ZeRO-1/2.
#
# **Trade-off:** extra AllGather communication, but memory savings dominate at scale.
#
# ### ZeRO-1 data flow in Noam notation
#
# ```
# After standard AllReduce:     dW_P     shape [P]   — every rank has the full gradient
#
# Rank r updates its shard:
#   g_Ps     = dW_P[shard_idx_Ps]        slice this rank's portion of dW
#   m_Ps, v_Ps updated locally           only 1/N of Adam state lives here
#   W_P.at[shard_idx_Ps] -= adam_step    only this rank's slice of W changes
#
# AllGather — reassemble full params:
#   [shard_0_Ps | shard_1_Ps | ... | shard_N_Ps]  →  W_P   shape [P]
#   jnp.concatenate(all updated shards) gives every rank the complete updated W
# ```
#

# %% [markdown]
# ---
# ### Exercise 4: ZeRO-1 — Local Adam Update + AllGather
#
# Implement the local Adam shard update and the AllGather reconstruction.
# Together these two functions complete one ZeRO-1 optimizer step.
#

# %%
def zero1_adam_step_shard(
    params_P:     Float[Array, "P"],
    grads_P:      Float[Array, "P"],
    m_Ps:         Float[Array, "Ps"],
    v_Ps:         Float[Array, "Ps"],
    shard_idx_Ps: Integer[Array, "Ps"],
    lr:           float,
    beta1:        float = 0.9,
    beta2:        float = 0.999,
    eps:          float = 1e-8,
    t:            int   = 1,
) -> Tuple[Float[Array, "P"], Float[Array, "Ps"], Float[Array, "Ps"]]:
    """
    Apply one Adam step to the parameter shard this rank owns.
    Only positions shard_idx_Ps of params_P are updated; the rest are unchanged.

    Args:
        params_P:     full parameter vector,          shape [P]
        grads_P:      full gradient vector,           shape [P]  (from AllReduce)
        m_Ps:         first moment for this shard,    shape [Ps]
        v_Ps:         second moment for this shard,   shape [Ps]
        shard_idx_Ps: global indices owned by rank,   shape [Ps]
        lr:           learning rate
        t:            current step (for bias correction)

    Returns:
        (updated_params_P [P], new_m_Ps [Ps], new_v_Ps [Ps])
    """
    raise NotImplementedError(
        "TODO:\n"
        "  g_Ps     = grads_P[shard_idx_Ps]\n"
        "  new_m_Ps = beta1 * m_Ps + (1 - beta1) * g_Ps\n"
        "  new_v_Ps = beta2 * v_Ps + (1 - beta2) * g_Ps ** 2\n"
        "  m_hat_Ps = new_m_Ps / (1 - beta1 ** t)\n"
        "  v_hat_Ps = new_v_Ps / (1 - beta2 ** t)\n"
        "  params_P = params_P.at[shard_idx_Ps].add(\n"
        "      -lr * m_hat_Ps / (jnp.sqrt(v_hat_Ps) + eps)\n"
        "  )\n"
        "  return params_P, new_m_Ps, new_v_Ps"
    )


def zero1_allgather(
    rank_shards:   List[Float[Array, "Ps"]],
    shard_indices: List[Integer[Array, "Ps"]],
    n_params:      int,
) -> Float[Array, "P"]:
    """
    Reconstruct full params from per-rank updated shards (simulates AllGather).

    After each rank calls zero1_adam_step_shard, it holds updated values only for
    its own shard. This function assembles all shards back into the full [P] vector,
    giving every rank the complete updated model.

    Args:
        rank_shards:   list of n arrays — updated param values per rank,  each [Ps]
        shard_indices: list of n arrays — global param indices per rank,  each [Ps]
        n_params:      total number of parameters P

    Returns:
        params_P of shape [P]
    """
    raise NotImplementedError(
        "TODO:\n"
        "  params_P = jnp.zeros(n_params)\n"
        "  for shard_vals_Ps, idx_Ps in zip(rank_shards, shard_indices):\n"
        "      params_P = params_P.at[idx_Ps].set(shard_vals_Ps)\n"
        "  return params_P"
    )


# ── Test ─────────────────────────────────────────────────────────────────────
P, N = 12, 4
shards_idx = [jnp.array(s) for s in np.array_split(np.arange(P), N)]

params_P = jnp.ones(P)
grads_P  = jnp.full(P, 0.1)
lr_zero  = 0.01

# Each rank updates only its own shard
updated_shards, shard_vals = [], []
for rank in range(N):
    idx_Ps = shards_idx[rank]
    m_Ps   = jnp.zeros(len(idx_Ps))
    v_Ps   = jnp.zeros(len(idx_Ps))
    p_updated_P, _, _ = zero1_adam_step_shard(
        params_P, grads_P, m_Ps, v_Ps, idx_Ps, lr=lr_zero, t=1
    )
    shard_vals.append(p_updated_P[idx_Ps])    # extract only this rank's updated values

# AllGather: reconstruct full params from all rank shards
final_params_P = zero1_allgather(shard_vals, shards_idx, P)

# Reference: plain Adam on all params at once
g_P      = grads_P
m_ref_P  = (1 - 0.9)   * g_P
v_ref_P  = (1 - 0.999) * g_P ** 2
m_hat_P  = m_ref_P / (1 - 0.9)
v_hat_P  = v_ref_P / (1 - 0.999)
ref_P    = jnp.ones(P) - lr_zero * m_hat_P / (jnp.sqrt(v_hat_P) + 1e-8)

judge.check("Ex4a: shard 0 updated",         bool(jnp.any(final_params_P[shards_idx[0]] != 1.0)), True)
judge.check("Ex4b: shard 1 updated",         bool(jnp.any(final_params_P[shards_idx[1]] != 1.0)), True)
judge.check("Ex4c: allgather shape correct", final_params_P.shape,                                 (P,))
judge.check("Ex4d: matches reference Adam",  final_params_P,                                        ref_P)

# %% [markdown]
# <details>
# <summary>💡 Hint — zero1_adam_step_shard</summary>
#
# ```python
# g_Ps     = grads_P[shard_idx_Ps]
# new_m_Ps = beta1 * m_Ps + (1 - beta1) * g_Ps
# new_v_Ps = beta2 * v_Ps + (1 - beta2) * g_Ps ** 2
# m_hat_Ps = new_m_Ps / (1 - beta1 ** t)
# v_hat_Ps = new_v_Ps / (1 - beta2 ** t)
# params_P = params_P.at[shard_idx_Ps].add(-lr * m_hat_Ps / (jnp.sqrt(v_hat_Ps) + eps))
# return params_P, new_m_Ps, new_v_Ps
# ```
# </details>
#
# <details>
# <summary>💡 Hint — zero1_allgather</summary>
#
# ```python
# params_P = jnp.zeros(n_params)
# for shard_vals_Ps, idx_Ps in zip(rank_shards, shard_indices):
#     params_P = params_P.at[idx_Ps].set(shard_vals_Ps)
# return params_P
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
# 1. **Gradient = partial sum.** `dW_DxF = einsum("bsd,bsf->df", X, δ)` sums over the
#    batch, giving shape `[D, F]` — same as W. Each entry `dW[d,f]` updates exactly `W[d,f]`.
#    Without AllReduce, each GPU applies a different `dW` and model copies diverge.
#
# 2. **AllReduce averages element-wise along the W axis:** `[W, D, F] → [D, F]`.
#    D and F are unchanged. Ring-AllReduce is bandwidth-optimal (`≈2P` per worker,
#    independent of N). JAX exposes this as `jax.lax.pmean` inside `jax.pmap` —
#    XLA emits the ring algorithm automatically.
#
# 3. **JAX is SPMD-only — no parameter server.** `pmap` maps computation across devices;
#    `lax.pmean/psum/all_gather/reduce_scatter` are the collective primitives.
#
# 4. **Gradient accumulation** sums `jax.value_and_grad` outputs over micro-batches
#    before a single `optimizer.update` — mathematically equivalent to training on the
#    full effective batch.
#
# 5. **ZeRO-1** shards optimizer states: each rank updates 1/N of the params locally,
#    then AllGather reconstructs the full model. ZeRO-2 additionally shards gradients;
#    ZeRO-3 shards weights too, AllGathering on demand at every layer.
#
# ---
# **Next:** [Chapter 3 — Model Parallelism](./chapter_03_model_parallelism.ipynb)
#
