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
# > **Course: Distributed Training in JAX** — JAX + Equinox + Optax + Orbax.
#
# ---
#
# ## What we build
#
# A **tiny GPT** (~1–5M params) trained on **TinyStories** across N devices using
# data parallelism. Architecture is deliberately *naive*: token + sinusoidal positional
# embeddings, vanilla multi-head attention, LayerNorm, GeLU MLP. Ch 10 revisits with a
# proper mini-Llama (RoPE, GQA, RMSNorm, SwiGLU, sharded init), so we keep this one
# minimal on purpose.
#
# **Real-world hook:** every modern LLM (Claude, GPT, Gemini) is this same shape, scaled
# up and trained DP. TinyStories (Eldan & Li, 2023) showed that even 1–10M-param
# transformers produce coherent prose — so we get a "wow" sample at the end without
# needing real scale.
#
# ## Learning Objectives
#
# By the end of this chapter you will be able to:
# - Write JAX functions and use `jit`, `grad`, `vmap` on them
# - Build models as `eqx.Module` and use `eqx.filter_jit` / `eqx.filter_grad`
# - Construct a `Mesh` and annotate tensors with `PartitionSpec` / `NamedSharding`
# - Implement the data-parallel training loop (replicated params, sharded batch)
# - Reason about the sync-vs-async tradeoff and why we don't use `pmap` anymore
#
# ---

# %% [markdown]
# ## Setup
#

# %%
import os
# Simulate 4 CPU devices so multi-device examples run on any machine (including Colab).
# Must be set before JAX is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import functools
from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from judge import Judge

judge = Judge("Chapter 2")
print(f"JAX devices: {jax.devices()}")


# %% [markdown]
# ---
# ## 1. The Data Parallel Idea
#
# Data parallelism is the simplest distributed training pattern.
#
# 1. Each device holds a **full copy** of the model.
# 2. The global batch is **split** across devices — each sees a different micro-batch.
# 3. Each device runs a forward + backward pass independently.
# 4. Gradients are **averaged** across devices (an AllReduce).
# 5. Each device applies the same update — model copies stay in sync.
#
# ```
# Global batch = [b0, b1, b2, b3]
#
#   GPU 0: model | b0 → dW_0 ─┐
#   GPU 1: model | b1 → dW_1 ─┤→ AllReduce → dW_avg → update all
#   GPU 2: model | b2 → dW_2 ─┤
#   GPU 3: model | b3 → dW_3 ─┘
# ```
#
# In modern JAX you do not write the AllReduce explicitly — the compiler inserts it for
# you when you tell it the params are replicated and the batch is sharded. Most of this
# chapter is about telling the compiler exactly that.
#

# %% [markdown]
# ---
# ## 2. JAX in Five Minutes — `jit`, `grad`, `vmap`
#
# JAX programs are **pure functions over arrays**. Three transforms do most of the work:
#
# - `jax.jit` — compile + optimize a function via XLA. First call traces; subsequent calls reuse the compiled artifact.
# - `jax.grad` — automatic differentiation. Returns a function computing gradients w.r.t. the first arg.
# - `jax.vmap` — vectorize over a batch dim without writing a loop.
#
# Examples:
#

# %%
# TODO: a 5-line demo of jit / grad / vmap on a tiny function.
#
# def f(x): return jnp.sin(x) ** 2
# print(jax.jit(f)(0.5))
# print(jax.grad(f)(0.5))
# print(jax.vmap(f)(jnp.arange(5.0)))


# %% [markdown]
# ---
# ## 3. Pytrees and PRNG
#
# - **Pytrees** are nested containers of arrays (tuples, lists, dicts, dataclasses). JAX
#   transforms operate over pytrees transparently.
# - **PRNG**: JAX has no global random state. You pass a `PRNGKey` and split it
#   explicitly with `jax.random.split` — this makes randomness deterministic and
#   parallelism-safe.
#

# %%
# TODO: brief demo:
# - jax.tree.map over a nested dict
# - key splitting: key, subkey = jax.random.split(key)
# - jax.random.normal(subkey, shape)


# %% [markdown]
# ---
# ## 4. Equinox: Modules as Pytrees
#
# Equinox is a tiny library: an `eqx.Module` is a frozen dataclass that is *also* a
# pytree. Fields can be `jax.Array`s, ints, strings, or sub-modules. Because modules
# **are** pytrees, JAX transforms work on them out of the box — and so does our sharding
# machinery later in the chapter.
#
# Two helpers we'll use everywhere:
#
# - `eqx.filter_jit(fn)` — like `jax.jit`, but only traces array leaves and treats
#   non-array leaves (ints, strings, booleans) as static. Avoids confusing tracing
#   errors when a module field isn't a `jax.Array`.
# - `eqx.filter_value_and_grad(fn)` — like `jax.value_and_grad`, but only differentiates
#   the array leaves of the model.
#
# Optional: `eqx.partition` / `eqx.combine` to split a module into "differentiable" and
# "static" halves explicitly. Useful for advanced patterns.
#

# %%
# TODO: tiny eqx.Module demo — a one-layer Linear:
#
# class Linear(eqx.Module):
#     w: jax.Array
#     b: jax.Array
#     def __init__(self, in_dim, out_dim, key):
#         k1, k2 = jax.random.split(key)
#         self.w = jax.random.normal(k1, (in_dim, out_dim)) / jnp.sqrt(in_dim)
#         self.b = jnp.zeros((out_dim,))
#     def __call__(self, x_BxI):
#         return x_BxI @ self.w + self.b


# %% [markdown]
# ---
# ## 5. The Tiny GPT
#
# Naive transformer block:
# - Token embedding `(V, E)` + sinusoidal positional encoding `(S, E)`
# - N × `Block`:
#     - LayerNorm → `eqx.nn.MultiheadAttention` → residual
#     - LayerNorm → MLP (`E → 4E → E`, GeLU) → residual
# - Final LayerNorm → LM head `(E, V)` (untied for simplicity)
#
# Tensor naming follows the project convention (Noam suffix, `x` separator):
# `inputs_BxS`, `embed_BxSxE`, `logits_BxSxV`, etc.
#

# %% [markdown]
# ### Exercise 1 — The transformer block
#
# Build a `Block` Equinox module containing pre-LayerNorm attention + pre-LayerNorm MLP,
# both with residual connections. Use `eqx.nn.MultiheadAttention` for the attention.
#

# %%
class Block(eqx.Module):
    # TODO: attn (eqx.nn.MultiheadAttention), mlp_in / mlp_out (eqx.nn.Linear),
    # ln1 / ln2 (eqx.nn.LayerNorm), dropout key handling.
    pass

    def __call__(self, x_SxE: jax.Array, *, key=None) -> jax.Array:
        # TODO: pre-LN attention + residual, pre-LN MLP + residual.
        ...


# %% [markdown]
# ### Exercise 2 — The tiny GPT
#
# Compose embedding + sinusoidal positions + N blocks + final LN + LM head into a
# `TinyGPT` Equinox module.
#

# %%
class TinyGPT(eqx.Module):
    # TODO: tok_emb (eqx.nn.Embedding), pos_emb (precomputed sinusoidal table),
    # blocks (list of Block), ln_f (eqx.nn.LayerNorm), lm_head (eqx.nn.Linear).
    pass

    def __call__(self, tokens_BxS: jax.Array, *, key=None) -> jax.Array:
        # TODO: embed → add positions → blocks → ln_f → lm_head → logits_BxSxV
        ...


def sinusoidal_positions(max_seq: int, dim: int) -> jax.Array:
    """Standard 'Attention is All You Need' positional encoding table of shape (max_seq, dim)."""
    # TODO
    ...


# %% [markdown]
# ---
# ## 6. Meshes and Sharding
#
# JAX's modern parallelism story is built on three things:
#
# - `Mesh` — a logical grid of devices with **named axes** (e.g. `data`, `model`).
# - `PartitionSpec` (alias `P`) — for each tensor dim, which mesh axis (if any) shards it.
#   `P('data')` shards dim 0 along the `data` axis. `P()` replicates everything.
# - `NamedSharding(mesh, spec)` — binds a `PartitionSpec` to a concrete `Mesh`.
#
# Once a tensor is placed with `jax.device_put(x, NamedSharding(mesh, spec))`, the XLA
# compiler tracks its sharding through `jit` and inserts the right collectives.
#

# %%
# TODO: build a 1D 'data' mesh over all devices.
#
# devices = np.array(jax.devices())
# mesh = Mesh(devices, axis_names=('data',))
#
# replicated = NamedSharding(mesh, P())
# sharded_batch = NamedSharding(mesh, P('data'))


# %% [markdown]
# ### Exercise 3 — Shard a batch and replicate params
#
# Given the `mesh` above and a `TinyGPT` instance, place the model with `replicated`
# sharding and a batch tensor with `sharded_batch`. Confirm with
# `jax.debug.visualize_array_sharding`.
#

# %%
# TODO


# %% [markdown]
# ---
# ## 7. The Data-Parallel Pattern
#
# The DP recipe in modern JAX:
#
# 1. Build the model. Place it under `NamedSharding(mesh, P())` (replicated).
# 2. Place each batch under `NamedSharding(mesh, P('data', ...))` (batch dim sharded).
# 3. Wrap the loss + step in `eqx.filter_jit`.
# 4. The compiler sees: replicated params, sharded batch → it inserts an AllReduce on
#    the gradients automatically. **You do not write `lax.pmean` yourself.**
#

# %% [markdown]
# ### Exercise 4 — The train step
#
# Implement `loss_fn(model, batch)` (next-token cross-entropy) and `train_step(model,
# opt_state, batch)` that returns updated `(model, opt_state, loss)`.
#

# %%
def loss_fn(model: TinyGPT, batch: dict) -> jax.Array:
    """Next-token cross-entropy. batch = {'tokens_BxS': int[B,S], 'targets_BxS': int[B,S]}."""
    # TODO: logits = jax.vmap(model)(batch['tokens_BxS'])
    #       cross-entropy against batch['targets_BxS'], mean over (B, S).
    ...


@eqx.filter_jit
def train_step(model, opt_state, batch, optimizer):
    # TODO: loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    #       updates, opt_state = optimizer.update(grads, opt_state, model)
    #       model = eqx.apply_updates(model, updates)
    #       return model, opt_state, loss
    ...


# %% [markdown]
# ---
# ## 8. Sync vs Async, Large-Batch Tradeoffs
#
# - **Synchronous SGD** (what we just built): every device waits for the AllReduce
#   before the next step. Deterministic, easy to reason about, dominant in practice.
# - **Asynchronous SGD**: workers update a parameter server as gradients arrive. Higher
#   throughput, but stale-gradient issues hurt convergence. Used historically (Hogwild!,
#   parameter servers); rare for modern LLMs.
# - **Large-batch tradeoff**: scaling DP to N devices grows the *global* batch size
#   N-fold. Linear LR scaling + warmup ([Goyal et al., 2017](https://arxiv.org/abs/1706.02677))
#   keeps convergence intact up to a point; beyond that, generalization degrades.
#

# %% [markdown]
# ---
# ## Sidebar: `pmap` is legacy
#
# Older JAX code (and the previous version of this chapter) used `jax.pmap` and
# `jax.lax.pmean` for data parallelism. Don't write new code with `pmap` — it doesn't
# compose with `Mesh`/`PartitionSpec`, doesn't generalize to TP/PP/FSDP, and is
# effectively in maintenance mode. Everything here is `jit` + sharding from the start.
#

# %% [markdown]
# ---
# ## Build: TinyGPT-on-TinyStories DP Trainer
#
# Now wire it all together. Pseudocode:
#
# ```python
# 1. Load / synthesize TinyStories tokens.  # use a small subset, hf_datasets or shipped sample
# 2. Build mesh = Mesh(jax.devices(), ('data',))
# 3. model = TinyGPT(...) ; place under NamedSharding(mesh, P())
# 4. optimizer = optax.adamw(3e-4) ; opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
# 5. for step in range(N):
#        batch = next_batch()                            # tokens_BxS, targets_BxS
#        batch = jax.device_put(batch, NamedSharding(mesh, P('data', None)))
#        model, opt_state, loss = train_step(model, opt_state, batch, optimizer)
# 6. Generate a 200-token sample from a prompt to confirm it learned something.
# ```
#

# %%
# TODO: full training loop.


# %% [markdown]
# ### Exercise 5 — Throughput benchmark
#
# Measure tokens/sec across `1, 2, 4, 8` (simulated) devices. Plot or print the scaling
# curve and explain where it falls off.
#

# %%
# TODO


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
# 1. Data parallelism replicates the model and shards the batch. Gradient AllReduce is
#    inserted by the JAX compiler when you correctly annotate sharding — you do not
#    write the collective.
# 2. Equinox modules are pytrees, so `Mesh` + `PartitionSpec` + `NamedSharding` works on
#    them with no special framework support.
# 3. `eqx.filter_jit` and `eqx.filter_value_and_grad` are the day-to-day workhorses.
# 4. `pmap` is legacy. Use `jit` + sharding for everything new.
# 5. DP scales until activation memory or per-step communication dominates — the next
#    several chapters are about pushing past those limits.
#
# ---
# **Next:** [Chapter 3 — Collectives & `shard_map`](./chapter_03_collectives_shard_map.ipynb) —
# learn the AllReduce / AllGather / ReduceScatter / ppermute primitives that the compiler
# was inserting for you, and write them yourself with `shard_map`.
#

# %%
