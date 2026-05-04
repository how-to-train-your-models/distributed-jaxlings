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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/notebooks/chapter_01_jax_equinox_intro.ipynb)
#
# # Chapter 1: JAX, Equinox & TinyGPT
#
# > **Course: Distributed Training in JAX** — built around JAX + Equinox + Optax + Orbax.
#
# ---
#
# ## Chapter Summary
#
# Before we distribute anything, we need to be comfortable with the tools we'll use throughout
# the course. This chapter is a hands-on tour of:
#
# - **JAX** — functional array programming with JIT, autodiff, and vectorisation
# - **Equinox** — neural networks as JAX pytrees
# - **TinyGPT** — the small transformer model we'll use as our running example
#
# ## Learning Objectives
#
# By the end of this chapter you will be able to:
# - Write JAX functions and apply `jit`, `grad`, and `vmap` to them
# - Explain the JAX PRNG model and use `jax.random.split` correctly
# - Build and inspect `eqx.Module` models as pytrees
# - Use `eqx.filter_jit` and `eqx.filter_value_and_grad`
# - Describe the TinyGPT architecture and locate it in `src/common/models.py`
#

# %% [markdown]
# ## Setup
#

# %%
import os
# Simulate 4 CPU devices so multi-device examples run on any machine (including Colab).
# Must be set before JAX is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import sys
import pathlib

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from judge import Judge

judge = Judge("Chapter 1")
print(f"JAX devices: {jax.devices()}")


# %% [markdown]
# ---
# ## 1. JAX in Five Minutes — `jit`, `grad`, `vmap`
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
def f(x): return jnp.sin(x) ** 2

print("jit:  ", jax.jit(f)(0.5))
print("grad: ", jax.grad(f)(0.5))          # d/dx sin²x = 2 sin x cos x = sin 2x
print("vmap: ", jax.vmap(f)(jnp.arange(5.0)))


# %% [markdown]
# ---
# ## 2. Pytrees and PRNG
#
# - **Pytrees** are nested containers of arrays (tuples, lists, dicts, dataclasses). JAX
#   transforms operate over pytrees transparently.
# - **PRNG**: JAX has no global random state. You pass a `PRNGKey` and split it
#   explicitly with `jax.random.split` — this makes randomness deterministic and
#   parallelism-safe.
#

# %%
tree = {"a": jnp.ones(3), "b": {"c": jnp.zeros(2)}}
doubled = jax.tree.map(lambda x: x * 2, tree)
print("doubled:", doubled)

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
print("random:", jax.random.normal(subkey, (3,)))


# %% [markdown]
# ---
# ## 3. Equinox: Modules as Pytrees
#
# Equinox is a tiny library: an `eqx.Module` is a frozen dataclass that is *also* a
# pytree. Fields can be `jax.Array`s, ints, strings, or sub-modules. Because modules
# **are** pytrees, JAX transforms work on them out of the box — and so does our sharding
# machinery later in the course.
#
# There are two helpers in equinox, which are worth knowing as we'll use them everywhere:
#
# - `eqx.filter_jit(fn)` — like `jax.jit`, but only traces array leaves and treats
#   non-array leaves (ints, strings, booleans) as static. This avoids confusing tracing
#   errors when a module has fields which are not jax.arrays.
# - `eqx.filter_value_and_grad(fn)` — like `jax.value_and_grad`, but only differentiates
#   the array leaves of the model. Non-array leaves are ignored for gradient computation.
#
# Optional: `eqx.partition` / `eqx.combine` to split a module into "differentiable" and
# "static" halves explicitly.
#

# %%
class Linear(eqx.Module):
    w: jax.Array
    b: jax.Array
    def __init__(self, in_dim, out_dim, key):
        k1, k2 = jax.random.split(key)
        self.w = jax.random.normal(k1, (in_dim, out_dim)) / jnp.sqrt(in_dim)
        self.b = jnp.zeros((out_dim,))
    def __call__(self, x_BxI):
        return x_BxI @ self.w + self.b

lin = Linear(4, 8, jax.random.PRNGKey(42))
print("Linear output shape:", lin(jnp.ones((2, 4))).shape)
print("Leaves:", jax.tree.leaves(lin))


# %% [markdown]
# ---
# ## 4. The TinyGPT Model
#
# We've implemented a tiny GPT model for our experiments throughout the course.
# `Block`, `TinyGPT`, and `sinusoidal_positions` are imported from `src.common.models`
# and reused across all chapters. The architecture is deliberately naive:
#
# - Token embedding `(V, E)` + sinusoidal positional encoding `(S, E)`
# - N × `Block`:
#     - LayerNorm → `eqx.nn.MultiheadAttention` → residual
#     - LayerNorm → MLP (`E → 4E → E`, GeLU) → residual
# - Final LayerNorm → LM head `(E, V)` (untied for simplicity)
#
# Have a look at `src/common/models.py` to familiarize yourself with the implementation.
#

# %%
from src.common.models import Block, TinyGPT, sinusoidal_positions, generate

# --- Hyperparams for the tiny GPT ---
VOCAB_SIZE = 256       # byte-level for simplicity
EMBED_DIM  = 128
NUM_HEADS  = 4
NUM_LAYERS = 2
MAX_SEQ    = 64

model = TinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ,
                key=jax.random.key(42))
eqx.tree_pprint(model)


# %% [markdown]
# ### Calling the model
#
# `TinyGPT` takes a single sequence of token ids `(S,)` and returns logits `(S, V)`.
# Use `jax.vmap` to process a batch:
#

# %%
dummy_tokens_S = jnp.ones((MAX_SEQ,), dtype=jnp.int32)
logits_SxV = model(dummy_tokens_S)
print("Single sequence logits shape:", logits_SxV.shape)

# Batched via vmap
dummy_batch_BxS = jnp.ones((4, MAX_SEQ), dtype=jnp.int32)
logits_BxSxV = jax.vmap(model)(dummy_batch_BxS)
print("Batched logits shape:", logits_BxSxV.shape)


# %% [markdown]
# ### Generating text
#
# `generate` (from `src.common.models`) runs greedy autoregressive decoding. It accepts a
# plain string prompt (encoded to UTF-8 bytes) and returns the full sequence as decoded text.
#

# %%
sample = generate(model, "Once upon a time", max_new=20)
print("Generated:", sample)


# %% [markdown]
# ### Inspecting parameters
#
# Because `TinyGPT` is an `eqx.Module` (a pytree), you can walk its leaves with standard
# JAX utilities:
#

# %%
param_arrays = jax.tree.leaves(eqx.filter(model, eqx.is_array))
total_params = sum(x.size for x in param_arrays)
print(f"TinyGPT parameter count: {total_params:,}")


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
# 1. **JAX = pure functions + XLA.** `jit`, `grad`, and `vmap` compose freely because
#    JAX functions have no side effects.
# 2. **PRNG is explicit.** Always split your key before use — never share a key between
#    two calls.
# 3. **Equinox modules are pytrees.** This means JAX transforms work on them directly,
#    and our sharding machinery in later chapters will too.
# 4. **TinyGPT is our running example.** It lives in `src/common/models.py` and is
#    imported by every subsequent chapter.
#
# ---
# **Next:** [Chapter 2 — Data Parallelism](./chapter_02_data_parallelism.ipynb) — replicate
# the model across devices, shard the batch, and let the JAX compiler insert the AllReduce.
#

# %%
