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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/notebooks/chapter_02_data_parallelism.ipynb)
#
# # Chapter 2: Data Parallelism
#
# > **Course: Distributed Training in JAX**
#
# ---
#
# ## Chapter Summary
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
import sys
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding



from judge import Judge

judge = Judge("Chapter 2", test_module="tests.test_chapter_02")
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
def f(x): return jnp.sin(x) ** 2

print("jit:  ", jax.jit(f)(0.5))
print("grad: ", jax.grad(f)(0.5))          # d/dx sin²x = 2 sin x cos x = sin 2x
print("vmap: ", jax.vmap(f)(jnp.arange(5.0)))


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
tree = {"a": jnp.ones(3), "b": {"c": jnp.zeros(2)}}
doubled = jax.tree.map(lambda x: x * 2, tree)
print("doubled:", doubled)

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
print("random:", jax.random.normal(subkey, (3,)))


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
# ## 5. The Tiny GPT
#
# Naive transformer block:
# - Token embedding `(V, E)` + sinusoidal positional encoding `(S, E)`
# - N × `Block`:
#     - LayerNorm → `eqx.nn.MultiheadAttention` → residual
#     - LayerNorm → MLP (`E → 4E → E`, GeLU) → residual
# - Final LayerNorm → LM head `(E, V)` (untied for simplicity)
#
#

# %% [markdown]
# ### Exercise 1 — The transformer block
#
# Build a `Block` Equinox module containing pre-LayerNorm attention + pre-LayerNorm MLP,
# both with residual connections. Use `eqx.nn.MultiheadAttention` for the attention.
#

# %%
class Block(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    mlp_in: eqx.nn.Linear
    mlp_out: eqx.nn.Linear
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, embed_dim: int, num_heads: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=embed_dim, inference=True, key=k1,
        )
        self.mlp_in = eqx.nn.Linear(embed_dim, 4 * embed_dim, key=k2)
        self.mlp_out = eqx.nn.Linear(4 * embed_dim, embed_dim, key=k3)
        self.ln1 = eqx.nn.LayerNorm(embed_dim)
        self.ln2 = eqx.nn.LayerNorm(embed_dim)

    def __call__(self, x_SxE: jax.Array, *, key=None) -> jax.Array:
        S = x_SxE.shape[0]
        mask_SxS = jnp.tril(jnp.ones((S, S)))          # causal mask

        # Pre-LN attention + residual
        h_SxE = jax.vmap(self.ln1)(x_SxE)
        h_SxE = self.attn(h_SxE, h_SxE, h_SxE, mask=mask_SxS)
        x_SxE = x_SxE + h_SxE

        # Pre-LN MLP + residual
        h_SxE = jax.vmap(self.ln2)(x_SxE)
        h_SxE = jax.nn.gelu(jax.vmap(self.mlp_in)(h_SxE))
        h_SxE = jax.vmap(self.mlp_out)(h_SxE)
        x_SxE = x_SxE + h_SxE
        return x_SxE


# %%
judge.check(Block)


# %% [markdown]
# ### Exercise 2 — The tiny GPT
#
# Compose embedding + sinusoidal positions + N blocks + final LN + LM head into a
# `TinyGPT` Equinox module.
#

# %%
class TinyGPT(eqx.Module):
    tok_emb: eqx.nn.Embedding
    pos_emb: jax.Array                     # non-trainable sinusoidal table
    blocks: tuple
    ln_f: eqx.nn.LayerNorm
    lm_head: eqx.nn.Linear

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, max_seq: int, *, key):
        keys = jax.random.split(key, num_layers + 2)
        self.tok_emb = eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0])
        self.pos_emb = sinusoidal_positions(max_seq, embed_dim)
        self.blocks = tuple(
            Block(embed_dim, num_heads, key=keys[i + 1])
            for i in range(num_layers)
        )
        self.ln_f = eqx.nn.LayerNorm(embed_dim)
        self.lm_head = eqx.nn.Linear(embed_dim, vocab_size, key=keys[-1])

    def __call__(self, tokens_S: jax.Array, *, key=None) -> jax.Array:
        """Single-example forward: tokens_S is int[S]. Returns logits_SxV."""
        S = tokens_S.shape[0]
        x_SxE = jax.vmap(self.tok_emb)(tokens_S) + self.pos_emb[:S]
        for block in self.blocks:
            x_SxE = block(x_SxE, key=key)
        x_SxE = jax.vmap(self.ln_f)(x_SxE)
        logits_SxV = jax.vmap(self.lm_head)(x_SxE)
        return logits_SxV


def sinusoidal_positions(max_seq: int, dim: int) -> jax.Array:
    """Standard 'Attention is All You Need' positional encoding table of shape (max_seq, dim)."""
    pos = jnp.arange(max_seq)[:, None]                       # (S, 1)
    i = jnp.arange(0, dim, 2)[None, :]                       # (1, dim//2)
    angle_rates = 1 / (10000 ** (i / dim))                    # (1, dim//2)
    angles = pos * angle_rates                                # (S, dim//2)
    # Interleave sin and cos: even indices = sin, odd indices = cos.
    pe = jnp.zeros((max_seq, dim))
    pe = pe.at[:, 0::2].set(jnp.sin(angles))
    pe = pe.at[:, 1::2].set(jnp.cos(angles))
    return pe


# %%
judge.check(TinyGPT)


# %% [markdown]
# ---
# ## 6. Meshes and Sharding
#
# JAX's modern parallelism story is built on three primitives:
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
devices = np.array(jax.devices())
mesh = Mesh(devices, axis_names=('data',))

replicated = NamedSharding(mesh, P())
sharded_batch = NamedSharding(mesh, P('data'))

print(f"Mesh: {mesh}")
print(f"Replicated sharding: {replicated}")
print(f"Sharded-batch sharding: {sharded_batch}")


# %% [markdown]
# ### Exercise 3 — Shard a batch and replicate params
#
# Given the `mesh` above and a `TinyGPT` instance, place the model with `replicated`
# sharding and a batch tensor with `sharded_batch`. Confirm with
# `jax.debug.visualize_array_sharding`.
#

# %%
# --- Hyperparams for the tiny GPT ---
VOCAB_SIZE = 256       # byte-level for simplicity
EMBED_DIM  = 128
NUM_HEADS  = 4
NUM_LAYERS = 2
MAX_SEQ    = 64
BATCH_SIZE = 8         # global batch (must be divisible by number of devices)

def shard_model_and_batch(model, tokens_BxS, mesh):
    """Replicate model arrays across devices; shard tokens along mesh axis 'data'."""
    replicated = NamedSharding(mesh, P())
    batch_sh = NamedSharding(mesh, P('data', None))
    model = jax.tree.map(
        lambda x: jax.device_put(x, replicated) if eqx.is_array(x) else x,
        model,
    )
    tokens_BxS = jax.device_put(tokens_BxS, batch_sh)
    return model, tokens_BxS


# Build model and put it through the sharding helper.
model = TinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ,
                key=jax.random.PRNGKey(0))
# Set inference mode so Dropout.inference is handled as static by eqx.filter_jit.
# (No dropout in this simple model — Ch 10 uses dropout with proper key handling.)
model = eqx.nn.inference_mode(model)

dummy_tokens_BxS = jnp.ones((BATCH_SIZE, MAX_SEQ), dtype=jnp.int32)
model, dummy_tokens_BxS = shard_model_and_batch(model, dummy_tokens_BxS, mesh)

print("Model param count:", sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))))
print("\nBatch sharding:")
jax.debug.visualize_array_sharding(dummy_tokens_BxS)


# %%
judge.check(shard_model_and_batch)


# %% [markdown]
# ---
# ## 7. The Data-Parallel Pattern
#
# The DP recipe in JAX:
#
# 1. Build the model. Place it under `NamedSharding(mesh, P())` (replicated).
# 2. Place each batch under `NamedSharding(mesh, P('data', ...))` (batch dim sharded).
# 3. Wrap the loss + step in a function to jitted. We are using: `eqx.filter_jit` for this.
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
    logits_BxSxV = jax.vmap(model)(batch['tokens_BxS'])
    # Per-token cross-entropy, then mean over all (B, S) positions.
    loss_BxS = optax.softmax_cross_entropy_with_integer_labels(
        logits_BxSxV, batch['targets_BxS']
    )
    return jnp.mean(loss_BxS)


@functools.partial(jax.jit, static_argnames=('static', 'optimizer'))
def train_step(params, static, opt_state, batch, optimizer):
    """DP train step. Params and static are the two halves of eqx.partition."""
    model = eqx.combine(params, static)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    grad_params, _ = eqx.partition(grads, eqx.is_array)
    updates, opt_state = optimizer.update(grad_params, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# %%
judge.check(loss_fn)
judge.check(train_step)


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
# Older JAX code  used `jax.pmap` and`jax.lax.pmean` for data parallelism. 
# Don't write new code with `pmap` — it doesn't compose with `Mesh`/`PartitionSpec`,
# doesn't generalize to TP/PP/FSDP, and is effectively in maintenance mode.
# Everything here is `jit` + sharding from the start.
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
import time

# --- Optimizer ---
optimizer = optax.adamw(3e-4)
params, static = eqx.partition(model, eqx.is_array)
opt_state = optimizer.init(params)

# --- Synthetic data: random token sequences (byte-level) ---
# In production you'd use TinyStories; random tokens suffice to demonstrate DP.
data_key = jax.random.PRNGKey(42)
NUM_STEPS = 50

batch_sharding = NamedSharding(mesh, P('data', None))

print(f"Training for {NUM_STEPS} steps  |  batch={BATCH_SIZE}×{MAX_SEQ}  |  devices={len(jax.devices())}")
print("-" * 60)

for step in range(NUM_STEPS):
    data_key, subkey = jax.random.split(data_key)
    # Random token sequences as synthetic training data
    tokens_BxS1 = jax.random.randint(subkey, (BATCH_SIZE, MAX_SEQ + 1), 0, VOCAB_SIZE)
    batch = {
        'tokens_BxS':  jax.device_put(tokens_BxS1[:, :-1], batch_sharding),
        'targets_BxS': jax.device_put(tokens_BxS1[:, 1:],  batch_sharding),
    }
    params, opt_state, loss = train_step(params, static, opt_state, batch, optimizer)
    if step % 10 == 0 or step == NUM_STEPS - 1:
        print(f"  step {step:3d}  loss={float(loss):.4f}")

print("-" * 60)

# Reconstruct the trained model for generation.
model = eqx.combine(params, static)

# --- Greedy generation to verify the model works ---
def generate(model, prompt_tokens, max_new: int = 50):
    """Greedy autoregressive generation."""
    tokens = list(prompt_tokens)
    for _ in range(max_new):
        x = jnp.array(tokens[-MAX_SEQ:])[None, :]          # (1, <=S)
        logits_1xSxV = jax.vmap(model)(x)
        next_token = int(jnp.argmax(logits_1xSxV[0, -1]))
        tokens.append(next_token)
    return tokens

sample = generate(model, [1, 2, 3], max_new=50)
print("Generated tokens (first 30):", sample[:30])


# %% [markdown]
# ### Exercise 5 — Throughput benchmark
#
# Measure tokens/sec across `1, 2, 4, 8` (simulated) devices. Plot or print the scaling
# curve and explain where it falls off.
#

# %%
# Throughput benchmark across device counts.
# We use sub-meshes of the available devices to simulate different scales.
all_devices = jax.devices()
device_counts = [d for d in [1, 2, 4] if d <= len(all_devices)]

NUM_BENCH_STEPS = 5
WARMUP_STEPS = 2

print(f"{'Devices':>8} {'Tokens/step':>12} {'Time/step (s)':>14} {'Tokens/sec':>12}")
print("-" * 50)

for n_dev in device_counts:
    sub_mesh = Mesh(np.array(all_devices[:n_dev]), axis_names=('data',))
    sub_replicated = NamedSharding(sub_mesh, P())
    sub_batch_sh = NamedSharding(sub_mesh, P('data', None))

    # Fresh model for fair comparison
    bench_model = TinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ,
                          key=jax.random.PRNGKey(0))
    bench_model = eqx.nn.inference_mode(bench_model)
    bench_model = jax.tree.map(
        lambda x: jax.device_put(x, sub_replicated) if eqx.is_array(x) else x,
        bench_model
    )
    bench_opt = optax.adamw(3e-4)
    bench_params, bench_static = eqx.partition(bench_model, eqx.is_array)
    bench_opt_state = bench_opt.init(bench_params)

    bench_bs = max(n_dev, 4)  # at least 1 sample per device
    bench_key = jax.random.PRNGKey(99)

    for i in range(WARMUP_STEPS + NUM_BENCH_STEPS):
        bench_key, sk = jax.random.split(bench_key)
        toks = jax.random.randint(sk, (bench_bs, MAX_SEQ + 1), 0, VOCAB_SIZE)
        b = {
            'tokens_BxS':  jax.device_put(toks[:, :-1], sub_batch_sh),
            'targets_BxS': jax.device_put(toks[:, 1:],  sub_batch_sh),
        }
        if i == WARMUP_STEPS:
            jax.block_until_ready(bench_params)
            t0 = time.perf_counter()
        bench_params, bench_opt_state, _ = train_step(
            bench_params, bench_static, bench_opt_state, b, bench_opt)

    jax.block_until_ready(bench_params)
    elapsed = time.perf_counter() - t0
    tokens_per_step = bench_bs * MAX_SEQ
    tok_per_sec = tokens_per_step * NUM_BENCH_STEPS / elapsed
    print(f"{n_dev:>8} {tokens_per_step:>12,} {elapsed/NUM_BENCH_STEPS:>14.4f} {tok_per_sec:>12,.0f}")


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
