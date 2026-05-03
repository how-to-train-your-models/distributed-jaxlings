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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/chapters/chapter_09_flax_optax_distributed.ipynb)
#
# # Chapter 9: Distributed Training with Flax & Optax
#
# > **Course: Distributed Training — From Concepts to JAX**
#
# ---
#
# ## Learning Objectives
#
# - Define a transformer model in Flax NNX
# - Set up a distributed optimizer with Optax
# - Write a complete sharded training loop using `jit` + `PartitionSpec`
# - Apply ZeRO-style optimizer state sharding via Optax
# - Run evaluation and compute perplexity in a distributed setting
#

# %%
# Install dependencies
# # !pip install -q flax optax

import os
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from functools import partial
from jax import jit, value_and_grad

print(f"JAX: {jax.__version__}, Devices: {jax.device_count()}")

# %% [markdown]
# ---
# ## 1. Flax for Distributed Training
#
# Flax is a neural network library for JAX. Its key design choices fit naturally with distributed training:
# - **Functional style:** parameters are plain Python dicts/pytrees — easy to shard
# - **`model.init(key, x)`:** returns a parameter pytree (no global state)
# - **`model.apply(params, x)`:** pure function, JIT-compilable
#
# ```python
# class MLP(nn.Module):
#     features: int
#
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(self.features)(x)
#         x = nn.gelu(x)
#         x = nn.Dense(self.features)(x)
#         return x
#
# model = MLP(features=256)
# params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 64)))['params']
# # params is a nested dict of arrays — shareable!
# ```
#
# ### Sharding Flax parameters
# Flax parameters are pytrees. You can shard them using `jax.tree.map`:
# ```python
# def shard_params(params, mesh, param_spec):
#     return jax.tree.map(
#         lambda p: jax.device_put(p, NamedSharding(mesh, param_spec(p))),
#         params
#     )
# ```
#

# %% [markdown]
# ---
# ## 2. Optax for Distributed Optimization
#
# Optax provides composable gradient transformations. Key properties for distributed training:
# - Optimizer state is a pytree — can be sharded just like parameters
# - Stateless updates: `new_params, new_state = optimizer.update(grads, state, params)`
#
# ```python
# optimizer = optax.chain(
#     optax.clip_by_global_norm(1.0),    # gradient clipping
#     optax.adamw(learning_rate=3e-4, weight_decay=0.1)
# )
#
# opt_state = optimizer.init(params)    # initialize optimizer state
# updates, new_state = optimizer.update(grads, opt_state, params)
# new_params = optax.apply_updates(params, updates)
# ```
#
# ### ZeRO-1 with Optax
# Shard optimizer states across data-parallel ranks:
# ```python
# # Each rank only stores optimizer state for its param shard
# sharded_opt_state = jax.tree.map(
#     lambda s: jax.device_put(s, NamedSharding(mesh, P('data', None))),
#     opt_state
# )
# ```
#

# %% [markdown]
# ---
# ## Judge Setup
#

# %%
import jax
import jax.numpy as jnp
import numpy as np
from judge import Judge

judge = Judge("Chapter 9", default_tol=1e-3)
print("Judge ready!")

# %% [markdown]
# ---
# ## Exercise 1: Define a Flax Transformer Block
#
# Implement a single Transformer block (self-attention + MLP + LayerNorm) in Flax.
#
# TODO: Implement `TransformerBlock`
#

# %%
import flax.linen as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        B, S, D = x.shape
        head_dim = self.d_model // self.n_heads

        # Q, K, V projections (fused)
        qkv = nn.Dense(3 * self.d_model, use_bias=False)(x)  # [B, S, 3D]
        q, k, v = jnp.split(qkv, 3, axis=-1)                 # each [B, S, D]

        # Reshape to [B, n_heads, S, head_dim]
        q = q.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, D)

        return nn.Dense(self.d_model, use_bias=False)(out)


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x, mask=None):
        """
        Pre-norm Transformer block: LN → Attn → residual → LN → MLP → residual
        
        Args:
            x:    [batch, seq_len, d_model]
            mask: optional causal mask
        Returns:
            [batch, seq_len, d_model]
        """
        # TODO: Implement pre-norm transformer block
        # Pattern:
        #   h = x + Attention(LayerNorm(x))
        #   out = h + MLP(LayerNorm(h))
        # MLP: Dense(d_ff) → GELU → Dense(d_model)
        
        # Attention sublayer
        h = None  # TODO: x + MultiHeadAttention(self.d_model, self.n_heads)(nn.LayerNorm()(x), mask)

        # MLP sublayer
        # TODO: h2 = nn.Dense(self.d_ff)(nn.LayerNorm()(h))
        #        h2 = nn.gelu(h2)
        #        out = h + nn.Dense(self.d_model)(h2)
        out = None  # TODO

        return out


# Test forward pass
key = jax.random.PRNGKey(0)
block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)

x_test = jnp.ones((2, 8, 64))  # [batch=2, seq=8, d=64]
params = block.init(key, x_test)
out = block.apply(params, x_test)

print(f"Input shape:  {x_test.shape}")
print(f"Output shape: {out.shape}")
judge.check("Ex1a: TransformerBlock output shape", out.shape, (2, 8, 64))
judge.check("Ex1b: Output is finite", bool(jnp.all(jnp.isfinite(out))), True)


# %% [markdown]
# ---
# ## Exercise 2: Initialize and Shard Parameters
#
# Initialize a small transformer and shard its parameters across devices for data parallelism (replicate weights, shard batch).
#
# TODO: Implement `init_and_shard_params`
#

# %%
class MiniGPT(nn.Module):
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    d_ff: int

    @nn.compact
    def __call__(self, token_ids):
        B, S = token_ids.shape
        x = nn.Embed(self.vocab_size, self.d_model)(token_ids)
        pos = jnp.arange(S)[None, :]
        x = x + nn.Embed(self.max_seq_len, self.d_model)(pos)

        # Causal mask
        mask = jnp.tril(jnp.ones((S, S), dtype=bool))[None, None, :, :]

        for _ in range(self.n_layers):
            x = TransformerBlock(self.d_model, self.n_heads, self.d_ff)(x, mask)

        x = nn.LayerNorm()(x)
        return nn.Dense(self.vocab_size, use_bias=False)(x)  # [B, S, vocab]


def init_and_shard_params(model, key, dummy_input, mesh):
    """
    Initialize model parameters and replicate them across all devices.
    For data parallelism, weights are replicated (P()) on all devices.
    
    Args:
        model:       Flax nn.Module
        key:         PRNG key
        dummy_input: example input array for shape inference
        mesh:        device mesh
    
    Returns:
        params pytree where every leaf is replicated on all devices
    """
    # Initialize params on single device
    params = model.init(key, dummy_input)['params']
    
    # TODO: Replicate all params across devices using jax.tree.map + jax.device_put
    # Each param leaf gets NamedSharding(mesh, P())  — fully replicated
    replicated_sharding = NamedSharding(mesh, P())
    sharded_params = None  # TODO: jax.tree.map(lambda p: jax.device_put(p, replicated_sharding), params)
    
    return sharded_params


# Set up mesh and model
n_dev = jax.device_count()
devices = mesh_utils.create_device_mesh((n_dev,))
dp_mesh = Mesh(devices, ('batch',))

model = MiniGPT(vocab_size=256, d_model=64, n_heads=4,
                n_layers=2, max_seq_len=32, d_ff=256)

dummy = jnp.zeros((1, 8), dtype=jnp.int32)
key = jax.random.PRNGKey(0)

params = init_and_shard_params(model, key, dummy, dp_mesh)

# Count params
n_params = sum(p.size for p in jax.tree.leaves(params))
print(f"Total parameters: {n_params:,}")

# Verify sharding: all params should be replicated
first_param = jax.tree.leaves(params)[0]
print(f"First param sharding: {first_param.sharding.spec}")

judge.check("Ex2a: params exist and are finite",
            bool(all(jnp.all(jnp.isfinite(p)) for p in jax.tree.leaves(params))), True)
judge.check("Ex2b: params fully replicated (P())",
            first_param.sharding.spec, P())

# %% [markdown]
# ---
# ## Exercise 3: Distributed Training Step
#
# Implement a complete distributed training step:
# 1. Shard the batch across devices (`P('batch', None)` for token ids)
# 2. Compute forward pass and cross-entropy loss
# 3. Compute gradients
# 4. Apply Optax optimizer update
#
# TODO: Implement `create_train_state` and `train_step`
#

# %%
from flax.training import train_state
import optax


def cross_entropy_loss(logits, labels):
    """
    Cross-entropy loss for language modeling.
    logits: [batch, seq, vocab]
    labels: [batch, seq]  (int token ids)
    Returns: scalar loss
    """
    vocab_size = logits.shape[-1]
    # Shift: predict token t+1 from token t
    logits = logits[:, :-1, :]   # [B, S-1, V]
    labels = labels[:, 1:]        # [B, S-1]
    
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -log_probs[jnp.arange(logits.shape[0])[:, None],
                      jnp.arange(logits.shape[1])[None, :],
                      labels]
    return jnp.mean(loss)


def create_train_state(params, learning_rate=1e-3):
    """
    Create Optax train state with AdamW optimizer.
    
    Returns:
        train_state.TrainState with params and optimizer state
    """
    # TODO: create optax.adamw optimizer and initialize train state
    # tx = optax.adamw(learning_rate)
    # return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    pass


@partial(jit, static_argnums=(0,))
def train_step(model, state, token_ids):
    """
    Single distributed training step.
    
    Args:
        model:     Flax nn.Module (static — not traced)
        state:     TrainState (params + optimizer state)
        token_ids: [batch, seq_len] int32, sharded over 'batch' axis
    
    Returns:
        (new_state, loss)
    """
    def loss_fn(params):
        logits = model.apply({'params': params}, token_ids)
        return cross_entropy_loss(logits, token_ids), logits
    
    # TODO: compute loss and gradients
    # (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    loss = None   # TODO
    grads = None  # TODO
    
    # TODO: apply gradients via state.apply_gradients
    # new_state = state.apply_gradients(grads=grads)
    new_state = None  # TODO
    
    return new_state, loss


# Initialize training
state = create_train_state(params, learning_rate=1e-3)

# Create sharded batch: [n_dev, seq_per_device]
seq_len = 16
key = jax.random.PRNGKey(1)
token_ids = jax.random.randint(key, (n_dev * 4, seq_len), 0, 256)
token_ids_sharded = jax.device_put(
    token_ids, NamedSharding(dp_mesh, P('batch', None)))

# Run training step
with dp_mesh:
    new_state, loss = train_step(model, state, token_ids_sharded)

print(f"Loss: {float(loss):.4f}")
print(f"Expected loss ≈ log(256) = {np.log(256):.3f} (random init)")

judge.check("Ex3a: loss is finite", bool(jnp.isfinite(loss)), True)
judge.check("Ex3b: loss is positive", bool(loss > 0), True)
judge.check("Ex3c: loss reasonable (< 10)", bool(loss < 10.0), True)


# %% [markdown]
# ---
# ## Exercise 4: Full Training Loop with Loss Curve
#
# Train the mini-GPT for a few steps on random data and verify the loss decreases.
#
# TODO: Implement `training_loop`
#

# %%
def training_loop(
    model,
    state,
    mesh,
    n_steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 42
):
    """
    Run n_steps of distributed training and return loss history.
    
    Returns:
        (final_state, losses: list of float)
    """
    losses = []
    key = jax.random.PRNGKey(seed)
    n_dev = jax.device_count()

    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        
        # TODO: generate a random batch of token ids [batch_size, seq_len]
        tokens = None  # TODO: jax.random.randint(subkey, (batch_size, seq_len), 0, vocab_size)
        
        # TODO: shard the batch across the 'batch' axis
        tokens_sharded = None  # TODO: jax.device_put(tokens, NamedSharding(mesh, P('batch', None)))
        
        # TODO: run one training step
        with mesh:
            state, loss = None, None  # TODO: train_step(model, state, tokens_sharded)
        
        losses.append(float(loss))
        if step % 10 == 0:
            print(f"Step {step:3d}: loss = {float(loss):.4f}")
    
    return state, losses


# Re-initialize with fresh state
params_fresh = init_and_shard_params(model, jax.random.PRNGKey(0), dummy, dp_mesh)
state_fresh  = create_train_state(params_fresh, learning_rate=3e-3)

final_state, losses = training_loop(
    model, state_fresh, dp_mesh,
    n_steps=30,
    batch_size=n_dev * 4,
    seq_len=16,
    vocab_size=256
)

print(f"\nInitial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")

judge.check("Ex4a: loss decreased over training", losses[-1] < losses[0], True)
judge.check("Ex4b: final loss is finite", bool(np.isfinite(losses[-1])), True)


# %% [markdown]
# ---
# ## Exercise 5: Compute Perplexity
#
# Perplexity is the standard metric for language models:
# $$\text{PPL} = e^{\text{loss}} = e^{-\frac{1}{N}\sum \log P(x_i)}$$
#
# A random model over vocab_size tokens has PPL = vocab_size.
#
# TODO: Implement `evaluate_perplexity`
#

# %%
@partial(jit, static_argnums=(0,))
def eval_step(model, params, token_ids):
    """Compute loss without gradient."""
    logits = model.apply({'params': params}, token_ids)
    return cross_entropy_loss(logits, token_ids)


def evaluate_perplexity(
    model,
    params,
    mesh,
    n_eval_batches: int = 5,
    batch_size: int = 8,
    seq_len: int = 16,
    vocab_size: int = 256,
    seed: int = 99
) -> float:
    """
    Evaluate model perplexity on random data.
    
    Returns:
        perplexity (float)
    """
    key = jax.random.PRNGKey(seed)
    total_loss = 0.0
    n_dev = jax.device_count()

    for _ in range(n_eval_batches):
        key, subkey = jax.random.split(key)
        tokens = jax.random.randint(subkey, (batch_size, seq_len), 0, vocab_size)
        tokens_sharded = jax.device_put(tokens, NamedSharding(mesh, P('batch', None)))
        
        with mesh:
            loss = eval_step(model, params, tokens_sharded)
        total_loss += float(loss)
    
    avg_loss = total_loss / n_eval_batches
    
    # TODO: compute perplexity from average loss
    perplexity = None  # TODO: float(jnp.exp(avg_loss))
    
    return perplexity


# Evaluate fresh (random) model — should be ≈ vocab_size = 256
ppl_random = evaluate_perplexity(model, params_fresh, dp_mesh, vocab_size=256)
print(f"Random model perplexity: {ppl_random:.1f}  (expected ≈ 256)")

# Evaluate trained model — should be lower
ppl_trained = evaluate_perplexity(model, final_state.params, dp_mesh, vocab_size=256)
print(f"Trained model perplexity: {ppl_trained:.1f}  (should be < random)")

judge.check("Ex5a: random PPL ≈ vocab_size", ppl_random, 256.0, tol=0.5)
judge.check("Ex5b: trained PPL < random PPL", ppl_trained < ppl_random, True)

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
# 1. **Flax** produces parameter pytrees — plain dicts of arrays that integrate seamlessly with JAX sharding.
# 2. **Optax** optimizer states are also pytrees and can be sharded just like parameters (ZeRO-style).
# 3. **Data parallelism** with JAX sharding: replicate weights (`P()`), shard batch (`P('batch', None)`), JAX handles gradient averaging automatically.
# 4. **`train_state.TrainState`** bundles params + optimizer state into one pytree for clean training loops.
# 5. Perplexity = exp(loss) is the standard LM metric; a random model has PPL ≈ vocab_size.
#
# ---
# **Next:** [Chapter 10 — Multi-host JAX Training in Practice](./chapter_10_multihost_training.ipynb)
#
