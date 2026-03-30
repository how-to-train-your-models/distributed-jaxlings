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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/chapters/chapter_07_jax_sharding.ipynb)
#
# # Chapter 7: JAX Sharding & Device Meshes
#
# > **Course: Distributed Training — From Concepts to JAX**
#
# ---
#
# ## Learning Objectives
#
# - Create device meshes with named axes
# - Annotate arrays with `PartitionSpec` to describe how they are sharded
# - Use `jax.sharding.NamedSharding` and `jax.device_put` to place tensors on devices
# - Inspect sharding with `jax.debug.visualize_array_sharding`
# - Express data parallelism, tensor parallelism, and replication using sharding specs
#

# %%
import os
# Simulate 8 devices on CPU for Colab (must be set before importing JAX)
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

print(f"JAX version: {jax.__version__}")
print(f"Devices available: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
# ---
# ## 1. Device Meshes
#
# A **Mesh** is a logical N-dimensional arrangement of devices with named axes. The names let you refer to axes symbolically in `PartitionSpec`.
#
# ```python
# # 8 devices → 2D mesh: 2 'data' rows × 4 'model' columns
# devices = mesh_utils.create_device_mesh((2, 4))
# mesh = Mesh(devices, axis_names=('data', 'model'))
# ```
#
# ```
#           model axis (4 devices)
#         ┌────┬────┬────┬────┐
# data  0 │ d0 │ d1 │ d2 │ d3 │
# axis  1 │ d4 │ d5 │ d6 │ d7 │
#         └────┴────┴────┴────┘
# ```
#
# Common mesh configurations:
#
# | Training strategy | Mesh shape | Axis names |
# |---|---|---|
# | Pure data parallel | (N,) | ('batch',) |
# | Pure tensor parallel | (N,) | ('model',) |
# | 2D (data + tensor) | (D, T) | ('data', 'model') |
# | 3D (data + tensor + pipeline) | (D, T, P) | ('data', 'model', 'pipe') |
#

# %% [markdown]
# ---
# ## 2. `PartitionSpec` — Describing Sharding
#
# `PartitionSpec` maps each array dimension to a mesh axis (or `None` for replicated).
#
# ```python
# # Mesh axes: ('data', 'model')
#
# # Data parallel: batch dimension sharded over 'data', features replicated
# P('data', None)       # shape [B, D] → each data-row gets B/2 rows, full D cols
#
# # Column-parallel weights: out-features sharded over 'model'
# P(None, 'model')      # shape [in, out] → each model-col gets full in, out/4 cols
#
# # Row-parallel weights: in-features sharded over 'model'
# P('model', None)      # shape [in, out] → each model-col gets in/4 rows, full out
#
# # Fully replicated (e.g. biases)
# P()                   # all devices have the full tensor
#
# # 2D sharded (both batch AND features split)
# P('data', 'model')    # shape [B, D] → each device gets B/2 rows, D/4 cols
# ```
#
# ### Applying sharding
# ```python
# sharding = NamedSharding(mesh, P('data', None))
# x_sharded = jax.device_put(x, sharding)
# ```
#
# JAX's JIT compiler respects sharding annotations — it will insert the necessary collectives automatically.
#

# %% [markdown]
# ---
# ## 3. `jit` with Sharding Constraints
#
# With the new JAX sharding API, `jit` is the primary parallelism primitive (replacing `pmap`). You annotate inputs and the compiler handles distribution:
#
# ```python
# with mesh:
#     @jit
#     def forward(params, x):
#         # x is already sharded across devices
#         # JAX compiler inserts collectives as needed
#         return x @ params['W'] + params['b']
#
#     result = forward(params, x_sharded)
#     # result.sharding tells you how the output is distributed
# ```
#
# You can also use `jax.lax.with_sharding_constraint` inside a function to **annotate intermediate tensors**:
# ```python
# from jax.lax import with_sharding_constraint as wsc
#
# @jit
# def f(x):
#     x = wsc(x, NamedSharding(mesh, P('data', None)))  # force this sharding
#     return x @ W
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
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

class Judge:
    def __init__(self):
        self.passed = 0; self.failed = 0

    def check(self, name, got, expected, tol=1e-4):
        got_np = np.array(got) if hasattr(got, 'shape') else got
        if isinstance(expected, bool):
            ok = bool(got) == expected
        elif isinstance(expected, tuple):
            ok = got == expected
        elif isinstance(expected, np.ndarray) or hasattr(expected, 'shape'):
            ok = np.allclose(got_np, np.array(expected), atol=tol)
        else:
            ok = abs(float(got_np.flat[0]) - float(expected)) / (abs(float(expected)) + 1e-9) < tol
        if ok:
            self.passed += 1; print(f"✅ {name}: PASSED")
        else:
            self.failed += 1; print(f"❌ {name}: FAILED — got {got!r}, expected {expected!r}")
        return ok

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*40}\n  Results: {self.passed}/{total} passed")
        print("  🎉 Chapter 7 complete!" if self.failed==0 else f"  {self.failed} remaining.")
        print('='*40)

judge = Judge()
print("Judge ready!")

# %% [markdown]
# ---
# ## Exercise 1: Create Device Meshes
#
# Create different mesh configurations for common parallelism strategies.
#
# TODO: Implement `make_data_parallel_mesh`, `make_tensor_parallel_mesh`, and `make_2d_mesh`
#

# %%
from jax.sharding import Mesh
from jax.experimental import mesh_utils


def make_data_parallel_mesh(n_devices: int) -> Mesh:
    """
    Create a 1D mesh for pure data parallelism.
    Axis name: 'batch'
    """
    # TODO: create a 1D device array and wrap in Mesh
    # devices = mesh_utils.create_device_mesh((n_devices,))
    # return Mesh(devices, axis_names=('batch',))
    pass


def make_tensor_parallel_mesh(n_devices: int) -> Mesh:
    """
    Create a 1D mesh for pure tensor parallelism.
    Axis name: 'model'
    """
    # TODO
    pass


def make_2d_mesh(n_data: int, n_model: int) -> Mesh:
    """
    Create a 2D mesh with 'data' and 'model' axes.
    Total devices = n_data * n_model
    """
    # TODO
    pass


# Tests
dp_mesh = make_data_parallel_mesh(8)
tp_mesh = make_tensor_parallel_mesh(8)
mesh_2d = make_2d_mesh(2, 4)   # 2 data × 4 model

print(f"DP mesh shape:   {dp_mesh.shape}  axes: {dp_mesh.axis_names}")
print(f"TP mesh shape:   {tp_mesh.shape}  axes: {tp_mesh.axis_names}")
print(f"2D mesh shape:   {mesh_2d.shape}  axes: {mesh_2d.axis_names}")

judge.check("Ex1a: DP mesh shape", dp_mesh.shape, (8,))
judge.check("Ex1b: DP mesh axis name", dp_mesh.axis_names, ('batch',))
judge.check("Ex1c: TP mesh axis name", tp_mesh.axis_names, ('model',))
judge.check("Ex1d: 2D mesh shape", mesh_2d.shape, (2, 4))
judge.check("Ex1e: 2D mesh axis names", mesh_2d.axis_names, ('data', 'model'))

# %% [markdown]
# ---
# ## Exercise 2: Shard Arrays with PartitionSpec
#
# Use `NamedSharding` and `jax.device_put` to place arrays on devices according to different parallelism strategies.
#
# TODO: Implement `shard_batch`, `shard_weights_column_parallel`, `replicate`
#

# %%
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = make_2d_mesh(2, 4)  # 2 data × 4 model


def shard_batch(x: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
    """
    Shard input activations for data parallelism:
    - batch dimension → 'data' axis
    - feature dimension → replicated (None)
    Shape: [B, D] where B is sharded across 'data'
    """
    # TODO: create NamedSharding with P('data', None) and device_put
    pass


def shard_weights_column_parallel(W: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
    """
    Shard weight matrix for column-parallel (tensor) parallelism:
    - input dim → replicated
    - output dim → 'model' axis
    Shape: [in, out] where out is sharded across 'model'
    """
    # TODO: P(None, 'model')
    pass


def replicate(x: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
    """
    Replicate an array across ALL devices (e.g., for biases or scalars).
    """
    # TODO: NamedSharding with P() (no sharding = fully replicated)
    pass


# Test
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (16, 64))   # batch=16, features=64
W = jax.random.normal(key, (64, 128))  # in=64, out=128
b = jnp.zeros(128)

X_sharded = shard_batch(X, mesh)
W_sharded = shard_weights_column_parallel(W, mesh)
b_rep     = replicate(b, mesh)

print(f"X sharding: {X_sharded.sharding}")
print(f"W sharding: {W_sharded.sharding}")
print(f"b sharding: {b_rep.sharding}")

# Verify shapes haven't changed (sharding is logical)
judge.check("Ex2a: X shape preserved", X_sharded.shape, (16, 64))
judge.check("Ex2b: W shape preserved", W_sharded.shape, (64, 128))

# Verify sharding specs
judge.check("Ex2c: X batch dim sharded over 'data'",
            X_sharded.sharding.spec[0], 'data')
judge.check("Ex2d: X feature dim replicated",
            X_sharded.sharding.spec[1], None)
judge.check("Ex2e: W out-dim sharded over 'model'",
            W_sharded.sharding.spec[1], 'model')
judge.check("Ex2f: bias fully replicated",
            b_rep.sharding.spec, P())

# %% [markdown]
# ---
# ## Exercise 3: Sharded Matrix Multiply
#
# With sharded arrays in place, JAX's JIT automatically inserts the required collectives when you compute `X @ W + b`. Implement the forward pass and verify the output sharding.
#
# TODO: Implement `sharded_linear_forward` inside a mesh context
#

# %%
from jax import jit


def sharded_linear_forward(X_sharded, W_sharded, b_rep):
    """
    Compute Y = X @ W + b where arrays are already sharded.
    JAX will handle the communication automatically.
    
    X_sharded: [B, in]   — batch sharded over 'data'
    W_sharded: [in, out] — output sharded over 'model'
    b_rep:     [out]     — replicated
    
    Expected output sharding: [B, out] — both 'data' and 'model' sharded
    """
    # TODO: Y = X_sharded @ W_sharded + b_rep
    # JAX handles required collectives; no manual comm needed!
    pass


# JIT-compile within the mesh context
with mesh:
    fwd_jit = jit(sharded_linear_forward)
    Y = fwd_jit(X_sharded, W_sharded, b_rep)

print(f"Output shape:    {Y.shape}")
print(f"Output sharding: {Y.sharding.spec}")

# Reference (unsharded)
Y_ref = np.array(X) @ np.array(W) + np.array(b)

judge.check("Ex3a: output shape correct", Y.shape, (16, 128))
judge.check("Ex3b: output values correct", np.array(Y), Y_ref, tol=1e-3)

# %% [markdown]
# ---
# ## Exercise 4: PartitionSpec for Different Parallelism Strategies
#
# Given a 2D mesh `('data', 'model')`, write the correct `PartitionSpec` for each scenario.
#
# TODO: Fill in the correct `PartitionSpec` for each tensor/strategy
#

# %%
# Mesh: ('data'=2, 'model'=4)  — 8 devices total
# For each tensor shape and strategy, provide the correct PartitionSpec.

# Tensor: activations [batch=16, seq=32, d_model=64]
# Strategy: batch over 'data', seq replicated, features replicated
spec_activations = None  # TODO: P('data', None, None)

# Tensor: attention weight [n_heads=8, d_model=64, head_dim=8]
# Strategy: heads sharded over 'model', rest replicated
spec_attn_weight = None  # TODO: P('model', None, None)

# Tensor: optimizer state (Adam m) [n_params=1000]
# Strategy: ZeRO-1 style — params partitioned along 'data'
spec_opt_state = None  # TODO: P('data',)

# Tensor: LM head weight [vocab=50000, d_model=64]
# Strategy: vocab sharded over 'model', d_model replicated
spec_lm_head = None  # TODO: P('model', None)

# Tensor: scalar loss value
# Strategy: fully replicated (every device sees the same scalar)
spec_loss = None  # TODO: P()


# Validate
judge.check("Ex4a: activations spec", spec_activations, P('data', None, None))
judge.check("Ex4b: attn weight spec", spec_attn_weight, P('model', None, None))
judge.check("Ex4c: opt state spec",   spec_opt_state,   P('data',))
judge.check("Ex4d: lm head spec",     spec_lm_head,     P('model', None))
judge.check("Ex4e: loss spec",         spec_loss,         P())

# %% [markdown]
# ---
# ## Exercise 5: Sharding Constraints Inside JIT
#
# Use `jax.lax.with_sharding_constraint` to guide the compiler about intermediate tensor placement. This is useful when the compiler's default choice is suboptimal.
#
# TODO: Implement `attention_with_sharding_hints`
#

# %%
from jax.lax import with_sharding_constraint as wsc


def attention_with_sharding_hints(
    Q: jnp.ndarray,   # [batch, n_heads, seq, head_dim]
    K: jnp.ndarray,   # [batch, n_heads, seq, head_dim]
    V: jnp.ndarray,   # [batch, n_heads, seq, head_dim]
    mesh: Mesh
) -> jnp.ndarray:
    """
    Compute scaled dot-product attention with sharding hints:
    - After computing attention scores, annotate as P('data', 'model', None, None)
      so heads are tensor-parallel and batch is data-parallel
    - Output annotated as P('data', 'model', None, None)
    
    Formula: softmax(Q @ K^T / sqrt(head_dim)) @ V
    """
    head_dim = Q.shape[-1]
    batch_head_spec = NamedSharding(mesh, P('data', 'model', None, None))

    # TODO: Compute attention scores: [batch, n_heads, seq, seq]
    # scores = Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(head_dim)
    scores = None  # TODO

    # TODO: Apply sharding constraint to scores
    # scores = wsc(scores, NamedSharding(mesh, P('data', 'model', None, None)))
    scores = None  # TODO

    # TODO: Softmax over last dim (key positions)
    attn_weights = None  # TODO: jax.nn.softmax(scores, axis=-1)

    # TODO: Weighted sum of values: [batch, n_heads, seq, head_dim]
    output = None  # TODO: attn_weights @ V

    # TODO: Apply sharding constraint to output
    output = None  # TODO: wsc(output, batch_head_spec)

    return output


# Test
key = jax.random.PRNGKey(5)
B, H, S, D = 4, 8, 16, 8   # batch, heads, seq, head_dim
Q = jax.random.normal(key, (B, H, S, D))
K = jax.random.normal(key, (B, H, S, D))
V = jax.random.normal(key, (B, H, S, D))

with mesh:
    out = jit(attention_with_sharding_hints)(Q, K, V, mesh)

# Reference (plain numpy)
import math
sc = np.array(Q) @ np.array(K).transpose(0, 1, 3, 2) / math.sqrt(D)
sc -= sc.max(axis=-1, keepdims=True)
w = np.exp(sc) / np.exp(sc).sum(axis=-1, keepdims=True)
ref = w @ np.array(V)

judge.check("Ex5a: attention output shape", out.shape, (B, H, S, D))
judge.check("Ex5b: attention output values", np.array(out), ref, tol=1e-3)

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
# 1. **`Mesh`** gives devices a logical N-dimensional layout with named axes — this is the foundation of all JAX parallelism.
# 2. **`PartitionSpec`** maps array dimensions to mesh axes: `P('data', None)` means batch-sharded, feature-replicated.
# 3. **`NamedSharding` + `device_put`** places tensors on devices as specified — JAX JIT inserts collectives automatically.
# 4. **`with_sharding_constraint`** lets you annotate intermediate tensors to guide the XLA compiler.
# 5. All communication patterns from chapters 3-4 (column-parallel, row-parallel, AllReduce, AllGather) map directly to combinations of `PartitionSpec` annotations.
#
# ---
# **Next:** [Chapter 8 — SPMD Programming in JAX](./chapter_08_spmd_jax.ipynb)
#
