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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/chapters/chapter_08_spmd_jax.ipynb)
#
# # Chapter 8: SPMD Programming in JAX
# ### `shard_map`, Custom Collectives, and GSPMD
#
# > **Course: Distributed Training — From Concepts to JAX**
#
# ---
#
# ## Learning Objectives
#
# - Understand SPMD (Single Program Multiple Data) vs MPMD
# - Use `shard_map` for explicit per-device computation
# - Write custom collective operations inside `shard_map`
# - Understand when `shard_map` is preferred over annotated `jit`
# - Understand GSPMD and how JAX's compiler implements it
#

# %%
import os
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils, shard_map
from jax.experimental.shard_map import shard_map as smap
from functools import partial
from jax import jit, lax

print(f"Devices: {jax.device_count()}")

# %% [markdown]
# ---
# ## 1. SPMD vs MPMD
#
# **SPMD (Single Program Multiple Data):** Every device runs the **same program** on **different data**. Communication happens through collective ops.
#
# **MPMD (Multiple Program Multiple Data):** Different devices can run entirely different programs. More flexible, much harder to program.
#
# JAX uses **SPMD exclusively**. The XLA compiler takes a single Python program and compiles it to run efficiently across N devices.
#
# ### Two SPMD styles in JAX
#
# | Style | API | When to use |
# |---|---|---|
# | **Implicit** | `jit` + `PartitionSpec` | Most cases; compiler chooses collectives |
# | **Explicit** | `shard_map` | Custom collectives, flash attention, ring attention |
#
# Implicit is easier; explicit gives more control.
#

# %% [markdown]
# ---
# ## 2. `shard_map` — Explicit Per-Device Computation
#
# `shard_map` is JAX's API for writing **explicit per-device code**. Each device runs the body function on its local shard.
#
# ```python
# from jax.experimental.shard_map import shard_map
#
# mesh = Mesh(devices, ('devices',))
#
# # in_specs:  how inputs are sharded going IN
# # out_specs: how outputs are sharded coming OUT
# @partial(shard_map, mesh=mesh,
#          in_specs=(P('devices'),),
#          out_specs=P('devices'))
# def f(x_local):
#     # x_local is the LOCAL shard on this device
#     # e.g., if x had 8 elements and 4 devices, x_local has 2 elements
#     return x_local * 2
# ```
#
# ### Collectives inside `shard_map`
# Inside the body, each device can communicate with others using `lax` collectives:
#
# ```python
# lax.psum(x, axis_name='devices')     # AllReduce sum
# lax.pmean(x, axis_name='devices')    # AllReduce mean
# lax.all_gather(x, axis_name='devices') # AllGather
# lax.psum_scatter(x, axis_name='devices') # ReduceScatter
# lax.ppermute(x, axis_name, perm)     # point-to-point (for ring algos)
# ```
#

# %% [markdown]
# ---
# ## 3. GSPMD — General SPMD
#
# GSPMD (Xu et al., 2021) is the **compiler algorithm** inside XLA that implements JAX's implicit sharding. Given:
# - Input sharding annotations (your `PartitionSpec`s)
# - The computation graph (from tracing your `jit`-ted function)
#
# GSPMD automatically:
# 1. Propagates sharding through every op (forward and backward)
# 2. Inserts the minimal set of collectives (AllReduce, AllGather, ReduceScatter)
# 3. Chooses layouts to maximize data locality
#
# **Why GSPMD matters:** You write a single-device program; the compiler generates the multi-device version. This is the key advantage over `pmap`, which required manual shard management.
#
# ```
# Your code:  Y = X @ W       (single-device semantics)
# Input spec: X ~ P('data', None), W ~ P(None, 'model')
#
# GSPMD:      Each device computes Y_local = X_local @ W_local
#             Output spec: Y ~ P('data', 'model')
#             → No collectives needed! Both dims already sharded.
#
# Different:  X ~ P('data', None), W ~ P('model', None)
#             → Row-parallel: each device needs ALL of W's rows for its slice
#             → GSPMD inserts AllGather on W before matmul
#             → OR uses psum after partial matmul (ReduceScatter)
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

judge = Judge("Chapter 8", default_tol=1e-4)
print("Judge ready!")

# %% [markdown]
# ---
# ## Exercise 1: `shard_map` Basics — Local Computation
#
# Use `shard_map` to apply a function to each local shard independently.
#
# TODO: Implement `apply_gelu_sharded` using `shard_map`
#

# %%
from jax.experimental.shard_map import shard_map as smap
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from functools import partial

devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh1d = Mesh(devices, ('devices',))


def apply_gelu_sharded(x: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
    """
    Apply GELU activation using shard_map.
    Each device applies GELU to its local shard independently.
    
    x shape: [N] where N is divisible by n_devices
    Output:  same shape as x, GELU applied element-wise
    """
    @partial(smap,
             mesh=mesh,
             in_specs=(P('devices'),),    # x is split along 'devices'
             out_specs=P('devices'))      # output same split
    def _gelu_local(x_local):
        # TODO: apply GELU to the local shard
        # jax.nn.gelu(x_local)
        pass
    
    return _gelu_local(x)


# Test
x = jnp.linspace(-3, 3, jax.device_count() * 4)
y_sharded = apply_gelu_sharded(x, mesh1d)
y_ref     = jax.nn.gelu(x)

print(f"Input:    {np.array(x).round(2)}")
print(f"GELU out: {np.array(y_sharded).round(4)}")
judge.check("Ex1: shard_map GELU", np.array(y_sharded), np.array(y_ref))


# %% [markdown]
# ---
# ## Exercise 2: AllReduce with `lax.psum`
#
# Implement a data-parallel gradient sync using `shard_map` + `lax.psum`.
#
# Each device has its local gradient. After `psum`, every device should have the **sum** of all local gradients.
#
# TODO: Implement `allreduce_sum_shardmap`
#

# %%
def allreduce_sum_shardmap(
    local_grads: jnp.ndarray,  # shape: [n_devices, n_params]
    mesh: Mesh
) -> jnp.ndarray:
    """
    AllReduce (sum) local gradients across all devices.
    
    Input:  [n_devices, n_params] — each row is one device's local gradient
    Output: [n_devices, n_params] — every row = sum of all rows
    
    Use shard_map with lax.psum inside.
    """
    @partial(smap,
             mesh=mesh,
             in_specs=(P('devices', None),),  # first dim sharded, params replicated
             out_specs=P('devices', None))
    def _sync(g_local):
        # g_local shape on each device: [1, n_params]
        # TODO: use lax.psum to sum across all devices
        # return lax.psum(g_local, axis_name='devices')
        pass
    
    return _sync(local_grads)


# Test
n_dev = jax.device_count()
key = jax.random.PRNGKey(0)
local_grads = jax.random.normal(key, (n_dev, 16))

synced = allreduce_sum_shardmap(local_grads, mesh1d)
expected_sum = jnp.sum(local_grads, axis=0, keepdims=True).repeat(n_dev, axis=0)

print(f"Local grad[0]:  {np.array(local_grads[0]).round(3)}")
print(f"Synced grad[0]: {np.array(synced[0]).round(3)}")
print(f"Expected sum:   {np.array(expected_sum[0]).round(3)}")

judge.check("Ex2a: AllReduce sum correct", np.array(synced), np.array(expected_sum))
judge.check("Ex2b: All devices identical after sync",
            np.allclose(np.array(synced[0]), np.array(synced[-1])), True)


# %% [markdown]
# ---
# ## Exercise 3: AllGather with `lax.all_gather`
#
# Implement parameter gathering for ZeRO-3 style: each device holds its shard, and after AllGather every device has the full tensor.
#
# TODO: Implement `allgather_shardmap`
#

# %%
def allgather_shardmap(
    param_shard: jnp.ndarray,  # [n_devices, shard_size]
    mesh: Mesh
) -> jnp.ndarray:
    """
    AllGather parameter shards: each device starts with its shard,
    ends with the full parameter tensor.
    
    Input:  [n_devices, shard_size]  — first dim sharded
    Output: [n_devices, n_devices * shard_size]  — every device has full params
    """
    n_dev = jax.device_count()

    @partial(smap,
             mesh=mesh,
             in_specs=(P('devices', None),),
             out_specs=P('devices', None))
    def _gather(shard_local):
        # shard_local: [1, shard_size] on each device
        # TODO: gather all shards using lax.all_gather
        # gathered shape: [n_devices, 1, shard_size] → reshape to [1, n_devices*shard_size]
        # Hint:
        #   gathered = lax.all_gather(shard_local, axis_name='devices', axis=0)
        #   return gathered.reshape(1, -1)
        pass
    
    return _gather(param_shard)


# Test
n_dev = jax.device_count()
shard_size = 4
shards = jnp.arange(n_dev * shard_size, dtype=jnp.float32).reshape(n_dev, shard_size)

gathered = allgather_shardmap(shards, mesh1d)
expected_full = jnp.arange(n_dev * shard_size, dtype=jnp.float32)

print(f"Input shards: {shards}")
print(f"Gathered[0]:  {np.array(gathered[0])}")
print(f"Expected:     {np.array(expected_full)}")

judge.check("Ex3a: AllGather shape", gathered.shape, (n_dev, n_dev * shard_size))
judge.check("Ex3b: AllGather content", np.array(gathered[0]), np.array(expected_full))
judge.check("Ex3c: All devices have same result",
            np.allclose(np.array(gathered[0]), np.array(gathered[-1])), True)


# %% [markdown]
# ---
# ## Exercise 4: Ring Communication with `lax.ppermute`
#
# `lax.ppermute` enables point-to-point communication: each device sends to and receives from a specific peer. This is the building block for ring algorithms.
#
# Use `ppermute` to implement a **ring shift**: each device sends its data to the next device (rank+1) % N.
#
# TODO: Implement `ring_shift`
#

# %%
def ring_shift(
    data: jnp.ndarray,  # [n_devices, chunk_size]
    mesh: Mesh,
    shift: int = 1      # how many positions to shift (positive = shift right)
) -> jnp.ndarray:
    """
    Ring-shift data: each device's data moves to the next device.
    Device 0 → Device 1, Device 1 → Device 2, ..., Device N-1 → Device 0.
    
    Input:  [n_devices, chunk_size] — first dim sharded
    Output: [n_devices, chunk_size] — shifted version
    """
    n_dev = jax.device_count()
    # Build the permutation: rank r sends to (r + shift) % n_dev
    # perm is a list of (src, dst) pairs
    perm = [(r, (r + shift) % n_dev) for r in range(n_dev)]

    @partial(smap,
             mesh=mesh,
             in_specs=(P('devices', None),),
             out_specs=P('devices', None))
    def _shift(local):
        # TODO: use lax.ppermute to send local data to (rank+shift) % n_dev
        # return lax.ppermute(local, axis_name='devices', perm=perm)
        pass
    
    return _shift(data)


# Test: shift by 1
n_dev = jax.device_count()
data = jnp.arange(n_dev, dtype=jnp.float32).reshape(n_dev, 1) * 10  # [0,10,20,...]

shifted = ring_shift(data, mesh1d, shift=1)
# After shift by 1: device 0 gets device N-1's data, device 1 gets device 0's data, ...
expected_shifted = jnp.roll(data, shift=1, axis=0)

print(f"Before shift: {data.flatten().tolist()}")
print(f"After shift:  {np.array(shifted).flatten().tolist()}")
print(f"Expected:     {expected_shifted.flatten().tolist()}")

judge.check("Ex4: ring shift by 1", np.array(shifted), np.array(expected_shifted))

# %% [markdown]
# ---
# ## Exercise 5: GSPMD Sharding Propagation
#
# Reason about what sharding GSPMD will produce for the output of operations given input shardings. Fill in the expected output `PartitionSpec`.
#
# Recall the rules:
# - `matmul(A, B)`: if A is `P('data', None)` and B is `P(None, 'model')`, output is `P('data', 'model')` (outer product-like)
# - `matmul(A, B)`: if A is `P('data', None)` and B is `P(None, None)`, output is `P('data', None)` (B replicated → output follows A)
# - `add(X, Y)`: output sharding = input sharding (element-wise)
# - `sum(X, axis=1)`: axis 1 is reduced away; its sharding disappears from output
#
# TODO: Fill in the expected output specs
#

# %%
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

mesh_2d = Mesh(mesh_utils.create_device_mesh((2, 4)), ('data', 'model'))

def infer_output_sharding_matmul(A_spec, B_spec):
    """
    Given PartitionSpecs for A [m, k] and B [k, n] in matmul C = A @ B,
    return the expected output PartitionSpec for C [m, n].
    
    Rules (assuming 2D mesh with axes 'data' and 'model'):
    - Output row axis (m) gets the sharding from A's row axis
    - Output col axis (n) gets the sharding from B's col axis
    - If A's col axis and B's row axis both sharded over same mesh axis
      → reduction happens → AllReduce needed, output loses that axis
    """
    # A_spec: P(row_A, col_A), B_spec: P(row_B, col_B)
    # Output m dim = A row shard, output n dim = B col shard
    row_out = A_spec[0] if len(A_spec) > 0 else None
    col_out = B_spec[1] if len(B_spec) > 1 else None
    return P(row_out, col_out)


# Question: what is the output sharding for each matmul?

# Case 1: A=[batch, d_model] P('data', None)  @  B=[d_model, d_ff] P(None, 'model')
out_spec_1 = None  # TODO: P('data', 'model')

# Case 2: A=[batch, d_model] P('data', None)  @  B=[d_model, d_ff] P(None, None)
out_spec_2 = None  # TODO: P('data', None)

# Case 3: A=[batch, d_ff] P('data', 'model') @  B=[d_ff, d_model] P('model', None)
# Contracting 'model' axis → needs AllReduce → output col has None
out_spec_3 = None  # TODO: P('data', None)

# Case 4: A=[n_heads, seq, d] P('model', None, None) @ B=[n_heads, d, seq] P('model', None, None)
# Both row dims are 'model', output is [n_heads, seq, seq]
out_spec_4 = None  # TODO: P('model', None, None)


judge.check("Ex5a: matmul out_spec_1", out_spec_1, P('data', 'model'))
judge.check("Ex5b: matmul out_spec_2", out_spec_2, P('data', None))
judge.check("Ex5c: matmul out_spec_3 (contracted axis)", out_spec_3, P('data', None))
judge.check("Ex5d: matmul out_spec_4", out_spec_4, P('model', None, None))


# Verify by actually running and checking sharding
key = jax.random.PRNGKey(0)
A = jax.device_put(jax.random.normal(key, (8, 16)),
                   NamedSharding(mesh_2d, P('data', None)))
B = jax.device_put(jax.random.normal(key, (16, 32)),
                   NamedSharding(mesh_2d, P(None, 'model')))

with mesh_2d:
    C = jit(lambda a, b: a @ b)(A, B)

print(f"\nActual output sharding: {C.sharding.spec}")
judge.check("Ex5e: actual GSPMD output sharding", C.sharding.spec, P('data', 'model'))

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
# 1. **SPMD** — all devices run the same program; JAX enforces this via `jit` and `shard_map`.
# 2. **`shard_map`** gives you explicit per-device control: you write the code that runs on each device's shard, and explicitly call collectives.
# 3. **`lax.psum`, `lax.all_gather`, `lax.psum_scatter`, `lax.ppermute`** are the collective primitives inside `shard_map`.
# 4. **GSPMD** automatically propagates sharding through ops and inserts minimal collectives — the output sharding of `A @ B` follows predictable rules based on input shardings.
# 5. **Use implicit `jit`+`PartitionSpec` for most cases; use `shard_map` when you need precise control** (custom attention, ring algorithms, custom gradients).
#
# ---
# **Next:** [Chapter 9 — Distributed Training with Flax & Optax](./chapter_09_flax_optax_distributed.ipynb)
#
