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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/chapters/chapter_06_jax_fundamentals.ipynb)
#
# # Chapter 6: JAX Fundamentals for Distributed Training
#
# > **Course: Distributed Training — From Concepts to JAX**
#
# ---
#
# ## Learning Objectives
#
# - Understand JAX's functional programming model and why it matters for distributed training
# - Use `jit` for compilation and understand what makes code JIT-compilable
# - Use `vmap` for automatic vectorization (batching)
# - Use `pmap` for single-program multiple-data (SPMD) across devices
# - Use `grad` and `value_and_grad` for automatic differentiation
# - Understand `jax.random` and why explicit keys are needed
#

# %%
# Install JAX (CPU version for Colab — GPU version auto-detected in Colab GPU runtime)
# # !pip install -q jax jaxlib
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad, pmap
import numpy as np

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")

# %% [markdown]
# ---
# ## 1. JAX's Functional Model
#
# JAX requires **pure functions** — no side effects, no in-place mutation. This enables:
# - **JIT compilation** via XLA — the entire computation graph is known at compile time
# - **Automatic differentiation** — differentiating through any pure function
# - **Parallelism** — safely replicate computation across devices
#
# ```python
# # ❌ Not JAX style (in-place mutation)
# def bad(x):
#     x[0] = 1.0   # JAX arrays are immutable!
#     return x
#
# # ✅ JAX style (returns new array)
# def good(x):
#     return x.at[0].set(1.0)
# ```
#
# ### JAX's tracing model
# When you call `jit(f)(x)`, JAX **traces** `f` with abstract values (ShapedArray), builds an XLA computation graph, compiles it, and caches it. Subsequent calls with same-shaped inputs reuse the compiled code.
#
# ```python
# @jit
# def f(x):
#     return jnp.sin(x) + jnp.cos(x)
#
# # First call: traces + compiles (~10ms)
# # Second call: runs cached XLA kernel (~0.1ms)
# ```
#

# %% [markdown]
# ---
# ## 2. `jit` — Just-In-Time Compilation
#
# ```python
# @jit
# def matmul_relu(W, x):
#     return jnp.maximum(0, W @ x)
# ```
#
# **What can't be JIT-compiled:**
# - Python control flow that depends on **values** (not shapes): `if x > 0: ...` → use `jnp.where`
# - Dynamic shapes (unknown at trace time)
# - I/O operations, global state
#
# **Static arguments:** Use `static_argnums` for arguments that affect compilation:
# ```python
# @partial(jit, static_argnums=(1,))
# def f(x, n_layers):  # n_layers controls Python loop → must be static
#     for _ in range(n_layers):
#         x = layer(x)
#     return x
# ```
#

# %% [markdown]
# ---
# ## 3. `vmap` — Vectorizing Map
#
# `vmap` transforms a function that operates on **single examples** into one that operates on **batches**, without manual batch dimensions.
#
# ```python
# def loss_single(params, x, y):   # works on one example
#     ...
#
# # Automatically batched:
# loss_batch = vmap(loss_single, in_axes=(None, 0, 0))
# #                               params don't vary, x and y vary along axis 0
# ```
#
# This is equivalent to a `for` loop but **compiled as a single XLA op** — much faster.
#

# %% [markdown]
# ---
# ## 4. `pmap` — Parallel Map
#
# `pmap` replicates a function across **multiple devices** (GPUs/TPUs). Each device runs the same function on a different slice of the input.
#
# ```python
# @pmap
# def forward(params, x):
#     return model.apply(params, x)
#
# # x shape: (n_devices, batch_per_device, ...)
# # params: must be replicated across devices using jax.device_put_replicated
# ```
#
# Inside `pmap`, you can use **collective operations**:
# ```python
# from jax import lax
#
# @pmap
# def sync_grads(grads):
#     return lax.pmean(grads, axis_name='batch')  # AllReduce mean
# ```
#
# The `axis_name` must match the `pmap` axis name:
# ```python
# @partial(pmap, axis_name='batch')
# def train_step(params, x, y):
#     loss, grads = value_and_grad(compute_loss)(params, x, y)
#     grads = lax.pmean(grads, axis_name='batch')  # sync gradients
#     return update_params(params, grads), loss
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

judge = Judge("Chapter 6", default_tol=1e-4)
print("Judge ready!")

# %% [markdown]
# ---
# ## Exercise 1: JIT a Linear Layer
#
# Implement a linear layer and JIT-compile it. Verify correctness and observe the compilation speedup.
#
# TODO: Implement `linear_layer` and JIT-compile it
#

# %%
import time
from jax import jit
import jax.numpy as jnp

def linear_layer(W, b, x):
    """
    Compute W @ x + b.
    
    Args:
        W: weight matrix [out_features, in_features]
        b: bias [out_features]
        x: input [in_features]
    Returns:
        output [out_features]
    """
    # TODO: implement y = W @ x + b
    pass


# TODO: create a JIT-compiled version
linear_layer_jit = None  # TODO: jit(linear_layer)


# Test correctness
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
W = jax.random.normal(k1, (8, 4))
b = jax.random.normal(k2, (8,))
x = jax.random.normal(k3, (4,))

y = linear_layer(W, b, x)
y_jit = linear_layer_jit(W, b, x)

print(f"Output shape: {y.shape}")
judge.check("Ex1a: linear layer output shape", y.shape[0], 8)
judge.check("Ex1b: jit version matches eager", np.array(y_jit), np.array(y))

# Timing
W_large = jax.random.normal(k1, (1024, 1024))
b_large = jax.random.normal(k2, (1024,))
x_large = jax.random.normal(k3, (1024,))

# Warmup JIT
_ = linear_layer_jit(W_large, b_large, x_large).block_until_ready()

t0 = time.perf_counter()
for _ in range(100):
    _ = linear_layer(W_large, b_large, x_large).block_until_ready()
t_eager = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(100):
    _ = linear_layer_jit(W_large, b_large, x_large).block_until_ready()
t_jit = time.perf_counter() - t0

print(f"Eager: {t_eager*10:.1f}ms/call, JIT: {t_jit*10:.1f}ms/call")
print(f"JIT speedup: {t_eager/t_jit:.1f}×")

# %% [markdown]
# ---
# ## Exercise 2: Automatic Differentiation with `grad`
#
# JAX's `grad` transforms a scalar-valued function into its gradient function.
#
# TODO: Implement a loss function and use `grad` to compute gradients
#

# %%
from jax import grad, value_and_grad

def mse_loss(params: dict, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared error loss for a linear model y_pred = X @ W + b.
    
    Args:
        params: dict with 'W' [out, in] and 'b' [out]
        X:      inputs [batch, in_features]
        y:      targets [batch]
    Returns:
        scalar MSE loss
    """
    # TODO: compute y_pred = X @ params['W'].squeeze() + params['b'].squeeze()
    # Then return mean squared error
    pass


# TODO: Create grad_fn that returns (loss, grads) using value_and_grad
# grad_fn = jit(value_and_grad(mse_loss))
grad_fn = None  # TODO


# Generate data: y = 2*x + 1
key = jax.random.PRNGKey(42)
X_data = jax.random.normal(key, (100, 1))
y_data = (2 * X_data.squeeze() + 1)

# Initialize params
params = {'W': jnp.ones((1, 1)), 'b': jnp.zeros(1)}
loss_val, grads = grad_fn(params, X_data, y_data)

print(f"Initial loss: {loss_val:.4f}")
print(f"Grad W: {grads['W']}")
print(f"Grad b: {grads['b']}")

judge.check("Ex2a: grad_fn returns scalar loss", loss_val.shape, ())
judge.check("Ex2b: grad W has same shape as W", grads['W'].shape, params['W'].shape)

# Simple gradient descent should decrease loss
lr = 0.1
for step in range(100):
    loss_val, grads = grad_fn(params, X_data, y_data)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

print(f"\nAfter 100 steps:")
print(f"  W ≈ {params['W'].flatten()[0]:.3f}  (target: 2.0)")
print(f"  b ≈ {params['b'][0]:.3f}  (target: 1.0)")
judge.check("Ex2c: W converged to 2.0", float(params['W'].flatten()[0]), 2.0, tol=0.1)
judge.check("Ex2d: b converged to 1.0", float(params['b'][0]), 1.0, tol=0.1)

# %% [markdown]
# ---
# ## Exercise 3: `vmap` for Batched Computation
#
# Write a per-example loss function, then use `vmap` to batch it.
#
# TODO: Implement `loss_single` and use `vmap` to create `loss_batched`
#

# %%
from jax import vmap
from functools import partial

def loss_single(params: dict, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Loss for a SINGLE example (no batch dimension).
    
    Args:
        params: dict with 'W' [out, in]
        x:      single input [in_features]
        y:      single target scalar
    Returns:
        scalar loss
    """
    # TODO: y_pred = params['W'] @ x; return (y_pred - y)**2
    pass


# TODO: Use vmap to create batched version.
# params are NOT batched (in_axes=None for params dict), x and y ARE batched.
# Hint: vmap(loss_single, in_axes=(None, 0, 0))
loss_batched = None  # TODO


# Test
key = jax.random.PRNGKey(1)
W = jax.random.normal(key, (1, 4))  # out=1, in=4
params2 = {'W': W}

X_test = jax.random.normal(key, (16, 4))   # batch of 16
y_test = jax.random.normal(key, (16,))

# Using vmap
per_example_losses = loss_batched(params2, X_test, y_test)

# Reference: loop version
ref_losses = jnp.array([float(loss_single(params2, X_test[i], y_test[i]))
                         for i in range(16)])

print(f"vmap losses shape: {per_example_losses.shape}")
print(f"First 3 losses: {per_example_losses[:3]}")
judge.check("Ex3a: vmap output shape", per_example_losses.shape, (16,))
judge.check("Ex3b: vmap matches loop", np.array(per_example_losses), np.array(ref_losses))


# %% [markdown]
# ---
# ## Exercise 4: JAX Random Keys
#
# JAX uses **explicit, functional random keys**. Unlike NumPy's global state, you must explicitly split and pass keys.
#
# TODO: Implement `init_model` using proper key splitting
#

# %%
def init_model(key: jax.Array, d_model: int, n_layers: int) -> dict:
    """
    Initialize a multi-layer MLP with proper key splitting.
    
    Each layer has:
        - 'W': normal(0, 0.02) of shape [d_model, d_model]
        - 'b': zeros of shape [d_model]
    
    Args:
        key:      initial PRNG key
        d_model:  model dimension
        n_layers: number of layers
    
    Returns:
        dict: {'layer_0': {'W': ..., 'b': ...}, 'layer_1': ...}
    """
    params = {}
    for i in range(n_layers):
        # TODO: split the key for each layer
        # key, subkey = jax.random.split(key)
        key, subkey = None, None  # TODO
        
        params[f'layer_{i}'] = {
            # TODO: 'W': 0.02 * jax.random.normal(subkey, (d_model, d_model))
            # TODO: 'b': jnp.zeros(d_model)
            'W': None,  # TODO
            'b': None,  # TODO
        }
    return params


# Test
key = jax.random.PRNGKey(0)
model_params = init_model(key, d_model=16, n_layers=4)

print(f"Layers: {list(model_params.keys())}")
print(f"W shape: {model_params['layer_0']['W'].shape}")
print(f"b shape: {model_params['layer_0']['b'].shape}")

# Each layer should have DIFFERENT weights (different keys)
W0 = model_params['layer_0']['W']
W1 = model_params['layer_1']['W']

judge.check("Ex4a: W shape correct", model_params['layer_0']['W'].shape, (16, 16))
judge.check("Ex4b: b initialized to zeros",
            np.allclose(model_params['layer_2']['b'], 0.0), True)
judge.check("Ex4c: Different layers have different weights",
            not np.allclose(W0, W1), True)
judge.check("Ex4d: Weight std ≈ 0.02",
            float(jnp.std(W0)), 0.02, tol=0.3)

# %% [markdown]
# ---
# ## Exercise 5: `pmap` for Data Parallelism
#
# Use `pmap` to implement a data-parallel training step. Each device processes a different batch shard and gradients are synchronized via `lax.pmean`.
#
# Note: On a CPU machine with 1 device, we can simulate multiple devices with `os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'` (run before importing JAX).
#

# %%
from jax import pmap, lax
from functools import partial

n_devices = jax.device_count()
print(f"Available devices: {n_devices}")

# If only 1 device, we'll demonstrate pmap behavior conceptually
# To simulate multiple devices in CPU: restart kernel after setting:
# import os; os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


def train_step_single(params, x_batch, y_batch, lr=0.01):
    """
    Single-device training step (for reference).
    Returns updated params and loss.
    """
    def loss_fn(p):
        y_pred = x_batch @ p['W'].T + p['b']
        return jnp.mean((y_pred - y_batch)**2)
    
    loss, grads = value_and_grad(loss_fn)(params)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


@partial(pmap, axis_name='devices')
def train_step_pmap(params, x_batch, y_batch, lr):
    """
    Data-parallel training step using pmap.
    Each device processes its own x_batch shard.
    Gradients are averaged across devices via pmean.
    
    TODO: Implement this function
    """
    def loss_fn(p):
        y_pred = x_batch @ p['W'].T + p['b']
        return jnp.mean((y_pred - y_batch)**2)
    
    loss, grads = value_and_grad(loss_fn)(params)
    
    # TODO: Synchronize gradients across devices using lax.pmean
    # grads = lax.pmean(grads, axis_name='devices')
    # loss  = lax.pmean(loss,  axis_name='devices')
    grads = None  # TODO
    loss  = None  # TODO
    
    new_params = jax.tree.map(lambda p, g: p - lr[0] * g, params, grads)
    return new_params, loss


# Test: train a simple linear model
key = jax.random.PRNGKey(0)
params = {'W': jnp.ones((1, 2)), 'b': jnp.zeros(1)}

# Replicate params across devices
params_rep = jax.device_put_replicated(params, jax.devices())
lr_rep = jax.device_put_replicated(jnp.array(0.05), jax.devices())

# Batch: [n_devices, batch_per_device, features]
X_all = jax.random.normal(key, (n_devices, 16, 2))
y_all = X_all[:, :, 0] * 2 + X_all[:, :, 1] * 3  # y = 2x1 + 3x2

# Run pmap training step
params_rep, losses = train_step_pmap(params_rep, X_all, y_all, lr_rep)

print(f"Loss shape: {losses.shape} (one per device)")
print(f"Losses: {losses}")
print(f"All devices have same loss: {jnp.allclose(losses, losses[0])}")

judge.check("Ex5a: losses shape", losses.shape, (n_devices,))
judge.check("Ex5b: all devices see same synced loss",
            bool(jnp.allclose(losses, losses[0])), True)

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
# 1. **JAX requires pure functions** — no in-place mutation, explicit PRNG keys, no global state.
# 2. **`jit`** compiles functions via XLA for significant speedups; shape must be static.
# 3. **`vmap`** vectorizes single-example functions into batched ones — like NumPy broadcasting but for arbitrary functions.
# 4. **`grad` / `value_and_grad`** give automatic differentiation through any JAX function.
# 5. **`pmap`** maps computation across devices; use `lax.pmean/psum` for gradient synchronization.
#
# ---
# **Next:** [Chapter 7 — JAX Sharding & Device Meshes](./chapter_07_jax_sharding.ipynb)
#
