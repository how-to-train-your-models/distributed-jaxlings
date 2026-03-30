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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/chapters/chapter_10_multihost_training.ipynb)
#
# # Chapter 10: Multi-host JAX Training in Practice
#
# > **Course: Distributed Training — From Concepts to JAX**
#
# ---
#
# ## Learning Objectives
#
# - Understand the JAX multi-host (multi-process) programming model
# - Configure distributed JAX initialization
# - Handle per-host data loading correctly
# - Implement global mesh construction across multiple hosts
# - Design a production-grade training loop with checkpointing and evaluation
# - Diagnose common distributed training failures
#

# %% [markdown]
# ---
# ## 1. JAX Multi-Host Architecture
#
# A **host** is a machine (node). A JAX **process** is a Python process. In multi-host JAX:
# - Each host runs **one Python process**
# - Each process controls its **local devices** (e.g., 8 GPUs per host)
# - All processes participate in the same **global device mesh**
# - Processes communicate through **XLA collective operations** (AllReduce etc.), not Python
#
# ```
# Host 0 (process 0):  GPU 0-7
# Host 1 (process 1):  GPU 8-15
# Host 2 (process 2):  GPU 16-23
# Host 3 (process 3):  GPU 24-31
#
# Global mesh: 32 GPUs total, shape (4, 8) → ('data'=4, 'model'=8)
# ```
#
# ### Initialization
# ```python
# # Every process must call this:
# jax.distributed.initialize(
#     coordinator_address='host0:1234',  # rank 0 address
#     num_processes=4,
#     process_id=MY_RANK,               # 0, 1, 2, or 3
#     local_device_ids=list(range(8))   # GPUs on this host
# )
# ```
#
# After this, `jax.devices()` returns ALL 32 devices globally.
#

# %% [markdown]
# ---
# ## 2. Per-Host Data Loading
#
# In multi-host training, each process loads **only its shard** of the data:
#
# ```python
# process_id = jax.process_index()  # 0, 1, 2, ...
# n_processes = jax.process_count()
#
# # Each host loads 1/n_processes of the dataset
# dataset_shard = dataset[process_id::n_processes]
#
# # Within the host, further split across local devices
# n_local = jax.local_device_count()   # devices on this host
# local_batch = get_batch(dataset_shard, batch_size=global_batch // n_processes)
# ```
#
# **Key rule:** Each process only creates arrays for its **local devices**. The global mesh then assembles the full logical array from all host shards.
#

# %% [markdown]
# ---
# ## 3. Global vs Local Arrays
#
# | | Local array | Global (sharded) array |
# |---|---|---|
# | Where it lives | One device | Spread across all devices |
# | Shape | Shard shape | Full logical shape |
# | Created by | Normal ops | `jax.make_array_from_process_local_data` |
# | Used for | Within-host ops | Cross-host distributed ops |
#
# ```python
# # Create a globally-sharded array from per-host data
# global_shape = (global_batch, seq_len)
# sharding = NamedSharding(global_mesh, P('data', None))
#
# global_tokens = jax.make_array_from_process_local_data(
#     sharding,
#     local_tokens   # shape: (local_batch, seq_len) on this host
# )
# ```
#

# %% [markdown]
# ---
# ## 4. Production Training Loop Design
#
# A production distributed training loop needs:
#
# ```
# Setup:
#   ✓ jax.distributed.initialize()
#   ✓ Create global mesh
#   ✓ Initialize or restore from checkpoint
#   ✓ Create optimizer with correct param sharding
#
# Per step:
#   ✓ Load data (process-local shard)
#   ✓ Create global array from local data
#   ✓ Forward + backward (jit-compiled, mesh-aware)
#   ✓ Gradient clipping
#   ✓ Optimizer step
#   ✓ Logging (only on process 0)
#
# Periodically:
#   ✓ Save distributed checkpoint
#   ✓ Run evaluation (separate eval mesh or same)
#   ✓ Adjust learning rate (schedule)
#
# On failure:
#   ✓ Detect NaN/Inf in loss → skip step or stop
#   ✓ Resume from last checkpoint
# ```
#

# %% [markdown]
# ---
# ## 5. Common Failure Modes
#
# | Problem | Symptoms | Fix |
# |---|---|---|
# | **Deadlock** | All processes hang | One process hit Python exception; check all logs |
# | **OOM** | GPU out of memory | Reduce batch, enable gradient checkpointing, check for un-freed allocations |
# | **NaN loss** | loss = nan | Enable `jax.config.update('jax_debug_nans', True)`, check LR, check loss scaling |
# | **Shape mismatch** | XLA error in jit | Verify PartitionSpec matches actual tensor shape |
# | **Slow AllReduce** | Low GPU utilization | Network bottleneck; check bandwidth, use gradient compression |
# | **Divergence** | Loss spike then NaN | LR too high, warmup too short, or bad batch |
#

# %% [markdown]
# ---
# ## Judge Setup
#

# %%
import os
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from functools import partial
from jax import jit

print(f"Devices: {jax.device_count()}")

class Judge:
    def __init__(self):
        self.passed = 0; self.failed = 0

    def check(self, name, got, expected, tol=1e-3):
        if isinstance(expected, bool):
            ok = bool(got) == expected
        elif isinstance(expected, tuple):
            ok = tuple(got) == tuple(expected)
        elif hasattr(expected, 'shape') or isinstance(expected, np.ndarray):
            ok = np.allclose(np.array(got), np.array(expected), atol=tol)
        else:
            ok = abs(float(np.array(got).flat[0]) - float(expected)) / (abs(float(expected)) + 1e-9) < tol
        if ok:
            self.passed += 1; print(f"✅ {name}: PASSED")
        else:
            self.failed += 1; print(f"❌ {name}: FAILED — got {got!r}, expected {expected!r}")
        return ok

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*40}\n  Results: {self.passed}/{total} passed")
        print("  🎉 Chapter 10 complete!" if self.failed==0 else f"  {self.failed} remaining.")
        print('='*40)

judge = Judge()
print("Judge ready!")


# %% [markdown]
# ---
# ## Exercise 1: Multi-host Mesh Construction
#
# Simulate the global mesh construction that would happen across multiple hosts. In a real setup, each host would call this with `jax.devices()` returning all global devices after `distributed.initialize()`.
#
# TODO: Implement `build_global_mesh`
#

# %%
def build_global_mesh(
    n_data_parallel: int,
    n_tensor_parallel: int,
) -> Mesh:
    """
    Build a 2D global mesh for 3D parallelism (data × tensor).
    Total devices must equal n_data_parallel * n_tensor_parallel.
    
    Axis ordering convention:
    - 'data' axis: outer dimension (varies across hosts)
    - 'model' axis: inner dimension (varies within a host)
    
    Args:
        n_data_parallel:   number of data-parallel replicas
        n_tensor_parallel: number of tensor-parallel shards
    
    Returns:
        Mesh with axis_names=('data', 'model')
    """
    total = n_data_parallel * n_tensor_parallel
    assert total <= jax.device_count(), \
        f"Need {total} devices but only have {jax.device_count()}"
    
    # TODO: create device mesh of shape (n_data_parallel, n_tensor_parallel)
    # and wrap in Mesh with axis_names=('data', 'model')
    # devices = mesh_utils.create_device_mesh((n_data_parallel, n_tensor_parallel))
    pass


# Test different configurations
mesh_2x4 = build_global_mesh(2, 4)   # 2 data × 4 model
mesh_4x2 = build_global_mesh(4, 2)   # 4 data × 2 model
mesh_8x1 = build_global_mesh(8, 1)   # pure data parallel

print(f"2×4 mesh: shape={mesh_2x4.shape}, axes={mesh_2x4.axis_names}")
print(f"4×2 mesh: shape={mesh_4x2.shape}, axes={mesh_4x2.axis_names}")
print(f"8×1 mesh: shape={mesh_8x1.shape}, axes={mesh_8x1.axis_names}")

judge.check("Ex1a: 2×4 mesh shape",  mesh_2x4.shape, (2, 4))
judge.check("Ex1b: axis names",       mesh_2x4.axis_names, ('data', 'model'))
judge.check("Ex1c: 4×2 data axis",   mesh_4x2.shape[0], 4)
judge.check("Ex1d: 8×1 pure DP",     mesh_8x1.shape, (8, 1))


# %% [markdown]
# ---
# ## Exercise 2: Per-host Data Sharding
#
# Simulate how each process loads its own data shard. In real multi-host training, each process only creates arrays for its local devices, then `make_array_from_process_local_data` assembles the global array.
#
# Here we simulate it by mimicking process-local data creation.
#
# TODO: Implement `get_local_batch` and `assemble_global_batch`
#

# %%
def get_local_batch(
    process_id: int,
    n_processes: int,
    global_batch_size: int,
    seq_len: int,
    vocab_size: int,
    step: int,
    base_seed: int = 0
) -> np.ndarray:
    """
    Simulate process-local data loading.
    Each process generates its own shard of the global batch.
    
    Returns:
        np.ndarray of shape [local_batch_size, seq_len]
        where local_batch_size = global_batch_size // n_processes
    """
    assert global_batch_size % n_processes == 0
    local_batch = global_batch_size // n_processes
    
    # TODO: Generate deterministic data for this process and step
    # Use seed = base_seed + step * n_processes + process_id for reproducibility
    seed = base_seed + step * n_processes + process_id
    rng = np.random.default_rng(seed)
    
    # TODO: return rng.integers(0, vocab_size, (local_batch, seq_len))
    pass


def assemble_global_batch(
    local_batches: list,   # list of process-local batches (simulating multi-host)
    mesh: Mesh
) -> jnp.ndarray:
    """
    Assemble a global sharded array from per-process local batches.
    In real multi-host: use jax.make_array_from_process_local_data.
    Here we simulate by stacking and applying sharding.
    
    Returns:
        Globally sharded array, shape [global_batch, seq_len]
        with batch axis sharded over 'data'
    """
    # TODO: Stack all local batches along axis 0 to get global batch
    global_batch = None  # TODO: np.concatenate(local_batches, axis=0)
    
    # TODO: Apply sharding: P('data', None) — shard batch over 'data' axis
    sharding = NamedSharding(mesh, P('data', None))
    return None  # TODO: jax.device_put(jnp.array(global_batch), sharding)


# Test
n_procs, global_batch, seq_len, vocab = 4, 16, 32, 256

# Simulate each process loading its shard
local_batches = [
    get_local_batch(pid, n_procs, global_batch, seq_len, vocab, step=0)
    for pid in range(n_procs)
]

print(f"Local batch shape (per process): {local_batches[0].shape}")
print(f"Different processes have different data: "
      f"{not np.array_equal(local_batches[0], local_batches[1])}")

mesh = build_global_mesh(4, 2)
global_tokens = assemble_global_batch(local_batches, mesh)

print(f"Global batch shape: {global_tokens.shape}")
print(f"Sharding spec: {global_tokens.sharding.spec}")

judge.check("Ex2a: local batch shape", local_batches[0].shape, (global_batch//n_procs, seq_len))
judge.check("Ex2b: processes have different data",
            not np.array_equal(local_batches[0], local_batches[1]), True)
judge.check("Ex2c: global batch shape", global_tokens.shape, (global_batch, seq_len))
judge.check("Ex2d: batch sharded over 'data'", global_tokens.sharding.spec[0], 'data')

# %% [markdown]
# ---
# ## Exercise 3: Learning Rate Schedule
#
# Production training uses a learning rate schedule: **linear warmup** followed by **cosine decay**.
#
# $$lr(t) = \begin{cases} \frac{t}{T_{warmup}} \cdot lr_{max} & t < T_{warmup} \\ lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})\left(1 + \cos\frac{\pi(t - T_{warmup})}{T_{total} - T_{warmup}}\right) & t \geq T_{warmup} \end{cases}$$
#
# TODO: Implement `warmup_cosine_schedule`
#

# %%
import math

def warmup_cosine_schedule(
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 0.0
) -> float:
    """
    Linear warmup followed by cosine decay.
    
    Args:
        step:          current training step
        warmup_steps:  number of warmup steps
        total_steps:   total training steps
        lr_max:        peak learning rate
        lr_min:        minimum learning rate (end of cosine decay)
    
    Returns:
        learning rate at this step
    """
    if step < warmup_steps:
        # TODO: Linear warmup
        return 0.0  # TODO: lr_max * step / warmup_steps
    else:
        # TODO: Cosine decay
        progress = 0.0  # TODO: (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.0      # TODO: lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# Create Optax schedule from our function
def make_lr_schedule(warmup_steps, total_steps, lr_max, lr_min=1e-5):
    """Create optax schedule using warmup_cosine_schedule."""
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0, lr_max, warmup_steps),
            optax.cosine_decay_schedule(lr_max, total_steps - warmup_steps, lr_min / lr_max)
        ],
        boundaries=[warmup_steps]
    )


# Test the schedule
total_steps = 1000
warmup_steps = 100
lr_max = 3e-4

steps = [0, 50, 100, 500, 999]
lrs = [warmup_cosine_schedule(s, warmup_steps, total_steps, lr_max) for s in steps]

print("Learning rate schedule:")
for s, lr in zip(steps, lrs):
    bar = '█' * int(lr / lr_max * 20)
    print(f"  step {s:4d}: lr={lr:.2e}  {bar}")

judge.check("Ex3a: lr at step 0 is 0",          lrs[0], 0.0)
judge.check("Ex3b: lr at warmup is lr_max",      lrs[2], lr_max, tol=0.01)
judge.check("Ex3c: lr at midpoint < lr_max",     lrs[3] < lr_max, True)
judge.check("Ex3d: lr at end ≈ 0",               lrs[4] < lr_max * 0.1, True)
judge.check("Ex3e: lr increases during warmup",  lrs[1] > lrs[0], True)


# %% [markdown]
# ---
# ## Exercise 4: NaN/Inf Detection and Safe Training Step
#
# Production training must handle numerical instability. Implement a safe training step that skips the update if the loss or gradients are non-finite.
#
# TODO: Implement `safe_train_step`
#

# %%
def grads_are_finite(grads) -> bool:
    """
    Check if all gradient arrays are finite (no NaN or Inf).
    
    Returns:
        True if all gradients are finite, False otherwise
    """
    # TODO: use jax.tree.leaves to get all grad arrays,
    # check jnp.all(jnp.isfinite(g)) for each, AND them together
    leaves = jax.tree.leaves(grads)
    # TODO: return all(jnp.all(jnp.isfinite(g)) for g in leaves)
    pass


def safe_train_step(
    model,
    state,
    tokens,
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_max: float
):
    """
    Training step with NaN detection and dynamic learning rate.
    
    - Computes loss and gradients
    - Skips update if grads contain NaN/Inf (returns unchanged state)
    - Applies warmup-cosine LR schedule
    - Returns (new_state, loss, skipped: bool)
    """
    # Compute current LR
    lr = warmup_cosine_schedule(step, warmup_steps, total_steps, lr_max)

    def loss_fn(params):
        logits = model.apply({'params': params}, tokens)
        # Simple next-token prediction loss
        log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
        targets = tokens[:, 1:]
        nll = -log_probs[
            jnp.arange(tokens.shape[0])[:, None],
            jnp.arange(tokens.shape[1] - 1)[None, :],
            targets
        ]
        return jnp.mean(nll)

    from jax import value_and_grad
    loss, grads = value_and_grad(loss_fn)(state.params)

    # TODO: check if loss and gradients are finite
    loss_ok   = None  # TODO: bool(jnp.isfinite(loss))
    grads_ok  = None  # TODO: grads_are_finite(grads)
    should_update = None  # TODO: loss_ok and grads_ok

    if should_update:
        # TODO: scale gradients by lr and apply via state.apply_gradients
        # Note: Optax handles LR internally; here we manually scale for simplicity
        scaled_grads = jax.tree.map(lambda g: g * lr / lr_max, grads)
        new_state = state.apply_gradients(grads=scaled_grads)
    else:
        new_state = state  # skip update

    return new_state, float(loss), not should_update


# Setup for test
import flax.linen as nn

class TinyModel(nn.Module):
    vocab: int; d: int
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.vocab)(nn.Embed(self.vocab, self.d)(x))

tiny = TinyModel(vocab=16, d=8)
key = jax.random.PRNGKey(0)
dummy_tokens = jnp.zeros((2, 4), jnp.int32)
tiny_params = tiny.init(key, dummy_tokens)['params']
tiny_state = train_state.TrainState.create(
    apply_fn=tiny.apply, params=tiny_params, tx=optax.adam(1e-3))

tokens_test = jax.random.randint(key, (2, 8), 0, 16)

# Normal step
new_state, loss, skipped = safe_train_step(
    tiny, tiny_state, tokens_test, step=50,
    warmup_steps=10, total_steps=100, lr_max=1e-3)

print(f"Normal step: loss={loss:.4f}, skipped={skipped}")
judge.check("Ex4a: normal step not skipped",  skipped, False)
judge.check("Ex4b: loss is finite",            bool(np.isfinite(loss)), True)

# Test NaN detection
judge.check("Ex4c: grads_are_finite with clean grads",
            grads_are_finite({'w': jnp.ones(5)}), True)
judge.check("Ex4d: grads_are_finite with NaN grad",
            grads_are_finite({'w': jnp.array([1.0, float('nan'), 2.0])}), False)

# %% [markdown]
# ---
# ## Exercise 5: Distributed Checkpoint with Orbax
#
# Implement checkpoint save and restore using a simple numpy-based approach (simulating Orbax-style distributed checkpointing).
#
# TODO: Implement `save_checkpoint` and `restore_checkpoint`
#

# %%
import json, os
import tempfile

def save_checkpoint(
    state,          # TrainState
    step: int,
    checkpoint_dir: str,
    keep_last: int = 3
) -> str:
    """
    Save training state to disk. Only saves from process 0.
    
    Saves:
      {checkpoint_dir}/step_{step:07d}/params.npz
      {checkpoint_dir}/step_{step:07d}/metadata.json
    
    Returns:
        path to the checkpoint directory
    """
    # In real training: only rank 0 saves, or use distributed checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}")
    os.makedirs(ckpt_path, exist_ok=True)
    
    # TODO: Flatten params pytree to numpy arrays and save as npz
    # Hint: use jax.tree_util.tree_leaves_with_path or manual flattening
    leaves, treedef = jax.tree_util.tree_flatten(state.params)
    arrays = {f'param_{i}': np.array(leaf) for i, leaf in enumerate(leaves)}
    
    # TODO: save arrays to {ckpt_path}/params.npz
    # np.savez(os.path.join(ckpt_path, 'params.npz'), **arrays)
    # TODO
    
    # TODO: save metadata (step, n_leaves)
    metadata = {'step': step, 'n_leaves': len(leaves)}
    # TODO: json.dump(metadata, open(os.path.join(ckpt_path, 'metadata.json'), 'w'))
    # TODO
    
    # TODO: cleanup old checkpoints (keep only the last `keep_last`)
    all_ckpts = sorted([
        d for d in os.listdir(checkpoint_dir) if d.startswith('step_')
    ])
    for old in all_ckpts[:-keep_last]:
        import shutil
        shutil.rmtree(os.path.join(checkpoint_dir, old), ignore_errors=True)
    
    return ckpt_path


def restore_checkpoint(
    state,           # TrainState (for structure)
    checkpoint_dir: str,
    step: int = None  # if None, restore latest
):
    """
    Restore training state from checkpoint.
    
    Returns:
        (restored_state, step)
    """
    if step is None:
        # Find the latest checkpoint
        ckpts = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith('step_')])
        if not ckpts:
            return state, 0
        step = int(ckpts[-1].split('_')[1])
    
    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}")
    
    # TODO: load metadata
    meta = None  # TODO: json.load(open(os.path.join(ckpt_path, 'metadata.json')))
    
    # TODO: load params.npz
    data = None  # TODO: np.load(os.path.join(ckpt_path, 'params.npz'))
    
    # TODO: reconstruct pytree from flat arrays
    leaves_orig, treedef = jax.tree_util.tree_flatten(state.params)
    restored_leaves = [jnp.array(data[f'param_{i}']) for i in range(meta['n_leaves'])]
    restored_params = treedef.unflatten(restored_leaves)
    
    new_state = state.replace(params=restored_params)
    return new_state, step


# Test
tmpdir = tempfile.mkdtemp()

# Save checkpoints at steps 10, 20, 30 (keep_last=2 → should delete step 10)
for s in [10, 20, 30]:
    path = save_checkpoint(tiny_state, step=s, checkpoint_dir=tmpdir, keep_last=2)
    print(f"Saved checkpoint at step {s}: {path}")

remaining = sorted(os.listdir(tmpdir))
print(f"Remaining checkpoints: {remaining}")
judge.check("Ex5a: old checkpoint deleted", 'step_0000010' not in remaining, True)
judge.check("Ex5b: last checkpoint kept",   'step_0000030' in remaining, True)

# Restore and verify params match
restored_state, restored_step = restore_checkpoint(tiny_state, tmpdir)
print(f"Restored from step: {restored_step}")

orig_leaves = jax.tree.leaves(tiny_state.params)
rest_leaves = jax.tree.leaves(restored_state.params)
judge.check("Ex5c: restored step is 30", restored_step, 30)
judge.check("Ex5d: restored params match original",
            all(np.allclose(np.array(a), np.array(b)) for a,b in zip(orig_leaves, rest_leaves)), True)

# %% [markdown]
# ---
# ## Summary
#

# %%
judge.summary()

# %% [markdown]
# ---
# ## Course Complete!
#
# Congratulations on finishing the **Distributed Training: From Concepts to JAX** course!
#
# ### What you've learned:
#
# | Chapter | Topics |
# |---------|--------|
# | 1 | Why distributed training: scaling laws, memory & compute math |
# | 2 | Data parallelism, Ring-AllReduce, gradient accumulation, ZeRO |
# | 3 | Tensor, pipeline, and sequence parallelism |
# | 4 | Collective operations: AllReduce, AllGather, ReduceScatter, Broadcast |
# | 5 | Mixed precision (FP16/BF16), loss scaling, gradient checkpointing |
# | 6 | JAX fundamentals: `jit`, `vmap`, `pmap`, `grad` |
# | 7 | JAX sharding: Device meshes, `PartitionSpec`, `NamedSharding` |
# | 8 | SPMD: `shard_map`, custom collectives, GSPMD reasoning |
# | 9 | Full distributed training with Flax + Optax |
# | 10 | Multi-host training, LR schedules, NaN handling, checkpointing |
#
# ### Next steps:
# - Study [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for production 3D parallelism
# - Study [MaxText](https://github.com/google/maxtext) for JAX-native LLM training
# - Experiment with [Orbax](https://github.com/google/orbax) for production checkpointing
# - Read the [GSPMD paper](https://arxiv.org/abs/2105.04663) for compiler internals
#
