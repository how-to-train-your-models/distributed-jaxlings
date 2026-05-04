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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/notebooks/chapter_03_collectives_shard_map.ipynb)
#
# # Chapter 3: Collectives & `shard_map`
#
# > **Course: Distributed Training in JAX**
# ---
#
# ## What we build
#
# A **hand-rolled Ring-AllReduce**, decomposed as `ReduceScatter + AllGather`, and
# benchmarked against `jax.lax.psum`. Along the way we touch every standard collective
# (Broadcast, Reduce, Scatter, Gather, AllGather, ReduceScatter, AllReduce, AllToAll,
# Permute) and learn `shard_map` — the API for writing per-device code with explicit
# collectives.
#
# **Real-world hook:** at 16k-GPU scale, AllReduce eats 30–40% of step time on frontier
# LLM training runs. NVIDIA built NCCL and SHARP **specifically** to make this faster —
# a 10% AllReduce regression at Meta/Google scale is millions of GPU-hours/year. In Ch 2
# the compiler hid the AllReduce from us. Now we look inside.
#
# ## Learning Objectives
#
# By the end of this chapter you will be able to:
# - Name every standard collective and its bandwidth cost
# - Use `shard_map` to write per-device functions with explicit collectives
# - Call `lax.psum`, `lax.all_gather`, `lax.psum_scatter`, `lax.ppermute`, `lax.all_to_all`
# - Decompose AllReduce as `ReduceScatter + AllGather` and explain *why* the ring algorithm
#   is bandwidth-optimal
# - Tell when to use implicit (`jit` + `PartitionSpec`) vs explicit (`shard_map`) SPMD
# - Sketch how the GSPMD compiler decides where to insert collectives
#
# ---

# %% [markdown]
# ## Setup
#

# %%
import os
# Simulate 4 CPU devices so multi-device examples run anywhere.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import sys
import pathlib

# Make the repo root and src/ importable so `from judge import Judge` resolves
# whether this runs from src/chapters/ (as a .py) or notebooks/ (as a .ipynb).
_HERE = pathlib.Path.cwd()
if _HERE.name == "chapters" and _HERE.parent.name == "src":
    _ROOT = _HERE.parent.parent          # src/chapters/ → repo root
elif _HERE.name in ("notebooks", "chapters", "src"):
    _ROOT = _HERE.parent                 # notebooks/ (notebook) → repo root
else:
    _ROOT = _HERE                        # already at repo root
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from judge import Judge

judge = Judge("Chapter 3")
print(f"JAX devices: {jax.devices()}")

# 1D mesh over all devices, axis named 'i' for "device index".
mesh = Mesh(np.array(jax.devices()), axis_names=('i',))


# %% [markdown]
# ---
# ## 1. What are collectives and why do we need them?
#
# When you are training a massive models, you need standard way to communicate between 
# devices. Theere are standardized way to do this (mostly derived from MPI - Message Passing Interface)
# and are generally called collectives.
#
# 
# There are several types of collectives, each with its own use case. In previous chapter we've seen AllReduce collective,
# where all devices contribute to a sum and each device gets the full sum. In this  chapter we'll look at other collectives
# as well.
# 
# Collectives are not specific to jax, they are generic concepts in distributed computing. Most deep learning
# frameworks (PyTorch, Tensorflow) have their own implementations of these collectives.
# However, we'll learn how to use them in the context of JAX.
#
#

# %% [markdown]
# ---
# ## 2. The Collective Zoo
#
# Notation: `D` = number of devices, `S` = total payload size (bytes).
#
# | Collective       | What it does                                              | Comm cost (per device, ring) |
# |------------------|-----------------------------------------------------------|------------------------------|
# | **Broadcast**    | One device sends a tensor; everyone else receives.        | `S`                          |
# | **Reduce**       | All devices contribute; root device gets the sum/mean.    | `S`                          |
# | **Scatter**      | One device holds an `(D, ...)` tensor; sends slice `i` to device `i`. | `S` |
# | **Gather**       | Each device holds slice `i`; root assembles `(D, ...)`.   | `S`                          |
# | **AllGather**    | Each device holds slice `i`; everyone ends up with all `D` slices. | `S` |
# | **ReduceScatter**| All devices contribute; each ends up with one reduced slice.        | `S` |
# | **AllReduce**    | All devices contribute; everyone gets the full reduction. | `2S` (= ReduceScatter + AllGather) |
# | **AllToAll**     | Each device sends slice `i→j` to device `j`.              | `S`                          |
# | **Permute**      | Each device sends to a fixed neighbor (e.g. ring shift).  | `S/D`                        |
#
# Key identity (used everywhere):
#
# > **`AllReduce  ==  ReduceScatter + AllGather`**
#
# This is why ring-based AllReduce achieves the bandwidth-optimal `2S(D-1)/D` per device.
#

# %% [markdown]
# ---
# ## 3. Bandwidth Analysis (Why ring beats tree at scale)
#
# Two AllReduce algorithms worth knowing:
#
# - **Tree-reduce + Broadcast:** O(log D) latency, but each step doubles the reducer's
#   incoming bandwidth — bottleneck is the root.
# - **Ring (NCCL default for medium-large messages):** every device sends and receives
#   simultaneously every step; per-device bandwidth is constant in `D`. Total volume per
#   device: `2S(D-1)/D ≈ 2S` for large `D`.
#
# Latency: rings need `2(D-1)` steps; trees need `2 log₂ D`. For tiny tensors (gradient
# tails, biases), trees win on latency. NCCL uses a **hybrid**: rings for big tensors,
# trees for small ones.
#

# %% [markdown]
# ---
# ## 4. Implicit vs Explicit SPMD
#
# Two ways to get collectives into your program:
#
# **Implicit (Ch 2 style):** annotate sharding with `PartitionSpec`, wrap in `jit`. The
# **GSPMD** compiler propagates shardings through the computation and inserts whatever
# collectives are needed. You never write `lax.psum` yourself.
#
# - Pros: less code, the compiler can be cleverer than you (fusion, overlap).
# - Cons: opaque — when you need to control *exactly* which collective happens where
#   (TP, ring attention, all-to-all dispatch), you can't.
#
# **Explicit (`shard_map`):** wrap a per-device function and call `lax.*` collectives by
# hand. Inside `shard_map`, each device runs the function on its local shard.
#
# - Pros: full control. Necessary for ring attention, all-to-all routing, anything where
#   the per-device program differs from the global program.
# - Cons: more code, less compiler magic.
#
# **Rule of thumb:** start with implicit; reach for `shard_map` when you need to *write*
# a collective (Ch 6 DLRM all-to-all, Ch 7 pipeline ppermute, Ch 8 ring attention).
#

# %% [markdown]
# ---
# ## 5. `shard_map` Basics
#
# Signature (sketch):
#
# ```python
# from jax.experimental.shard_map import shard_map
#
# fn_sharded = shard_map(
#     fn,
#     mesh=mesh,
#     in_specs=(P('i'),),       # how each input is sharded across the mesh
#     out_specs=P('i'),         # how the output is sharded
#     check_rep=False,
# )
# ```
#
# Inside `fn`, you see only your local shard. `lax.psum`, `lax.all_gather`, etc. then
# operate over the named mesh axis (`'i'`).
#

# %%
# TODO: a 5-line shard_map demo — local doubling, no collective.
#
# def local_double(x):
#     return x * 2
#
# x = jnp.arange(8.0)
# y = shard_map(local_double, mesh=mesh, in_specs=P('i'), out_specs=P('i'))(x)
# print(y)


# %% [markdown]
# ---
# ## 6. JAX Collective APIs (cheat sheet)
#
# Inside a `shard_map`-wrapped function, over a mesh axis named `'i'`:
#
# | What you want      | Call                                          |
# |--------------------|-----------------------------------------------|
# | AllReduce (sum)    | `lax.psum(x, 'i')`                            |
# | AllReduce (mean)   | `lax.pmean(x, 'i')`                           |
# | AllReduce (max)    | `lax.pmax(x, 'i')`                            |
# | AllGather          | `lax.all_gather(x, 'i')`                      |
# | ReduceScatter      | `lax.psum_scatter(x, 'i', tiled=True)`        |
# | All-to-all         | `lax.all_to_all(x, 'i', split_axis, concat_axis, tiled=True)` |
# | Ring shift / send  | `lax.ppermute(x, 'i', perm=[(i, (i+1)%D) for i in range(D)])` |
# | Index of this device | `lax.axis_index('i')`                       |
# | Size of axis       | `lax.axis_size('i')`                          |
#

# %% [markdown]
# ---
# ## Exercise 1 — Broadcast, Reduce, Scatter, Gather under `shard_map`
#
# Implement each of the simple collectives by composing `lax.axis_index`,
# `lax.all_gather`, and `lax.psum`. (Some will use a "select my slice from the gather"
# pattern.)
#

# %%
# TODO: shard_map implementations of broadcast / reduce / scatter / gather.


# %% [markdown]
# ---
# ## Exercise 2 — AllGather and ReduceScatter
#
# Wrap `lax.all_gather` and `lax.psum_scatter` in clean `shard_map`-based helpers and
# verify the shapes against the cheat sheet.
#

# %%
# TODO


# %% [markdown]
# ---
# ## 7. Ring Algorithms — Decomposing AllReduce
#
# The bandwidth-optimal AllReduce: split the tensor into `D` chunks. Then:
#
# 1. **ReduceScatter** (D−1 ring steps): after step `k`, device `i` holds the partial
#    sum for chunk `(i − k − 1) mod D`. After `D−1` steps, device `i` owns the *fully
#    reduced* chunk `i`.
# 2. **AllGather** (D−1 ring steps): device `i` rotates its reduced chunk around the
#    ring; after `D−1` steps everyone has every reduced chunk.
#
# Each step moves `S/D` bytes per device. Total per device: `2(D−1)·S/D ≈ 2S`. For large
# `D` you cannot do better — every device must send and receive at least `S(D−1)/D`
# bytes for AllReduce.
#

# %% [markdown]
# ### Exercise 3 — Ring AllReduce from scratch
#
# Compose your AllGather and ReduceScatter (or use `lax.ppermute` directly) to produce a
# `ring_all_reduce(x)` and verify it matches `lax.psum(x, 'i')` numerically.
#

# %%
# TODO


# %% [markdown]
# ### Exercise 4 — Ring shift with `ppermute`
#
# Implement a simple ring shift: each device passes its tensor to the next neighbor.
# Confirm with `lax.axis_index('i')` that the result really did rotate.
#

# %%
# TODO


# %% [markdown]
# ---
# ## 8. GSPMD Intuition
#
# When you write `jit` + `PartitionSpec`, the **GSPMD** compiler runs a constraint-based
# propagation: it walks the dataflow graph and figures out a sharding for every
# intermediate. When two operands disagree, it inserts a collective to reconcile them.
#
# Roughly:
#
# - Operand sharded `P('data')` + operand sharded `P()` (replicated) → no collective; one
#   side stays sharded, the other broadcasts trivially.
# - Operand sharded `P('model')` × operand sharded `P('model')` along the contracting dim
#   → ReduceScatter or AllReduce after the matmul (this is row-parallel TP, Ch 5).
# - Reading a tensor sharded `P('data')` from inside code that expects `P('model')` →
#   AllToAll.
#
# We'll use this constantly in Chs 4–9. For now, the takeaway: the implicit path works
# because GSPMD is doing real work; when it makes the wrong call, drop into `shard_map`.
#

# %% [markdown]
# ---
# ## Build: Hand-rolled Ring-AllReduce vs `lax.psum`
#
# Wrap your Exercise 3 implementation in a `jit`able function and benchmark against
# `lax.psum` for tensor sizes spanning small (1 KB) to large (100 MB+). Plot or print
# the throughput. Reflect on:
#
# - Where does the hand-rolled version win, lose, or match?
# - What does NCCL/cuCollective do that you don't?
# - Why does the gap shrink for very large tensors?
#

# %%
# TODO


# %% [markdown]
# ### Exercise 5 — Benchmark and explain
#
# Produce a one-paragraph explanation (in a markdown cell below) of the gap between your
# hand-rolled Ring-AllReduce and `lax.psum`. Reference at least one of: launch overhead,
# fusion / overlap with compute, in-network reduction (SHARP), tree-vs-ring crossover.
#

# %% [markdown]
# *Your answer here.*
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
# 1. There are ~9 standard collectives; you need to know all of them by name and rough
#    cost.
# 2. **`AllReduce == ReduceScatter + AllGather`** — this identity reappears in FSDP (Ch 4),
#    TP (Ch 5), and ring attention (Ch 8).
# 3. Implicit SPMD (`jit` + `PartitionSpec`) is the default; reach for `shard_map` when
#    you need to write a specific collective by hand.
# 4. `lax.psum`, `lax.all_gather`, `lax.psum_scatter`, `lax.ppermute`, `lax.all_to_all`
#    are the per-device APIs you'll use for the rest of the course.
# 5. Ring is bandwidth-optimal but latency-suboptimal — production libraries (NCCL) pick
#    the algorithm based on tensor size.
#
# ---
# **Next:** [Chapter 4 — FSDP / ZeRO](./chapter_04_fsdp_zero.ipynb) — convert Ch 2's
# tiny-GPT trainer through ZeRO-1 → 2 → 3 and watch peak HBM drop at each stage.
#

# %%
