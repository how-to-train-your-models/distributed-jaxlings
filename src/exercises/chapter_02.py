"""Chapter 2 exercise stubs.

Mirrors the empty exercises in `notebooks/chapter_02_data_parallelism.ipynb` so
students can implement them either in the notebook or here. Reference solutions
live in `src.solutions.chapter_02`.

Model components (Block, TinyGPT, sinusoidal_positions) are demos provided by
`src.common.models` — not exercises.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


# --- Exercise 3: Shard a batch and replicate params -----------------------

def shard_model_and_batch(model, tokens_BxS, mesh: Mesh):
    """Replicate model arrays across devices; shard tokens along mesh axis 'data'.

    Returns (sharded_model, sharded_tokens_BxS).
    """
    # TODO: build a `replicated` NamedSharding (P()) and a `batch_sh` NamedSharding
    # (P('data', None)). Walk the model with jax.tree.map and jax.device_put each
    # array leaf onto `replicated`. Place tokens_BxS onto `batch_sh`.
    raise NotImplementedError


# --- Exercise 4: loss_fn and train_step -----------------------------------

def loss_fn(model, batch: dict) -> jax.Array:
    """Next-token cross-entropy. batch = {'tokens_BxS': int[B,S], 'targets_BxS': int[B,S]}."""
    # TODO: vmap the model across the batch dim to produce logits_BxSxV, then take the
    # mean of optax.softmax_cross_entropy_with_integer_labels(logits_BxSxV, targets).
    raise NotImplementedError


@functools.partial(jax.jit, static_argnames=('static', 'optimizer'))
def train_step(params, static, opt_state, batch, optimizer):
    """One DP train step. Params and static are the two halves of eqx.partition."""
    # TODO: recombine params + static, take eqx.filter_value_and_grad of loss_fn,
    # filter the grads to array leaves, call optimizer.update, then optax.apply_updates.
    # Return (params, opt_state, loss).
    raise NotImplementedError
