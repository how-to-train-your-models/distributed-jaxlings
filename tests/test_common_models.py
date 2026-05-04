"""Tests for common model components (Block, TinyGPT).

Convention: each test takes one argument, named after the exercise function/class
it grades. The notebook's `Judge.check(fn)` matches `test_<fn.__name__>(__...)?` and
calls each match with `fn` as the sole positional arg. Pytest gets the same name
from a fixture defined in `conftest.py`.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import math

import jax
import jax.numpy as jnp
import optax


VOCAB_SIZE = 32
EMBED_DIM = 16
NUM_HEADS = 2
NUM_LAYERS = 2
MAX_SEQ = 8
BATCH_SIZE = 4


# -- Block -----------------------------------------------------------------

def test_Block__output_shape(Block):
    block = Block(EMBED_DIM, NUM_HEADS, key=jax.random.PRNGKey(0))
    x_SxE = jax.random.normal(jax.random.PRNGKey(1), (MAX_SEQ, EMBED_DIM))
    out = block(x_SxE)
    assert out.shape == x_SxE.shape, f"expected {x_SxE.shape}, got {out.shape}"


def test_Block__residual_changes_input(Block):
    """Output must differ from input — confirms attention/MLP actually run."""
    block = Block(EMBED_DIM, NUM_HEADS, key=jax.random.PRNGKey(0))
    x_SxE = jax.random.normal(jax.random.PRNGKey(1), (MAX_SEQ, EMBED_DIM))
    out = block(x_SxE)
    assert not jnp.allclose(out, x_SxE), "Block output equals input — residual-only path?"


# -- TinyGPT ---------------------------------------------------------------

def test_TinyGPT__output_shape(TinyGPT):
    model = TinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ,
                    key=jax.random.PRNGKey(0))
    tokens_S = jnp.zeros((MAX_SEQ,), dtype=jnp.int32)
    logits_SxV = model(tokens_S)
    assert logits_SxV.shape == (MAX_SEQ, VOCAB_SIZE)


def test_TinyGPT__num_blocks(TinyGPT):
    model = TinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ,
                    key=jax.random.PRNGKey(0))
    assert len(model.blocks) == NUM_LAYERS


def test_TinyGPT__loss_near_log_vocab_at_init(TinyGPT):
    """Random init should produce ~uniform logits → CE loss ≈ ln(V)."""
    model = TinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ,
                    key=jax.random.PRNGKey(0))
    tokens_BxS = jax.random.randint(jax.random.PRNGKey(1),
                                    (BATCH_SIZE, MAX_SEQ), 0, VOCAB_SIZE)
    targets_BxS = jax.random.randint(jax.random.PRNGKey(2),
                                     (BATCH_SIZE, MAX_SEQ), 0, VOCAB_SIZE)
    logits_BxSxV = jax.vmap(model)(tokens_BxS)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        logits_BxSxV, targets_BxS))
    expected = math.log(VOCAB_SIZE)
    assert abs(float(loss) - expected) < 1.0, (
        f"init loss {float(loss):.3f} far from ln(V)={expected:.3f}")
