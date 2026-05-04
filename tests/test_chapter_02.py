"""Chapter 2 tests.

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
import equinox as eqx
import optax
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


VOCAB_SIZE = 32
EMBED_DIM = 16
NUM_HEADS = 2
NUM_LAYERS = 2
MAX_SEQ = 8
BATCH_SIZE = 4

mesh = Mesh(np.array(jax.devices()), axis_names=('data',))


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


# -- shard_model_and_batch -------------------------------------------------

def test_shard_model_and_batch__batch_sharded_on_data(shard_model_and_batch):
    from src.exercises.chapter_02 import TinyGPT as RefTinyGPT
    model = RefTinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ,
                       key=jax.random.PRNGKey(0))
    tokens_BxS = jnp.zeros((BATCH_SIZE, MAX_SEQ), dtype=jnp.int32)
    _, sharded_BxS = shard_model_and_batch(model, tokens_BxS, mesh)
    assert sharded_BxS.sharding == NamedSharding(mesh, P('data', None))


def test_shard_model_and_batch__model_replicated(shard_model_and_batch):
    from src.exercises.chapter_02 import TinyGPT as RefTinyGPT
    model = RefTinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ,
                       key=jax.random.PRNGKey(0))
    tokens_BxS = jnp.zeros((BATCH_SIZE, MAX_SEQ), dtype=jnp.int32)
    sharded_model, _ = shard_model_and_batch(model, tokens_BxS, mesh)
    arrays = [x for x in jax.tree.leaves(sharded_model) if eqx.is_array(x)]
    replicated = NamedSharding(mesh, P())
    for a in arrays:
        assert a.sharding == replicated, f"array with sharding {a.sharding}, expected replicated"


# -- loss_fn ---------------------------------------------------------------

def _ref_model():
    from src.exercises.chapter_02 import TinyGPT as RefTinyGPT
    return RefTinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ,
                      key=jax.random.PRNGKey(0))


def _dummy_batch(seed=0):
    k = jax.random.PRNGKey(seed)
    toks = jax.random.randint(k, (BATCH_SIZE, MAX_SEQ + 1), 0, VOCAB_SIZE)
    return {'tokens_BxS': toks[:, :-1], 'targets_BxS': toks[:, 1:]}


def test_loss_fn__returns_scalar(loss_fn):
    loss = loss_fn(_ref_model(), _dummy_batch())
    assert loss.shape == (), f"expected scalar, got shape {loss.shape}"


def test_loss_fn__near_log_vocab_at_init(loss_fn):
    loss = float(loss_fn(_ref_model(), _dummy_batch()))
    expected = math.log(VOCAB_SIZE)
    assert abs(loss - expected) < 1.0, (
        f"init loss {loss:.3f} far from ln(V)={expected:.3f}")


# -- train_step ------------------------------------------------------------

def _opt_setup():
    model = _ref_model()
    model = eqx.nn.inference_mode(model)
    optimizer = optax.adamw(3e-4)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)
    return params, static, opt_state, optimizer


def test_train_step__returns_three(train_step):
    params, static, opt_state, optimizer = _opt_setup()
    out = train_step(params, static, opt_state, _dummy_batch(), optimizer)
    assert isinstance(out, tuple) and len(out) == 3


def test_train_step__decreases_loss_on_repeated_batch(train_step):
    """Training on the same batch repeatedly should drop the loss."""
    params, static, opt_state, optimizer = _opt_setup()
    batch = _dummy_batch(seed=42)
    losses = []
    for _ in range(5):
        params, opt_state, loss = train_step(params, static, opt_state, batch, optimizer)
        losses.append(float(loss))
    assert losses[-1] < losses[0], (
        f"loss did not decrease: {losses[0]:.3f} → {losses[-1]:.3f}")
