"""Shared model components used across chapters.

These are deliberately *naive* transformer building blocks — token + sinusoidal
positional embeddings, vanilla MHA, LayerNorm, GeLU MLP. They are introduced as
demos in Chapter 2 and reused by later chapters that build distributed trainers
on top of the same architecture. Chapter 10 swaps in a proper mini-Llama
(RoPE, GQA, RMSNorm, SwiGLU).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx


def sinusoidal_positions(max_seq: int, dim: int) -> jax.Array:
    """Standard 'Attention is All You Need' positional table of shape (max_seq, dim)."""
    pos = jnp.arange(max_seq)[:, None]
    i = jnp.arange(0, dim, 2)[None, :]
    angle_rates = 1 / (10000 ** (i / dim))
    angles = pos * angle_rates
    pe = jnp.zeros((max_seq, dim))
    pe = pe.at[:, 0::2].set(jnp.sin(angles))
    pe = pe.at[:, 1::2].set(jnp.cos(angles))
    return pe


class Block(eqx.Module):
    """Pre-LN transformer block: attention + MLP, both with residual connections."""
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
        mask_SxS = jnp.tril(jnp.ones((S, S)))

        h_SxE = jax.vmap(self.ln1)(x_SxE)
        h_SxE = self.attn(h_SxE, h_SxE, h_SxE, mask=mask_SxS)
        x_SxE = x_SxE + h_SxE

        h_SxE = jax.vmap(self.ln2)(x_SxE)
        h_SxE = jax.nn.gelu(jax.vmap(self.mlp_in)(h_SxE))
        h_SxE = jax.vmap(self.mlp_out)(h_SxE)
        x_SxE = x_SxE + h_SxE
        return x_SxE


class TinyGPT(eqx.Module):
    """Naive decoder-only transformer: embeddings + N blocks + LN + LM head."""
    tok_emb: eqx.nn.Embedding
    pos_emb: jax.Array
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
