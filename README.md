# Distributed Training in JAX

A hands-on course on distributed training, built around JAX + Equinox + Optax.

Each chapter is motivated by a system running in production today, and the chapter's
deliverable is a small working implementation that mirrors it. Concepts and the JAX
APIs that implement them are introduced together, in the chapter where they earn
their keep — there is no standalone "JAX primitives" tour.

## Stack

- **JAX** — `jit`, `grad`, `vmap`, `Mesh`, `PartitionSpec`, `NamedSharding`, `shard_map`, `lax` collectives
- **Equinox** — pytree-native modules; a model is just a pytree, so sharding is `jax.tree.map`
- **Optax** — composable optimizers
- **Orbax** — sharded checkpointing for multi-host training

## Chapters

| #  | Chapter                                       | What we build (mirrors)                                                                 |
|----|-----------------------------------------------|------------------------------------------------------------------------------------------|
| 1  | Why Distributed Training                      | Memory + FLOP calculator; single-GPU OOM as we scale (sizing a Llama-3 70B run)         |
| 2  | Data Parallelism                              | Tiny GPT on TinyStories trained DP across devices (every modern LLM is this, scaled up) |
| 3  | Collectives & `shard_map`                     | Hand-rolled Ring-AllReduce vs `lax.psum` (why NCCL/SHARP exist at 16k-GPU scale)        |
| 4  | FSDP / ZeRO                                   | Convert Ch 2's trainer through ZeRO-1 → 2 → 3 (how Llama-2/3 fit in GPU memory)         |
| 5  | Tensor Parallelism                            | Column/row-parallel MLP + head-parallel attention (Megatron-LM, Llama-3 405B FFN)       |
| 6  | Embedding Parallelism (DLRM/DCN)              | Mini DLRM with row-sharded embeddings + all-to-all (Meta DLRM / Google DCN-v2 ad ranking)|
| 7  | Pipeline Parallelism                          | 4-stage pipelined transformer with `lax.scan`: GPipe then 1F1B (DeepSeek-V3 on H800s)   |
| 8  | Sequence Parallelism & Long Context           | Ring attention from scratch (Gemini 1M, Claude 200K context)                            |
| 9  | Mixed Precision (bf16 → fp8)                  | bf16 + dynamic loss scaling, then fp8 sketch (H100/B200 + Transformer Engine)           |
| 10 | End-to-End Sharded Transformer                | Mini-Llama trainer: DP × TP × FSDP × bf16 (frontier-lab stack at small scale)           |
| 11 | Multi-host Training & Checkpointing           | Per-host data + Orbax sharded checkpoints; simulate node failure (MaxText on TPU pods)  |
| 12 | MoE & Expert Parallelism *(optional)*         | Mini-MoE block with all-to-all expert dispatch (Mixtral, DeepSeek-V3)                   |

## Layout

```
src/
  chapters/   # jupytext .py source for each chapter (edit here)
  exercises/  # exercise stubs / reference solutions per chapter
  solutions/  # (reserved for future solution files)
  judge.py    # shared exercise validator
notebooks/    # generated .ipynb notebooks (Colab-runnable)
tests/        # pytest test suite
scripts/      # notebook generation
```

## Setup

This project uses `uv`.

```bash
uv sync
```

Each chapter is a self-contained Jupyter notebook runnable on Google Colab.
