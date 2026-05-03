# Course Plan

A writing reference. For each chapter: goal, motivating build, sections, concepts, JAX/Equinox APIs introduced, and exercises.

Conventions:
- **Goal**: one-sentence outcome the reader should achieve.
- **Build**: the concrete artifact the chapter produces (mirrors a real production system).
- **Sections**: the chapter's outline, in order.
- **APIs**: only those *first introduced* in this chapter. Earlier APIs are reused freely.
- **Exercises**: 4–5 graded tasks. Aim for one per major section.

---

## Chapter 1 — Why Distributed Training

**Goal.** Reader can size a training run on paper and explain why distribution becomes mandatory above a certain scale.

**Build.** A model-sizing calculator: takes `(n_layers, hidden, n_heads, seq, batch)` → params, training memory, FLOPs/step, single-GPU OOM verdict. Show an OOM as we scale a transformer.

**Real system.** Sizing a Llama-3 70B (or DLRM with 10TB embedding tables) — neither fits on a single H100.

**Sections.**
1. The scale problem — modern model sizes vs HBM ceilings
2. Memory math — params, gradients, optimizer state, activations (the 16-bytes/param rule)
3. Compute math — `6ND` FLOPs, throughput vs hardware
4. Hardware tour — HBM, NVLink, InfiniBand, TPU pods
5. The four parallelism axes — DP, TP, PP, EP (preview only)
6. Build: the sizing calculator

**Concepts.** Scaling laws, memory budget, FLOP budget, hardware roofs, parallelism preview.

**APIs introduced.** None (pure theory + light `jax.numpy` for the calculator).

**Exercises.**
1. Compute params + memory for Llama-7B / 70B / 405B / GPT-3 / DLRM.
2. Estimate wall-clock training time given a target FLOPs/s.
3. For a given bottleneck (param OOM, batch OOM, activation OOM), pick the right parallelism axis.
4. Implement the sizing calculator from scratch.

---

## Chapter 2 — Data Parallelism

**Goal.** Reader can train an Equinox language model across multiple devices using DP, with mesh + sharding spec — no `pmap`.

**Build.** A **tiny GPT** (~1–2 layers, hidden 128–256, ~1–5M params) trained on **TinyStories**. Deliberately *naive* architecture: sinusoidal positional encoding, vanilla MHA via `eqx.nn.MultiheadAttention`, LayerNorm, GeLU MLP. Train DP across N devices, throughput chart vs device count, generate a few sample stories at the end.

**Real system.** Every modern LLM (Claude, GPT, Gemini) is this same shape, scaled up and trained DP. TinyStories (Eldan & Li, 2023) showed that even 1–10M-param transformers can produce coherent prose, so the "wow" lands without needing scale.

> Why this exact model? It's the simplest thing that's recognizably an LLM. Ch 10 revisits with a *production* mini-Llama (RoPE, GQA, RMSNorm, SwiGLU, sharded init) — clearly a different artifact, so no overlap.

**Sections.**
1. Why DP is the natural starting point — replicate model, shard data
2. JAX functional model — pure functions, `jit`, `grad`, `vmap`
3. Pytrees and PRNG — `jax.tree`, `jax.random.split`
4. Equinox: modules as pytrees — `eqx.Module`, `eqx.filter_jit`, `eqx.filter_grad`, `eqx.partition`/`combine`
5. The tiny GPT — embedding, sinusoidal positions, transformer block (`eqx.nn.MultiheadAttention` + LayerNorm + MLP), LM head
6. Meshes & sharding — `Mesh`, `PartitionSpec`, `NamedSharding`, `jax.device_put`
7. The DP pattern — params replicated `P()`, batch sharded `P('data')`, implicit AllReduce via `jit`
8. Train loop — `train_step` under `jit`; cross-entropy loss on next-token prediction
9. Sync vs async, large-batch tradeoffs (theory)
10. Sidebar: `pmap` is legacy; do not use
11. Build: tiny GPT DP trainer + sample generations + throughput benchmark

**Concepts.** DP, gradient averaging, synchronous SGD, large-batch tradeoffs, replication, next-token prediction, Noam shape suffix in practice.

**APIs introduced.** `jit`, `grad`, `vmap`, pytrees, `jax.random.split`, `eqx.Module`, `eqx.filter_jit`, `eqx.filter_grad`, `eqx.partition`/`combine`, `eqx.nn.MultiheadAttention`, `eqx.nn.LayerNorm`, `eqx.nn.Embedding`, `eqx.nn.Linear`, `Mesh`, `PartitionSpec`, `NamedSharding`, `jax.device_put`, `jax.debug.visualize_array_sharding`.

**Exercises.**
1. Build the transformer block as an `eqx.Module` using `eqx.nn.MultiheadAttention`.
2. Compose blocks + token/positional embeddings + LM head into the tiny GPT.
3. Write `train_step` with `eqx.filter_value_and_grad` and cross-entropy loss.
4. Construct a 1D `data` mesh; shard the batch, replicate params; visualize with `jax.debug.visualize_array_sharding`.
5. Throughput benchmark — sweep 1, 2, 4, 8 devices; sample a story from the trained model.

---

## Chapter 3 — Collectives & `shard_map`

**Goal.** Reader knows every standard collective, can implement them with `shard_map`, and can reason about bandwidth.

**Build.** Hand-rolled Ring-AllReduce as ReduceScatter + AllGather, benchmarked against `lax.psum`.

**Real system.** At 16k-GPU scale, AllReduce eats 30–40% of step time — NCCL and SHARP exist for exactly this reason.

**Sections.**
1. Why collectives matter at scale
2. The collective zoo — Broadcast, Reduce, Scatter, Gather, AllGather, ReduceScatter, AllReduce, AllToAll, Permute
3. Bandwidth analysis — per-rank cost, why AllReduce ≈ 2S
4. Implicit collectives (`jit` + PartitionSpec) vs explicit (`shard_map` + `lax.*`) — when to choose which
5. `shard_map` — per-device function, axis names, in_specs/out_specs
6. JAX collective APIs — `lax.psum`, `lax.all_gather`, `lax.psum_scatter`, `lax.ppermute`, `lax.all_to_all`
7. Ring algorithms — Ring-AllReduce decomposition
8. GSPMD intuition — the compiler inserts collectives for you
9. Build: hand-rolled Ring-AllReduce vs `lax.psum`

**Concepts.** All collectives, bandwidth analysis, ring decomposition, implicit vs explicit SPMD, GSPMD propagation.

**APIs introduced.** `shard_map`, `lax.psum`, `lax.all_gather`, `lax.psum_scatter`, `lax.ppermute`, `lax.all_to_all`.

**Exercises.**
1. Implement Broadcast, Reduce, Scatter, Gather under `shard_map`.
2. Implement AllGather and ReduceScatter.
3. Compose them to recover AllReduce; verify against `lax.psum`.
4. Implement a ring shift with `lax.ppermute`.
5. Benchmark hand-rolled Ring-AllReduce vs `lax.psum`; explain the gap.

---

## Chapter 4 — FSDP / ZeRO

**Goal.** Reader can shard optimizer state, gradients, and parameters across devices, and understands the memory/comm tradeoffs at each stage.

**Build.** Take Ch 2's tiny GPT trainer; convert progressively through ZeRO-1 → ZeRO-2 → ZeRO-3 (FSDP). Measure peak HBM drop at each stage.

**Real system.** Meta open-sourced FSDP because Llama-2/3 optimizer state alone (Adam = 12 bytes/param) exceeded GPU memory. Mistral, DeepSeek, Qwen all train with FSDP-style sharding today.

**Sections.**
1. The DP memory wall — why pure DP can't train large models
2. ZeRO stages overview — what's sharded at each level
3. ZeRO-1 — sharded optimizer state (biggest single saving, no extra comm)
4. ZeRO-2 — also shard gradients (ReduceScatter instead of AllReduce)
5. ZeRO-3 / FSDP — also shard params; the gather/release pattern
6. Comm cost analysis — FSDP adds AllGather but trades off with replication
7. Sharding pytrees in JAX — `jax.tree.map` to apply NamedShardings to an Equinox module
8. Gradient checkpointing (rematerialization) — `jax.checkpoint`, the memory/compute tradeoff
9. Build: progressive ZeRO conversion + memory measurement

**Concepts.** ZeRO-1/2/3, FSDP gather/release, comm volume tradeoff, rematerialization.

**APIs introduced.** `jax.checkpoint` (rematerialization), tree-level sharding patterns over Equinox modules.

**Exercises.**
1. Convert Ch 2's tiny GPT trainer to ZeRO-1 (shard optimizer state).
2. Empirically verify memory savings (peak HBM per device).
3. Add ZeRO-2 (shard grads with `ReduceScatter`).
4. Add ZeRO-3 (shard params with gather-on-use, release-after-use).
5. Apply `jax.checkpoint` to a layer; measure the memory/compute tradeoff.

---

## Chapter 5 — Tensor Parallelism

**Goal.** Reader can split individual matmuls across devices for transformer MLP and attention.

**Build.** Column-parallel + row-parallel MLP, then head-parallel attention, dropped into a small transformer block. Combined with DP on a 2D mesh.

**Real system.** Llama-3 405B's MLP weights are ~200 GB per layer; Megatron-LM exists because no GPU holds a single FFN of a frontier model.

**Sections.**
1. When TP becomes necessary — single-layer params exceed HBM
2. Column-parallel linear — no comm in forward, AllReduce in backward
3. Row-parallel linear — AllReduce in forward, no comm in backward
4. The Megatron MLP pattern — column-parallel up + row-parallel down (one AllReduce per MLP)
5. Attention TP — heads as the parallel dim; Q/K/V split, output projection row-parallel
6. Equinox modules with sharded leaves
7. `with_sharding_constraint` — annotating intermediate activations
8. Combining DP × TP — 2D mesh with named axes
9. Build: a TP transformer block on a 2D mesh

**Concepts.** Column/row TP, fused MLP pattern, head-parallel attention, 2D meshes, activation sharding hints.

**APIs introduced.** `with_sharding_constraint`, einsum sharding rules.

**Exercises.**
1. Column-parallel `Linear` as `eqx.Module`.
2. Row-parallel `Linear`.
3. Fused TP MLP (column-up + row-down) with one AllReduce.
4. TP attention block (head-parallel Q/K/V + row-parallel output).
5. Combine DP × TP on a 2D mesh; verify sharding placement.

---

## Chapter 6 — Embedding Parallelism (DLRM / DCN)

**Goal.** Reader understands recommendation-scale parallelism — sparse lookups, table-wise sharding, all-to-all dispatch.

**Build.** A mini DLRM: row-sharded embedding tables + all-to-all to assemble per-batch features + dense top tower. Optionally a DCN-v2 cross network.

**Real system.** Meta DLRM and Google DCN-v2 power ad ranking — billions of dollars/year hinge on this pattern. Embedding tables are sharded *row-wise across hosts*; lookups need *all-to-all*, not all-reduce. A different parallelism pattern from LLMs.

**Sections.**
1. Recommendation models 101 — embedding lookups + dense tower (DLRM, DCN-v2)
2. Why this is different from LLM TP — sparse, lookup-heavy, table-shaped
3. Sharding strategies — table-wise, row-wise, column-wise
4. The all-to-all pattern — each device looks up its local rows, then exchanges so each device has all features for its local batch
5. The DLRM bottom MLP + interaction layer + top MLP
6. DCN cross network — explicit feature interactions
7. Sharded embedding modules in Equinox
8. Mixing parallelism — embeddings model-parallel, dense tower data-parallel
9. Build: mini DLRM end-to-end on synthetic CTR data

**Concepts.** Table-wise sharding, sparse lookups, all-to-all routing, mixed sparse/dense parallelism.

**APIs introduced.** `lax.all_to_all`, sharded `gather`/`take`, sparse-friendly `shard_map` patterns.

**Exercises.**
1. Build a row-sharded embedding table around `eqx.nn.Embedding`.
2. All-to-all to assemble per-batch features on each device.
3. Implement DLRM bottom MLP + dot-product interaction + top MLP.
4. Implement DCN cross layers.
5. Train mini DLRM on a synthetic CTR task; verify multi-device speedup.

---

## Chapter 7 — Pipeline Parallelism

**Goal.** Reader can build a pipelined transformer with GPipe and 1F1B scheduling using `lax.scan`, and reason about the pipeline bubble.

**Build.** A 4-stage pipelined transformer with `lax.scan`. GPipe scheduling first, then 1F1B. Visualize the bubble shrinking as micro-batches grow.

**Real system.** DeepSeek-V3 (671B) was trained on export-restricted H800s with weak NVLink — pipelining (and DualPipe) is the only reason it fit at all. Same story for any cross-node frontier training.

**Sections.**
1. When TP isn't enough — cross-node bandwidth limits
2. Pipeline basics — stages, micro-batches, the bubble
3. Bubble fraction — `(p−1) / (m + p − 1)`
4. GPipe scheduling — fill, steady state, drain
5. 1F1B scheduling — interleaved forward/backward to cap activation memory
6. Implementation in JAX — `lax.scan` over micro-batches; `lax.ppermute` for stage handoff
7. Memory tradeoffs — peak activations vs throughput
8. Sidebar: DualPipe (DeepSeek-V3) overview
9. Build: pipelined transformer with both schedules

**Concepts.** Pipeline stages, bubble fraction, GPipe vs 1F1B, micro-batching, activation memory.

**APIs introduced.** `lax.scan`, `lax.ppermute` for stage handoff (deeper use than Ch 3).

**Exercises.**
1. Compute bubble fraction for given `p`, `m`; chart it.
2. Build a 4-stage pipeline with `lax.scan` (forward only first).
3. Add GPipe schedule (forward all, then backward all).
4. Add 1F1B schedule (interleaved).
5. Measure bubble empirically; compare schedules' peak memory.

---

## Chapter 8 — Sequence Parallelism & Long Context

**Goal.** Reader understands how to shard the sequence dimension and can implement ring attention.

**Build.** Ring attention from scratch on a small transformer; show activation memory scales sub-linearly with sequence length.

**Real system.** Gemini 2.5 (1M+ context) and Claude (200K) cannot fit attention activations on any single GPU — ring attention is the only way to serve them.

**Sections.**
1. The long-context problem — attention activations are O(s²); 1M tokens fit nowhere
2. Sequence parallelism for non-attention ops (LayerNorm, dropout) — easy: shard along seq
3. Why attention is the hard part — every Q row must see every K row
4. Ring attention — pass K/V blocks around the ring; accumulate attention with online softmax
5. Online softmax — numerically stable streaming variant
6. Implementation with `lax.ppermute`
7. Memory analysis — activation memory now O(s² / D)
8. Comparison to Flash Attention (single-device, IO-aware)
9. Build: ring attention block

**Concepts.** SP, ring attention, online softmax, activation memory math.

**APIs introduced.** Heavy use of `lax.ppermute` along seq axis; `with_sharding_constraint` along seq.

**Exercises.**
1. Apply SP to LayerNorm and dropout.
2. Implement online (streaming) softmax; verify equivalence to standard softmax.
3. Implement ring attention forward pass.
4. Verify equivalence to standard attention on small inputs.
5. Measure activation memory vs sequence length; chart sub-linear scaling.

---

## Chapter 9 — Mixed Precision (bf16 → fp8)

**Goal.** Reader can convert a working trainer to bf16 with NaN-safe training, and understands fp8 + accumulation patterns.

**Build.** Convert a trainer to bf16 + dynamic loss scaling. Sketch fp8 with fp32 accumulation. Verify convergence is preserved.

**Real system.** H100/B200 with NVIDIA Transformer Engine cuts training cost ~30% by going to fp8. Llama-3, Nemotron, recent DeepSeek runs all use fp8.

**Sections.**
1. The precision spectrum — fp32, bf16, fp16, fp8 (E4M3, E5M2)
2. Why bf16 won — same exponent range as fp32, no loss scaling needed
3. fp16 + dynamic loss scaling — historical pattern (still useful intuition)
4. fp8 — H100/B200 era; per-tensor scaling factors; fp32 accumulation
5. What stays in fp32 — master weights, optimizer state, loss, accumulators
6. JAX dtype mechanics — `jnp.bfloat16`, `jax.lax.Precision`, `jnp.promote_types`
7. Numerical pitfalls — gradient underflow, overflow, NaN handling
8. Build: bf16 trainer with NaN-safe `train_step`

**Concepts.** bf16/fp16/fp8, dynamic loss scaling, fp32 accumulation, master weights, NaN handling.

**APIs introduced.** `jax.lax.Precision`, `jnp.bfloat16`, dtype policies, `jnp.promote_types`.

**Exercises.**
1. Detect fp16 overflow/underflow in toy ops.
2. Implement a dynamic loss scaler (skip step on NaN, halve scale; double after N clean steps).
3. Convert a model to bf16 — decide which tensors stay fp32.
4. Add NaN/Inf detection to `train_step`; skip the step on detection.
5. Sketch an fp8 matmul with fp32 accumulation and per-tensor scale.

---

## Chapter 10 — End-to-End Sharded Transformer

**Goal.** Reader builds a complete distributed transformer trainer combining DP × TP × FSDP × bf16 in one Equinox + Optax recipe.

**Build.** A mini-Llama trainer — same recipe Anthropic / Mistral / DeepSeek use, just smaller meshes.

**Real system.** The actual frontier-lab training stack pattern.

**Sections.**
1. Architecture — RMSNorm, RoPE, SwiGLU MLP, GQA attention (Llama-style)
2. Mesh design — 2D or 3D mesh with named axes (`data`, `model`, optionally `pipeline`)
3. Sharding strategy — which mesh axis maps to which weight dim
4. Optax — composable optimizer chains (AdamW + grad clip + warmup-cosine LR)
5. The `train_step` — combining DP gradient avg, TP collectives, FSDP gather/release, bf16 cast
6. Initialization at scale — sharded init (never materialize the full param tree on one device)
7. Evaluation — perplexity loop on a held-out shard
8. Build: end-to-end mini-Llama trainer

**Concepts.** Synthesis of all prior chapters; sharded init; production trainer structure.

**APIs introduced.** None new — full Equinox + Optax integration. Optax: `optax.adamw`, `optax.chain`, `optax.warmup_cosine_decay_schedule`, `optax.clip_by_global_norm`.

**Exercises.**
1. Implement RMSNorm, RoPE, SwiGLU, GQA as Equinox modules.
2. Compose them into a sharded transformer block.
3. Wire up Optax (AdamW + grad clip + warmup-cosine).
4. Sharded init — initialize weights directly into their shards, no full materialization.
5. End-to-end train + eval loop; chart loss curve.

---

## Chapter 11 — Multi-host Training & Checkpointing

**Goal.** Reader can run a multi-host JAX job, load data per-host, and save/restore Orbax sharded checkpoints with fault recovery.

**Build.** Per-host data pipeline + Orbax sharded checkpoints; simulate a node failure mid-training and resume from the last checkpoint.

**Real system.** Google MaxText runs on TPU v5p pods with 1000s of hosts; Meta Llama training spans weeks on 24k H100s. A single node failure without proper checkpointing = hours of lost work × $1000s/hour.

**Sections.**
1. The multi-host model — one process per host, single global mesh, XLA collectives
2. Process initialization — `jax.distributed.initialize`, `process_index`, `process_count`
3. Global vs local arrays — what every process sees
4. Per-host data loading — each process reads its own shard; assemble via `make_array_from_process_local_data`
5. Building a global mesh that spans hosts
6. Orbax — async sharded checkpointing
7. Fault recovery — detect failure, restart, resume from last checkpoint
8. Production loop structure — init, train, periodic ckpt, eval, recovery
9. Build: multi-host trainer with Orbax + simulated node-failure recovery

**Concepts.** Multi-process model, global vs local arrays, async checkpointing, fault recovery.

**APIs introduced.** `jax.distributed.initialize`, `jax.process_index`, `jax.process_count`, `jax.make_array_from_process_local_data`, Orbax (`CheckpointManager`, `PyTreeCheckpointer`, async options).

**Exercises.**
1. Build a global mesh across (simulated) hosts.
2. Per-host data loader + assemble a global batch.
3. Save a checkpoint with Orbax (sharded + async).
4. Restore from checkpoint into the same mesh.
5. Simulate failure mid-training; verify exact resume (loss curve continuity).

---

## Chapter 12 — MoE & Expert Parallelism *(optional)*

**Goal.** Reader can build a mini-MoE block with expert parallelism and all-to-all token dispatch.

**Build.** A mini-MoE block: top-k router + expert MLPs + weighted combine, with experts sharded across devices and tokens dispatched via all-to-all.

**Real system.** Mixtral-8x7B and DeepSeek-V3 (256 experts) — sparse activation gives ~5× parameter efficiency. Routing tokens *across devices* is its own parallelism axis.

**Sections.**
1. Why MoE — sparse activation; parameter-efficient scaling
2. The MoE block — router (top-k), expert MLPs, weighted combine
3. Load balancing — auxiliary loss, capacity factor, token dropping
4. Expert parallelism — experts sharded across devices; tokens routed via all-to-all
5. The all-to-all pattern — dispatch tokens to expert devices, gather results back
6. Capacity overflow handling
7. Combining EP with DP/TP/FSDP
8. Build: mini-MoE block on a 2D mesh

**Concepts.** Sparse routing, top-k, capacity factor, expert parallelism, all-to-all routing.

**APIs introduced.** `lax.all_to_all` (production-grade use), advanced sharding patterns.

**Exercises.**
1. Implement a top-k router with softmax gating.
2. Compute the load-balancing auxiliary loss.
3. All-to-all dispatch: route tokens to their assigned expert devices.
4. All-to-all gather: collect expert outputs back to original token positions.
5. Train a mini-MoE on a toy task; chart per-expert utilization.

---

## Cross-cutting conventions

- **Tensor naming**: Noam shape suffix notation, `x` separator (per CLAUDE.md). E.g., `inputs_BxLxD`, `logits_BxLxV`.
- **Standard suffixes**: B (batch), S (seq), E (embed), H (heads), D (head dim), V (vocab), Dh (head_dim/2 for RoPE), BS (batch×seq flat), Sq/Sk (asymmetric attention).
- **Einsum**: prefer `jnp.einsum` over chained reshape/matmul where it clarifies intent.
- **Module style**: every model is an `eqx.Module`. Sharding lives in a parallel pytree of `NamedSharding`s with the same structure.
- **Train step style**: every chapter's `train_step` is `eqx.filter_jit`-wrapped; gradients via `eqx.filter_value_and_grad`.
- **Mesh axis names**: `data`, `model`, `seq`, `pipeline`, `expert` — consistent across chapters.
