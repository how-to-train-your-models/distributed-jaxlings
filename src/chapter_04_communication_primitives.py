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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/how-to-train-your-models/distributed-jaxlings/blob/main/chapters/chapter_04_communication_primitives.ipynb)
#
# # Chapter 4: Communication Primitives
#
# > **Course: Distributed Training — From Concepts to JAX**
#
# ---
#
# ## Learning Objectives
#
# - Name and describe the six core collective operations
# - Implement Broadcast, Gather, Scatter, AllGather, ReduceScatter, and AllReduce
# - Calculate communication volume and bandwidth requirements
# - Map each collective to its use in distributed training
#

# %% [markdown]
# ---
# ## 1. The Collective Operations Zoo
#
# Collective operations involve **all ranks** in a communicator. The six core collectives:
#
# ```
# BROADCAST    root → all         [A]     → [A][A][A][A]
# REDUCE       all → root (sum)   [A][B][C][D] → [A+B+C+D]
# ALLREDUCE    all → all (sum)    [A][B][C][D] → [A+B+C+D] on every rank
# GATHER       all → root         [A][B][C][D] → root has [A,B,C,D]
# SCATTER      root → all (chunk) root:[A,B,C,D] → [A][B][C][D]
# ALLGATHER    all → all          [A][B][C][D] → [A,B,C,D] on every rank
# REDUCESCATTER all → all(chunk)  [A][B][C][D] → [A+..][B+..][C+..][D+..] one chunk each
# ```
#
# ### Where each collective appears in distributed training
#
# | Collective | Used in |
# |---|---|
# | AllReduce | DDP gradient sync, row-parallel linear output |
# | AllGather | ZeRO-3 parameter fetch, column-parallel output, sequence parallel |
# | ReduceScatter | ZeRO-2 gradient sync, sequence parallel input |
# | Broadcast | Model weight initialization from rank 0 |
# | Scatter | Distributing data shards to workers |
# | Reduce | Aggregating metrics to rank 0 |
#
# ### Bandwidth analysis
#
# For message size $S$ across $N$ ranks:
#
# | Collective | Data sent per rank | Bottleneck |
# |---|---|---|
# | Broadcast | $S$ (root sends $N-1$×) | Root outbound |
# | AllReduce | $2S(N-1)/N ≈ 2S$ | Balanced |
# | AllGather | $S(N-1)/N ≈ S$ | Balanced |
# | ReduceScatter | $S(N-1)/N ≈ S$ | Balanced |
#

# %% [markdown]
# ---
# ## Judge Setup
#

# %%
import numpy as np
from typing import List, Optional
from judge import Judge

judge = Judge("Chapter 4", default_tol=1e-6)
print("Judge ready!")


# %% [markdown]
# ---
# ## Exercise 1: Broadcast and Reduce
#
# Implement the two basic one-to-all and all-to-one collectives.
#
# TODO: Implement `broadcast` and `reduce`
#

# %%
def broadcast(buffers: List[np.ndarray], root: int) -> List[np.ndarray]:
    """
    Broadcast root's buffer to all ranks.
    
    Args:
        buffers: list of arrays, one per rank (only buffers[root] is used)
        root:    rank to broadcast from
    Returns:
        list where every element equals the original buffers[root]
    """
    # TODO: return a list where every element is a copy of buffers[root]
    pass


def reduce(buffers: List[np.ndarray], root: int, op=np.add) -> List[np.ndarray]:
    """
    Reduce all buffers to root using op (default: sum).
    
    Args:
        buffers: list of arrays, one per rank
        root:    rank that receives the result
        op:      reduction operator (np.add, np.maximum, etc.)
    Returns:
        list where buffers[root] holds the reduced result;
        all other elements are unchanged from input.
    """
    # TODO: compute the reduction of all buffers and place result at root
    # Non-root ranks keep their original values
    pass


# Tests
data = [np.array([1.0, 2.0]), np.array([3.0, 4.0]),
        np.array([5.0, 6.0]), np.array([7.0, 8.0])]

bc = broadcast(data, root=2)
judge.check("Ex1a: broadcast — rank 0 gets root value", bc[0], data[2])
judge.check("Ex1b: broadcast — all ranks equal",
            all(np.allclose(bc[i], data[2]) for i in range(4)), True)

rd = reduce(data, root=0)
judge.check("Ex1c: reduce — root has sum", rd[0], np.array([16.0, 20.0]))
judge.check("Ex1d: reduce — non-root unchanged", rd[1], data[1])


# %% [markdown]
# ---
# ## Exercise 2: Scatter and Gather
#
# TODO: Implement `scatter` and `gather`
#

# %%
def scatter(send_buf: np.ndarray, root: int, n_ranks: int) -> List[np.ndarray]:
    """
    Split send_buf (at root) evenly across all ranks.
    
    Args:
        send_buf: 1D array at root, length must be divisible by n_ranks
        root:     rank that holds the data to scatter
        n_ranks:  total number of ranks
    Returns:
        list of n_ranks arrays, each of length len(send_buf)//n_ranks
    """
    # TODO: split send_buf into n_ranks equal chunks
    pass


def gather(recv_bufs: List[np.ndarray], root: int) -> np.ndarray:
    """
    Gather each rank's buffer at root, concatenated in rank order.
    
    Args:
        recv_bufs: list of arrays, one per rank
        root:      rank to gather to (unused in this sim, just return full array)
    Returns:
        1D array — concatenation of all recv_bufs in rank order
    """
    # TODO: concatenate all buffers
    pass


# Tests
full = np.array([10., 20., 30., 40., 50., 60., 70., 80.])
chunks = scatter(full, root=0, n_ranks=4)

judge.check("Ex2a: scatter chunk size", chunks[0].shape, (2,))
judge.check("Ex2b: scatter rank 2 chunk", chunks[2], np.array([50., 60.]))

reassembled = gather(chunks, root=0)
judge.check("Ex2c: gather reconstructs original", reassembled, full)


# %% [markdown]
# ---
# ## Exercise 3: AllGather
#
# Every rank starts with its own chunk. After AllGather, **every rank** has all chunks concatenated.
#
# Used in: ZeRO-3 (gather weights before forward), column-parallel linear output, sequence parallel.
#
# TODO: Implement `allgather`
#

# %%
def allgather(buffers: List[np.ndarray]) -> List[np.ndarray]:
    """
    Every rank ends up with the concatenation of all buffers.
    
    Args:
        buffers: list of 1D arrays, one per rank
    Returns:
        list of n_ranks arrays, each identical: concatenation of all inputs
    """
    # TODO: each rank gets np.concatenate(buffers)
    pass


chunks = [np.array([1., 2.]), np.array([3., 4.]),
          np.array([5., 6.]), np.array([7., 8.])]
results = allgather(chunks)

expected_full = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
judge.check("Ex3a: AllGather rank 0 has all data", results[0], expected_full)
judge.check("Ex3b: AllGather rank 3 same as rank 0", results[3], results[0])
judge.check("Ex3c: AllGather — all ranks identical",
            all(np.allclose(results[i], results[0]) for i in range(4)), True)


# %% [markdown]
# ---
# ## Exercise 4: ReduceScatter
#
# ReduceScatter is the complement of AllGather: each rank starts with a **full-size** buffer. The buffers are summed, then the result is **split** and each rank gets one chunk.
#
# Used in: ZeRO-2 gradient sync, sequence parallel (before tensor-parallel region).
#
# Note: `AllReduce = ReduceScatter + AllGather`
#
# TODO: Implement `reducescatter`
#

# %%
def reducescatter(buffers: List[np.ndarray]) -> List[np.ndarray]:
    """
    Sum all buffers element-wise, then scatter evenly to all ranks.
    
    Args:
        buffers: list of n_ranks arrays, each of same shape (n,)
                 n must be divisible by n_ranks
    Returns:
        list of n_ranks arrays, each of shape (n//n_ranks,)
        rank i holds the i-th chunk of the element-wise sum
    """
    n_ranks = len(buffers)
    n = len(buffers[0])
    assert n % n_ranks == 0
    chunk_size = n // n_ranks

    # TODO: Step 1 — element-wise sum all buffers
    total = None  # TODO: np.sum(np.stack(buffers), axis=0)

    # TODO: Step 2 — scatter: rank i gets total[i*chunk_size:(i+1)*chunk_size]
    result = []  # TODO

    return result


# Each rank holds a full gradient vector
bufs = [
    np.array([1., 2., 3., 4.]),
    np.array([5., 6., 7., 8.]),
    np.array([9., 10., 11., 12.]),
    np.array([13., 14., 15., 16.]),
]
rs = reducescatter(bufs)

# Sum = [28, 32, 36, 40]; chunks: rank0=[28,32], rank1=[36,40]
judge.check("Ex4a: ReduceScatter rank 0 chunk", rs[0], np.array([28., 32.]))
judge.check("Ex4b: ReduceScatter rank 1 chunk", rs[1], np.array([36., 40.]))
judge.check("Ex4c: ReduceScatter chunk size",   rs[2].shape, (2,))

# Verify: AllGather of ReduceScatter = AllReduce
ag = allgather(rs)
expected_allreduce = np.array([28., 32., 36., 40.])
judge.check("Ex4d: RS + AG = AllReduce", ag[0], expected_allreduce)


# %% [markdown]
# ---
# ## Exercise 5: AllReduce = ReduceScatter + AllGather
#
# Implement AllReduce by composing ReduceScatter and AllGather (this is exactly what Ring-AllReduce does).
#
# TODO: Implement `allreduce_via_rs_ag`
#

# %%
def allreduce_via_rs_ag(buffers: List[np.ndarray]) -> List[np.ndarray]:
    """
    AllReduce by composing ReduceScatter then AllGather.
    All ranks end up with the element-wise sum of all inputs.
    """
    # TODO: Call reducescatter then allgather
    pass


np.random.seed(5)
grads = [np.random.randn(8) for _ in range(4)]
expected_sum = np.sum(grads, axis=0)

ar = allreduce_via_rs_ag(grads)

judge.check("Ex5a: AllReduce rank 0 = sum",  ar[0], expected_sum)
judge.check("Ex5b: AllReduce rank 3 = sum",  ar[3], expected_sum)
judge.check("Ex5c: AllReduce all equal",
            all(np.allclose(ar[i], ar[0]) for i in range(4)), True)


# %% [markdown]
# ---
# ## Exercise 6: Communication Volume Analysis
#
# Calculate the communication volume (bytes sent per rank) for each collective.
#
# TODO: Implement `comm_volume_bytes`
#

# %%
def comm_volume_bytes(
    collective: str,
    n_params: int,
    n_ranks: int,
    bytes_per_elem: int = 2  # BF16
) -> float:
    """
    Calculate bytes sent per rank for a given collective operation.
    
    Use the ring-algorithm optimal bandwidth formulas:
      allreduce:     2 * S * (N-1)/N   ≈ 2S   (S = total tensor bytes)
      allgather:     S * (N-1)/N       ≈ S
      reducescatter: S * (N-1)/N       ≈ S
      broadcast:     S                 (root sends S, others receive)
      reduce:        S                 (all send S to root)
    
    Args:
        collective:     one of 'allreduce','allgather','reducescatter','broadcast','reduce'
        n_params:       number of elements in the tensor
        n_ranks:        number of ranks
        bytes_per_elem: bytes per element (2 for BF16, 4 for FP32)
    
    Returns:
        bytes sent/received per rank
    """
    S = n_params * bytes_per_elem  # total tensor size in bytes
    
    # TODO: implement each case
    if collective == 'allreduce':
        return 0  # TODO: 2 * S * (n_ranks - 1) / n_ranks
    elif collective == 'allgather':
        return 0  # TODO: S * (n_ranks - 1) / n_ranks
    elif collective == 'reducescatter':
        return 0  # TODO: same as allgather
    elif collective == 'broadcast':
        return 0  # TODO: S
    elif collective == 'reduce':
        return 0  # TODO: S
    else:
        raise ValueError(f"Unknown collective: {collective}")


# GPT-3 175B params, BF16, 1024 ranks
N = 175_000_000_000
n = 1024

ar_vol = comm_volume_bytes('allreduce', N, n, bytes_per_elem=2)
ag_vol = comm_volume_bytes('allgather', N, n, bytes_per_elem=2)

print(f"AllReduce per rank:     {ar_vol/1e9:.2f} GB")
print(f"AllGather per rank:     {ag_vol/1e9:.2f} GB")
print(f"AllReduce ≈ 2× AllGather: {abs(ar_vol / ag_vol - 2) < 0.01}")

# AllReduce should be ~2 × message size (≈ 2 × 350GB = 700GB)
judge.check("Ex6a: AllReduce volume ≈ 2S",
            ar_vol / (N * 2), 2.0, tol=0.01)  # close to 2 * S
judge.check("Ex6b: AllGather == ReduceScatter volume",
            comm_volume_bytes('allgather', N, n),
            comm_volume_bytes('reducescatter', N, n))

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
# 1. **AllReduce = ReduceScatter + AllGather** — understanding this decomposition is key to ZeRO and Ring-AllReduce.
# 2. **AllReduce volume ≈ 2S** per rank — unavoidable lower bound for gradient synchronization.
# 3. **AllGather and ReduceScatter each cost ≈ S** — ZeRO-3 replaces one AllReduce with RS+AG but adds an extra AllGather for weights, increasing total volume by 50%.
# 4. **Choose collectives wisely**: unnecessary communication is the #1 scaling bottleneck in large clusters.
#
# ---
# **Next:** [Chapter 5 — Mixed Precision & Fault Tolerance](./chapter_05_mixed_precision_checkpointing.ipynb)
#
