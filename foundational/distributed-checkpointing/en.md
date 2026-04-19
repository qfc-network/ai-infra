# Async Checkpointing and PyTorch DCP

- **Relevant systems**: PyTorch Distributed Checkpoint (`torch.distributed.checkpoint`), DeepSpeed ZeRO checkpoint, Megatron-Core async checkpoint, MegaScale
- **PyTorch DCP stable since**: PyTorch 2.1 (2023)
- **Links**: [PyTorch DCP docs](https://pytorch.org/docs/stable/distributed.checkpoint.html) · [DCP design RFC](https://github.com/pytorch/pytorch/issues/88083) · [DeepSpeed zero_to_fp32](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/zero_to_fp32.py)

## TL;DR

At frontier scale, checkpointing is not a housekeeping task — it is a recurring idle-time tax on every GPU in the job. A 70B BF16 model carries ~420 GB of checkpoint data (weights + Adam state); a 405B model, ~2.4 TB. Written synchronously to object storage at 10 GB/s aggregate, these require 42 s and 240 s of dead GPU time respectively. Async in-memory checkpointing eliminates the GPU stall by staging data to CPU DRAM on a background thread while training continues. PyTorch's `torch.distributed.checkpoint` (DCP) standardizes the sharded save/load API, handles resharding across topology changes, and provides pluggable storage backends. DeepSpeed ZeRO, Megatron-Core, and MegaScale all implement the same async pattern, differing mainly in shard granularity and CPU memory management.

## Why Checkpointing Is a Bottleneck

The checkpoint size for an Adam-trained model in BF16/FP32 mixed precision breaks down as follows:

```
model weights (BF16):         2 bytes × P parameters
gradients (BF16):             2 bytes × P          [transient, not checkpointed]
Adam m  (FP32 master copy):   4 bytes × P
Adam v  (FP32 master copy):   4 bytes × P
FP32 master weights:          4 bytes × P
─────────────────────────────────────────────────
Total checkpoint:            ≈ 10 bytes × P
  → 70B model:   10 × 70×10⁹  ≈ 700 GB  (BF16 weights only: 140 GB)
  → 405B model:  10 × 405×10⁹ ≈ 4.0 TB
```

In practice, optimizer state is the dominant term. For a 70B model with Adam, the optimizer state alone (m + v + master weights) is ~560 GB. Many frameworks checkpoint only BF16 weights + optimizer state, landing at roughly 420 GB. For 405B the total is approximately 2.4 TB.

Now consider the idle-time arithmetic. If aggregate write bandwidth to object storage is 10 GB/s (a reasonable figure for a large NFS or distributed object store under parallel write load), synchronous checkpointing costs:

```
70B:   420 GB / 10 GB/s  = 42 s  per checkpoint
405B: 2400 GB / 10 GB/s  = 240 s per checkpoint
```

At 1 iteration per second (typical for large models), checkpointing every 500 iterations with synchronous writes gives:

```
idle fraction = 42 s / 500 s = 8.4%   (70B)
idle fraction = 240 s / 500 s = 48%   (405B — clearly unacceptable)
```

This is why async checkpointing is not optional at scale above ~100B parameters.

## Synchronous vs Asynchronous Checkpointing

**Synchronous** checkpointing is the naive baseline: training halts, all ranks write their shard, training resumes. The GPU sits idle for the entire write duration. Correct, simple, and untenable above a few hundred billion parameters.

**Async in-memory** checkpointing eliminates the GPU stall:

1. At checkpoint step N, each rank copies its parameter and optimizer state tensors from GPU HBM to CPU DRAM. This copy takes ~1–3 seconds (H100 NVLink bandwidth is ~900 GB/s to host; PCIe Gen5 is ~64 GB/s — PCIe is the realistic constraint for most systems).
2. Training immediately resumes on the GPU, now at step N+1.
3. A background thread serializes CPU DRAM tensors and writes them to object storage. This write happens in parallel with training steps N+1, N+2, ... and completes before the next checkpoint at step N+K.

The GPU stall is replaced by a 1–3 s copy latency to CPU DRAM. That cost is paid once; the background write is free from the GPU's perspective. The pattern is used by DeepSpeed ZeRO (in-memory checkpoint via `checkpoint_activations=True` equivalent for state), Megatron-Core's `AsyncCheckpointingDataWriterWrapper`, and MegaScale (described in the ByteDance system paper as "in-memory checkpointing with warm spares").

**Memory cost**: CPU DRAM must hold one full checkpoint copy. For a 70B model, that is ~420 GB of CPU RAM. Frontier training servers (H100 DGX nodes) typically have 1–2 TB of DRAM, so this is feasible. At 405B scale, the ~2.4 TB requirement becomes tight — some systems carry only the latest checkpoint in DRAM, or compress before CPU-side storage.

## PyTorch Distributed Checkpoint (DCP)

`torch.distributed.checkpoint` is the PyTorch 2.x API for sharded, reshardable checkpoint save and load. Its key design principle: **each rank writes only its local shard**. There is no rank-0 gather bottleneck — the approach that made vanilla `torch.save` unusable above ~10B parameters (rank 0 would need to materialize the entire model in memory).

### Save

```python
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemWriter

state = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
}
dist_cp.save(
    state_dict=state,
    storage_writer=FileSystemWriter("/checkpoint/step-1000"),
)
```

Each rank writes only its owned parameter shards. The output is a directory containing:

- `.metadata`: a JSON-like manifest listing all tensor names, shapes, dtypes, and the byte ranges across the data files where each shard lives.
- `.distcp` data files: one or more per rank per tensor group, containing the raw tensor bytes. Total file count scales as `O(ranks × tensor_groups)`.

### Load with Resharding

The decisive advantage of DCP over per-rank `torch.save` files: `dist_cp.load` can load a checkpoint saved with one topology into a different topology. A checkpoint written with TP=4 can be resumed with TP=8 — DCP reads the `.metadata` manifest, figures out how the old shards map to the new shard layout, and streams only the needed byte ranges from storage:

```python
dist_cp.load(
    state_dict=state,
    storage_reader=FileSystemReader("/checkpoint/step-1000"),
)
```

The resharding logic is handled by DCP's `DefaultLoadPlanner`, which computes the tensor-chunk-to-rank mapping from the metadata and issues targeted reads. No extra code is needed in the training script to handle topology changes.

### Storage Backends

DCP uses pluggable storage writers/readers:

- `FileSystemWriter` / `FileSystemReader`: local or NFS paths; uses Python `os` for file I/O.
- `S3StorageWriter` / `S3StorageReader`: writes directly to S3-compatible object storage.
- `FSSpecStorageWriter`: uses `fsspec` for any filesystem supported by that library (GCS, Azure Blob, HDFS).

The backend is swapped without changing any other checkpoint code — useful when moving from NFS-based development to S3-based production.

### Async Wrapper

DCP provides `AsyncCheckpointingDataWriterWrapper`, which wraps any `StorageWriter` and implements the async pattern:

```python
from torch.distributed.checkpoint import AsyncCheckpointingDataWriterWrapper

async_writer = AsyncCheckpointingDataWriterWrapper(
    FileSystemWriter("/checkpoint/step-1000"),
    async_snapshot_timeout=180,  # seconds
)
dist_cp.save(state_dict=state, storage_writer=async_writer)
# Returns immediately; write completes in background thread
```

After `save` returns, training continues. The wrapper stages tensors to CPU in the main call and delegates the file write to a background thread. `async_snapshot_timeout` sets a deadline — if the background write has not completed before the next checkpoint, the framework will wait for it to finish before taking the new snapshot, preventing checkpoint overlap.

## ZeRO Sharded Checkpointing

Under ZeRO-3 (or FSDP full shard), each DP rank owns only `1/N` of optimizer state and parameters. Checkpointing in this regime means each rank saves its own shard — an organic fit for DCP's sharded save model. On resume, DCP reassembles the shards. The checkpoint is consistent and correct without any rank needing to materialize the full model.

The `zero_to_fp32.py` utility (DeepSpeed) handles the special case of *inference*: given a ZeRO-3 sharded checkpoint directory, it concatenates the shards, strips the optimizer state, and writes a single contiguous FP32 weight file suitable for inference frameworks that do not understand ZeRO sharding. This conversion is CPU-only and can take 10–30 minutes for a 70B model but is a one-time cost.

## Recovery Bandwidth and Warm Spare Nodes

Recovery time from checkpoint on failure is determined by read bandwidth:

```
70B (420 GB):   420 GB / 10 GB/s  = 42 s  (from object storage)
405B (2.4 TB): 2400 GB / 10 GB/s  = 240 s (from object storage)
```

Four minutes of GPU idle time on a 16,000 H100 cluster costs roughly $5,000–$10,000 in GPU-hours at prevailing cloud rates. For long runs — Llama 3's 54-day run on ~16,000 GPUs — minimizing recovery time matters.

**Warm spare nodes** solve this: a small pool of standby nodes continuously receives the latest checkpoint into CPU DRAM via a reliable distributed write (e.g., RDMA). When a training node fails, a warm spare joins the job, already holding the checkpoint in DRAM. Resume takes seconds, not minutes. MegaScale uses this pattern explicitly.

The tradeoff: warm spare nodes consume cluster resources continuously (typically 1–5% of the training pool). For large enough jobs and high failure rates (H100 clusters can have GPU failure rates of ~1 per day at 10,000-node scale), the economics favor warm spares.

## Checkpoint Frequency Optimization

The optimal checkpoint interval K (in iterations) balances two competing costs:

```
Expected wasted compute on failure  = (K/2) × (GPUs × GPU-hours per iter)
Checkpointing I/O overhead          = checkpoint_size / (K × iter_time × write_bw)
```

Solving for the minimum total cost:

```
K_optimal = sqrt(checkpoint_size / (iter_time × write_bw × failure_rate × GPU_cost))
```

For a 70B model, 1 iter/s, 10 GB/s write, daily failure probability 0.01 on the full cluster (conservative for 1000 GPUs), and GPU cost $3/hr:

```
K_optimal ≈ sqrt(420 × 10⁹ / (1 × 10 × 10⁹ × 0.01 × 3/3600)) ≈ 800 iterations
```

With async checkpointing, the I/O term effectively vanishes (GPU idle time is near zero), so K can be set purely by recovery cost — typically 200–1000 iterations depending on failure rates observed in the cluster.

## Engineering Tradeoffs

| Approach | GPU idle time per checkpoint | CPU DRAM required | Recovery time | Reshardable | Implementation complexity |
|---|---|---|---|---|---|
| Synchronous full write | High: proportional to checkpoint size / write_bw (42 s for 70B) | Minimal: tensors stay on GPU | Proportional to checkpoint size / read_bw | With DCP, yes | Low |
| Async in-memory | ~1–3 s (GPU-to-CPU copy only) | Full checkpoint size in DRAM (~420 GB for 70B) | Same as sync if from storage; seconds with warm spares | With DCP, yes | Medium: background thread, timeout management |
| ZeRO sharded (per-rank shard) | Low: each rank writes only its shard; parallelizes naturally | Per-rank shard only (420 GB / N) | Requires all shards present; DCP handles reassembly | Yes, DCP's core feature | Medium: shard manifest tracking |
| Streaming checkpoint (write while training) | Near-zero: overlaps with compute | Low: streamed out incrementally | Same as sync | Depends on implementation | High: requires reliable ordered streaming |

## Practical Notes

At 70B scale with async DCP: expect 1–2 s GPU stall for the CPU copy, 30–60 s background write to NFS, and zero write-caused training interruption. The `.metadata` file is small (<1 MB); the data files scale linearly. A 16-rank TP=8, PP=2 job will produce 16 data files per checkpoint step.

At 405B scale: CPU DRAM pressure is real. A single DGX H100 node has 2 TB DRAM; holding one 2.4 TB checkpoint requires either compressing in DRAM or spreading the CPU-side buffer across multiple nodes. Megatron-Core supports both approaches. Warm spare nodes become economically attractive above ~200B parameters.

## Cross-References

- `../zero-fsdp/` — ZeRO-3 shards optimizer state across DP ranks; DCP handles the sharded save/load for ZeRO-3 checkpoints
- `../../deepseek/open-source-week/3fs/` — 3FS is a storage substrate designed for exactly this checkpoint I/O pattern: high-aggregate-bandwidth random writes from many concurrent processes
- `../../meta/llama3/` — Llama 3's 54-day run on 16,000 H100s required robust checkpointing with rapid recovery from GPU failures
- `../../deepseek/open-source-week/dualpipe/` — DualPipe mentions async checkpointing as a co-requirement for zero-bubble training
- `../torch-compile/` — `torch.compile` and DCP are both PyTorch 2.x production features that are used together in `torchtitan`

## References

- [1] Zhao et al. _PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel._ arXiv:2304.11277, 2023.
- [2] Rajbhandari et al. _ZeRO: Memory Optimizations Toward Training Trillion Parameter Models._ SC '20 / arXiv:1910.02054.
- [3] Jiang et al. _MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs._ arXiv:2402.15627, 2024. (Section 4: fault tolerance and in-memory checkpointing.)
- [4] Dubey et al. _The Llama 3 Herd of Models._ arXiv:2407.21783, 2024. (Section 3.3.1: training reliability and checkpointing.)
- [5] PyTorch Distributed Checkpoint documentation: `pytorch.org/docs/stable/distributed.checkpoint.html`
- [6] DeepSpeed ZeRO checkpoint utilities: `github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/zero_to_fp32.py`
