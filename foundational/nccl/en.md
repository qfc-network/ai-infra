# NCCL Internals

- **Org**: NVIDIA
- **Full name**: NVIDIA Collective Communications Library
- **Links**: [GitHub](https://github.com/NVIDIA/nccl) · [Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

## TL;DR

NCCL is the collective communications library that sits underneath every distributed deep learning framework. It implements AllReduce, AllGather, ReduceScatter, Broadcast, and AllToAll on GPU, and it is the reason ZeRO-3 can scatter a 70B model across 512 GPUs and reassemble it layer-by-layer during a training step. The library handles topology detection (PCIe, NVLink, NVSwitch, InfiniBand) at init time, selects ring or tree algorithms based on message size, and exposes protocol variants (Simple, LL, LL128) that trade latency against throughput. On modern multi-node clusters, it integrates with NVLink Switch (NVLS) for intra-node hardware-accelerated reduction and SHARP for IB fabric offload — each path targeting a different point on the latency/throughput tradeoff curve.

## Context & Motivation

Distributed training frameworks do not call RDMA verbs or NVLink APIs directly. They call NCCL collectives. The abstraction boundary is clean: PyTorch's `DistributedDataParallel`, Megatron-LM, and DeepSpeed all reduce gradients via NCCL AllReduce; ZeRO-3's shard/gather/scatter pattern issues AllGather and ReduceScatter; pipeline parallel send/recv uses point-to-point primitives that NCCL also provides.

The challenge NCCL solves is that optimal communication on a multi-GPU, multi-node cluster requires awareness of the underlying topology: two GPUs on the same NVLink domain should never bounce their traffic through the CPU; two GPUs connected via IB should use RDMA and not TCP; the algorithm choice (ring vs tree) should depend on message size and GPU count. Getting this wrong by a factor of 2–3× in effective bandwidth is easy. NCCL centralizes all of this topology awareness and algorithm selection.

## Ring AllReduce

Ring AllReduce is NCCL's bandwidth-optimal algorithm for large messages. With N GPUs arranged in a logical ring, the algorithm proceeds in two phases:

**Phase 1 — ReduceScatter**: the data vector of size D is divided into N equal chunks. In each of N−1 steps, each GPU sends one chunk to its right neighbor and receives one chunk from its left neighbor, accumulating the received data into its local buffer. After N−1 steps, each GPU holds the fully reduced result for exactly 1/N of the data.

**Phase 2 — AllGather**: each GPU broadcasts its reduced chunk to all others. In N−1 steps, each GPU sends its chunk to the right and receives a chunk from the left. After N−1 steps, all GPUs hold the full reduced data.

Total data sent per GPU:

```
Phase 1:  (N-1)/N · D   (send) + (N-1)/N · D   (recv)
Phase 2:  (N-1)/N · D   (send) + (N-1)/N · D   (recv)

Total per GPU = 2 · (N-1)/N · D  ≈  2D  for large N
```

Latency model for ring AllReduce:

```
T = 2·(N-1)·α  +  2·(N-1)/N · D/β
```

where α is per-step latency (startup cost per message), β is effective bandwidth per link, D is data size. For large D, the bandwidth term dominates and efficiency approaches 100%: ring sends exactly the minimum data required. For small D, the 2(N-1) latency steps make ring slow.

Ring is therefore the right choice when D is large relative to α/β. NCCL's crossover threshold to tree algorithms is empirically around 256 KB; below that, tree's O(log N) steps win on latency.

## Tree AllReduce

Tree AllReduce uses a binary (or binomial) tree topology. The reduce phase aggregates values up the tree in log₂N steps; the broadcast phase propagates the result down in another log₂N steps. Total latency:

```
T = 2·log₂(N)·α  +  2·D/β
```

For large N, 2·log₂(N) << 2·(N-1), so tree wins on latency for small messages. However, the bandwidth term for tree is 2D/β rather than 2(N-1)/N · D/β ≈ 2D/β, so tree is asymptotically bandwidth-equivalent — but in practice, tree is less bandwidth-efficient because intermediate nodes must both send and receive simultaneously, creating contention.

NCCL selects between ring and tree (and NVLS/CollNet) automatically. Override with:

```
NCCL_ALGO=Ring          # force ring
NCCL_ALGO=Tree          # force tree
NCCL_ALGO=CollNet       # force CollNet (SHARP offload)
NCCL_ALGO=NVLS          # force NVLink Switch
```

## Protocols: Simple, LL, LL128

NCCL implements three transfer protocols, selected via `NCCL_PROTO`:

**Simple**: lowest startup overhead. Uses standard DMA copies. Best for large messages where pipelining is less critical.

**LL (Low Latency)**: uses 128-byte aligned flag-based signaling. Each 128-byte chunk is tagged with a 4-byte flag; the receiver spins on the flag to detect arrival before processing. This decouples data arrival detection from DMA completion, reducing effective latency by allowing the receiving GPU to start processing arriving data before the full transfer completes. Suited for medium messages.

**LL128**: like LL but with 8-byte headers (128-byte chunks with 8 bytes of header/flag, 120 bytes of data). Better utilization per cache line than LL's 4-byte flag. Preferred on NVLink where link latency is very low and cache-line efficiency matters.

Protocol selection order by message size (default): LL → LL128 → Simple. The crossover points depend on link type; override with `NCCL_PROTO=Simple|LL|LL128`.

## Intra-Node Transport: NVLink, PXN, and NVLS

Within a single node, NCCL probes NVLink connectivity at init and preferentially routes GPU-to-GPU transfers over NVLink rather than PCIe because NVLink provides 3–7× more bandwidth:

| Link type        | Bandwidth (H100 NVLink 4.0) | Latency        |
|------------------|-----------------------------|----------------|
| NVLink 4.0       | 900 GB/s bidirectional       | ~1 μs          |
| PCIe Gen5 ×16    | 128 GB/s bidirectional       | ~3–5 μs        |

**PXN (P2P crossing NVLink)**: when two GPUs on different NVLink domains need to communicate and cannot establish direct peer access, NCCL can route through a third GPU that is connected to both domains via NVLink. This avoids a CPU bounce, which would involve PCIe traversal on both sides.

**NVLS (NVLink Switch AllReduce)**: NVSwitch 3 (the switch chip inside DGX H100 and HGX H100 nodes) implements a hardware multicast/reduce mechanism in the switch fabric. Instead of a software ring, GPUs write to a shared NVLink multicast address; the switch aggregates the values in-fabric and broadcasts the result back to all GPUs. NCCL on H100 nodes detects NVSwitch 3 at init and uses NVLS for intra-node AllReduce.

NVLS performance vs software ring (8-GPU DGX H100, 1 MB message):

| Method          | Latency    | Effective bandwidth |
|-----------------|------------|---------------------|
| Software ring   | ~10–15 μs  | ~200–250 GB/s       |
| NVLS            | ~2–3 μs    | ~600–700 GB/s       |

The latency improvement is the more impactful figure for training: the gradient AllReduce in tensor-parallel training is on the critical path and its latency directly serializes computation.

## Inter-Node Transport: InfiniBand and IBGDA

Between nodes, NCCL uses RDMA over InfiniBand (or RoCE). The standard path uses a CUDA proxy thread that posts RDMA send/receive work requests on behalf of the GPU, introducing one CPU round-trip per operation.

**IBGDA (InfiniBand GPUDirect Async)**: the GPU posts RDMA work requests directly to the NIC doorbell from GPU kernel code, bypassing the CPU proxy entirely. The GPU streams RDMA operations without waiting for CPU intervention. NCCL uses IBGDA internally when available (requires InfiniBand NIC with GPUDirect RDMA support and CUDA >= 11.8). Latency for small messages drops from ~5–10 μs (proxy path) to ~1–2 μs.

This mechanism is central to DeepEP's expert-dispatch all-to-all: when routing tokens to MoE experts on remote GPUs, the latency of each dispatch round matters enormously, and IBGDA at ~1–2 μs vs proxy at ~5–10 μs represents a 3–5× latency improvement on the communication-bound expert dispatch path.

## SHARP: Switch-Offloaded AllReduce

SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) offloads AllReduce computation to InfiniBand Quantum-2 switch ASICs. Rather than having GPUs reduce data through a software ring, the IB switches perform the floating-point reduction in-fabric as data passes through the switch tree.

From the GPU's perspective: send data to the IB NIC; receive the fully-reduced result back. The switch handles all intermediate aggregation. CPU memory bandwidth and GPU compute are entirely off the critical path for the reduction operation.

Requirements and constraints:
- InfiniBand Quantum-2 (or later) switches with SHARP enabled.
- Dedicated SHARP trees allocated per job — not available in shared-fabric environments without reservation.
- NCCL selects SHARP via `NCCL_ALGO=CollNet` (CollNet is NCCL's abstraction for in-network computation).
- Enable at runtime: `NCCL_ALGO=CollNet NCCL_COLLNET_ENABLE=1`.

SHARP advantages are largest at medium-to-large message sizes (>1 MB) on IB-connected clusters, where it can reduce the effective AllReduce time by 2–4× compared to software ring by eliminating host memory traffic.

## Topology Detection and Initialization

At communicator init (`ncclCommInitRank`), NCCL:

1. Probes PCIe topology by reading `/sys/bus/pci/devices/` to determine NUMA affinity and PCIe switch connectivity.
2. Queries NVLink topology via NVML to determine which GPU pairs are directly connected and at what bandwidth.
3. Detects NVSwitch presence and NVLink multicast capability.
4. Enumerates IB HCAs (Host Channel Adapters) and probes RDMA capabilities.
5. Constructs a topology graph and runs a ring/tree assignment algorithm to determine the optimal GPU ordering for ring collectives.

The detected topology can be dumped for debugging:

```bash
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml  # write topology XML
NCCL_TOPO_FILE=/tmp/custom_topo.xml      # override with a custom topology
```

Incorrect topology detection (e.g., because the OS doesn't expose PCIe topology correctly) leads to NCCL using suboptimal paths — often routing over PCIe when NVLink is available, a silent 3–7× bandwidth regression.

## Key Environment Variables

```bash
# Debugging
NCCL_DEBUG=INFO           # enable info-level logging
NCCL_DEBUG=WARN           # warnings only (default)
NCCL_DEBUG=TRACE          # verbose trace logging (very noisy)
NCCL_DEBUG_SUBSYS=COLL    # filter to collective operations
NCCL_DEBUG_SUBSYS=NET     # filter to network transport
NCCL_DEBUG_SUBSYS=P2P     # filter to peer-to-peer operations
NCCL_DEBUG_SUBSYS=INIT    # filter to initialization

# Algorithm and protocol override
NCCL_ALGO=Ring|Tree|CollNet|NVLS
NCCL_PROTO=Simple|LL|LL128

# Network interface selection
NCCL_SOCKET_IFNAME=eth0       # use specific network interface
NCCL_IB_HCA=mlx5_0:1         # use specific IB adapter and port
NCCL_IB_DISABLE=1             # disable IB, fall back to socket transport

# Tuning
NCCL_NTHREADS=512             # number of CUDA threads per NCCL block
NCCL_MAX_NCHANNELS=32         # max parallel communication channels
NCCL_MIN_NCHANNELS=1          # min parallel communication channels
NCCL_BUFFSIZE=4194304         # ring buffer size (bytes)
NCCL_BLOCKING_WAIT=1          # block CPU instead of spinning during wait

# SHARP / CollNet
NCCL_COLLNET_ENABLE=1         # enable SHARP/CollNet offload
```

## Memory and Bandwidth Accounting

NCCL allocates persistent GPU memory at communicator init: ring buffers (`NCCL_BUFFSIZE`, default 4 MB), proxy thread buffers, and algorithm state. For a 256-GPU communicator this can reach ~2–4 GB of GPU memory across all channels. In ZeRO-3 training, the per-layer AllGather of a 70B model layer might transfer 100–500 MB at once; NCCL streams this through the ring buffers, not in a single DMA.

The bandwidth utilization observable with `nccl-tests` (busbw metric, which accounts for ring algorithm overhead) is the canonical reference:

```
effective busbw = measured_bytes_per_second × algorithm_factor
algorithm_factor (AllReduce, ring) = 2(N-1)/N
```

For 8 GPUs, `2×7/8 = 1.75` — the ring sends 1.75× the data volume in total traffic to deliver one AllReduce. This factor approaches 2 for large N and is already at 1.75 for a single DGX node.

## Engineering Tradeoffs

| Algorithm / Path | Message size sweet spot | Hardware requirement | Latency (8-GPU, 1 MB) | Bandwidth efficiency | Fault tolerance |
|---|---|---|---|---|---|
| Ring AllReduce | > 256 KB | Any GPU interconnect | 10–15 μs (PCIe), 3–5 μs (NVLink ring) | Near-optimal for large N; 2(N-1)/N factor | No dedicated HW dependency; any link failure breaks the ring |
| Tree AllReduce | < 256 KB | Any GPU interconnect | 1–3 μs (within node, small msg) | Less bandwidth-efficient than ring for large messages | Tree branches can be rebalanced around failures |
| NVLS (NVLink Switch) | 1 KB – 8 MB (intra-node) | NVSwitch 3 (H100 / GH200 nodes) | 2–3 μs | Very high — hardware reduction in-fabric | NVSwitch failure affects whole node; N+1 switch redundancy on DGX H100 |
| SHARP (CollNet) | > 1 MB (inter-node IB) | InfiniBand Quantum-2 with SHARP trees | < ring on IB; ~2–4× lower than software ring | Eliminates host memory bandwidth on reduction | SHARP tree failure falls back to software ring; requires reservation |

## Failure Modes and Diagnostics

**Hang at collective**: most commonly caused by one rank not calling the collective (code path divergence, OOM on one rank). `NCCL_DEBUG=TRACE` combined with `NCCL_DEBUG_SUBSYS=COLL` shows which ranks are blocked. `NCCL_BLOCKING_WAIT=1` ensures the CPU thread blocks, making the hang visible in `nvidia-smi` as a GPU idle state.

**Suboptimal bandwidth**: topology detection failure causes NCCL to use PCIe instead of NVLink. Diagnose with `NCCL_TOPO_DUMP_FILE` and compare the detected topology against expected NVLink connectivity. `nccl-tests` with `--check 1` validates correctness; `-b 1G -e 1G` measures peak single-size bandwidth.

**IB transport errors**: `NCCL_DEBUG_SUBSYS=NET` exposes RDMA transport issues. `NCCL_IB_HCA` selects specific HCAs if auto-detection picks the wrong one in a multi-rail setup. `NCCL_IB_GID_INDEX` may need to be set for RoCE v2 setups.

**NCCL_SOCKET_IFNAME misconfiguration**: if NCCL falls back to socket transport (no IB), the wrong network interface can be selected, routing traffic over a 1 GbE management NIC instead of the 100 GbE data fabric. Always set this explicitly in multi-homed environments.

## Commentary

NCCL occupies a curious engineering position: it is among the most performance-critical software in large-scale training, yet it is largely invisible. Every framework calls it; almost no one reads its source code. Understanding NCCL at the level described here — ring vs tree thresholds, LL protocol flag mechanics, NVLS path activation, IBGDA latency budgets — matters when you are debugging a 512-GPU training run that is achieving 60% of theoretical communication bandwidth and you need to understand whether the bottleneck is the algorithm, the protocol, the topology detection, or the fabric. The `nccl-tests` repository is the mandatory diagnostic tool; always run it before attributing poor training throughput to other causes. The addition of NVLS and SHARP represents the ongoing trend of offloading reduction work closer to the data — into the switch fabric, away from host CPUs and GPU compute units — a pattern that will continue with future interconnect generations.

## Cross-References

- [`../gpu-interconnect/`](../gpu-interconnect/) — NVLink 4.0 / NVSwitch 3 hardware, RDMA, IBGDA transport layer
- [`../zero-fsdp/`](../zero-fsdp/) — ZeRO-3 issues one AllGather + one ReduceScatter per layer per forward/backward step; NCCL is the implementation
- [`../megatron-lm/`](../megatron-lm/) — Tensor Parallelism uses AllReduce; Pipeline Parallelism uses P2P send/recv; Sequence Parallelism uses ReduceScatter + AllGather
- [`../../deepseek/open-source-week/deep-ep/`](../../deepseek/open-source-week/deep-ep/) — DeepEP uses IBGDA directly for MoE expert dispatch, bypassing NCCL's overhead for all-to-all

## References

- [1] NVIDIA. _NCCL Documentation._ https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- [2] Rabenseifner, R. _Optimization of Collective Reduction Operations._ ICCS 2004.
- [3] Patarasuk & Yuan. _Bandwidth optimal all-reduce algorithms for clusters of workstations._ JPDC 2009.
- [4] Graham, R. et al. _Scalable Hierarchical Aggregation Protocol (SHArP): A Hardware Architecture for Efficient Data Reduction._ 2016.
- [5] NVIDIA. _Magnum IO GPUDirect RDMA._ White paper, 2021.
- [6] nccl-tests GitHub. https://github.com/NVIDIA/nccl-tests
