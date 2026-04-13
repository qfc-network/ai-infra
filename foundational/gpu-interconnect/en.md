# GPU Interconnect — NVLink, NVSwitch, RDMA, IBGDA

_A reference primer, not a paper write-up. The interconnect hierarchy is assumed by almost every entry in this repo — MoE all-to-all, ring attention, PD-disaggregation, Pathways, 3FS, DeepEP. This file names the pieces and what each is for._

## TL;DR

Modern LLM training and serving run on a **three-tier interconnect hierarchy**:

1. **Intra-node**: NVLink (and NVSwitch, which aggregates it) — GPU-to-GPU inside one box at hundreds of GB/s to low-single-digit TB/s.
2. **Intra-pod / rack**: NVLink Switch System (NVL72) — extends NVLink across a rack of nodes at the same bandwidth class.
3. **Inter-node**: InfiniBand or RoCE Ethernet — GPU-to-GPU across racks at 100-400 Gb/s per NIC, via GPUDirect RDMA.

Alongside the fabric, a software stack (CUDA, NCCL, NVSHMEM, IBGDA) decides how much of the physical capability user code can actually use. The difference between "400 GB/s of fabric" and "400 GB/s of achieved app bandwidth" can be 10×, depending on which abstractions you use.

This primer maps the pieces and flags the decisions that every infra paper in this repo implicitly makes.

## Intra-node: NVLink and NVSwitch

### NVLink generations (rough aggregate per-GPU bandwidth)

| Gen | GPU | Year | Links × GB/s/link | Aggregate per GPU |
|-----|-----|------|-------------------|---------------------|
| 1 | P100 | 2016 | 4 × 20 GB/s | 80 GB/s |
| 2 | V100 | 2017 | 6 × 25 GB/s | 150 GB/s |
| 3 | A100 | 2020 | 12 × 25 GB/s | 300 GB/s (600 bi) |
| 4 | H100 | 2022 | 18 × 25 GB/s | 450 GB/s (900 bi) |
| 4 | H800 | 2022 | restricted subset | ~200 GB/s (400 bi) |
| 5 | Blackwell | 2024 | 18 × 50 GB/s | 900 GB/s (1.8 TB/s bi) |

"Bi" = bidirectional. The H800 gap — roughly half of H100's NVLink — is the specific hardware constraint that motivates DeepSeek's DualPipe, DeepEP's asymmetric forwarding, and FlashMLA's seesaw schedule. Same chips, different fabric, fundamentally different software.

### NVSwitch

NVLink alone is point-to-point: each GPU has N links, each going to one peer. With 8 GPUs per node, you'd need either (a) a ring topology (bandwidth per step) or (b) all-to-all direct links (not enough ports). NVSwitch solves this: a crossbar that lets **every GPU talk to every other GPU at full NVLink bandwidth simultaneously**.

Since H100: 4 NVSwitches per 8-GPU DGX, fully-bi-sectional bandwidth for any traffic pattern. This is what makes tensor parallelism across 8 GPUs "effectively free" communication-wise.

### NVL72 (Blackwell)

Scaled NVSwitch to **72 GPUs connected at NVLink bandwidth** across a rack. Creates a "super-node" where what used to require InfiniBand (cross-node) now runs at NVLink speeds. Implications for training: TP and model sharding can span 72 GPUs, not 8. Implications for serving: PD-disaggregation across larger pools at internal-bus speeds.

This is why **2025+ papers start talking about "super-node" as a unit**, not "node."

## Inter-node: RDMA

### InfiniBand (IB) and RoCE

GPUs in different nodes communicate via network cards. Two fabrics dominate:

- **InfiniBand** — HPC heritage, purpose-built for RDMA, lowest latency, higher cost, vendor (Mellanox / NVIDIA) lock-in. CX7 400 Gb/s is the current standard for frontier clusters.
- **RoCE (RDMA over Converged Ethernet)** — RDMA semantics over standard Ethernet. Slightly higher latency, slightly weaker congestion control, commodity switches. Meta uses it for Llama 3's 16k-GPU cluster; cost advantage at scale.

Both deliver **RDMA (Remote Direct Memory Access)** — the network card reads/writes peer memory without the peer CPU being involved. Key for microsecond-class communication.

### GPUDirect RDMA

Without GPUDirect: GPU → host CPU → NIC → wire → NIC → host CPU → GPU. The CPU hops add ~20 µs each direction.

GPUDirect RDMA: NIC reads/writes GPU memory directly via PCIe. **Host CPU is out of the path**. Latency drops to the fabric's own latency (~2-5 µs for IB). This is the foundation every modern inter-node GPU communication builds on.

Requirements: a PCIe topology that lets the NIC see GPU memory (often means NIC and GPU on the same PCIe root complex / NUMA node).

### IBGDA — InfiniBand GPUDirect Async

GPUDirect RDMA still has the CPU *initiate* transfers (the kernel driver posts work requests to the NIC). For fine-grained communication inside a kernel — dispatching tokens to experts, sending a token from prefill to decode — the CPU round-trip to post is itself the bottleneck.

**IBGDA** lets the **GPU itself post work requests to the NIC**, via memory-mapped queue pairs. The CPU is entirely out of the communication loop.

DeepEP's low-latency kernel uses IBGDA. So does any modern MoE inference stack where per-token latency matters. The fabric must support it (CX6+ NICs, IB or RoCE with the right firmware).

## Software abstractions

The fabric is only half the story; the programming model is the other half.

### NCCL

NVIDIA Collective Communications Library. The default for collectives (`all-reduce`, `all-gather`, `reduce-scatter`, etc.) in PyTorch / Megatron / JAX. Tuned for NVLink + NVSwitch + IB topology. Autodetects topology and picks algorithms per collective + shape.

When people say "PyTorch DDP uses NCCL under the hood," this is what they mean. NCCL abstracts away the difference between "this is NVLink" and "this is InfiniBand" — a single `all_reduce()` call maps to the right primitive.

Limits: NCCL is collective-oriented. Point-to-point patterns like MoE all-to-all are expressible but not always optimal. Custom kernels with NVSHMEM or IBGDA can beat NCCL on specific shapes.

### NVSHMEM

A PGAS-style programming model: treat all GPUs in a cluster as one logically unified memory, read and write any address from any GPU with `get`/`put` / atomic operations. Under the hood, NVSHMEM uses NVLink for intra-node and GPUDirect RDMA (or IBGDA) for inter-node.

When to prefer NVSHMEM over NCCL:

- **Irregular or fine-grained communication** — MoE dispatch, graph neural networks, irregular sparse attention.
- **Custom kernels** where you want to issue communication mid-kernel.
- **Asymmetric patterns** — MoE all-to-all with non-uniform per-rank volume.

DeepEP is built on NVSHMEM. So are many custom MoE and graph kernels.

### PyTorch distributed primitives

Above NCCL, PyTorch exposes `dist.all_reduce`, `dist.send`, `dist.recv`, `dist.all_to_all`, etc. FSDP and DDP use these. Usage patterns are collective-centric.

Newer: `DTensor` gives PyTorch a sharding abstraction (similar to JAX's shard_map), and the compiler can route to the right collectives automatically. This is PyTorch's converge-toward-GSPMD path.

## Topology: the hidden variable

Every serious training or serving deployment has a **topology map**: which GPUs are on the same NVLink island, which NUMA node owns which NIC, which racks share a leaf switch. The performance gap between a topology-aware placement and a naive one is routinely 2-5× on communication-heavy workloads.

Examples from this repo's other entries:

- **DeepEP's forwarder pattern** — put forwarders on the rank that owns the RDMA NIC; everyone else fans in over NVLink.
- **V3's ≤4-nodes-per-token routing** — bounded RDMA out-degree, designed around the fabric's bisection bandwidth.
- **3FS's chain tables** — balance chains so failures redistribute load across *many* peers, not the immediate NVLink neighbors.
- **Ring Attention** — ring ordering chosen to match the physical NVLink/IB topology, so each hop is cheap.

## How it shows up in papers

A quick map for reading the rest of the repo:

| Paper / System | Interconnect it's exploiting |
|---|---|
| Megatron TP | NVLink + NVSwitch (intra-node) |
| Megatron PP | Inter-node RDMA, small messages, latency-tolerant |
| FSDP / ZeRO | NCCL `all-gather` + `reduce-scatter`, NVLink + IB |
| Ring Attention / CP | Ring over NVLink primarily, IB for cross-node |
| DeepEP normal | Asymmetric NVLink (fanout) + RDMA (cross-node) |
| DeepEP low-latency | Pure IBGDA (later with NVLink for intra-node) |
| Mooncake / DistServe | RDMA between prefill and decode pools |
| 3FS | RDMA throughout, locality-oblivious |
| Pathways | Per-pod resource manager, abstracts fabric details |

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| NVLink + NVSwitch intra-node | TB/s-class flat bandwidth, 8+ GPUs act like one | Fixed to NVIDIA's topology; no escape |
| InfiniBand over RoCE | Lowest latency, mature RDMA | Cost, vendor lock-in |
| RoCE over IB | Commodity, cheaper at scale | Extra tuning for congestion, slightly higher latency |
| GPUDirect RDMA | CPU out of the path | PCIe topology constraints |
| IBGDA | GPU-initiated RDMA, no CPU involvement | Firmware / driver requirements; not universal |
| NCCL (high-level) | Portable, autotuned | Collective-shaped only |
| NVSHMEM | Fine-grained, irregular patterns | Harder to reason about; debugging is specialist work |
| Topology-aware placement | 2-5× on comm-heavy workloads | Deploy complexity, per-cluster engineering |

## Commentary

Reading an LLM infra paper without a rough mental model of the fabric it runs on is like reading a distributed systems paper without thinking about latency. The specific choices — "why does V3 cap tokens to 4 nodes?" "why does DeepEP have three separate kernels?" "why can Llama 3 use RoCE but DeepSeek needs IB?" — all trace back to **interconnect shape + bandwidth + latency + software abstraction**, not to algorithmic preferences.

Two trends to watch going forward:

1. **NVL72 and beyond.** Scale-up (bigger super-nodes) is taking over from scale-out (more smaller nodes) for frontier workloads. A 72-GPU NVLink domain changes what's possible — TP can span 72, MoE EP can stay in-domain, PD-disaggregation can cross pools at NVLink speed. Papers from 2025+ will assume this; papers from 2022-2024 assume 8-GPU nodes.

2. **Compute-in-network and congestion control.** As models grow, the aggregate bisection bandwidth is the bottleneck. Future work will push reductions into switches (SHARP), co-design congestion control with collective algorithms, and treat the network as a first-class accelerator target.

For anyone building infra in 2026: **know your fabric**. Most "why is this training run slow" mysteries resolve at this layer. Most "this paper doesn't reproduce for us" gaps resolve here too.

## References

- NVIDIA NVLink technical overview: https://www.nvidia.com/en-us/data-center/nvlink/
- NVSwitch architecture: NVIDIA whitepapers (per-gen)
- GPUDirect RDMA: https://docs.nvidia.com/cuda/gpudirect-rdma/
- IBGDA: https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/
- NVSHMEM: https://developer.nvidia.com/nvshmem
- NCCL: https://github.com/NVIDIA/nccl
- Meta's RoCE Llama 3 cluster post: https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/
