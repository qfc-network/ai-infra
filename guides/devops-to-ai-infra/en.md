# From DevOps to AI Infrastructure

> A career-path guide, not a paper analysis. Target audience: engineers with 2–5 years of DevOps / SRE / platform engineering experience looking to move into AI infrastructure.

## One-line conclusion

**The core move for DevOps → AI Infra is: layer your existing cluster, networking, storage, and observability skills on top of the GPU and distributed training / inference stack.** You don't need to become an ML algorithm expert first. The entry point is serving operations; how deep into ML systems you go depends on how far you want to push.

---

## I. Advantage inventory — what you already have

DevOps experience transfers to AI Infra further than most people expect:

| DevOps skill | AI Infra equivalent |
|---|---|
| Kubernetes / container orchestration | GPU cluster scheduling (GPU Operator, MIG partitioning, device plugin) |
| Network engineering (VLAN, BGP, ECMP) | RDMA, InfiniBand, RoCE topology — senior network engineers are genuinely scarce here |
| Linux systems / NUMA awareness | PCIe topology, GPU-NIC affinity, driver management, hugepages |
| Prometheus / Grafana / alerting | GPU utilization, MFU, training throughput, serving latency dashboards |
| Distributed storage / NFS / Ceph | High-throughput training data reads + checkpoint storage (think 3FS, Lustre) |
| CI/CD pipelines | Model training pipelines, image builds, serving canary releases |
| Incident response / oncall | Training job restarts after node failures, NCCL timeout triage |

**Networking background is the most underrated advantage.** Most ML engineers don't understand RDMA. They know "all-reduce uses InfiniBand" but not why ECN tuning affects NCCL throughput, or what PFC deadlock looks like, or what IBGDA removes from the critical path. If you have network engineering experience, this is a genuine moat. See the [GPU Interconnect primer](../../foundational/gpu-interconnect/) in this repo — the majority of it is squarely in network engineer territory.

---

## II. Hard gaps — what you need to fill

Skipping these will stall your career trajectory:

### 2.1 GPU programming model (conceptual, not kernel-writing)

You need to understand:
- SM (Streaming Multiprocessor) / warp / thread block hierarchy
- The bandwidth gap between HBM (device memory) and SRAM (on-chip cache)
- Why "keeping data in SRAM" is the core of nearly every kernel optimization
- What Tensor Cores are, and what FP16 / BF16 / FP8 mean in practice

You don't need (short-term): CUDA C++ syntax, PTX assembly, `__shared__` memory declarations.

**Resources**: First two chapters of NVIDIA's CUDA programming model docs; Simon Boehm's "How to Optimize a CUDA Matmul Kernel" (read for intuition, don't worry about running the code).

### 2.2 Distributed training basics

You need to understand:
- **Data Parallelism (DP)**: each GPU holds a full copy; gradients are all-reduced
- **Tensor Parallelism (TP)**: weight matrices split column/row-wise; communicates over NVLink within a node
- **Pipeline Parallelism (PP)**: layers split across nodes; activations transferred over RDMA between stages
- **ZeRO**: optimizer state / gradients / weights sharded across GPUs; reduce-scatter + all-gather

Understand the **communication patterns and memory shapes** of each. You don't need to implement them. See [Megatron-LM](../../foundational/megatron-lm/) and [ZeRO / FSDP](../../foundational/zero-fsdp/).

### 2.3 Inference serving internals

You need to understand:
- What KV cache is and why it dominates inference memory consumption
- That prefill (processing the prompt) and decode (generating tokens) are two fundamentally different compute shapes
- How continuous batching keeps GPUs from idling
- Why batch size is gated by KV cache size

See [PagedAttention / vLLM](../../foundational/paged-attention/) and [DistServe](../../foundational/distserve/).

### 2.4 PyTorch at a working level

Not mastery — just enough to read code:
- Tensor operations, shapes, dtypes
- `model.load_state_dict()` and `torch.save()` — how checkpointing actually works
- Why `DataLoader` worker count and `pin_memory` affect training throughput

---

## III. Entry path — by phase

### Step 1: Get hands-on (1–2 months)

**Goal: run a real inference service on a real GPU and observe it through an ops lens.**

```bash
# Install
pip install vllm

# Start a serving endpoint (requires GPU)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9

# Benchmark
python benchmarks/benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 200
```

Things to observe:
- `nvidia-smi` / `nvitop`: how memory is allocated, what the GPU utilization curve looks like
- Lower `--max-model-len` and watch batch size increase as KV cache shrinks
- Send two concurrent long requests and separate out TTFT (Time to First Token) from TPOT (Time per Output Token)

You don't need to understand ML here — pure ops perspective is sufficient.

### Step 2: Read the key papers in this repo (2–3 months, in order)

Ordered by friendliness to DevOps background:

1. **[PagedAttention / vLLM](../../foundational/paged-attention/)** — KV cache paging, directly analogous to OS virtual memory management
2. **[DistServe](../../foundational/distserve/)** — goodput as the right metric, why prefill/decode disaggregation makes sense
3. **[GPU Interconnect primer](../../foundational/gpu-interconnect/)** — NVLink / IB / RDMA / IBGDA — network engineer home turf
4. **[ZeRO / FSDP](../../foundational/zero-fsdp/)** — how memory gets sharded, where all-reduce runs
5. **[SGLang](../../foundational/sglang/)** — RadixAttention prefix caching, analogous to CDN cache hierarchies
6. **[Megatron-LM](../../foundational/megatron-lm/)** — communication shapes of the three parallelisms, selective recompute

Further reading (after specializing):
- [DeepEP](../../deepseek/open-source-week/deep-ep/) — RDMA + IBGDA for expert parallelism; network background makes the motivation immediately legible
- [3FS](../../deepseek/open-source-week/3fs/) — RDMA-native distributed storage; storage background makes this directly accessible
- [Mooncake](../../moonshot/mooncake/) — KV cache pooling, analogous to CDN edge caching architecture

### Step 3: Pick one vertical and go deep

Don't spread across all three paths. Choose the one closest to your existing background.

---

## IV. Three vertical paths

### Path A: GPU cluster operations (fastest to land)

**Best for**: engineers with Kubernetes, bare-metal ops, or data center networking backgrounds.

**Core work**:
- GPU node provisioning: driver / CUDA / NCCL version matrix management
- Kubernetes GPU Operator: device plugin, MIG partitioning, node labeling
- Health monitoring: DCGM exporter + custom probes (GPU temperature, HBM ECC errors, NVLink error counters)
- Topology-aware scheduling: colocate communication-heavy GPUs within the same NVLink domain / leaf switch
- Incident triage: distinguishing GPU hardware failure vs driver issues vs NCCL deadlock

**Tech stack**: Kubernetes + GPU Operator, SLURM (many training clusters use it), DCGM, NCCL debug tooling.

**Target roles**: GPU Platform Engineer, HPC Cluster Engineer, AI Infrastructure Engineer (Ops)

---

### Path B: Inference platform (best engineering depth)

**Best for**: engineers with SRE, microservices ops, or API gateway experience.

**Core work**:
- Deploy and operate vLLM / SGLang / TensorRT-LLM
- Autoscaling strategies: based on queue depth / GPU utilization / TTFT SLO
- Multi-model scheduling: prefix cache sharing, routing, model version management
- Cost optimization: spot instances + checkpoint recovery, KV cache offloading to CPU / SSD
- Observability: TTFT p99, TPOT, token throughput, KV cache hit rate

**Tech stack**: vLLM / SGLang, Kubernetes, Prometheus, custom load balancer.

**Target roles**: Inference Platform Engineer, LLM Serving Engineer, ML Platform SRE

---

### Path C: Training infrastructure (hardest, highest ceiling)

**Best for**: engineers with distributed systems, storage, and full-stack networking backgrounds who are willing to go deep into PyTorch internals.

**Core work**:
- Job scheduling and resource management: SLURM / Volcano / custom schedulers
- Checkpoint management: async writes, recovery speed optimization, resume-across-failure
- Communication profiling: NCCL all-reduce bottlenecks, bandwidth utilization, topology-aware ring construction
- Storage I/O optimization: training data prefetch, parallel reads, checkpoint write bandwidth
- Fault recovery: automated detection of node loss + restart from last checkpoint

**Tech stack**: PyTorch, Megatron-LM / FSDP, NCCL, SLURM, high-throughput storage (Lustre / 3FS).

**Target roles**: Training Infrastructure Engineer, ML Systems Engineer

---

## V. Common traps

**Trap 1: Getting pulled into CUDA kernel writing too early**

Writing CUDA kernels is a separate career track (ML Systems / Kernel Engineer). It's not the DevOps entry path into AI Infra. Your advantage is at the systems layer, not the kernel layer. Knowing the CUDA execution model conceptually is enough for now.

**Trap 2: Starting from ML algorithms**

What attention is, how transformers work, loss function derivations — useful eventually, but starting there is a detour. DevOps engineers' advantage is at the systems layer. Leading with algorithms erodes your differentiation.

**Trap 3: Reading without running**

AI Infra knowledge needs to be validated on real GPUs. "KV cache fills up memory" means nothing until you watch `nvidia-smi` show KV cache consuming the device's HBM in real time.

**Trap 4: Undervaluing your networking skills**

Most ML engineers don't understand RDMA. They know "all-reduce uses InfiniBand" but don't know why ECN configuration affects NCCL throughput, what a PFC deadlock looks like, or what the IBGDA optimization actually removes. This is where DevOps / network engineers have a real moat that ML-background engineers can't easily replicate.

---

## VI. This repo as a learning map

Every entry in this repo is written from an engineering perspective with minimal assumption of ML algorithm background. Suggested reading order:

**Tier 1 (80%+ accessible with DevOps background)**
- [PagedAttention / vLLM](../../foundational/paged-attention/)
- [GPU Interconnect primer](../../foundational/gpu-interconnect/)
- [3FS walkthrough](../../deepseek/open-source-week/3fs/)
- [DistServe](../../foundational/distserve/)

**Tier 2 (requires some GPU model background)**
- [ZeRO / FSDP](../../foundational/zero-fsdp/)
- [SGLang](../../foundational/sglang/)
- [Megatron-LM](../../foundational/megatron-lm/)
- [Mooncake](../../moonshot/mooncake/)

**Tier 3 (read after specializing)**
- [FlashAttention 1/2/3](../../foundational/flash-attention/)
- [DeepEP walkthrough](../../deepseek/open-source-week/deep-ep/)
- [DeepSeek V3 Technical Report](../../deepseek/v3-tech-report/)

---

## VII. 6-month milestones

| Month | Target |
|---|---|
| 1 | vLLM running locally or in cloud; can explain KV cache memory consumption; can read TTFT / TPOT metrics |
| 2 | Tier 1 papers read; can explain PagedAttention and what goodput means |
| 3 | Vertical path selected; going deep in one area (cluster / serving / training) |
| 4–5 | Tier 2 papers read; can explain the communication shapes of TP / DP / ZeRO |
| 6 | One concrete project or contribution in the chosen vertical (benchmark, postmortem, tooling) |

---

## References

- [vLLM docs](https://docs.vllm.ai/) — best entry point for inference serving
- [Megatron-LM README](https://github.com/NVIDIA/Megatron-LM) — de facto standard for distributed training
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/) — debugging collective communication
- [DCGM docs](https://docs.nvidia.com/datacenter/dcgm/) — GPU cluster monitoring
- [Efficient Large Scale Language Modeling with Megatron (paper)](https://arxiv.org/abs/2104.04473) — prerequisite for Tier 2
- [GPU Interconnect primer](../../foundational/gpu-interconnect/) in this repo — the networking foundation for DevOps → AI Infra
