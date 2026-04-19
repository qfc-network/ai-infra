# xAI Grok + Colossus Cluster

- **Org**: xAI (Elon Musk)
- **Key dates**: Grok-1 open-weight release March 2024; Colossus Phase 1 announced October 2024; Grok-3 launched February 2025
- **Links**: [Grok-1 weights (HuggingFace)](https://huggingface.co/xai-org/grok-1) · [xAI API](https://api.x.ai) · [Colossus announcement](https://x.ai/blog/colossus)

## TL;DR

xAI built a 100,000 H100 single-site training cluster (Colossus Phase 1) in approximately 122 days in Memphis, Tennessee — the largest single-campus GPU cluster publicly announced at that time. The system trained Grok-3, xAI's frontier reasoning model. The decision to pack all compute into one physical site was an explicit infrastructure bet: eliminate cross-datacenter communication latency for synchronous distributed training at scales that no multi-site cluster had previously attempted with H100s. This document covers the Grok model family's architecture, the Colossus build-out, the parallelism and storage decisions behind Grok-3's training, and the inference patterns created by tight integration with the X (Twitter) platform.

## 1. Grok Model Family

### Grok-1: First Frontier Open MoE

Released open-weight in March 2024 under an Apache 2.0 license, Grok-1 is a **314B parameter Mixture-of-Experts transformer** with 8 experts and top-2 routing per token. At release it was the largest open-weight model publicly available and the first frontier-scale MoE released with full weights.

Architectural summary:
- 314B total parameters; roughly 86B activated per token (top-2 of 8 experts)
- MoE routing similar in spirit to Mixtral 8x7B but at far larger scale
- 128k token context window
- Rotary positional embeddings (RoPE)
- No publicly released training hyperparameters or dataset card

The open-weight release was a strategic move: xAI had no prior reputation in the open-source ML community and releasing Grok-1 established credibility at a moment when Mixtral had just shown that sparse MoE could match dense models at lower activation cost.

### Grok-1.5 and Grok-2: Multimodal Expansion

Grok-1.5 (April 2024) and Grok-2 (August 2024) were closed-source releases. Grok-2 is competitive with GPT-4o on standard benchmarks (MMMU, GPQA, HumanEval, MATH) and introduced the **Aurora image understanding module** — a vision encoder integrated into the base LLM for image captioning, chart reading, and visual reasoning. Grok-2 weights were not publicly released; the model is served exclusively through xAI's API and X platform.

Architecture notes for Grok-2+:
- Aurora module: a vision transformer encoder fused with the language backbone at the cross-attention layer, similar in pattern to LLaVA-style visual instruction tuning but integrated at pretraining time
- Long context retained at 128k tokens across modalities
- Inference optimized for sub-second latency at social-media query volumes

### Grok-3: Trained on Colossus

Launched February 2025 with state-of-the-art reasoning benchmark claims, Grok-3 was trained entirely on Colossus Phase 2 (the 200k H100 expansion). xAI described the training as using roughly ten times the compute of Grok-2. Grok-3 competes directly with GPT-4o and Claude 3.5 Sonnet in coding, math, and science tasks. A "Grok-3 mini" variant optimized for latency was released alongside the full model.

FP8 training was used for Grok-3 — by 2024–2025 this had become standard practice at frontier labs. FP8 forward passes with BF16 optimizer state provide roughly 2× throughput improvement over pure BF16 on H100s.

## 2. Colossus Phase 1 — 100,000 H100 Single-Site Cluster

### Build Timeline and Scale

Colossus Phase 1 is located in Memphis, Tennessee. Construction started in mid-2024 and the cluster reportedly reached operational status in approximately **122 days** — a buildout speed that required parallel rather than sequential procurement, power infrastructure, and rack installation. xAI announced the cluster in October 2024.

Hardware summary:

| Metric | Value |
|---|---|
| GPU model | H100 SXM5 |
| Total GPUs | 100,000 |
| GPUs per node | 8 (NVLink domain) |
| Total nodes | ~12,500 |
| Intra-node bandwidth | NVLink 4.0, 900 GB/s bidirectional |
| Inter-node fabric | InfiniBand HDR/NDR, RDMA |
| Peak cluster power draw | ~200 MW |
| Storage scale | Tens of PB, flash-based distributed |
| Context window supported | 128k tokens |

### Why Single-Site?

The case for concentrating all compute on one physical campus is rooted in the physics of collective communication in distributed training.

All-to-all collectives (AllReduce, AllGather, ReduceScatter) are bounded by the latency of the **slowest link in the topology**. For a ring-AllReduce over N ranks, the time per step is proportional to:

$$T_{\text{allreduce}} \approx 2 \cdot \frac{N-1}{N} \cdot \frac{M}{B} + 2(N-1) \cdot \alpha$$

where M is message size, B is bisection bandwidth, and α is per-hop latency. Adding a WAN link between two datacenters inserts an α of 1–10 ms; within a single campus over InfiniBand, α is under 5 μs — three orders of magnitude lower. For a 100k-GPU training job running 10,000 gradient steps, even a 1 ms inter-DC latency penalty translates to roughly 3 hours of dead time per run.

Multi-site training mitigation strategies (gradient compression, asynchronous DP, hierarchical AllReduce) each impose correctness or throughput penalties. xAI's bet was that a single physical site eliminates the problem rather than managing it.

**Comparison with contemporaneous clusters:**

| Cluster | GPUs | Configuration | Notable training |
|---|---|---|---|
| Colossus P1 (xAI) | 100,000 H100 | Single site, Memphis | Grok-3 |
| Meta (Llama 3 405B) | ~16,000 H100 | Multi-cluster, multiple sites | Llama 3 405B |
| Google TPU v5p | ~8,960 TPU chips | Pod, multiple sites | Gemini Ultra |
| Microsoft / OpenAI | ~20,000 A100 | Azure, multiple sites | GPT-4 (est.) |

Colossus P1 is roughly 6× the GPU count of Meta's Llama 3 training cluster and operates as a single synchronous training domain rather than a federated arrangement.

### Network Fabric

Within each node, all 8 H100 SXM5 GPUs are connected in an NVLink mesh providing 900 GB/s bidirectional bandwidth. This is sufficient for tensor parallelism (TP) within the node: gradient and activation shards are exchanged within the NVLink domain without hitting the slower InfiniBand fabric.

Between nodes, InfiniBand HDR (200 Gb/s per port) and NDR (400 Gb/s per port) links provide RDMA-capable fabric for pipeline parallelism (PP) and data parallelism (DP) communication. At 12,500 nodes, the switch fabric requires a multi-tier fat-tree topology to maintain full bisection bandwidth; exact fabric details have not been published by xAI.

### Storage

At frontier training scale, checkpoint and dataset I/O impose independent storage requirements:

- **Checkpoint storage**: A 314B-parameter model in BF16 requires ~630 GB per checkpoint. With Adam optimizer state (two additional BF16 tensors per parameter), a full optimizer checkpoint is approximately 2.5 TB. At 100,000-GPU scale, checkpointing every 500 training steps (a conservative interval) against a cluster that may consume $50M/month in compute means checkpoint write bandwidth and storage IOPS are first-class constraints, not afterthoughts. Flash-based distributed storage (NVMe over RDMA) is necessary to avoid checkpointing becoming a training bottleneck.
- **Dataset I/O**: Pre-tokenized datasets read sequentially; bandwidth per GPU is modest (~500 MB/s aggregate from flash) but sustained for weeks.

## 3. Colossus Phase 2 — 200,000 H100 Target

Announced alongside Grok-3's training, Phase 2 expanded the Memphis campus toward 200,000 H100s, doubling peak power to approximately **400 MW**. This places Colossus Phase 2 at the scale of a small municipal power supply.

### Power and Cooling Challenges

At 400 MW, the cluster's power demand exceeds what commercial utility feeds can supply reliably at single-building densities. xAI addressed this through:

- **On-site power generation**: Gas turbines or supplemental generation providing dedicated capacity beyond the utility grid; this is a prerequisite for planning clusters above ~100 MW in the US market where grid interconnect queues run 3–5 years.
- **Direct liquid cooling (DLC) at rack level**: H100 SXM5 nodes dissipate roughly 700W per GPU (5.6 kW per 8-GPU node). At 25,000 nodes, air cooling becomes physically infeasible above roughly 30–40 kW per rack; DLC allows 60–100 kW rack densities. Rear-door heat exchangers or direct-to-chip cold plates are both deployed at Colossus scale.

### GPU Procurement

Sourcing 100,000 additional H100s in the 2024–2025 period operated against NVIDIA's constrained supply: H100 SXM5 allocations were effectively rationed to priority hyperscaler and government customers. xAI's relationship with NVIDIA — and Elon Musk's personal leverage — was a competitive input to the build. Standard procurement cycles would not have permitted this timeline.

## 4. Training Infrastructure Decisions for Grok-3

### H100 Over H200 or Blackwell

The choice to build on H100 SXM5 rather than waiting for H200 or Blackwell (B100/B200) was a timeline decision:

- H100 SXM5 was available at volume in mid-2024
- H200 (with HBM3e, providing ~1.4× memory bandwidth improvement over H100) began shipping in small quantities mid-2024 but was not available at 100k-unit scale in time for Phase 1
- B100/B200 Blackwell was delayed into 2025 due to mask revisions; committing to Blackwell would have pushed the training window out by 12+ months

The cost of waiting for next-generation hardware was effectively ceding the Grok-3 training window to competitors. The H100 cluster built on time delivers more aggregate compute than a smaller next-generation cluster built later.

### 4D Parallelism Strategy

For a 314B+ MoE model on 100,000 GPUs, the parallelism decomposition is constrained by topology and memory:

- **Tensor Parallelism (TP=8)**: Within NVLink domain. Shards attention heads and FFN rows/columns across 8 GPUs. NVLink bandwidth (900 GB/s) is sufficient for the AllGather and ReduceScatter operations in each transformer layer.
- **Pipeline Parallelism (PP)**: Across nodes via InfiniBand. Micro-batching amortizes the pipeline bubble. At 126+ layers, PP=8 or PP=16 keeps per-stage memory tractable.
- **Data Parallelism (DP)**: Across pipeline replica groups. At 100k GPUs with TP=8 and PP=16, the DP dimension is approximately 781 — the number of independent gradient replicas being reduced each step.
- **Expert Parallelism (EP)**: For MoE layers, the 8 experts are distributed across EP ranks. Expert dispatch (token routing to expert GPUs) introduces all-to-all communication not present in dense models; this is the primary MoE-specific training overhead.

FP8 training reduces both memory pressure (activations stored in FP8 where precision allows) and compute pressure (H100 FP8 Tensor Core throughput is ~2× BF16).

### Checkpoint Overhead at Scale

With 100,000 GPUs and optimizer state:

- Per-checkpoint write: ~2.5 TB of optimizer state + ~630 GB of model weights = ~3.2 TB total
- At 10 GB/s aggregate write bandwidth (flash distributed storage): ~5 minutes per checkpoint
- At checkpoint interval of 500 steps, checkpointing consumes roughly 1% of training time at this bandwidth — acceptable, but only with parallel distributed write from all GPUs simultaneously

Hardware failures at 100k GPU scale (mean time between failures estimated at one per few hours for any single GPU) make frequent checkpointing non-optional. The storage system must absorb checkpoint I/O without throttling training.

## 5. Inference Infrastructure

### Dual-Use Cluster

Colossus operates in a dual-use pattern: training runs during periods of model development; inference workloads serve the X platform and xAI API between or alongside training runs. This is operationally complex — a training job and an inference serving tier have very different memory allocation, batching, and fault-tolerance requirements — but is economically necessary at $500M+ build costs.

### X Platform Demand Patterns

Grok's integration with X (Twitter) creates an inference demand profile that differs from pure API traffic:

- **Query bursts**: Viral events on X generate simultaneous spikes from millions of users. Unlike API traffic that follows organizational work patterns, social media inference demand is bursty and correlated — a breaking news event creates load that spikes in minutes.
- **Real-time augmentation**: Grok has live access to X posts and web search, meaning each query may involve external retrieval before inference. This adds latency to the critical path and creates non-deterministic I/O wait in the serving path.
- **Latency targets**: Consumer social media users expect p50 response times under 1 second for short completions. This requires aggressive continuous batching, speculative decoding (small draft model predicts tokens verified by Grok in parallel), and KV cache management tuned for sub-second SLA.

### Speculative Decoding at Social Scale

For a 314B+ model to serve sub-second p50 latency, speculative decoding using a smaller draft model (estimated at 7–13B parameters) runs in parallel with Grok's verification pass. This amortizes the memory bandwidth bound of autoregressive decoding — bandwidth utilization per token goes up, wall-clock time per token goes down. The tradeoff is additional memory occupied by the draft model on each inference instance.

## 6. Engineering Tradeoffs

| Decision | Gained | Gave Up |
|---|---|---|
| Single-site 100k GPU cluster | Synchronous collective communication within campus latency (<5 μs InfiniBand); no WAN penalty in training collectives | Single point of geographic failure; power and cooling must be solved locally; limits total addressable power grid capacity |
| H100 SXM5 over H200/Blackwell | Available at volume now; cluster operational 12+ months before Blackwell at scale | ~1.4× memory bandwidth of H200 foregone; FP8 on Blackwell (B200) would be meaningfully faster per chip |
| MoE architecture (Grok-1/3) | ~86B activated params from 314B total; lower inference compute per token at serving scale | Expert parallelism overhead during training; all-to-all routing latency; harder to fine-tune than dense |
| On-site power generation | Independence from utility grid queue; enables 400 MW demand at single site | Capital cost of turbines and grid infrastructure; permitting and time-to-first-power |
| FP8 training (Grok-3) | ~2× throughput vs BF16 on H100; reduced activation memory | Numerical stability requires careful loss scaling; tooling maturity lower than BF16 in 2024 |
| Dual-use training/inference cluster | Hardware utilization improves; inference revenue offsets training capex | Operational complexity of context-switching between training and serving; interference in memory and network scheduling |

## Cross-References

- [`../../foundational/gpu-interconnect/`](../../foundational/gpu-interconnect/) — NVLink and InfiniBand fabric topology underpinning Colossus node and cluster networking
- [`../../foundational/blackwell-b200/`](../../foundational/blackwell-b200/) — next-generation NVIDIA hardware xAI is likely to integrate in future Colossus expansions
- [`../../meta/llama3/`](../../meta/llama3/) — detailed 4D parallelism implementation at 16k H100 scale; benchmark for comparative cluster analysis
- [`../../foundational/megatron-lm/`](../../foundational/megatron-lm/) — TP/PP/SP parallelism primitives that underpin the training stack likely used for Grok-3
