# DeepSpeed — MoE, Chat, and Inference Engine

- **Authors / Org**: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase et al. — Microsoft
- **Published**: 2022-01 (MoE) · 2022-06 (Inference) · 2023-04 (Chat)
- **Links**: [arXiv:2201.05596 (MoE)](https://arxiv.org/abs/2201.05596) · [arXiv:2207.00032 (Inference)](https://arxiv.org/abs/2207.00032) · [arXiv:2308.01320 (Chat)](https://arxiv.org/abs/2308.01320) · [code](https://github.com/microsoft/DeepSpeed) · [blog](https://github.com/microsoft/DeepSpeed)

## TL;DR

DeepSpeed extends beyond ZeRO (covered in `foundational/zero-fsdp/`) into three specialized systems: **DeepSpeed-MoE** for training and serving Mixture-of-Experts models with expert parallelism and hierarchical all-to-all communication; **DeepSpeed-Chat** for end-to-end RLHF with a hybrid engine that switches the same model between generation mode and training mode within a single PPO loop; and the **DeepSpeed Inference Engine** for high-throughput serving via fused CUDA kernels and INT8/FP16 quantized matmuls. Each addresses a distinct bottleneck that ZeRO alone cannot solve.

## Context & Motivation

ZeRO solved the memory wall for dense model training. Three gaps remained once you move beyond that baseline:

**Gap 1 — MoE training.** ZeRO partitions parameters uniformly across data-parallel ranks. Mixture-of-Experts adds a router that assigns each token to one or more experts, which may live on entirely different GPUs. This demands all-to-all communication — a fundamentally different pattern from ZeRO's reduce-scatter and all-gather. Running MoE on top of ZeRO naively collapses all experts onto every rank, negating the parameter-efficiency advantage of sparse activation.

**Gap 2 — RLHF training.** PPO requires the actor model to alternate between two incompatible operational modes: (a) autoregressive generation, which benefits from KV caching, continuous batching, and high-throughput inference kernels; and (b) gradient computation, which benefits from ZeRO-3 parameter sharding and activation checkpointing. No training framework in 2022–2023 handled this duality gracefully. The common workaround — a separate inference server for generation and a separate training job for updates — doubled infrastructure cost and added complex weight-sync engineering.

**Gap 3 — Inference serving.** Training-optimized kernels are wasteful at inference time. PyTorch's autograd graph, cuBLAS's general GEMM kernels, and layer-by-layer execution leave significant memory bandwidth on the table. A transformer block at inference involves repeated reads of the same weight matrices for QKV projections, attention, and FFN — fusing these into a single CUDA kernel and expressing attention with a custom tiling strategy can double effective throughput on the same hardware.

## Core Method

### DeepSpeed-MoE: Expert Parallelism with Hierarchical All-to-All

**Expert parallelism (EP)** assigns a disjoint subset of experts to each GPU (or group of GPUs). During the forward pass, a top-k gating function assigns each token to k experts:

$$\text{output} = \sum_{i \in \text{top-k}} G_i(x) \cdot E_i(x)$$

where $G_i(x)$ is the softmax gate score for expert $i$ and $E_i(x)$ is that expert's feed-forward computation. Tokens must physically travel to the GPUs holding their assigned experts, requiring an **all-to-all collective**: each GPU sends a variable-length batch of tokens to every other GPU.

The key systems challenge is that all-to-all across many nodes is expensive and unpredictable. DeepSpeed-MoE addresses this with a **hierarchical all-to-all** in two stages:

1. **Intra-node all-to-all**: GPUs within a node communicate over NVLink (~600 GB/s bidirectional on A100s). Expert groups are packed so that each node holds a contiguous subset of experts, maximizing NVLink utilization.
2. **Inter-node all-to-all**: Only the tokens that genuinely need to cross nodes travel over InfiniBand (~25 GB/s effective per link). By designing the expert assignment to minimize cross-node traffic, the slow link is rarely the bottleneck.

**Combining EP with ZeRO**: Dense layers (attention, layer norm, embedding) are still sharded with ZeRO-3 across all ranks. Expert weights use EP. The two strategies are composed: a rank may own 1/N of attention parameters (ZeRO shard) and 1/E of expert parameters (EP shard), where N is the DP degree and E is the EP degree.

**PR-MoE (Pyramid-Residual MoE)**: DeepSpeed-MoE introduces a model architecture variant that uses fewer experts in early transformer layers (where representations are less specialized) and more in later layers. A residual connection between the dense MLP and the MoE layer allows the expert pathway to specialize on residuals rather than full representations. This design reduces total parameter count by up to 8× at the same quality compared to a dense baseline, because sparse activation is most efficient when experts have clear specialization — which emerges later in the network.

**Load balancing**: The router must avoid routing all tokens to the same expert (collapse). DeepSpeed-MoE uses an auxiliary load-balancing loss standard in MoE literature, plus a capacity-factor mechanism that drops tokens exceeding a per-expert buffer size to bound memory usage and communication volume.

### DeepSpeed-Chat: Hybrid Engine for RLHF

PPO in RLHF requires four computational phases per step:

1. **Generation (rollout)**: actor model generates responses autoregressively. Favors: KV cache, tensor parallelism for decode throughput, contiguous batching.
2. **Reward scoring**: a frozen reward model scores each (prompt, response) pair.
3. **KL computation**: reference model computes log probabilities for KL divergence penalty.
4. **PPO update**: actor and critic run forward and backward passes. Favors: ZeRO-3 parameter sharding, gradient checkpointing, optimizer state offload.

The **hybrid engine** makes a single actor model switch between these modes without maintaining two separate model copies:

**Inference mode**: the actor loads weights into a layout optimized for generation — contiguous weight buffers, KV cache allocated, tensor parallelism shards merged for fast matmul. DeepSpeed's own inference kernels (fused attention, INT8 projection) are activated.

**Training mode**: the actor switches to ZeRO-3 layout — weights scattered across ranks, gradients tracked, activation checkpointing enabled. The inference kernel is deactivated; standard autograd runs.

The mode switch involves an in-place weight rearrangement. For a 7B model this takes roughly 100–200 ms, acceptable given that a full generation step takes seconds. The key engineering insight is that the weight data does not need to be copied — the same GPU memory buffers are reinterpreted with different stride and partition metadata.

**Memory management for reward and reference models**: during the actor training phase, the reward model and reference model are offloaded to CPU memory. During generation, they are restored to GPU. At 7B scale, CPU offload saves 30–40% of GPU memory, allowing a larger actor to fit. The transfer latency (~0.5 s per transition at 7B) is absorbed into the pipeline since reward scoring and reference computation happen between generation and training, not on the critical path.

The full RLHF pipeline implemented in DeepSpeed-Chat runs end-to-end from a pre-trained or SFT-fine-tuned base model to a PPO-trained policy in a single training script, without requiring the user to manage separate serving and training infrastructure.

### DeepSpeed Inference Engine: Fused Kernels and Quantized Serving

The inference engine targets the throughput bottleneck in serving: repeated memory bandwidth pressure from reading large weight matrices for each token.

**Kernel fusion strategy**: a standard transformer forward pass reads weight matrices for QKV projection, applies attention (which reads KV cache), reads the output projection weight, reads two FFN weight matrices, and reads layer-norm parameters — roughly 7–8 distinct global memory reads per transformer block. DeepSpeed's fused transformer kernel collapses this to 2–3 reads by:

- Fusing QKV projection into a single batched GEMM
- Fusing softmax, scaled dot-product attention, and dropout into one kernel with tiled SRAM computation
- Fusing the output projection + residual add + layer norm into the same kernel launch
- Pre-computing layer-norm fused with the following projection

For a model with 32 transformer blocks, this halves the number of CUDA kernel launches and reduces the memory bandwidth footprint proportionally.

**INT8 quantization**: the engine supports per-channel symmetric quantization of weight matrices. Weights are stored in INT8 (halving weight memory), and dequantized on-the-fly to FP16 during each matmul using a custom CUDA kernel that performs the scale factor multiply in the same warp as the GEMM. Activations remain in FP16 throughout. The effective memory bandwidth for weight reads drops from 2 bytes/element (FP16) to 1 byte/element (INT8), directly doubling throughput for memory-bandwidth-bound layers.

**Tensor parallelism without framework overhead**: the inference engine implements Megatron-style tensor parallelism for attention heads and FFN columns, but strips away the PyTorch distributed primitives and implements the required all-reduce directly in CUDA using NCCL primitives, avoiding the overhead of Python-level collective scheduling.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Expert parallelism vs tensor parallelism for MoE | Each expert is a complete, unpartitioned module on one GPU — no cross-device splits within expert computation; simple expert-level kernels | All-to-all communication volume grows with EP degree; beyond EP=16, cross-node all-to-all becomes the throughput bottleneck, especially on InfiniBand clusters |
| Hierarchical all-to-all (intra then inter-node) | NVLink saturation for local expert routing; InfiniBand only carries genuine cross-node traffic; 2–3× lower all-to-all latency vs flat all-to-all | Expert-to-node assignment must be designed in advance; dynamic load imbalance can force cross-node traffic anyway; debugging communication stalls is harder with two-level schedules |
| Hybrid engine mode-switching in Chat | Single actor model for both generation and training; no separate inference deployment or weight sync between separate processes | In-place weight rearrangement adds ~100–200 ms overhead per mode switch at 7B scale; custom weight layout code is fragile across model architectures |
| CPU offload for reward and reference models in Chat | Fits a larger actor on GPU; 30–40% memory reduction during training phase without a separate GPU budget for frozen models | CPU→GPU transfer latency (~0.5 s at 7B) on every PPO step; PCIe bandwidth becomes a bottleneck at 70B+ scale where transfer time approaches generation time |
| Fused inference kernels (INT8 + attention fusion) | 2–4× throughput improvement over standard PyTorch at the same hardware; lower memory bandwidth pressure per token | Hand-written CUDA; architecture-specific (A100/H100); not composable with arbitrary HuggingFace model changes; requires recompilation when model architecture changes |
| Per-channel INT8 quantization (weights only) | Halves weight memory bandwidth cost; minimal accuracy degradation vs FP16 for most transformer architectures | Cannot quantize activations without calibration data and more complex kernel logic; for MoE models, expert routing decisions made in FP16 can interact poorly with INT8 weight reads in low-capacity experts |

## Experiments & Results

**DeepSpeed-MoE**: training a 52B MoE model (with 128 experts) achieves 1.45× throughput compared to a dense model of equivalent quality (350M active parameters, 350M dense baseline). PR-MoE achieves 8× reduction in total model parameters at the same perplexity on language modeling benchmarks compared to a dense model, by concentrating parameters in later layers where expert specialization is highest. On serving, MoE models are 4.5× cheaper to inference than dense models at equivalent quality because only a fraction of parameters are activated per token.

**DeepSpeed-Chat**: RLHF training on OPT-13B is 15× faster than a naive HuggingFace TRL implementation on the same hardware, primarily due to the hybrid engine eliminating the separate inference server. OPT-66B PPO training completes in under 9 hours on 64 A100s; the same workload on a naive baseline would require multiple days. The end-to-end pipeline — from SFT checkpoint to PPO-trained model — runs in a single `deepspeed` launch command, which lowered the practical barrier to RLHF experimentation significantly in 2023.

**Inference Engine**: on BERT-style models, DeepSpeed's fused kernels achieve 1.9× throughput improvement over NVIDIA's FasterTransformer. On GPT-style decoder-only models, the improvement is 1.56× on average, with higher gains for longer sequences where attention becomes the bottleneck. INT8 quantization adds a further 1.3–1.5× improvement on top of kernel fusion by reducing weight-read bandwidth.

## Reproducibility Notes

All three systems are fully open source at [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed).

**Installation**: `pip install deepspeed`. Custom CUDA extensions are compiled at install time; requires CUDA 11.6+ and a compatible GCC version. Pre-compiled wheels are available for common CUDA/PyTorch combinations.

**DeepSpeed-MoE**: available via `deepspeed.moe.layer.MoE` — a drop-in replacement for standard FFN layers. PR-MoE requires using the `DeepSpeedMoEConfig` class. Expert parallelism degree is set as a `deepspeed_config.json` parameter.

**DeepSpeed-Chat**: the `examples/chat` directory contains a three-step training script (SFT → reward model training → PPO). Running on a single 8× A100 node with OPT-1.3B is achievable with the provided configuration. Scaling to 13B or 66B requires adjusting the ZeRO stage and offload settings in the config JSON.

**Inference Engine**: `deepspeed.init_inference(model, mp_size=N, dtype=torch.int8)` wraps any HuggingFace model and activates the fused kernel path. Tensor parallelism requires launching with `deepspeed --num_gpus N`. Fused kernels are validated on A100 and H100; V100 support is partial (no INT8 on older CUDA cores).

**Known caveats**: the hybrid engine in DeepSpeed-Chat is sensitive to model architecture — custom attention implementations in newer HuggingFace model variants may bypass the fused kernel path and silently fall back to slower PyTorch execution. Always check the `ds_report` output to verify kernel activation.

## Commentary

DeepSpeed's trajectory illustrates what a systems research lab looks like when it tracks the frontier in real time. Each component addresses the bottleneck that emerged as the previous one was solved: ZeRO cleared the memory wall for dense training, which enabled large dense models, which created the MoE demand; RLHF emerged as the alignment method of choice, which created the mode-switching demand; scaling to production serving created the inference kernel demand. The components are reactive to the research agenda rather than speculative.

The hybrid engine in DeepSpeed-Chat is conceptually the same insight as verl's HybridFlow (covered separately): the actor model must serve two masters — generation throughput and training memory efficiency — and you need a single framework that handles both. DeepSpeed-Chat's answer (in-place weight rearrangement, CPU offload for frozen models) is older and less composable than verl's (per-model parallelism groups, FSDP + vLLM separation). DeepSpeed-Chat works well for single-node or small-cluster RLHF; verl's architecture becomes necessary at 70B+ multi-node scale.

For inference, DeepSpeed's fused kernels were competitive with FasterTransformer in 2022 and ahead of vanilla PyTorch by a wide margin. By 2023, vLLM's PagedAttention plus FlashAttention-2 surpassed DeepSpeed Inference for variable-length serving workloads. The fundamental reason: DeepSpeed's inference engine assumes fixed-length batches and a static KV cache layout, which works well for offline throughput benchmarks but poorly for production serving with heterogeneous request lengths. PagedAttention's dynamic KV cache management fills that gap.

DeepSpeed remains the most accessible entry point for practitioners who want to experiment with custom RLHF pipelines at small-to-mid scale without building distributed infrastructure from scratch. Its `deepspeed_config.json` abstraction — where all parallelism, offload, and quantization decisions are expressed in a single JSON file — is a significant usability advantage over Megatron's C++/Python hybrid configuration.

## References

- [1] Rajbhandari et al. "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale." arXiv:2201.05596, 2022.
- [2] Aminabadi et al. "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale." arXiv:2207.00032, 2022.
- [3] Yao et al. "DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales." arXiv:2308.01320, 2023.
- [4] Rajbhandari et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC '20 / arXiv:1910.02054.
- [5] Lepikhin et al. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." arXiv:2006.16668, 2021.
- [6] Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR 2022 / arXiv:2101.03961.
- [7] Sheng et al. "HybridFlow: A Flexible and Efficient RLHF Framework." arXiv:2409.19256, 2024.
