# verl — HybridFlow: A Flexible and Efficient RLHF Training Framework

- **Authors / Org**: Guangming Sheng, Chi Zhang, Zilingfeng Ye et al. — Bytedance / Seed
- **Published**: 2024-09
- **Links**: [paper (arXiv:2409.19256)](https://arxiv.org/abs/2409.19256) · [code](https://github.com/volcengine/verl) · [blog](https://github.com/volcengine/verl/blob/main/README.md)

## TL;DR

verl (Volcano Engine Reinforcement Learning) is Bytedance's open-source RLHF training framework. It solves the "4-model memory wall" of PPO and GRPO at scale by introducing **HybridFlow**: a programming model where a single Python process drives the RL control loop while execution is distributed across a GPU cluster via **heterogeneous parallelism strategies per model**. The actor, critic, reference, and reward model can each run with an independent tensor-parallel / pipeline-parallel / data-parallel configuration, coexisting via CPU offloading and automatic tensor resharding. verl is the infrastructure that makes algorithms like GRPO and PPO feasible at 70B+ parameter scale — it is, concretely, the framework underlying the open reproduction of DeepSeek-R1-style training.

## Context & Motivation

After DPO simplified offline preference learning and GRPO (DeepSeekMath/R1) eliminated the critic model, the remaining hard problem is **online RL at scale**. PPO requires four models simultaneously:

1. **Actor** (policy being trained)
2. **Reference policy** (frozen SFT model, for KL regularization)
3. **Critic** (value network, for advantage estimation)
4. **Reward model** (scorer for rollout responses)

GRPO eliminates the critic but still requires actor + reference + reward model. At 70B:

- Actor weights: 140 GB in BF16 → needs 4+ H100s with tensor parallelism just to hold the parameters
- Reference model: another 140 GB
- Reward model: another 70+ GB
- All three must run within a single training step

The systems challenge is coordination under memory pressure. Existing approaches made conflicting choices:

**TRL / OpenRLHF (Ray-based)**: serialize model transitions via a central Ray coordinator. One model runs at a time; GPUs idle between transitions. Simple to implement; throughput limited by serialization overhead. At 70B, GPU utilization during the RL loop is often around 40–50%.

**DeepSpeed-Chat**: co-locate all models on the same GPU pool via ZeRO-3. Flexible but forces all models to share the same parallelism strategy. A reward model that fits on 4 GPUs with TP=4 is over-parallelized if forced to match the actor's TP=8, wasting interconnect bandwidth.

**Megatron-based custom stacks**: high efficiency but poor extensibility. Implementing a new RL algorithm (e.g., GRPO → DAPO → future variants) requires deep modifications to the training loop.

verl's key insight: the bottleneck is not any single model but the *interface* between models. If you decouple the RL loop logic (Python control flow) from the distributed execution (GPU cluster) and allow each model to choose its own parallelism, you recover efficiency without sacrificing extensibility.

## Core Method

### HybridFlow Programming Model

verl's architecture has two layers:

**Single-controller RL loop**: A Python process (the controller) implements the RL algorithm as normal imperative code — a `for` loop over training steps, with function calls to `actor.generate()`, `reward_model.score()`, `reference.log_probs()`, `critic.get_values()`, and `actor.update()`. This looks like writing a sequential training script; the RL algorithm logic is fully decoupled from the distributed execution.

**Multi-controller execution**: Each model (actor, critic, reference, reward) is backed by an independent distributed **worker group** with its own process group, TP/PP/DP configuration, and NCCL communicator. When the controller calls `actor.generate()`, the actor worker group runs vLLM-style generation internally using its own TP=8, DP=4 configuration. When it calls `reward_model.score()`, the reward model worker group runs its own TP=4, DP=8 computation. The controller simply awaits results.

This separation means the RL algorithm author writes sequential Python; the distributed complexity lives inside each worker group, not in the RL loop.

**Per-model parallelism configurations**: Each model picks its own optimal configuration based on its size and compute profile:

| Model | Typical config at 70B scale | Rationale |
|---|---|---|
| Actor (70B) | TP=8, DP=4, pipeline=1 | Large; needs wide TP for generation throughput |
| Reference (70B) | TP=8, DP=4 | Same size as actor; offloaded to CPU between uses |
| Critic (70B) | TP=4, DP=8 | Inference-heavy; DP favored for value estimates |
| Reward model (7B–13B) | TP=2, DP=16 | Smaller; wide DP for high-throughput scoring |

**CPU offloading between steps**: Models not actively computing are offloaded to CPU DRAM between their forward/backward calls. On H100s with NVLink 4.0 (900 GB/s bidirectional GPU bandwidth) and PCIe 5.0 (128 GB/s CPU↔GPU), a 140 GB actor can be fully transferred in under 1 second. The RL step typically takes 30–120 seconds; offloading latency is negligible relative to computation time.

This allows verl to fit 2 models on GPU at a time rather than 4, halving the GPU memory pressure at the cost of transfer overhead.

**FSDP + vLLM integration for actor**: The actor has dual roles — it must generate completions efficiently (rollout phase) and update its parameters with gradient steps (training phase). These have conflicting requirements:

- **Generation** favors vLLM: PagedAttention for memory-efficient KV cache management, continuous batching, tensor parallelism optimized for high-throughput decode.
- **Training** favors FSDP (ZeRO-3): parameter sharding across data-parallel ranks, gradient accumulation, optimizer state distribution.

verl maintains the actor in two representations simultaneously. During rollout, actor weights are loaded into a vLLM engine instance. After the PPO/GRPO update via FSDP, verl synchronizes the updated weights from FSDP shards to the vLLM workers. The weight sync uses NCCL all-gather followed by parameter assignment into the vLLM model's state dict. This sync is the main engineering complexity in verl's actor implementation.

**Hybrid resharding for cross-model data**: When data passes between models with different parallelism configs — e.g., actor generates (TP=8, DP=4) responses that must be scored by the reward model (TP=2, DP=16) — the tensor sharding must change. verl handles this with automatic resharding: tensors are gathered on the controller, redistributed to the target worker group, and re-sharded to the new layout. For small tensors (response sequences), this is cheap; for large tensors (weight gradients), verl avoids cross-model weight transfers entirely by design.

**The GRPO case**: GRPO removes the critic, simplifying the loop to:

1. Actor generates G rollouts per prompt (e.g., G=8)
2. Reward model scores all G rollouts
3. Advantage = (reward − mean(group rewards)) / std(group rewards)
4. PPO-style clipped objective applied to actor using group-relative advantages

verl implements GRPO with the same HybridFlow abstraction — the controller loop is ~50 lines of Python; the distributed complexity is unchanged. This is the configuration used for reproducing DeepSeek-R1-Zero style training.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Per-model parallelism configs | Each model uses its optimal TP/PP/DP; no single bottleneck; actor generation and reward scoring can both run at peak throughput | Configuration complexity; engineers must understand each model's compute/memory profile to tune effectively |
| CPU offloading between steps | 2 models on GPU at a time instead of 4; 2× larger effective model fits | CPU↔GPU transfer latency per step; requires fast PCIe / NVLink; memory bandwidth becomes a constraint at very large batch |
| vLLM for rollout generation | Full vLLM throughput (PagedAttention, continuous batching) during actor rollout; high GPU utilization | Weight resharding overhead at each policy update (FSDP shards → vLLM state dict); adds ~5–15% per-step overhead |
| Single Python controller | Readable RL loop; trivial to implement new algorithm variants (GRPO, DAPO, etc.); controller is framework-agnostic | Controller is a centralized bottleneck; very fine-grained parallelism decisions must happen inside worker groups |
| FSDP for actor training | ZeRO-3 memory efficiency; integrates naturally with HuggingFace models and standard optimizers | Less flexible than Megatron-LM's 3D parallelism for models > 100B; pipeline parallelism integration is limited |
| Ray for cluster orchestration | Elastic scaling; heterogeneous worker groups; fault tolerance at cluster level | Ray overhead; Ray actor model adds latency for frequent small messages; not ideal for very latency-sensitive workloads |

## Experiments & Results

verl benchmarks on a 32× H100 cluster (4 nodes, 8 H100s each, NVLink 4.0 intra-node):

**Throughput vs baselines** (70B actor + 70B critic, PPO with LLaMA 3 70B):
- verl: 1.4× higher tokens/second throughput vs OpenRLHF
- verl: 2.1× higher throughput vs TRL (HuggingFace PPO)
- Rollout GPU utilization: 82% for verl vs 43% for the Ray-serialized baseline

The utilization gap is the key result: naive single-model-at-a-time execution wastes over half of GPU compute on idle time during model transitions. HybridFlow's offloading + per-model parallelism fills this gap.

**GRPO at 70B scale**: verl's GRPO implementation on LLaMA 3 70B with a math verifier reward model achieves similar training dynamics to the DeepSeek-R1-Zero setup described in the R1 technical report. The reproducibility demonstration is arguably verl's most impactful contribution — it makes frontier RL training accessible outside proprietary clusters.

**Memory efficiency**: At 32× H100 (2.5 TB total HBM), verl can run PPO with a 70B actor + 70B critic + 13B reward model simultaneously, with room for sequence lengths up to 8k. The same configuration on OpenRLHF hits OOM at sequence length 4k due to insufficient memory headroom.

**Algorithm flexibility**: verl ships with out-of-the-box support for PPO, GRPO, ReMax, and PRIME (process reward models with implicit MLE). The controller-level abstraction means adding a new RL variant typically requires modifying ~100 lines of Python without touching the distributed infrastructure.

## Reproducibility Notes

verl is fully open source at [github.com/volcengine/verl](https://github.com/volcengine/verl). The repository includes:

- Pre-built Docker images for H100 setups (CUDA 12.1 + PyTorch 2.3 + vLLM 0.4)
- GRPO recipes matching the DeepSeek-R1-Zero math training setup
- PPO recipes for general instruction following
- HuggingFace-compatible model loading for LLaMA, Mistral, and Qwen families
- Ray cluster configuration examples for AWS and GCP

**Requirements**: Ray cluster with NCCL, CUDA-capable GPUs (tested on H100 and A100). The FSDP↔vLLM weight sync requires PyTorch 2.3+ for reliable `state_dict` handling at TP scale.

**Known limitations**: The vLLM weight sync step currently serializes through the Ray controller, which adds latency on clusters with slow inter-node bandwidth. For 8-GPU single-node setups, this is negligible; for 32+ GPU multi-node setups, it can add 5–10 seconds per policy update. Work is ongoing to implement direct NCCL-based weight broadcast from FSDP workers to vLLM workers, bypassing the controller.

**Calibration**: verl's performance advantage is most pronounced at large batch sizes and long sequences (where KV cache pressure makes generation the bottleneck). At small batch / short sequence, the overhead of offloading and resharding can erode the advantage. The recommended minimum configuration for meaningful throughput gains over TRL is 8× H100s with sequence length ≥ 2k.

## Commentary

verl is the missing engineering layer between RLHF algorithm papers (theory) and production RL training (practice). It answers a question that most GRPO and PPO papers leave implicit: "okay, the algorithm is right — but how do you actually run it on 32 H100s with a 70B model without the cluster falling over?"

The HybridFlow insight — that different models in the RL loop should have different parallelism configurations — is non-obvious but obvious in retrospect. Every serious ML engineer knows that a 70B model and a 7B reward model have different optimal TP/DP splits. What verl does is make that intuition operationally concrete: the framework handles the resharding glue so engineers don't have to.

The FSDP + vLLM actor architecture is the sharpest engineering decision in verl. The problem it solves — "I need the actor to be fast at generation AND fast at gradient updates, and these want different parallelism strategies" — is real and not obvious to solve cleanly. verl's answer (maintain two representations, sync after each update) is pragmatic and works, at the cost of sync overhead. An alternative (use Megatron throughout, implement vLLM-style generation inside Megatron) is more efficient but far less portable.

From a research perspective, verl matters because it democratizes the training recipe. Before verl, reproducing DeepSeek-R1-style GRPO training at 70B required either proprietary infrastructure or significant custom engineering. verl lowers this barrier substantially — any team with access to 16+ H100s can now run frontier RL training with a few days of engineering effort rather than months.

The broader lesson for systems builders: **RL training has fundamentally different systems requirements than pretraining**. Pretraining is a single-model, single-forward-pass, single-backward-pass loop. RLHF/GRPO involves multiple models, generation (inference), scoring (inference), and gradient updates (training) in a single step. Systems designed for one are wrong for the other. verl is the first framework to take this difference seriously at the architecture level rather than bolting inference onto a training framework after the fact.

## References

- [1] Sheng et al. _HybridFlow: A Flexible and Efficient RLHF Framework._ arXiv:2409.19256, 2024.
- [2] Schulman et al. _Proximal Policy Optimization Algorithms._ arXiv:1707.06347, 2017.
- [3] Shao et al. _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models._ arXiv:2402.03300, 2024.
- [4] DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ arXiv:2501.12948, 2025.
- [5] Ouyang et al. _Training Language Models to Follow Instructions with Human Feedback._ arXiv:2203.02155, 2022.
- [6] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180.
- [7] Rajbhandari et al. _ZeRO: Memory Optimizations Toward Training Trillion Parameter Models._ SC '20 / arXiv:2101.06840.
