# verl —— HybridFlow：灵活高效的 RLHF 训练框架

- **作者 / 机构**：Guangming Sheng、Chi Zhang、Zilingfeng Ye 等 —— 字节跳动 / Seed
- **发表时间**：2024-09
- **链接**：[论文 (arXiv:2409.19256)](https://arxiv.org/abs/2409.19256) · [代码](https://github.com/volcengine/verl) · [博客](https://github.com/volcengine/verl/blob/main/README.md)

## 一句话总结

verl（火山引擎强化学习）是字节跳动开源的 RLHF 训练框架。它通过引入 **HybridFlow** 解决了 PPO/GRPO 规模化时的"四模型内存墙"：一个 Python 进程驱动 RL 控制循环，执行则分布到 GPU 集群上，**每个模型可独立配置张量并行 / 流水线并行 / 数据并行策略**。Actor、Critic、Reference 和 Reward Model 各自以独立分布式进程组运行，通过 CPU offloading 和自动张量 resharding 协同工作。verl 是让 GRPO 和 PPO 在 70B+ 参数规模下切实可行的工程基础——具体来说，它是开放复现 DeepSeek-R1 风格训练的核心框架。

## 背景与动机

DPO 简化了离线偏好学习，GRPO（DeepSeekMath/R1）消除了 Critic 模型，但剩下的硬问题依然是：**在规模上做在线强化学习**。PPO 同时需要四个模型：

1. **Actor**（被训练的策略）
2. **Reference policy**（冻结的 SFT 模型，用于 KL 正则化）
3. **Critic**（价值网络，用于优势估计）
4. **Reward model**（对 rollout 回复打分）

GRPO 去掉了 Critic，但仍需要 Actor + Reference + Reward Model。在 70B 规模下：

- Actor 权重：BF16 下 140 GB → 仅持有参数就需要 4 张以上 H100 加上张量并行
- Reference model：另外 140 GB
- Reward model：另外 70+ GB
- 三者都必须在单个训练步内运行

系统层面的挑战是在内存压力下的协调调度。现有方案各有取舍：

**TRL / OpenRLHF（基于 Ray）**：通过中央 Ray 协调器串行化模型切换。同一时间只有一个模型在运行，模型切换期间 GPU 空闲。实现简单，吞吐受串行化开销限制。在 70B 下，RL 循环的 GPU 利用率常在 40–50% 左右。

**DeepSpeed-Chat**：通过 ZeRO-3 将所有模型共置于同一个 GPU 池。灵活，但强制所有模型共享相同的并行策略。一个用 TP=4 就够的 7B Reward Model，被迫跟随 Actor 的 TP=8，浪费互联带宽。

**基于 Megatron 的定制栈**：效率高，但可扩展性差。实现一个新的 RL 算法变体（如 GRPO → DAPO → 未来变种）需要深度修改训练循环。

verl 的核心洞察：瓶颈不在任何单一模型，而在模型**之间的接口**。如果将 RL 循环逻辑（Python 控制流）与分布式执行（GPU 集群）解耦，并允许每个模型选择自己的并行策略，就能在不牺牲可扩展性的情况下恢复效率。

## 核心方法

### HybridFlow 编程模型

verl 的架构分为两层：

**单控制器 RL 循环**：一个 Python 进程（控制器）以普通命令式代码实现 RL 算法——一个遍历训练步骤的 `for` 循环，依次调用 `actor.generate()`、`reward_model.score()`、`reference.log_probs()`、`critic.get_values()` 和 `actor.update()`。这看起来像写一个顺序训练脚本；RL 算法逻辑与分布式执行完全解耦。

**多控制器执行**：每个模型（Actor、Critic、Reference、Reward）背后是一个独立的分布式**工作组**，拥有自己的进程组、TP/PP/DP 配置和 NCCL 通信子。当控制器调用 `actor.generate()` 时，Actor 工作组在内部以 TP=8、DP=4 的配置运行类 vLLM 生成；调用 `reward_model.score()` 时，Reward Model 工作组以 TP=2、DP=16 运行自己的计算。控制器只需等待结果。

这种分离意味着 RL 算法作者写的是顺序 Python；分布式复杂性封装在各工作组内部，不外泄到 RL 循环。

**逐模型并行配置**：每个模型根据自身大小和计算特性选择最优配置：

| 模型 | 70B 规模下的典型配置 | 原因 |
|---|---|---|
| Actor（70B） | TP=8，DP=4，pipeline=1 | 大模型；需要宽 TP 支撑生成吞吐 |
| Reference（70B） | TP=8，DP=4 | 与 Actor 等大；不用时 offload 到 CPU |
| Critic（70B） | TP=4，DP=8 | 推理密集；价值估计偏向宽 DP |
| Reward Model（7B–13B） | TP=2，DP=16 | 更小；宽 DP 实现高吞吐打分 |

**步间 CPU Offloading**：暂时不参与计算的模型在其前向/反向调用之间 offload 到 CPU DRAM。H100 配备 NVLink 4.0（900 GB/s 双向 GPU 带宽）和 PCIe 5.0（128 GB/s CPU↔GPU），140 GB 的 Actor 可在不到 1 秒内完成完整传输。RL 步通常耗时 30–120 秒；offload 延迟相对于计算时间可忽略不计。

这使得 verl 同一时刻只需在 GPU 上保留 2 个模型而非 4 个，以传输开销换取一半的 GPU 内存压力。

**Actor 的 FSDP + vLLM 集成**：Actor 承担双重角色——既要高效生成补全（rollout 阶段），又要执行梯度更新（训练阶段）。两者需求相互矛盾：

- **生成**偏向 vLLM：PagedAttention 高效管理 KV cache、连续批处理、针对高吞吐 decode 优化的张量并行。
- **训练**偏向 FSDP（ZeRO-3）：参数跨数据并行 rank 分片、梯度累积、优化器状态分布。

verl 同时维护 Actor 的两种表示。在 rollout 阶段，Actor 权重加载进 vLLM 引擎实例。经 FSDP 完成 PPO/GRPO 更新后，verl 将更新后的权重从 FSDP 分片同步到 vLLM worker。权重同步使用 NCCL all-gather 后将参数赋值回 vLLM 模型的 `state_dict`。这一同步步骤是 verl Actor 实现中最主要的工程复杂度所在。

**跨模型数据的混合 Resharding**：当数据在并行配置不同的模型间传递时——例如 Actor 以 TP=8、DP=4 生成的回复需要传给 TP=2、DP=16 的 Reward Model——张量的分片方式必须改变。verl 通过自动 resharding 处理：张量在控制器侧 gather，再分发到目标工作组，按新的布局 re-shard。对于小张量（回复序列），这很廉价；对于大张量（权重梯度），verl 通过设计完全避免了跨模型的权重传输。

**GRPO 场景**：GRPO 去掉了 Critic，将循环简化为：

1. Actor 对每个 prompt 生成 G 个 rollout（如 G=8）
2. Reward Model 对全部 G 个 rollout 打分
3. 优势 = (reward − 组内平均) / 组内标准差
4. 对 Actor 应用基于组相对优势的 PPO 风格裁剪目标

verl 用同样的 HybridFlow 抽象实现 GRPO——控制器循环约 50 行 Python；分布式复杂性不变。这是用于复现 DeepSeek-R1-Zero 风格训练的配置。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 逐模型并行配置 | 每个模型使用最优 TP/PP/DP；无单一瓶颈；Actor 生成和 Reward 打分均可跑满吞吐 | 配置复杂度；工程师需理解每个模型的计算/内存特性才能有效调优 |
| 步间 CPU Offloading | 同一时刻 GPU 上只保留 2 个模型而非 4 个；有效模型容量翻倍 | 每步 CPU↔GPU 传输延迟；需要快速 PCIe / NVLink；超大 batch 时内存带宽成为约束 |
| vLLM 负责 rollout 生成 | Actor rollout 期间拥有完整 vLLM 吞吐（PagedAttention、连续批处理）；GPU 利用率高 | 每次策略更新需做权重 resharding（FSDP 分片 → vLLM state dict）；每步增加约 5–15% 开销 |
| 单 Python 控制器 | RL 循环可读；新算法变体（GRPO、DAPO 等）几乎零成本实现；控制器框架无关 | 控制器是集中式瓶颈；极细粒度的并行决策必须内化在工作组中 |
| FSDP 负责 Actor 训练 | ZeRO-3 内存效率；与 HuggingFace 模型和标准优化器无缝集成 | 对 100B+ 模型不如 Megatron-LM 的 3D 并行灵活；流水线并行集成有限 |
| Ray 负责集群编排 | 弹性伸缩；异构工作组；集群级容错 | Ray 开销；Ray Actor 模型对频繁小消息增加延迟；对极低延迟工作负载不理想 |

## 实验与结果

verl 在 32× H100 集群（4 节点，每节点 8 张 H100，节点内 NVLink 4.0）上做了基准测试：

**与基线对比吞吐量**（70B Actor + 70B Critic，PPO，LLaMA 3 70B）：
- verl：tokens/second 吞吐比 OpenRLHF 高 1.4×
- verl：吞吐比 TRL（HuggingFace PPO）高 2.1×
- rollout GPU 利用率：verl 达 82%，Ray 串行基线仅 43%

利用率差距是核心结论：朴素的单模型串行执行在模型切换期间浪费了超过一半的 GPU 算力。HybridFlow 的 offloading + 逐模型并行填补了这一空白。

**70B 规模 GRPO**：verl 在 LLaMA 3 70B 上以数学验证器 Reward Model 实现的 GRPO，与 DeepSeek-R1 技术报告中描述的 R1-Zero 训练动态高度吻合。这一复现演示可以说是 verl 最具影响力的贡献——它让前沿 RL 训练在专有集群之外变得可及。

**内存效率**：在 32× H100（共 2.5 TB HBM）上，verl 可同时运行 70B Actor + 70B Critic + 13B Reward Model，且序列长度最高可达 8k。同等配置下，OpenRLHF 在序列长度 4k 时就因内存不足而 OOM。

**算法灵活性**：verl 开箱支持 PPO、GRPO、ReMax 和 PRIME（基于过程奖励模型的隐式 MLE）。控制器层的抽象意味着添加新 RL 变体通常只需修改约 100 行 Python，无需触碰分布式基础设施。

## 复现要点

verl 完全开源，地址：[github.com/volcengine/verl](https://github.com/volcengine/verl)。仓库包含：

- H100 环境预构建 Docker 镜像（CUDA 12.1 + PyTorch 2.3 + vLLM 0.4）
- 对应 DeepSeek-R1-Zero 数学训练配置的 GRPO 方案
- 通用指令跟随的 PPO 方案
- 对 LLaMA、Mistral、Qwen 系列的 HuggingFace 兼容模型加载
- AWS 和 GCP 的 Ray 集群配置示例

**依赖环境**：配备 NCCL 的 Ray 集群，支持 CUDA 的 GPU（在 H100 和 A100 上测试过）。FSDP↔vLLM 权重同步需要 PyTorch 2.3+ 以在 TP 规模下可靠地处理 `state_dict`。

**已知局限**：vLLM 权重同步步骤目前通过 Ray 控制器串行进行，在节点间带宽较慢的集群上会增加延迟。对于 8 卡单节点，这可以忽略；对于 32 卡以上多节点，每次策略更新可能增加 5–10 秒。社区正在推进直接用 NCCL 从 FSDP worker 广播权重到 vLLM worker，绕过控制器中转。

**调优建议**：verl 的性能优势在大 batch size 和长序列（KV cache 压力使生成成为瓶颈）时最为突出。在小 batch / 短序列下，offloading 和 resharding 的开销可能抵消优势。相对于 TRL 取得有意义吞吐提升的推荐最低配置是 8× H100、序列长度 ≥ 2k。

## 个人评注

verl 是 RLHF 算法论文（理论）与生产 RL 训练（工程实践）之间缺失的那一层。它回答了大多数 GRPO 和 PPO 论文留而未答的问题："好，算法对了——但你怎么在 32 张 H100 上跑 70B 模型而不让集群崩掉？"

HybridFlow 的洞察——RL 循环中不同模型应有不同并行配置——在事后看来显而易见，但在提出时并不平凡。每个有经验的 ML 工程师都知道 70B 模型和 7B Reward Model 有不同的最优 TP/DP 分配。verl 的贡献在于让这一直觉在操作层面具体化：框架处理 resharding 的胶水代码，工程师无需手写。

FSDP + vLLM Actor 架构是 verl 最锋利的工程决策。它解决的问题——"我需要 Actor 在生成时快，在梯度更新时也快，而这两者想要不同的并行策略"——真实存在且不容易漂亮地解决。verl 的答案（维护两种表示，每次更新后同步）务实有效，代价是同步开销。替代方案（全程用 Megatron，在 Megatron 内实现 vLLM 风格的生成）效率更高，但可移植性大幅下降。

从研究角度看，verl 的意义在于它民主化了训练方案。在 verl 之前，在 70B 规模复现 DeepSeek-R1 风格的 GRPO 训练，要么需要专有基础设施，要么需要大量定制工程。verl 显著降低了这一门槛——任何拥有 16 张以上 H100 的团队，现在只需几天工程努力而非几个月，即可运行前沿 RL 训练。

对系统构建者更大的启示是：**RL 训练与预训练在系统层面有根本不同的需求**。预训练是单模型、单前向、单反向的循环。RLHF/GRPO 在单个步骤内包含多个模型、生成（推理）、打分（推理）和梯度更新（训练）。为一者设计的系统用到另一者就是错的。verl 是第一个在架构层面认真对待这一差异的框架——而不是在训练框架上事后拼接推理能力。

## 参考

- [1] Sheng et al. _HybridFlow: A Flexible and Efficient RLHF Framework._ arXiv:2409.19256，2024。
- [2] Schulman et al. _Proximal Policy Optimization Algorithms._ arXiv:1707.06347，2017。
- [3] Shao et al. _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models._ arXiv:2402.03300，2024。
- [4] DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ arXiv:2501.12948，2025。
- [5] Ouyang et al. _Training Language Models to Follow Instructions with Human Feedback._ arXiv:2203.02155，2022。
- [6] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180。
- [7] Rajbhandari et al. _ZeRO: Memory Optimizations Toward Training Trillion Parameter Models._ SC '20 / arXiv:2101.06840。
