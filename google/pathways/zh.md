# Pathways —— 面向 ML 的异步分布式数据流

- **作者 / 机构**：Barham 等，Google
- **发表时间**：MLSys '22
- **链接**：[论文 (arXiv:2203.12533)](https://arxiv.org/abs/2203.12533) · [Google 博客](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)

## 一句话总结

Pathways 是 Google 的分布式 ML 运行时，专为**上千 TPU、跨 pod**训练设计，采用**单控制器**架构。与 Megatron 风的多控制器系统（MPI 式，每个 worker 跑同一份 Python）不同，Pathways 用**一个调度器驱动跨多个 pod 的数据流图**，通过分片数据流抽象通信。核心主张：**在极大规模下仍可接近硬件峰值利用率，同时保留单客户端编程模型**，且能跑多控制器系统难以表达的**异构、稀疏、多任务计算**。Pathways 支撑了 **PaLM（540B）**以及其后所有 Google 前沿训练（Gemini 也是，细节未公开）。它也是 JAX `pmap` 时代多控制器风格的架构反命题——现在 JAX 通过 `jit` / `pjit` / `shard_map` 把 Pathways 当作后端。

## 背景与动机

Pathways 之前，前沿规模 ML 训练分两派：

1. **多控制器（SPMD）** —— Megatron、DeepSpeed、Horovod：每个 worker 跑同一份程序，经 collective 协调。简单、密集均匀计算上扩展好，但每个 worker 必须为每一步都在线，异构 / 稀疏 / 异步模式很别扭。分片是隐式的（每个人知道"我的 rank"）。
2. **单控制器（MPMD / MPP）** —— TensorFlow 1.x、`tf.distribute`：一个客户端建图，worker 执行片段。表达复杂图优雅，但历史上慢：控制器成瓶颈，per-op RPC 延迟毁掉小 op workload。

Google 除了单控制器的性能问题，还面对两个 forcing function：

- **TPU pod 物理上异构** —— 训练要特定 pod 形状，不是"任意 512 芯片"。系统必须理解拓扑。
- **未来 workload 是稀疏、多任务、异构的** —— MoE、多任务、投机计算、RL 的 learner/actor 非对称。多控制器难处理。

Pathways 的赌注：**把单控制器性能修好**，就能在前沿规模保住它的表达力。

## 核心方法

### 分片数据流计算

Pathways 程序是一张**数据流图**，节点是**分片计算**——每个节点吃 shard、产出 shard、跑在特定设备子集上。调度器把这张图翻译成设备级操作。

关键：图是**每程序的**，不是全局的：多个程序可共享一个 pod，调度器知道如何共调度，不要求形状相同。

### 异步 gang-scheduled 分发

对每次计算，Pathways：

1. **Gang schedule** —— 所有参与设备承诺在同一逻辑时间跑。保证 collective 对齐。
2. **从控制器异步分发**到每 pod 的**资源管理器**；pod 再发设备级指令。
3. **分发与执行流水线化** —— 当前步跑着，下一步已经在送。把控制器→pod 延迟摊到多步。

这是关键性能技巧：单控制器每步分发慢，但**与计算流水线化后**，只要 per-step 计算 > 分发延迟，延迟就被遮蔽。前沿规模下 per-step 计算永远很大。

### 并行异步分发（PAD）

对有多条独立分支的程序（多任务、投机求值），Pathways 不等前序分支就并行分发。控制器不被不存在的依赖串行化。

### 编译与形状特化

控制器按形状 JIT 编译 shard——像 XLA，但跨 pod。编译产物缓存。多数训练 step 命中缓存。

### 部署：pod 资源管理器

每个 TPU pod 运行一个资源管理器，拥有本地调度、显存、拓扑。Pathways 控制器和资源管理器对话，不直接对芯片。这**把 pod 内部拓扑从控制器抽象掉**——控制器只说"跑这个分片计算"，pod RM 决定具体用哪些芯片、怎么跑。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 单控制器 | 对异构 / 多任务 / 稀疏 workload 表达力强 | 控制器吞吐要扩，历史上是瓶颈 |
| 流水线式异步分发 | 遮蔽 per-step 延迟 | 仅当计算 ≫ 分发时有效；小 op 仍贵 |
| Gang schedule | Collective 正确 | 掉队者拖停 gang；step 级时序无弹性 |
| 每 pod 资源管理器 | 拓扑抽象 | 多一层软件；pod 级调度复杂度 |
| TPU pod 专属 | 与 TPU 拓扑 / 网络深度集成 | 不经重写无法移植到 GPU fabric |
| XLA 做编译器 | 成熟、峰值好 | 编译时间；调试 XLA 是独立技能 |

## 实验与结果

- PaLM（540B）在两个 TPU v4 pod（共 6144 芯片）上由 Pathways 训练——首次公开确认 Pathways 的前沿能力。
- PaLM 训练 **57.8% 硬件 FLOP 利用率**——如此规模下极高，得益于单控制器能调度异构 comm/compute 模式。
- Pathways 展示了**跨 pod 两 pod 训练的 50% 效率**——跨数据中心规模网络，多控制器会差得多。

## 复现要点

- Pathways 本身**未开源**。无公开实现、无代码。
- JAX 的 `pjit` / `shard_map` 是公开面上的编程模型，在 Google Cloud TPU 上背后走 Pathways。JAX TPU 用户隐式地在用 Pathways。
- **MaxText**（Google 开源的 JAX LLM 训练框架）是最接近的公开对照品；可跑在 TPU Pathways 或 GPU-JAX 上。
- **Ray**（UC Berkeley → Anyscale）在更高层次上哲学类似——单控制器分布式 actor 图——但面向更广域、权衡不同。

## 个人评注

Pathways 是 **Megatron / PyTorch FSDP 学派的反命题**。PyTorch/Megatron 派说："一切 SPMD，别挡控制器的路，靠 collective 和精细并行配置扩展。"Pathways 派说："聪明的控制器能调度 SPMD 派调度不了的 workload，只要分发够快，表达力就赢了。"

哪个对？按 workload：

- **密集均匀训练（Llama 3、V3 预训练）** —— SPMD 赢。没有异构性可挖，不需要单控制器开销。
- **稀疏 / 混合 / 异构（差异巨大的 MoE 专家、多任务、学习者-执行者不对称的异步 RL、投机分支）** —— Pathways 架构天然契合。

2026 年两派在收敛。PyTorch `torch.distributed` 添了更像 Pathways 的特性（collective group、异构 process group）。JAX 的 `shard_map` 把显式分片带回多控制器风格。思想距离在缩小。

更深的启示是**什么时候集中化**：Pathways 说明，如果你的调度足够有趣（异构、异步、稀疏），控制器有价值；如果调度很无聊（密集均匀），去中心化。今天多数 LLM 预训练在"无聊"区。agentic / RL / 多任务 workload 正在进入"有趣"区，Pathways 式系统会更重要。

对没有 Google TPU 的读者：读论文为的是架构想法，不是直接使用。Ray 生态、MaxText、JAX on TPU 是实务带走项。

## 参考

- [1] Barham et al. _Pathways: Asynchronous Distributed Dataflow for ML._ MLSys '22 / arXiv:2203.12533.
- [2] Chowdhery et al. _PaLM._ JMLR '23.
- [3] Xu et al. _GSPMD._ arXiv:2105.04663, 2021.（配套并行编译器）
- [4] MaxText：https://github.com/AI-Hypercomputer/maxtext
- [5] JAX：https://github.com/jax-ml/jax
