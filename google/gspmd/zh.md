# GSPMD —— 编译器驱动的自动并行化

- **作者 / 机构**：Xu 等，Google
- **发表时间**：2021-05（arXiv）
- **链接**：[论文 (arXiv:2105.04663)](https://arxiv.org/abs/2105.04663) · [XLA SPMD partitioner](https://github.com/openxla/xla)

## 一句话总结

GSPMD（Generalized SPMD）是 **XLA 编译器 pass**，把 JAX / TensorFlow 程序 + **几个关键张量的 sharding 标注**，自动生成跨数千设备的 SPMD 并行程序——自动插入 collective、re-sharding、优化通信。这就是你写 `w = jax.lax.with_sharding_constraint(w, P('data', 'model'))` 后编译器能从那里推出 TP、FSDP、混合策略的原因。GSPMD 支撑 **PaLM、Gemini 以及所有现代 Google 规模的 JAX 训练**，也是当今所有非 Google 的 JAX TPU / 多 GPU 用户的并行底座。与 Pathways（运行时）并列，GSPMD 是 Google 对"如何给 pod 编程"的另一半答案。

## 背景与动机

Megatron、DeepSpeed 那样的手工并行要求程序员：

- 沿正确轴把权重矩阵切到各设备。
- 在正确位置插 `all-reduce` 或 `all-gather`。
- 保证 collective 跨设备对齐。
- 改 TP 大小、PP 布局、加 ZeRO 时全部重来。

小规模易错，前沿规模不靠一小撮专家根本不可能。TF 1.x `tf.distribute` 处理数据并行不处理模型并行。GPipe 和 Mesh-TensorFlow 有答案但要大改代码。

GSPMD 的目标：**给几个关键张量标 sharding，剩下交给编译器**。程序员声明意图，编译器处理机制。

## 核心方法

### Sharding 作为类型

每个张量带一个 **sharding 规格** —— 哪些 mesh 轴切哪些张量轴。Mesh 是命名的多维设备网格（例如 32 设备摆成 8×4，就是 `{'data': 8, 'model': 4}`）。

2D 张量的 sharding 规格可能是 `P('data', 'model')`：第 1 轴沿 `data` mesh 维切分，第 2 轴沿 `model` 切分。`P(None, 'model')` 沿 `data` 复制、沿 `model` 切分。`P()` 完全复制。

### 传播

给定输入 sharding 和几个显式约束，GSPMD **在计算图上传播 sharding**：

- `a` 是 `P('data', None)`、`b` 是 `P(None, 'model')`，那么 `a @ b` 自然成为 `P('data', 'model')`——编译器自动推出。
- Reduce、reshape、broadcast 各有传播规则。
- 多条传播冲突时，编译器**选一条并插入 re-sharding**（`all-to-all`、`all-gather` 等 collective）调和。

传播在 XLA HLO 上迭代重写完成。一个 collective 的**成本模型**决定哪种 re-sharding 摆放最便宜。

### Partitioner pass

每个 HLO op 的 sharding 一致后，**partitioner** 把每个 op 改写成分片形式：

- 输入分片的 `matmul` 变为分片 matmul + 必要时对收缩维做 `all-reduce`。
- `conv` 类似分片，空间 reduction 专门处理。
- 逐元素 op 自然分片。
- 改形 op（reshape、transpose）可能需要 re-sharding；编译器发所需 collective。

结果：一个 SPMD 程序，每设备跑同一份代码、读写自己那片。与人工在 Megatron 里写出来的完全一致——只是由几十条标注生成。

### 流水线并行走显式 rewrite

GSPMD 不自动做流水线并行；PP 通过**另一条 rewrite**（`pipeline_parallelism_pass`）加入，把图切成 stage、插入 stage 边界。PP 不进 GSPMD 的核心 sharding 逻辑，因为它需要不同的成本推理（bubble、调度）。

### 嵌套与混合策略

因为 sharding 是可组合的，GSPMD **天然支持混合策略**：

- 权重 `P('data', 'model')` = 张量并行。
- 权重 `P('fsdp', None)` + 前向 `all-gather` = FSDP。
- 不同张量组合 = ZeRO-3 + TP。
- 一块代码外加 `shard_map` 标注 = 局部 SPMD 嵌在全局 SPMD 里。

手写这些在 Megatron 里要几千行、小心的状态管理。GSPMD 当标注接受。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 标注驱动自动并行 | 轻松切换 TP/FSDP/混合 | 关键位置仍需标注；不是零配置 |
| 编译器成本模型 | 选到接近最优的 re-sharding | 成本模型可能错；对用户不透明 |
| 核心只 SPMD | 代码模型简单；每设备同代码 | PP 要单独 pass |
| 基于 XLA | 成熟编译器、峰值好 | XLA 编译时间；调试是技能 |
| 传播 + partitioner | 用户标注最少 | partitioner bug 表现为微妙的性能问题 |
| Sharding 作为类型 | 组合干净 | 需要框架配合（仅 JAX、TF） |

## 实验与结果

- 通过 GSPMD + Pathways 训 **PaLM（540B）**，MFU 57.8%。
- 2021 年起 Google 内部**所有生产 JAX 训练**的分片都由 GSPMD 处理。
- 公开对照品：JAX 的 `jit`+`pjit`+`shard_map` 把 GSPMD 行为开放给所有 JAX 用户。生产框架（MaxText、T5X、Praxis）基于 GSPMD。

## 复现要点

- **编译器开源** —— `openxla/xla` 包含 SPMD partitioner。每个 JAX TPU / 多 GPU workload 都在用。
- JAX 文档覆盖用户标注模型；编译器内部在 XLA HLO pass 里。
- JAX 外，OpenXLA 通过 PyTorch/XLA 被 PyTorch 采纳——GSPMD 式并行也在渗入 PyTorch。

## 个人评注

GSPMD 是**前沿 ML 里最安静的顶梁工具**。多数人没听过；多数 JAX 用户每天在用却不自知。它是"消除一整类手工活"的编译器 pass 中最干净的例子之一。GSPMD 之前，每个主要并行训练框架都自己发明切张量、插 collective 的机制；GSPMD 之后，你标注一下就完事。

更有意思的对照是 PyTorch 的轨迹。PyTorch 历史上押**eager 心智模型**——一切显式定义。并行由用户掌管（DDP、FSDP、PyTorch 版 Megatron）。对研究灵活性好，对策略组合糟糕。PyTorch 现在正加入越来越 GSPMD 化的编译器路径（`torch.compile`、`torch.distributed._tensor`、`DTensor`）。收敛是真的但慢——PyTorch 已沉淀的 eager 风格代码基数巨大，不能简单重写。

哲学要点：**如果你的编译器理解 sharding，并行就成超参；如果不理解，并行就是重写**。前者是长期赢家位，GSPMD 在前沿规模上首先兑现了它。预期 PyTorch 生态在 2025–2026 年通过 DTensor + `torch.compile` 追平，但设计语言不管承认与否都在抄 GSPMD。

## 参考

- [1] Xu et al. _GSPMD: General and Scalable Parallelization for ML Computation Graphs._ arXiv:2105.04663, 2021.
- [2] Barham et al. _Pathways._ MLSys '22.（运行时对照）
- [3] Lepikhin et al. _GShard._ 2020.（分片 MoE 的先驱——"GSPMD" 是泛化）
- [4] OpenXLA：https://github.com/openxla/xla
- [5] JAX `shard_map` 与 `pjit` 文档：https://jax.readthedocs.io/
