# DeepSpeed —— MoE、Chat 与推理引擎

- **作者 / 机构**：Samyam Rajbhandari、Jeff Rasley、Olatunji Ruwase 等 —— 微软
- **发表时间**：2022-01（MoE）· 2022-06（推理引擎）· 2023-04（Chat）
- **链接**：[arXiv:2201.05596 (MoE)](https://arxiv.org/abs/2201.05596) · [arXiv:2207.00032 (推理)](https://arxiv.org/abs/2207.00032) · [arXiv:2308.01320 (Chat)](https://arxiv.org/abs/2308.01320) · [代码](https://github.com/microsoft/DeepSpeed) · [博客](https://github.com/microsoft/DeepSpeed)

## 一句话总结

DeepSpeed 在 ZeRO（已在 `foundational/zero-fsdp/` 中单独介绍）之外，衍生出三个专项系统：**DeepSpeed-MoE**，通过专家并行（Expert Parallelism）和分层 all-to-all 通信训练和推理 MoE 模型；**DeepSpeed-Chat**，在单个 PPO 循环中用混合引擎（Hybrid Engine）让同一个模型在生成模式和训练模式间切换，实现端到端 RLHF；**DeepSpeed 推理引擎**，通过融合 CUDA 内核与 INT8/FP16 量化矩乘实现高吞吐服务。三者各自攻克了 ZeRO 无法解决的独立瓶颈。

## 背景与动机

ZeRO 解决了稠密模型训练的显存墙问题。但走出这一基础之后，三个缺口依然存在：

**缺口一 —— MoE 训练。** ZeRO 在数据并行 rank 之间均匀切分参数。而 Mixture-of-Experts 引入了路由器，将每个 token 分配给一个或多个特定专家，这些专家可能分布在完全不同的 GPU 上。这要求 all-to-all 通信——与 ZeRO 的 reduce-scatter / all-gather 根本不同的通信模式。在 ZeRO 之上直接运行 MoE 会把所有专家堆到每张卡上，完全抵消稀疏激活带来的参数效率优势。

**缺口二 —— RLHF 训练。** PPO 要求 actor 模型在两种截然对立的运行模式之间交替：（a）自回归生成，适合 KV 缓存、连续批处理和高吞吐推理内核；（b）梯度计算，适合 ZeRO-3 参数分片和激活检查点。2022–2023 年没有任何训练框架能优雅地兼顾这种二元性。当时常见的做法是用独立的推理服务器做生成、用独立的训练任务做参数更新，不仅基础设施成本翻倍，还需要复杂的权重同步工程。

**缺口三 —— 推理服务。** 为训练优化的内核在推理时是浪费的。PyTorch 的 autograd 图、cuBLAS 的通用 GEMM 内核、逐层顺序执行，留下大量显存带宽余量。一个 transformer block 在推理时要反复读取同样的权重矩阵（QKV 投影、attention、FFN）——将这些操作融合进单个 CUDA 内核，再结合自定义 tiling 策略，可以在相同硬件上实现翻倍的有效吞吐。

## 核心方法

### DeepSpeed-MoE：分层 All-to-All 的专家并行

**专家并行（EP）** 将专家子集分配给每张 GPU（或一组 GPU）。前向传播时，top-k 门控函数将每个 token 路由到 k 个专家：

$$\text{output} = \sum_{i \in \text{top-k}} G_i(x) \cdot E_i(x)$$

其中 $G_i(x)$ 是专家 $i$ 的 softmax 门控得分，$E_i(x)$ 是该专家的前馈计算。Token 必须物理传输到持有对应专家的 GPU，这需要 **all-to-all 集合通信**：每张 GPU 向其他所有 GPU 发送变长 token 批次。

核心系统挑战在于：跨多节点的 all-to-all 代价高且不可预测。DeepSpeed-MoE 通过**分层 all-to-all** 分两阶段解决：

1. **节点内 all-to-all**：同一节点内的 GPU 通过 NVLink 通信（A100 上双向约 600 GB/s）。专家组按节点连续排列，最大化 NVLink 利用率。
2. **节点间 all-to-all**：只有真正需要跨节点的 token 才走 InfiniBand（有效约 25 GB/s）。通过设计专家分配策略，最小化跨节点流量，使慢速链路不成为瓶颈。

**EP 与 ZeRO 的组合**：注意力、layer norm、embedding 等稠密层继续用 ZeRO-3 在所有 rank 间分片；专家权重用 EP。两种策略叠加使用：一张卡可以同时持有 1/N 的注意力参数（ZeRO 分片）和 1/E 的专家参数（EP 分片），N 为 DP 度，E 为 EP 度。

**PR-MoE（Pyramid-Residual MoE）**：DeepSpeed-MoE 提出一种模型架构变体，在早期 transformer 层（表示特化程度较低）使用较少专家，在后期层使用更多专家。稠密 MLP 与 MoE 层之间引入残差连接，让专家路径专注于残差而非完整表示。这一设计在相同质量下将总参数量减少最多 8×，因为稀疏激活的效率在专家有明确分工时最高——而这在网络后期才会出现。

**负载均衡**：路由器必须避免所有 token 集中到同一个专家（坍塌）。DeepSpeed-MoE 使用 MoE 文献中标准的辅助负载均衡损失，并引入容量因子（capacity factor）机制——超出单专家缓冲上限的 token 会被丢弃，以约束显存使用和通信量。

### DeepSpeed-Chat：RLHF 混合引擎

PPO 的每个训练步包含四个计算阶段：

1. **生成（rollout）**：actor 自回归生成响应。需要：KV 缓存、用于解码吞吐的张量并行、连续批处理。
2. **奖励打分**：冻结的奖励模型对每个（prompt, response）对打分。
3. **KL 计算**：参考模型计算 log 概率用于 KL 散度惩罚。
4. **PPO 更新**：actor 和 critic 运行前向与反向传播。需要：ZeRO-3 参数分片、梯度检查点、优化器状态 offload。

**混合引擎**让同一个 actor 模型在这些模式间切换，无需维护两份模型副本：

**推理模式**：actor 将权重加载到为生成优化的布局——连续权重缓冲区、KV 缓存已分配、张量并行分片合并以加速矩乘。DeepSpeed 自己的推理内核（融合 attention、INT8 投影）被激活。

**训练模式**：actor 切换到 ZeRO-3 布局——权重分散到各 rank，开启梯度追踪和激活检查点。推理内核关闭，标准 autograd 接管。

模式切换涉及原地权重重排。7B 模型大约需要 100–200 ms，相对于完整生成步耗时数秒，这个开销是可接受的。关键工程洞察：**权重数据本身不需要复制**——相同的 GPU 显存缓冲区用不同的 stride 和分区元数据重新解读即可。

**奖励模型与参考模型的显存管理**：actor 训练阶段，奖励模型和参考模型 offload 到 CPU 内存；生成阶段恢复到 GPU。在 7B 规模下，CPU offload 节省 30–40% 的 GPU 显存，使更大的 actor 能够放入。传输延迟（7B 约 0.5 s 每次）被吸收到流水线中，因为奖励打分和参考计算发生在生成与训练之间，不在关键路径上。

DeepSpeed-Chat 实现的完整 RLHF 流水线——从预训练或 SFT 微调 checkpoint 到 PPO 训练模型——通过单个 `deepspeed` 命令端到端运行，无需用户管理独立的服务和训练基础设施。

### DeepSpeed 推理引擎：融合内核与量化服务

推理引擎针对服务时的吞吐瓶颈：读取大型权重矩阵的重复显存带宽压力。

**内核融合策略**：标准 transformer 前向传播需要读取 QKV 投影权重矩阵、attention（读取 KV 缓存）、输出投影权重、两个 FFN 权重矩阵和 layer norm 参数——每个 transformer block 大约 7–8 次独立的全局显存读取。DeepSpeed 的融合 transformer 内核将其压缩到 2–3 次读取，方法是：

- 将 QKV 投影融合为单个批量 GEMM
- 将 softmax、缩放点积 attention 和 dropout 融合到一个使用 SRAM tiling 计算的内核
- 将输出投影 + 残差加法 + layer norm 融合到同一个内核启动
- 将 layer norm 与后续投影预融合计算

对于拥有 32 个 transformer block 的模型，这将 CUDA 内核启动次数减半，并相应降低显存带宽占用。

**INT8 量化**：推理引擎支持权重矩阵的逐通道对称量化。权重以 INT8 存储（显存占用减半），在每次矩乘时通过自定义 CUDA 内核实时反量化为 FP16——缩放因子乘法在与 GEMM 同一个 warp 中完成。激活值全程保持 FP16。权重读取的有效显存带宽从 2 字节/元素（FP16）降至 1 字节/元素（INT8），对于显存带宽受限的层，直接实现翻倍吞吐。

**无框架开销的张量并行**：推理引擎为注意力头和 FFN 列实现了 Megatron 风格的张量并行，但去掉了 PyTorch 分布式原语，直接通过 NCCL 原语在 CUDA 中实现所需的 all-reduce，避免 Python 层集合通信调度的开销。

## 工程权衡

| 决策 | 获得 | 放弃 |
|---|---|---|
| MoE 使用专家并行而非张量并行 | 每个专家是单张 GPU 上完整、无分割的模块，专家内核实现简单 | all-to-all 通信量随 EP 度增长；EP 超过 16 时，跨节点 all-to-all 成为吞吐瓶颈，在 InfiniBand 集群上尤为明显 |
| 分层 all-to-all（先节点内后节点间） | NVLink 承载本地专家路由；InfiniBand 只传输真正的跨节点流量；比平铺 all-to-all 延迟低 2–3× | 专家到节点的分配必须提前设计；动态负载不均衡可能仍然触发跨节点流量；两级通信调度调试更难 |
| Chat 混合引擎模式切换 | 单个 actor 模型同时负责生成和训练；无需独立部署推理服务或维护跨进程权重同步 | 7B 规模下每次模式切换约 100–200 ms 原地权重重排开销；自定义权重布局代码在不同模型架构间脆弱 |
| Chat 中对奖励模型和参考模型做 CPU Offload | actor 可使用更大的 GPU 显存；训练阶段显存节省 30–40%，无需为冻结模型额外预算 GPU | 每个 PPO 步需要 CPU→GPU 传输（7B 约 0.5 s）；70B+ 规模下 PCIe 带宽成为瓶颈，传输时间接近生成时间 |
| 融合推理内核（INT8 + attention 融合） | 相同硬件下吞吐较标准 PyTorch 提升 2–4×；每 token 显存带宽压力降低 | 手写 CUDA；架构特定（A100/H100）；与任意 HuggingFace 模型修改不可组合；模型架构变动时需重新编译 |
| 仅对权重做逐通道 INT8 量化 | 权重显存带宽成本减半；对大多数 transformer 架构精度损失相对 FP16 可忽略 | 激活量化需要校准数据和更复杂的内核逻辑；对 MoE 模型，FP16 路由决策与低容量专家的 INT8 权重读取可能产生不良交互 |

## 实验与结果

**DeepSpeed-MoE**：训练一个 52B 参数的 MoE 模型（128 个专家）相比等质量稠密模型（350M 活跃参数，350M 稠密基线）实现 1.45× 吞吐提升。PR-MoE 在语言建模基准上以相同困惑度实现总参数量 8× 缩减，因为参数集中在专家分工最明确的后期层。在服务端，MoE 模型相比等质量稠密模型推理成本降低 4.5×，因为每个 token 只激活一小部分参数。

**DeepSpeed-Chat**：OPT-13B 的 RLHF 训练比在相同硬件上的朴素 HuggingFace TRL 实现快 15×，主要得益于混合引擎消除了独立推理服务器。OPT-66B 的 PPO 训练在 64 张 A100 上 9 小时内完成；朴素基线需要数天。端到端流水线——从 SFT checkpoint 到 PPO 训练模型——通过单个 `deepspeed` 启动命令运行，在 2023 年大幅降低了 RLHF 实验的门槛。

**推理引擎**：在 BERT 类模型上，DeepSpeed 融合内核相比 NVIDIA FasterTransformer 实现 1.9× 吞吐提升。在 GPT 风格的 decoder-only 模型上，平均提升 1.56×，长序列下收益更高（attention 成为瓶颈时）。在内核融合基础上，INT8 量化再叠加 1.3–1.5× 提升，来自显存带宽降低。

## 复现说明

三个系统均已完全开源，地址：[github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)。

**安装**：`pip install deepspeed`。自定义 CUDA 扩展在安装时编译，需要 CUDA 11.6+ 和兼容版本的 GCC。常见 CUDA/PyTorch 组合可用预编译 wheel。

**DeepSpeed-MoE**：通过 `deepspeed.moe.layer.MoE` 使用——可作为标准 FFN 层的直接替换。PR-MoE 需要使用 `DeepSpeedMoEConfig` 类。专家并行度在 `deepspeed_config.json` 中配置。

**DeepSpeed-Chat**：`examples/chat` 目录包含三步训练脚本（SFT → 奖励模型训练 → PPO）。在单个 8× A100 节点上用 OPT-1.3B 即可完整运行。扩展到 13B 或 66B 需要在 config JSON 中调整 ZeRO stage 和 offload 设置。

**推理引擎**：`deepspeed.init_inference(model, mp_size=N, dtype=torch.int8)` 可包装任何 HuggingFace 模型并激活融合内核路径。张量并行需要通过 `deepspeed --num_gpus N` 启动。融合内核在 A100 和 H100 上验证通过；V100 支持不完整（旧版 CUDA core 无 INT8）。

**已知注意事项**：DeepSpeed-Chat 的混合引擎对模型架构敏感——新版 HuggingFace 模型变体中的自定义 attention 实现可能绕过融合内核路径，静默回退到较慢的 PyTorch 执行。始终检查 `ds_report` 输出以确认内核激活状态。

## 评注

DeepSpeed 的演化轨迹展示了一个系统研究团队实时跟踪前沿时的样子。每个组件都在解决前一个组件被解决后涌现的新瓶颈：ZeRO 清除了稠密训练的显存墙，使大型稠密模型成为可能，进而催生了 MoE 需求；RLHF 成为主流对齐方法，催生了模式切换需求；规模化到生产服务，催生了推理内核需求。各组件是对研究议程的响应，而非预见性投机。

DeepSpeed-Chat 的混合引擎与 verl 的 HybridFlow（见单独介绍）在概念上是同一个洞察：actor 模型必须同时满足两个主人——生成吞吐和训练显存效率——需要一个框架同时处理两者。DeepSpeed-Chat 的方案（原地权重重排、冻结模型 CPU offload）更早、但可组合性更弱；verl 的方案（逐模型并行组、FSDP + vLLM 分离）在 70B+ 多节点规模下更加必要。

在推理方面，DeepSpeed 的融合内核在 2022 年与 FasterTransformer 旗鼓相当，远优于原生 PyTorch。到 2023 年，vLLM 的 PagedAttention + FlashAttention-2 在变长服务负载下超越了 DeepSpeed 推理引擎。根本原因：DeepSpeed 推理引擎假定固定长度批次和静态 KV 缓存布局，这在离线吞吐基准测试中表现良好，但在请求长度异构的生产服务中表现不佳。PagedAttention 的动态 KV 缓存管理填补了这一空白。

对于希望在小到中等规模进行定制 RLHF 实验而不从零构建分布式基础设施的工程师，DeepSpeed 仍然是最易上手的入口。其 `deepspeed_config.json` 抽象——将所有并行、offload 和量化决策表达在单个 JSON 文件中——相比 Megatron 的 C++/Python 混合配置，在可用性上有显著优势。

## 参考文献

- [1] Rajbhandari 等. "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale." arXiv:2201.05596, 2022.
- [2] Aminabadi 等. "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale." arXiv:2207.00032, 2022.
- [3] Yao 等. "DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales." arXiv:2308.01320, 2023.
- [4] Rajbhandari 等. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC '20 / arXiv:1910.02054.
- [5] Lepikhin 等. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." arXiv:2006.16668, 2021.
- [6] Fedus 等. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR 2022 / arXiv:2101.03961.
- [7] Sheng 等. "HybridFlow: A Flexible and Efficient RLHF Framework." arXiv:2409.19256, 2024.
