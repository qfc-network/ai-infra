# ZeRO 与 FSDP

- **作者 / 机构**：ZeRO — Rajbhandari 等，Microsoft（SC '20）。FSDP — Meta（PyTorch）。
- **发表时间**：ZeRO — 2019-10 / 2020-08 · FSDP 论文 — 2023-04
- **链接**：[ZeRO (arXiv:1910.02054)](https://arxiv.org/abs/1910.02054) · [ZeRO-Offload](https://arxiv.org/abs/2101.06840) · [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) · [FSDP (arXiv:2304.11277)](https://arxiv.org/abs/2304.11277) · [DeepSpeed](https://github.com/microsoft/DeepSpeed) · [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)

## 一句话总结

标准数据并行把整个模型在每张 GPU 上复制——小模型没问题，大规模下灾难性。**ZeRO**（Microsoft）把 **optimizer state（ZeRO-1）**、**梯度（ZeRO-2）**、**参数（ZeRO-3）**依次切分到各 DP rank，前向/反向按需 gather。**FSDP**（Meta）是 PyTorch 原生的 ZeRO-3 重写版，overlap、打平、工具链更好——现已是 PyTorch 生态的默认分布式训练原语。两者定义了现代大规模训练的"显存轴"，并与 Megatron 的 TP/PP 正交组合。

## 背景与动机

经典 DDP：每 rank 持有完整参数、梯度、optimizer state。混合精度下 Adam 训练 1B 模型，每 rank 约 16 GB（参数 2B + 梯度 2B + optimizer 12B）。100B 参数时每 rank 1.6 TB——不可能。

Megatron 的 TP/PP 切分的是 **模型内部**（intra-model）；ZeRO/FSDP 切分的是 **跨 DP 副本**（消除复制）。两者正交。组合式的轴分解是现代大规模训练的基础。

## 核心方法

### ZeRO-1：切分 optimizer state

每 rank 只存 `1/N` 的 optimizer state（Adam 的 `m`、`v`、FP32 master weights）。更新时各 rank 只算自己那一片，然后 all-gather 得到完整更新后的参数。梯度、参数仍全复制。

Adam 场景下显存省约 4×（optimizer 是大头）。

### ZeRO-2：梯度也切分

梯度按 optimizer-state 的切分，用 **reduce-scatter** 分发给对应 rank（同通信量替代 DDP 的 all-reduce）。再省约 2×。

### ZeRO-3：参数也切分

每 rank 只存 `1/N` 参数。前向/反向在每层前要 **all-gather** 物化完整参数，算完即释放。反向同理：all-gather 参数、算梯度、reduce-scatter 梯度、释放。

通信量上升（相对 DDP，每层前向/反向各多一次 all-gather），但显存随 `N` 线性下降——单卡放不下的参数也能训。

### ZeRO-Offload 与 ZeRO-Infinity

- **Offload**：把 optimizer state + master weights 移到 CPU RAM，Adam step 由 CPU 算。用 PCIe 带宽换显存——中小规模可行。
- **Infinity**：offload 延伸到 NVMe，可训远超聚合 GPU 显存的模型，代价是对存储带宽巨大压力。

### FSDP：PyTorch 原生 ZeRO-3

概念与 ZeRO-3 相同，工程上有关键差异：

- **FlatParameter**：每个 unit 的参数打平成一个 1-D 张量，一次 all-gather 搞定而非多次。
- **Unit 粒度**：用户把模型包进若干 "FSDP unit"（常按 transformer block）。每个 unit 独立 gather，允许**下一个 unit 的 all-gather 与当前 unit 的计算 overlap**。
- **混合精度、CPU offload、激活 checkpoint** 原生集成。
- **Full shard vs hybrid shard**：hybrid 在节点内切、节点间复制，换取更少的跨节点 gather。

FSDP-2（2024）更进一步：按参数切分（不打平），与 TP/PP 组合更干净，state-dict 语义更清晰。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| ZeRO-1 → 2 → 3（渐进） | 显存单调下降，按需采用 | 每阶段通信量增加 |
| ZeRO-3 / FSDP 完整切分 | 显存随 `1/N` 下降，纯 DP 也能训巨型 dense | 每层前向/反向多一次 all-gather |
| FSDP FlatParameter | 每 unit 一次通信，overlap 好做 | 部分加载、state-dict、混合 dtype 更复杂 |
| Unit 粒度 | 显存/通信可调 | 包装方式选错会悄悄掉吞吐 |
| Hybrid shard | 跨节点 all-gather 减少 | 节点内显存压力更大；复制因子成本 |
| CPU / NVMe offload | 超越聚合 GPU 显存 | PCIe / 存储带宽成为新瓶颈 |

## 实验与结果

- **ZeRO 原论文**：400 张 V100 训 170B，部分配置下超线性加速。首次展示了无模型并行也能训 100B+ dense。
- **FSDP 论文**：512 张 A100 近线性扩展，175B Llama 风格模型配合激活 checkpoint MFU >55%。
- 采用度：FSDP 是 torchtitan、llama-recipes 及多数 PyTorch 原生训练栈的默认；DeepSpeed ZeRO 在重 offload、HuggingFace 流水线场景仍活跃。

## 复现要点

- 均开源、文档良好。
- 正确性通常没问题，难点在**性能调优**——unit 大小、overlap、精度、与 TP/PP 的交互都要权衡。
- 未来走 FSDP-2；FSDP-1 仍在大量生产中。

## 个人评注

ZeRO 与 FSDP 是大规模训练的"沉默骨干"。Megatron 因支撑万亿参数而出名，但没有 ZeRO 式切分，DDP 下连 7B dense 都很难在常规硬件上跑起来。设计思想简单——**能切的就别复制，按需 gather**——工程全在 overlap 与 sizing 上。最被低估的点是：**ZeRO-3 / FSDP 与 Megatron 正交**，`TP × PP × FSDP` 的组合立方体是现代前沿训练可组合性的基础。向前看：FSDP-2 的按参数切分抹平了最后几处粗糙，FSDP + TP 的 overlap（激活切分等）仍是活跃研究方向。如果在 PyTorch 里训 ~1B 以上的模型而没用 FSDP，基本是在浪费显存或吞吐。

## 参考

- [1] Rajbhandari et al. _ZeRO: Memory Optimizations Toward Training Trillion Parameter Models._ SC '20 / arXiv:1910.02054.
- [2] Ren et al. _ZeRO-Offload: Democratizing Billion-Scale Model Training._ USENIX ATC '21 / arXiv:2101.06840.
- [3] Rajbhandari et al. _ZeRO-Infinity._ SC '21 / arXiv:2104.07857.
- [4] Zhao et al. _PyTorch FSDP._ arXiv:2304.11277, 2023.
