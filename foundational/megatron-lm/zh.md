# Megatron-LM：张量、流水线与序列并行

- **作者 / 机构**：Shoeybi 等，NVIDIA
- **发表时间**：v1 — 2019-09 · v2 — 2021-04 · v3（序列并行 + 选择性重算） — 2022-05
- **链接**：[v1 (arXiv:1909.08053)](https://arxiv.org/abs/1909.08053) · [v2 (arXiv:2104.04473)](https://arxiv.org/abs/2104.04473) · [v3 (arXiv:2205.05198)](https://arxiv.org/abs/2205.05198) · [代码](https://github.com/NVIDIA/Megatron-LM)

## 一句话总结

NVIDIA 的三部曲，定义了 **transformer 如何在规模下做并行**。v1 提出**张量并行（TP）**——把单个 matmul 按精心选择的通信模式切到多张 GPU（每层两个 all-reduce）。v2 加入**流水线并行（PP）**，用 interleaved 1F1B 调度，并证明 TP + PP + DP 组合的 "3D 并行" 能撑到 1T+ 参数。v3 加入**序列并行（SP）**按序列维度切分激活显存，并引入**选择性激活重算**以低算力代价降显存。此后的所有大规模训练框架（DeepSpeed、Colossal-AI、MindSpore，乃至 V3 与 Llama 3）都继承了 Megatron 的并行原语。

## 背景与动机

2019 年时 ZeRO 与数据并行能处理 optimizer / 梯度切分，但到 GPT-2-XL 规模，**单个 transformer 层**已无法放进一张 GPU。流水线并行（GPipe）存在但 bubble 很大。缺的是**把单层切到多张 GPU 且不拖垮吞吐**的方法。

v2 的背景：业界开始挑战 500B+ 模型。纯 TP 过不了节点边界（NVLink），纯 PP bubble 太多，纯 DP 显存爆。只能组合，问题是怎么组合。

v3 的背景：即便有了 3D 并行，长序列下激活显存仍爆。完整 checkpoint 重算已是标配，但吃掉 ~30% 算力。需要**更细粒度地控制重算什么**。

## 核心方法

### 张量并行（v1）

对 transformer 块：

- **MLP**：`Y = GeLU(XA) · B`。`A` 按列切分 → 每卡产出 `GeLU(XA)` 的一部分；`B` 按行切分 → 每卡计算自己这部分 `B` 的贡献。一次 **all-reduce** 合并。GeLU 之前前向不通信。
- **Self-attention**：head 按卡切分（Q/K/V 投影按列并行，输出投影按行并行）。同样每块每方向一次 all-reduce。

合计：每个 transformer 块前向 2 次 all-reduce、反向 2 次。只有在节点内带宽（NVLink）可用时高效——TP 一般 ≤ 节点大小（4 或 8）。

### 流水线并行（v2）

层按 PP stage 切分。bubble 来自流水线填充/排空。两个关键点：

1. **1F1B 调度**：某 stage 处理完一个 micro-batch 的前向后，立刻处理之前某个 micro-batch 的反向。in-flight 激活更少，流水线保持活跃。
2. **Interleaved 1F1B**：给每个 stage 分配不连续的层组（如 stage 0 持有 0–3 层和 16–19 层）。更多 micro-batch 并行，bubble 更小，显存更多。

Bubble 占比约 `(PP − 1) / (m + PP − 1)`，`m` 为 micro-batch 数——所以要把 `m` 做大。

### 3D 并行

`total_gpus = TP × PP × DP`。经验法则：
- TP 在节点内（NVLink 充裕），一般 8。
- PP 跨节点，按层数 / 显存定。
- DP（+ ZeRO）在最外层。

V3 把这一套扩到 ~2k H800；Llama 3 扩到 16k H100（额外加 CP）。

### 序列并行（v3）

TP=8 时，dropout / LayerNorm / 残差这些操作在所有 TP rank 上是复制的——不吃 TP 收益，却每张卡占用完整激活显存。SP **把这些非 matmul 的激活沿序列维度切分**。要加 all-gather / reduce-scatter，但整体通信量接近纯 TP（一个 ring 被拆成两半），激活显存减少 ~1/TP。

### 选择性激活重算（v3）

完整 checkpoint 重算会把所有激活在反向时重算一遍 → 约 30% 算力开销。v3 剖析后发现：少量层（尤其是 attention 矩阵）占用大量激活显存，但重算极便宜。只重算这些，昂贵的激活保留在 cache。结果：显存损失几乎为零，算力开销 <5%。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 张量并行 | 单层可切到多卡 | 跨节点就被通信拖垮；TP 大小受 NVLink 拓扑限 |
| 流水线并行 | 可扩到任意层数 | 有 bubble；需要大量 micro-batch；跨 stage 激活传输 |
| Interleaved 1F1B | bubble 更小 | 并发 micro-batch 更多 → 显存更大 |
| 序列并行 | 激活显存节省 ~1/TP | 多几次通信（但 ring 变短，总量接近） |
| 选择性重算 | 算力开销降到完整重算的 ~1/5 | 需要对 workload 做 profiling 来挑选 |
| 3D 并行整体 | 可训万亿参数 | 启动/配置复杂度；框架绑定；调试困难 |

## 实验与结果

- v1：8.3B GPT-2，在 512 张 V100 上持续 15.1 PetaFLOPs，相对 DP 有 76% scaling 效率。
- v2：3072 张 A100 训练 1T 模型，达峰值 52%；详细给出 TP/PP/DP 取值的 scaling 规律。
- v3：序列并行 + 选择性重算在 22B 与 175B 配置上把激活显存降 5×，吞吐相对完整重算提升 ~29%。
- Megatron 是业界参考点——后续的框架论文多以它为对照基线。

## 复现要点

- 完全开源；NVIDIA 以 Megatron-Core 作为长期维护库。
- 主流生产训练栈要么直接用 Megatron，要么重写其原语（DeepSpeed、Colossal-AI、基于 FSDP 的 Meta 栈）。
- 配置（TP、PP、DP、micro-batch、序列长度）是 workload 相关的，需调参；NVIDIA 对常见规模给出了 recipe。

## 个人评注

Megatron 的三篇论文是分布式 transformer 训练最重要的单一工作集合。真正新颖的地方在于**每一步事后看都很简单**：TP 是线性代数 + all-reduce；PP 1F1B 是"能反向就立刻反向"；SP 是"把另一个还没被切的轴切掉"。工程艺术在于 (a) 让通信模式对齐网络拓扑，(b) 识别出哪些激活真正重要。此后所有工作——FSDP、ZeRO-3、ring attention、context parallelism——都是在回应 Megatron 的边界，而非替换它的抽象。2026 年设计训练系统，仍从 TP/PP/DP 的分解出发，偏离要有理由。先读 v2（每页信息密度最高），再读 v3 看现代技巧；v1 如今主要是历史价值。

## 参考

- [1] Shoeybi et al. _Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism._ arXiv:1909.08053, 2019.
- [2] Narayanan et al. _Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM._ SC '21 / arXiv:2104.04473.
- [3] Korthikanti et al. _Reducing Activation Recomputation in Large Transformer Models._ arXiv:2205.05198, 2022.
- [4] Huang et al. _GPipe._ NeurIPS 2019.
- [5] Rajbhandari et al. _ZeRO._ SC '20.
