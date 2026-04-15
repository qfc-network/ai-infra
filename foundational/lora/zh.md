# LoRA / QLoRA —— 参数高效微调的低秩适配

- **作者 / 机构**：Hu 等，Microsoft（LoRA）；Dettmers 等，华盛顿大学（QLoRA）
- **发表时间**：2021-06 / ICLR 2022（LoRA，arXiv:2106.09685）；2023-05 / NeurIPS 2023（QLoRA，arXiv:2305.14314）
- **链接**：[LoRA 论文](https://arxiv.org/abs/2106.09685) · [QLoRA 论文](https://arxiv.org/abs/2305.14314) · [HuggingFace PEFT](https://github.com/huggingface/peft) · [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

## 一句话总结

对一个 7B 参数模型做全量微调，需要存储并更新 70 亿个梯度和优化器状态——AdamW 在 BF16 下仅优化器状态就约 112 GB，激活还没算。65B 模型在多节点 A100 集群以外根本跑不动。LoRA（低秩适配）绕过了这个问题：**微调时权重更新的内在秩很低**——与其更新完整权重矩阵 W，不如训练一对小矩阵 B 和 A，用它们的乘积 BA 近似 ΔW；推理时把 BA 合并回 W，零额外开销。QLoRA 再进一步：把冻结的基础模型量化为 4 比特（NF4），LoRA 适配器保持 BF16，优化器状态在 GPU 内存不够时换页到 CPU。结果：在**单张 48 GB A100** 上微调 65B Llama 模型。LoRA 和 QLoRA 现已成为开源大模型生态的默认微调原语。

## 背景与动机

### 全量微调的代价

把预训练大模型在下游任务上做全量微调，经典做法是更新全部参数：模型中每个权重都要计算梯度、存储梯度、存储优化器状态（Adam 每个参数需要两个动量）。在 BF16 下：

- 7B 模型：~14 GB 权重 + ~28 GB 优化器状态 + 激活 ≈ 80+ GB
- 13B 模型：~26 GB + ~52 GB ≈ 100+ GB
- 65B 模型：~130 GB + ~260 GB ≈ 单节点不可行

即便有梯度检查点和混合精度，65B 的全量微调也至少需要 8× A100-80GB，绝大多数研究组和几乎所有工程实践者根本够不到。

### 已有方法的问题

LoRA 之前，参数高效微调的主流方法是：

- **Adapter 层**：在 Transformer 层之间插入小型瓶颈 MLP，只训这些模块。问题：推理时增加延迟（额外的串行操作），且难以移除。
- **前缀调优 / 软提示调优**：在输入或每层 KV 前插入可训练的软 token。问题：压缩有效上下文长度；优化不稳定；无法合并消除。
- **BitFit**：只训偏置参数。参数极少但表达能力有限，不能很好地泛化到多样任务。

三者要么带来推理开销，要么牺牲质量。LoRA 的设计目标是：**匹配全量微调的质量，零推理开销，参数量只用一小部分**。

### 低秩假设

驱动 LoRA 的核心经验观察：微调时，权重矩阵的梯度更新具有很低的**内在维数**——优化轨迹大部分时间只在全参数空间的某个低维子空间里移动。如果这成立，ΔW 的低秩分解就足以捕获微调信号，不需要表示完整的 d×k 个维度。

Aghajanyan 等（2020）通过内在维度分析为此提供了经验支持：许多 NLP 任务可以通过在预训练模型适当子空间里优化不到 1,000 个参数来解决。LoRA 把这一观察工程化为可用的训练技术。

## 核心方法

### LoRA：权重更新的低秩分解

对预训练权重矩阵 W₀ ∈ ℝ^{d×k}，不学习完整的 ΔW ∈ ℝ^{d×k}，改为学习低秩分解：

```
ΔW = BA，其中 B ∈ ℝ^{d×r}，A ∈ ℝ^{r×k}，r ≪ min(d, k)
```

训练时修改后的前向传播为：

```
h = W₀x + (α/r) · BAx
```

其中 α 是缩放超参数（通常设为 r 或 2r；α/r 相当于对 LoRA 更新的学习率缩放）。W₀ 在整个训练过程中**保持冻结**，只有 A 和 B 被更新。

初始化：A 从高斯分布初始化（使得训练开始时 ΔW = BA ≈ 0，保留预训练行为）；B 初始化为全零。

**参数量对比**：对 d=k=4096、r=8 的单个权重矩阵，全量微调需要 4096×4096 = 1,670 万参数；LoRA 只需 2×8×4096 = 65,536 个参数——**原来的 0.4%**。对 7B 模型所有注意力投影矩阵（32 层各 Q/K/V/O），r=8 的 LoRA 只新增约 400–800 万可训参数，不到总量的 0.1%。

### 应用于哪些层

LoRA 通常施加在注意力机制的线性投影上：Q（query）、K（key）、V（value）、O（output）投影。原始论文发现只用 Q 和 V 通常已够；后来的工作（包括 QLoRA）发现把四个加上 MLP 层（Llama 风格模型中的 gate、up、down 投影）一起做，在指令跟随和代码生成任务上质量更好。

### 推理时合并

推理时，更新可以直接合并回权重：

```
W = W₀ + (α/r) · BA
```

这是部署前的一次性操作。合并后的模型与基础模型在形状和延迟上完全相同——无额外计算、无额外内存。**合并后 LoRA 的推理开销恰好为零**。

也可以保持**不合并**、动态叠加适配器。这实现了 **LoRA 多路复用**：单一基础模型的 GPU 部署可以按请求切换适配器，服务多个微调变体（适配器足够小，毫秒级加载）。S-LoRA（Sheng 等，2023）在规模上实现了这一点。

### QLoRA：量化基础模型 + BF16 适配器

QLoRA 在 LoRA 基础上把冻结的基础模型量化为 4 比特，大幅降低内存占用，同时让 LoRA 适配器保持 BF16 以保证梯度计算的稳定性。

**NF4（Normal Float 4）**：专为正态分布权重设计的 4 比特数据类型。16 个量化级别按标准正态分布的分位数放置——对近似 N(0, σ²) 分布的权重是最优的。NF4 对正态数据的量化误差低于标准 INT4。

**双重量化**：量化常数本身（每 64 个参数一个，存为 FP32）再被量化为 FP8，平均每参数额外节省约 0.37 bit。

**分页优化器**：QLoRA 用 NVIDIA 统一内存（`cudaMallocManaged`）在 GPU 内存不足时把优化器状态透明换页到 CPU，处理反向传播中的内存峰值而不崩溃——对在单 GPU 上塞进 65B 至关重要。

**前向传播**：训练时，NF4 权重在矩阵乘法前实时反量化为 BF16，计算完即丢弃。实际计算在 BF16 进行；只有存储用 NF4。量化误差只累积在冻结的基础模型权重上，不影响被训练的 LoRA 适配器。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| LoRA vs 全量微调 | 可训参数减少 10–100×；单 GPU 可装；迭代更快 | 对需要大范围权重改变的任务（如领域迁移）有轻微质量上限；更新只能表达秩为 r 的方向空间 |
| LoRA 秩 r（4 → 64） | 较高 r：表达能力更强，能表示更复杂的更新 | 较高 r：参数更多，内存占用大，收敛更慢；超过 r=64 几乎无益；r=8–16 是常用甜点 |
| 施加 LoRA 的层范围 | 全部投影（Q/K/V/O + MLP）：质量更好，尤其在复杂任务上 | 适配器更多 = 参数更多；对简单分类 / NLU 任务，只用 Q+V 已足够 |
| NF4 基础（QLoRA）vs BF16 基础 | 基础模型内存减少 4×；65B 可在 1× A100-80GB 上运行 | 部分 benchmark 上相比 BF16 全量微调有约 1–3% 的质量差距；反量化增加约 10% 计算开销 |
| 推理时合并 vs 不合并 | 合并：零延迟开销，与基础模型完全相同 | 不合并：适配器可按请求切换、一个基础服务多任务；多个适配器无法在不做算术的情况下合并 |

## 实验与结果

**LoRA（Hu 等，2022）：**
- 在 GLUE、SuperGLUE 和 E2E NLG benchmark 上，用 LoRA 微调 GPT-3（175B）。
- r=4 时，LoRA 在多数任务上与全量微调相当，而只训练了 0.01% 的参数（175B 中约 470 万）。
- 推理延迟：合并后与基础模型完全相同（零开销，经 profiling 确认）。
- 消融：Q+V 优于只用 Q；四个注意力矩阵（Q/K/V/O）一起用比仅 Q+V 再进一步，但参数量翻倍。

**QLoRA（Dettmers 等，2023）：**
- 在单张 NVIDIA A100-80GB 上微调 Llama-65B（之前需要约 8× A100）。训练速度：Alpaca 数据集上约 24 GPU 小时 / 10K 步。
- Guanaco-65B（用 QLoRA 在 OASST1 上微调 Llama-65B）：据人工评估，在 Vicuna benchmark 上达到 ChatGPT 99.3% 的表现。
- Guanaco-7B（单张 24 GB 消费级 GPU 上的 QLoRA）：与在 8× A100 上全量微调的模型竞争力相当。
- 质量对比：7B BF16 全量微调 > 7B QLoRA，差距约 1–2%（MMLU）；但在同等 GPU 预算下，QLoRA 13B 或 33B 通常优于全量微调的 7B——规模的收益覆盖了量化的损耗。

## 复现要点

LoRA 和 QLoRA 均已完全开源、生产级可用：

- **HuggingFace PEFT**（`pip install peft`）：标准 LoRA 实现。`LoraConfig` 指定目标模块、秩、alpha、dropout，直接与 `Trainer` 集成。
- **bitsandbytes**（`pip install bitsandbytes`）：提供 NF4 量化、双重量化、分页优化器。PEFT 的 QLoRA 路径使用此库。
- **Axolotl**、**LLaMA-Factory**、**Unsloth**：更高层的封装，处理 PEFT/bitsandbytes 配置与数据预处理。

7B 模型在单张 A100-40GB 上的典型 QLoRA 配置：

```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"])
model = get_peft_model(model, lora_config)
```

常见坑：
- **忘记启用梯度检查点**：没有它，即使用 QLoRA，大模型的激活也会 OOM。
- **目标模块名称错误**：不同模型家族投影层命名不同，始终先检查 `model.named_modules()`。
- **Alpha 设置过低**：`lora_alpha` 远小于 `r` 时，适配器更新量太小；惯例是 alpha = r 或 2r。

## 个人评注

LoRA 的影响力来自 ML 论文中罕见的一个特性：**它让现有大系统在部署端完全没有代价地变得更好**。合并后的模型在形状和延迟上与全量微调无法区分。这比"同等资源下质量相当"的说法更强——意味着采用 LoRA 完全不需要改动任何推理基础设施。

QLoRA 随后做了更激进的事：把冻结基础模型的内存边界往下移了四倍，代价只是轻微的质量损耗。实际后果是："能对 65B 模型做微调的是谁"这个问题，从"只有超大型机构"变成了"能租到一张 A100 的任何人"。这种民主化效应复利增长：开源社区从此可以对 65B 级模型做实验，催生了 2023–2024 年基于 Llama 的微调爆炸式增长。

剩余局限是真实的，但范围很窄：

- **秩的限制**：LoRA 无法表示需要超过 r 个独立方向的更新。对需要大范围分布迁移的任务（如向模型加入全新语言），需要全量微调或更高秩的适配器。
- **NF4 质量差距**：对绝对质量敏感的任务（竞赛数学、复杂推理），QLoRA 1–3% 的差距有影响。实用答案是：在同等 GPU 预算内用更大的 QLoRA 模型——规模通常能弥补差距。
- **适配器组合**：把在不同任务上训练的多个 LoRA 适配器合并并不简单。LoraHub、TIES-merging、DARE 在探索这个问题，但仍是活跃的研究方向。

2026 年对工程师的建议：LoRA（r=8–16，施加到所有注意力 + MLP 投影）是任何微调任务的正确默认选择。如果模型 BF16 下装不进 GPU，QLoRA 是正确的默认选择。只有当你有足够计算预算且确认 LoRA 的秩约束是质量瓶颈时，才考虑全量微调。

## 参考

- [1] Hu et al. _LoRA: Low-Rank Adaptation of Large Language Models._ ICLR 2022 / arXiv:2106.09685.
- [2] Dettmers et al. _QLoRA: Efficient Finetuning of Quantized LLMs._ NeurIPS 2023 / arXiv:2305.14314.
- [3] Aghajanyan et al. _Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning._ ACL 2021.
- [4] Sheng et al. _S-LoRA: Serving Thousands of Concurrent LoRA Adapters._ arXiv:2311.03285, 2023.
- [5] HuggingFace PEFT 库. https://github.com/huggingface/peft
- [6] bitsandbytes. https://github.com/TimDettmers/bitsandbytes
- [7] Zhao et al. _TIES-Merging._ NeurIPS 2023 / arXiv:2306.01708.
