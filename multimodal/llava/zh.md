# LLaVA / 视觉语言模型

- **作者 / 机构**：Haotian Liu、Chunyuan Li、Qingyang Wu、Yong Jae Lee（威斯康星大学麦迪逊分校 / 微软研究院）
- **发表时间**：LLaVA — 2023-04-17 · LLaVA-1.5 — 2023-10-05 · LLaVA-1.6/NeXT — 2024-01-30
- **链接**：[LLaVA arXiv:2304.08485](https://arxiv.org/abs/2304.08485) · [LLaVA-1.5 arXiv:2310.03744](https://arxiv.org/abs/2310.03744) · [代码](https://github.com/haotian-liu/LLaVA)

## 一句话总结

LLaVA（大型语言与视觉助手）是构建视觉语言模型最主流的开源方案：冻结预训练的 [CLIP](../clip/zh.md) ViT-L/14 编码器和预训练 LLM（Vicuna/LLaMA），仅训练一个轻量 MLP 投影器将视觉特征映射到 LLM 的嵌入空间，再对整个系统进行指令微调。LLaVA-1.5 证明这个极简方案——将投影器从线性层升级为两层 MLP——就已在 11 个基准上达到最优，仅需约 120 万训练样本。服务侧代价非常直接：单张 336px 图像向提示词贡献 576 个视觉 token，高分辨率变体（LLaVA-1.6）则高达每图 2880 个 token。

## 背景与动机

GPT-4V 在 2023 年底的发布证明多模态 LLM 已不只是研究玩具——它们能读图表、描述复杂场景、对图像进行推理，而纯文本模型从根本上无法做到这些。开源社区需要一套可复现的方案。LLaVA 的洞察在于：视觉表示和语言理解两大难题已分别由 CLIP 和 LLaMA 解决，唯缺一座连接两者的桥梁。

这种数据高效的方式与同期更复杂的交叉注意力架构（Flamingo、InstructBLIP）形成对比。LLaVA 表明两层 MLP 已足够，这对基础设施有深远意义：参数量在 25 万到 100 万之间的 2 层 MLP，其显存和 FLOPs 可忽略不计。瓶颈在于传递给 LLM 的视觉 token 数量，而非投影器本身。

## 架构设计

### 投影器设计

投影器将 CLIP 视觉特征映射到 LLM 嵌入空间：

- **LLaVA（原版）**：单线性层。W_proj ∈ R^{D_CLIP × D_LLM}。对于 CLIP ViT-L/14（D=1024）→ LLaMA-13B（D=5120）：权重矩阵约 2000 万参数。
- **LLaVA-1.5**：带 GELU 激活的两层 MLP。在 VQA 基准上较线性层高 4–8 个点，但相对 7B/13B LLM 仍然极小。

投影器是唯一需要图像和语言同时出现在同一训练目标下的组件。其他所有内容均可独立预训练，这是 LLaVA 数据效率的关键所在。

传递哪些 CLIP token？LLaVA 使用**全部空间 patch token**（ViT-L/14@336px 的 N=576），而非 [CLS] token。这保留了每个位置的视觉信息，使 LLM 能回答空间性问题（"左上角是什么？"）。关于为何 patch token 比池化 [CLS] 携带更多信息，参见 [ViT](../vit/zh.md)。

### 两阶段训练

**第一阶段——特征对齐（训练投影器）：**
- 冻结：CLIP 编码器（全部权重）、LLM（全部权重）。
- 训练：仅投影器。
- 数据：约 59.5 万图文对（过滤后的 CC3M），简单描述文本而非指令。
- 目标：在视觉 token 条件下对描述进行下一 token 预测。
- 目的：教投影器生成冻结 LLM 能"读懂"的视觉嵌入。
- 成本：7B 模型约需 4 小时 H100。

**第二阶段——视觉指令微调：**
- 冻结：CLIP 编码器。
- 训练：投影器 + LLM（全量微调，或使用 LoRA 提高效率）。
- 数据：约 66.5 万条指令遵循样本（LLaVA-Instruct-158K + VQA/描述/OCR 数据，1.5 版本）。
- 目标：根据图像和文本问题生成指令遵循的回复。
- 成本：LLaVA-1.5-7B 约需 20 小时 H100。

第二阶段使用 [LoRA](../../foundational/lora/zh.md) 可将 13B 模型的显存从约 80GB 降至约 40GB，因为无需计算 LLM 的完整梯度。投影器始终使用完整梯度训练（其参数量足够小）。

### 视觉 Token 数量——核心基础设施变量

| 配置 | 分辨率 | Patch token | 视觉 token 总数 |
|---|---|---|---|
| LLaVA-1.0 ViT-L/14 | 224px | N=256 | 256 |
| LLaVA-1.5 ViT-L/14@336 | 336px | N=576 | 576 |
| LLaVA-1.6 NeXT（2×2 分块） | 4 × 336px | 4×576 | 2304 |
| LLaVA-1.6 NeXT（3×3 分块） | 9 块 | 9×576 | 5184 |

576 个视觉 token 已经相当可观。典型文本提示为 50–200 个 token；图像贡献的 token 数是文本的 3–10 倍。这颠覆了计算配置：**预填充由视觉 token 主导，而非文本**。

### LLaVA-1.6 / LLaVA-NeXT：高分辨率分块

LLaVA-1.5 固定的 336px 输入会丢失高分辨率图像（截图中的文字、细小物体）的细粒度细节。LLaVA-NeXT 通过**动态分块**解决这一问题：

1. 将输入图像调整大小以适应 tile 网格（最多 4 块：1×1、1×2、2×1、2×2）。
2. 每块 tile 为 336×336px，由 ViT-L/14 独立编码 → 每块 576 个 token。
3. 始终附加一张降采样的全局缩略图（336px）→ 另外 576 个 token。
4. 合计：最多 4 块 × 576 + 576 全局 = **2880 个 token** 每张图像。

tile 数量根据纵横比动态选择，避免不必要的填充。竖向图像使用 1×2 网格；横向全景使用 2×1；密集截图使用 2×2。

每图 2880 个 token 意味着一个图像-问题对的预填充阶段，耗时相当于生成约 2880 个文本 token。这正是[分块预填充](../../foundational/chunked-prefill/zh.md)的动机：将 2880 token 的视觉预填充分割为小块，使其可与其他请求的解码步骤交错执行。

## 服务侧影响

### LLaVA-1.5-7B 显存分解

| 组件 | 显存 |
|---|---|
| CLIP ViT-L/14@336 权重 | ~1.2 GB |
| 投影器 MLP | ~40 MB |
| LLaMA-7B 权重（FP16） | ~14 GB |
| KV 缓存（1024 上下文，bs=8） | ~4 GB |
| **合计** | **~20 GB** |

KV 缓存随视觉 token 数量增长：576 个视觉 token 在 KV 缓存中占用的空间与 576 个文本 token 相同。对于 8 个请求各含一张图像、2048 上下文的批量：KV 缓存 ≈ 8 × 2048 × 隐藏维度 × 2（K+V）× 层数 × 2 字节。

### 视觉编码器作为独立推理过程

在生产 VLM 服务中，CLIP 编码器在 LLM 前向传播之前作为**独立预处理步骤**运行：

1. 图像到达 → 用 CLIP 编码 → 576 个嵌入向量（无自回归，可并行）。
2. 通过投影器 MLP → 576 个 LLM 空间 token 嵌入。
3. 拼接 [视觉 token] + [文本 token] → 输入 LLM 预填充。
4. 正常执行 LLM 解码。

步骤 1 对批次中的图像完全可并行，可与后续请求的步骤 3 进行流水线处理。生产服务模式详见 [VLM 服务](../vlm-serving/zh.md)。

### 图像前缀缓存

若同一图像出现在多个请求中（例如针对同一产品图像提出不同问题的问答机器人），视觉 token 在 KV 缓存中可以通过[前缀缓存](../../foundational/prefix-caching/zh.md)复用，以图像内容的哈希值作为缓存键。缓存命中 → 完全跳过这些 token 的预填充计算，直接复用存储的 K 和 V 张量。前提是视觉 token 始终位于序列开头（LLaVA 通过约定保证了这一点）。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 冻结 CLIP+LLM，仅训练投影器（第一阶段） | 训练成本极低；4 GPU 小时；用极少数据实现图文对齐 | 投影器须跨越两个独立训练的嵌入空间；限制了对齐深度 |
| 第二阶段 LLM 全量微调 | LLM 适应视觉指令；基准性能更好 | 显存需求 2–4×；训练更长；存在遗忘纯文本能力的风险 |
| 第二阶段使用 LoRA | 显存减半；原始 LLM 权重保留（可合并） | 相比全量微调，复杂视觉推理任务上略有差距 |
| 传递所有 576 个 patch token（非 [CLS]） | 丰富的空间特征；支持定位问答和 OCR | 576 个 token 主导预填充；KV 缓存正比增长 |
| 动态分块（LLaVA-NeXT） | 高分辨率细节；精细 OCR/图表识别 | 每图最多 2880 个 token；预填充计算量较 LLaVA-1.5 增加 5× |
| 单一视觉编码器（无重采样） | 架构简单；无 Q-Former 复杂度 | token 数量固定于 patch 数量；无法自适应压缩视觉信息 |

## 实验与结果

- **LLaVA-1.5-7B**：VQAv2 78.5%，MMBench 38.9%，POPE 幻觉基准 66.8%——彼时 7B 模型的最优水平。
- **LLaVA-1.5-13B**：仅用 120 万训练样本，在 11/11 个基准上超越包括 InstructBLIP-13B 在内的所有 7B 和 13B 模型，而 InstructBLIP 使用了 1.29 亿训练样本。
- **LLaVA-1.6 / NeXT**：文档理解（DocVQA、ChartQA）大幅提升，得益于高分辨率分块；2024 年初多项基准上的最优开源结果。

## 个人评注

LLaVA 建立了几乎所有后续开源 VLM 都遵循的模板：CLIP 骨干 + 轻量投影器 + 指令微调 LLM。简洁本身就是意义所在。更复杂的架构（InstructBLIP 的 Q-Former，Flamingo 的交叉注意力层）带来的边际收益有限，但复杂度和训练成本大幅提升。对于实践者来说，LLaVA 方案意味着用几百 GPU 小时和 100 万条数据就能构建出竞争力强的 VLM——瓶颈不是模型方案，而是指令数据的质量。

视觉 token 膨胀问题是 LLaVA 无意中带来的核心服务挑战。当每图 576 个 token 刚被引入时，看起来尚在可控范围内。LLaVA-NeXT 的分块将其推至 2880。现代多图 VLM（Qwen-VL、Llama 4）每次请求可处理 4–8 张图像，在任何文本之前就已产生 1 万–2 万个视觉 token。这要求服务栈做出架构调整：[分块预填充](../../foundational/chunked-prefill/zh.md)防止 GPU 被长视觉预填充独占，[前缀缓存](../../foundational/prefix-caching/zh.md)复用重复图像，注意力核优雅处理异构序列长度。

发展方向指向自适应 token 压缩：LLaVA-HR、TokenPacker 等方法和可学习视觉重采样器将 576 个 token 压缩到 64–144 个，且准确率损失可接受。这种压缩发生在投影器或专用重采样器中，很可能成为生产部署的标配——在高分辨率分块 5× 预填充成本令人难以接受的场景尤为如此。[Llama 4](../../meta/llama4/zh.md) 架构已采用可学习交叉注意力方式来控制 token 数量。

## 参考文献

- [1] Liu et al. _Visual Instruction Tuning (LLaVA)._ arXiv:2304.08485, 2023.
- [2] Liu et al. _Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)._ arXiv:2310.03744, 2023.
- [3] Liu et al. _LLaVA-NeXT: Improved reasoning, OCR, and world knowledge._ 博客文章, 2024.
- [4] 相关词条：[CLIP](../clip/zh.md)、[ViT](../vit/zh.md)、[LoRA](../../foundational/lora/zh.md)、[分块预填充](../../foundational/chunked-prefill/zh.md)、[前缀缓存](../../foundational/prefix-caching/zh.md)、[Llama 4](../../meta/llama4/zh.md)
