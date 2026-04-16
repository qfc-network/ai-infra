# 投机解码变体 — Medusa / EAGLE

- **作者 / 机构**: Cai 等（普林斯顿 + Together AI）· Li 等（上交 + 微软研究院）
- **发表时间**: Medusa — 2024-01 (arXiv:2401.10774) · EAGLE — 2024-01 (arXiv:2401.15077) · EAGLE-2 — 2024-06 (arXiv:2406.16858)
- **链接**: [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077) · [EAGLE-2](https://arxiv.org/abs/2406.16858)

## TL;DR

原始投机解码方法（Leviathan、Chen）使用**独立的草稿模型**以线性链方式生成候选 token。Medusa 和 EAGLE 去掉了独立草稿模型，改用**树形结构草稿**：目标模型的单次前向传播可同时验证一棵分支候选树。Medusa 附加多个并行解码头；EAGLE 在**特征层面**（隐状态）而非 token 层面进行轻量级自回归草稿，获得更高的接受率。EAGLE-2 进一步引入**自适应树扩展**——对置信度低的分支剪枝，将预算集中在高置信分支。两种方法现已默认集成于 vLLM 和 TensorRT-LLM。投机解码的基础（拒绝采样保证等）见[投机解码 primer](../speculative-decoding/zh.md)，本文在此基础上展开。

## 背景与动机

原始投机解码存在两个实际摩擦点：

1. **独立草稿模型**：需要为每个目标模型维护一个配套的蒸馏草稿模型，保持分布对齐是额外的运维负担。
2. **线性链验证**：顺序生成 `K` 个候选 token；首次拒绝后其余全部丢弃。若接受率 `α = 0.7`，`K = 5`，期望接受 token 数约为 `(1 - 0.7^5) / (1 - 0.7) ≈ 2.8`——早期拒绝会浪费整个投机预算。

树形结构解决了这两个问题：提出一组分支候选续写；一次前向验证所有路径；接受最长一致路径。自草稿方法则彻底消除了配套模型。

## Medusa — 并行多头草稿

### 架构

Medusa 在**冻结或轻量微调**的目标模型上附加 `H` 个额外的 LM 头。每个头 `h` 是一个两层 MLP 加线性投影，训练目标是根据 `t` 位置的最后隐状态预测位置 `t + h` 的 token。

```
目标隐状态 at t
    │
    ├── head-1 → t+1 的 logits   (取 top-s₁ 候选)
    ├── head-2 → t+2 的 logits   (取 top-s₂ 候选)
    ├── head-3 → t+3 的 logits   (取 top-s₃ 候选)
    └── head-4 → t+4 的 logits   (取 top-s₄ 候选)
```

四个头在**同一次前向传播**中激发——头之间没有自回归依赖。这是与独立草稿模型的核心区别：不存在顺序草稿生成步骤。

### 树的构建

候选树通过对各头输出取笛卡尔积并剪枝到固定预算来构建。典型配置：head-1 输出 `s₁ = 3` 个 top token；head-2 输出 `s₂ = 3`；head-3 `s₃ = 2`；head-4 `s₄ = 2`，最多 `3 × 3 × 2 × 2 = 36` 条路径，实际剪枝后更少。

### 树注意力

所有候选路径通过**单次目标模型前向传播 + 自定义注意力掩码**联合验证：

- 每个节点可以 attend 到全部原始前缀 token（已验证的上下文）。
- 节点只能 attend 到其树中祖先节点，**不能** attend 到兄弟分支。

这可实现为块稀疏因果掩码。树共有 `|节点数|` 个 token；前向传播代价与节点数成正比，而非路径数。

验证后，目标模型在每个节点位置的 token 概率与 Medusa 头的提议通过标准拒绝采样规则比较。**最长已接受的根到叶路径**成为输出，状态更新至该端点。

### 训练

Medusa 头以交叉熵损失在训练数据上训练，预测正确的未来 token。目标模型主体可冻结（Medusa-1）或联合微调（Medusa-2）。Medusa-1 参数量增幅不足 1%，单节点几小时即可完成训练。

## EAGLE — 特征级自回归草稿

### 特征级草稿的洞察

Medusa 头是非自回归的：head-2 对 `t+2` 的预测不使用 `t+1` 的预测结果，这限制了接受率——对 `t+3` 的预测仅依赖 `t` 的隐状态，无法感知 `t+1`、`t+2` 是什么。

EAGLE 在**隐状态（特征）层面**恢复了自回归结构。一个轻量草稿网络根据目标模型最后一层的 `h_t` 和当前 token 嵌入 `e_t` 预测隐状态 `h_{t+1}`，再通过目标模型共享的 **LM 头**得到 token 概率：

```
h_t（来自目标模型）──┐
                      ├──► 草稿网络 ──► ĥ_{t+1} ──► [共享 LM 头] ──► p̂(x_{t+1})
e_t（token 嵌入）  ──┘
```

草稿网络仅是单个 transformer 层或小 MLP，比运行完整目标模型要轻得多。核心观察：**特征（隐状态）比 token 更可预测**——单层在 `h_t` 上操作捕获了远多于仅凭 `h_t` 无法获取的上下文信息。

### EAGLE 的树构建

EAGLE 的草稿是自回归的，因此树逐层扩展：

1. 运行一次草稿网络：预测 `ĥ_{t+1}`，采样 top-`s₁` token → 第一层节点。
2. 对每个第一层节点，再次运行草稿网络：预测 `ĥ_{t+2}`，采样 top-`s₂` token → 第二层节点（分支为 `s₁ × s₂`）。
3. 重复至深度 `d`。

生成具有结构化分支的树。验证时使用与 Medusa 相同的树注意力掩码。

### EAGLE-2 — 自适应树扩展

固定树浪费预算：深度 1 高置信度的预测应更大范围扩展；低置信度的预测应尽早剪枝。

EAGLE-2 引入基于草稿 token 概率的**草稿置信度得分**。树扩展过程中，节点仅在概率超过动态阈值时才展开；否则剪枝。总树规模被约束在固定预算 `B`（例如 60 个节点），但形状自适应地聚焦在草稿模型置信度高的区域。

关键结论：通过将预算集中在高接受率分支上，EAGLE-2 在相同计算预算下比 EAGLE-1 吞吐提升约 20%。

## 树验证算法

所有树形方法共享的核心原语：

```python
def tree_verify(prefix_kv, tree_nodes, target_model, draft_probs):
    # tree_nodes: (token, parent_idx) 列表 — 候选树
    # 构建树注意力掩码：节点 i 可 attend 前缀 + 节点 i 的祖先
    mask = build_tree_mask(tree_nodes)

    # 对所有树节点做单次前向传播
    target_logits = target_model(prefix_kv, tree_nodes, attn_mask=mask)

    # 遍历每条根到叶路径；对每个 token 拒绝采样
    best_path = []
    for path in enumerate_paths(tree_nodes):
        accepted = []
        for depth, (token, node_idx) in enumerate(path):
            p_target = softmax(target_logits[node_idx])[token]
            p_draft  = draft_probs[node_idx][token]
            u = random.uniform(0, 1)
            if u < p_target / p_draft:
                accepted.append(token)
            else:
                # 从残差分布中采样替换 token
                replacement = sample_residual(target_logits[node_idx], p_draft)
                accepted.append(replacement)
                break
        if len(accepted) > len(best_path):
            best_path = accepted

    return best_path
```

掩码构建是实现细节：深度 `d`、分支 `b` 上的节点 `i` 只能 attend 分支 `b` 上的 `d` 个祖先加上全部 `|前缀|` 个 token，存储为拼接后序列（前缀 + 树节点）的块稀疏三角掩码。

## 工程权衡

| 决策 | 收益 | 代价 |
|---|---|---|
| Medusa 头（并行、非自回归） | 草稿零延迟；训练简单 | 接受率低于顺序草稿；head-2 看不到 head-1 的输出 |
| EAGLE 特征级草稿（自回归） | 更高接受率；共享 LM 头 | 每次草稿步多一次轻量前向；训练更复杂 |
| EAGLE-2 自适应树 | 最佳计算预算下的吞吐 | 动态形状使批处理复杂化；需要调整阈值 |
| 树注意力掩码 | 单次前向验证整棵树 | 需要自定义注意力实现；不是标准因果掩码 |
| 仅微调 Medusa 头 | 可快速附加到任意已有模型 | 若目标模型未针对 Medusa 训练，效果次优 |
| 多请求批处理 | 吞吐随批量增长 | 不同请求接受路径长度不同；调度器需处理可变长度步骤 |

## 实验与结果

| 方法 | 加速比（单流） | 备注 |
|---|---|---|
| 原始投机解码（独立草稿模型） | 2.0–3.0× | 基线；70B 目标配 7B 草稿 |
| Medusa-1（冻结） | 2.2–2.8× | Vicuna / Llama-2 |
| EAGLE-1 | 2.5–3.5× | Llama-2 / 3；同等树预算下优于 Medusa |
| EAGLE-2 | 3.0–4.0× | 自适应树；当前无损解码加速的最优方法 |

所有方法在拒绝采样保证下精确保留目标分布。

在**批处理 serving**（vLLM 生产工作负载）中，加速比为 1.3–1.8×——低于单流基准，但稳定存在，且与 PagedAttention 和连续批处理叠加。

## 生产集成

**vLLM**：通过 `--speculative-model` 和 `--num-speculative-tokens` 支持 Medusa 和 EAGLE；指定 Medusa/EAGLE 权重时启用树解码。调度器通过补齐验证批次来处理可变接受 token 数。

**TensorRT-LLM**：Medusa 头在编译时内置进模型引擎；树注意力掩码融合进注意力 kernel。

**Hugging Face TGI**：v1.4 起支持 Medusa。

实际操作：在已有 Llama 3 70B 部署上添加 EAGLE，只需下载对应的 EAGLE 草稿权重（单个 transformer 层，约 1 亿参数），在推理服务器中同时指向两套权重并开启树解码，无需修改目标模型。

## 深度解析

Medusa → EAGLE → EAGLE-2 的演进是针对特定瓶颈持续收紧的典型案例。原始投机解码确立了正确的抽象（草稿 + 验证）；变体则证明**草稿分布的质量**才是核心杠杆——而特征比 token 携带更多信号。树形结构是在接受单条草稿路径低效这一前提后的自然延伸。

最深层的开放问题是**与解耦 prefill/decode 的组合性**（见 [DistServe](../distserve/zh.md) 和[前缀缓存](../prefix-caching/zh.md)）：在解耦系统中，prefill 节点和 decode 节点分开运行，可能在不同硬件上。树验证要求目标模型看到整棵树——这将验证绑定在 decode 节点，使任何 prefill/decode 异步共享状态的方案都更复杂。

第二个开放问题是**自适应草稿模型选择**：对简单请求（高 `α`），深树加大投机最优；对难请求（低 `α`），开销占主导，回退标准解码可能更好。针对此的运行时 oracle 方法是活跃的研究方向。

## 参考文献

- [1] Cai et al. _Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads._ arXiv:2401.10774, 2024.
- [2] Li et al. _EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty._ arXiv:2401.15077, 2024.
- [3] Li et al. _EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees._ arXiv:2406.16858, 2024.
- [4] Leviathan et al. _Fast Inference from Transformers via Speculative Decoding._ ICML '23. → 见[投机解码](../speculative-decoding/zh.md)。
- [5] Miao et al. _SpecInfer: Accelerating LLM Serving with Speculative Inference and Token Tree Verification._ MLSys '24.
