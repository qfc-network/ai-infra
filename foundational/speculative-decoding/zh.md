# 投机解码（Speculative Decoding）

- **作者 / 机构**：Leviathan 等（Google，ICML '23）与 Chen 等（DeepMind，2023，同期独立工作）。后续：Medusa（Princeton/Together）、EAGLE、Lookahead（Meta）。
- **发表时间**：2022-11 — 2024
- **链接**：[Leviathan (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) · [Chen (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) · [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077)

## 一句话总结

LLM 解码是显存带宽瓶颈：每步都要把全部模型权重从 HBM 读一遍才产出一个 token。投机解码的做法是：用一个**便宜的 draft 模型**顺序提出 `K` 个 token，再让**昂贵的 target 模型在一次 forward 内校验全部 `K` 个**。两篇论文独立导出的 rejection sampling 方案，在数学上保证被接受的 token **与从 target 直接采样的分布完全一致**——无质量损失。典型提速 2–3×（贪心、采样同量级）。现已成为 vLLM、TensorRT-LLM 等生产栈的标配；后续变体（Medusa、EAGLE）直接省掉独立 draft 模型。

## 背景与动机

解码阶段的 transformer 推理由**权重从 HBM 读取**主导，不由算力主导。一次 forward 产一个 token；算术强度极低。但对 `K` 个 token 做一次 forward，wall time 与单 token 相差无几——权重反正要读，tensor core 还有巨大余量。

问题：没真正批处理 workload 时，这份余量能换成加速吗？答案是：先投机生成 `K` 个候选 token，再在一次 batched forward 里校验。如果多数被接受，有效 token/step 就上去了。

## 核心方法

### Draft + verify

- **Draft 模型** `q(x_t | x_<t)`：小、快；可以是从 70B target 蒸馏出的 1B，也可以只是更短的版本。
- **Target 模型** `p(x_t | x_<t)`：目标分布。

每个外层 step：

1. Draft 模型贪心或随机生成 `K` 个 token `x_1, ..., x_K`。
2. Target 模型对前缀 + 这 `K` 个 token 做**一次 forward**，得到 `p(· | x_<t+i)`，`i = 0, ..., K`。
3. 对每个 draft token 做 **rejection sampling** 校验：
   - 以概率 `min(1, p_i(x_i) / q_i(x_i))` 接受 `x_i`。
   - 拒绝时，从残差分布 `(p_i − q_i)_+ / norm` 采样一个替换 token，并终止。
4. 如果 `K` 个全部被接受，再从 `p_{K+1}` 多采样一个（免费，因为 target forward 已算出）。

每个外层 step 接受 1 到 `K+1` 个 token，代价只有一次 target forward。数学上可证边际分布等于 target。对贪心、temperature、top-k、top-p 都有对应版本。

### 接受率决定加速

加速受限于：

- Draft 与 target 一致的比例（接受率 `α`）。
- 成本比 `c = time(draft) / time(target)`。

期望加速约 `(1 − α^(K+1)) / ((1 − α) · (K·c + 1))`。调参：`K` 取到再多也没意义的临界点，典型 `K = 4–8`。

### 去掉独立 draft 模型

- **Medusa**：在 target 上挂几个额外 LM head 直接预测 `+2, +3, +4` 位置的 token。draft 是并行的，非自回归。无额外模型权重，仅需少量微调。
- **EAGLE**：预测未来 token 的**特征**而非 token 本身，再喂给 target 的 LM head 校验。相似成本下接受率更高；目前是无损解码加速的 SOTA。
- **Lookahead decoding**：用 Jacobi 迭代从 target 本身生成 draft 轨迹，无 draft 模型、无训练。接受率弱一点，但零配置。

### 批处理与 serving

生产 serving 下，投机解码与批处理的交互并不平凡：不同请求在同一外层 step 接受不同数量的 token，对齐被打破。vLLM、TensorRT-LLM 用 padding + masked verification 处理；真实 batch 下仍有收益，但小于单流 benchmark。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 独立 draft 模型 | 简单、模型无关 | 要训/选 draft；多一份显存；与 target 分布对齐影响 `α` |
| Medusa / EAGLE（自 draft） | 无额外模型；`α` 更高 | 需要微调；方法绑到具体权重 |
| Lookahead（无需训练） | 零配置 | `α` 较低，加速较小 |
| 精确 rejection sampling | 无质量损失，数学清爽 | 比"直接接受 top-1"复杂；每步小额开销 |
| `K` 较大 | 峰值加速高 | `α` 不够时浪费算力；校验成本上升 |
| 批 serving | 真实 batch 下仍可用 | 有效加速变小；调度复杂度增加 |

## 实验与结果

- 原始 Leviathan：T5-XXL、PaLM 变体上 wall-clock 提速 2–3×。
- Chen：Chinchilla 上同量级。
- Medusa：Vicuna / Llama 上 2–3×，无 draft 模型。
- EAGLE-2：Llama 2 / 3 上可达 ~4×，分布无损；目前无损解码加速的 SOTA。

所有方法都**严格保留 target 输出分布**（对应指定采样器）。这一点是它们与量化、蒸馏等有损方法的分水岭。

## 复现要点

- vLLM、TensorRT-LLM、TGI 都在生产支持。
- Medusa / EAGLE 对主流基座模型提供开源权重和 head。
- 正确性测试很微妙——验证分布保留需要统计检验，不能只靠肉眼对比输出。

## 个人评注

投机解码是少见的**真正"免费"**的推理技巧——无质量损失、推理时无额外参数（自 draft 变体）、wall-clock 实打实收益。概念上的妙处在：**解码是带宽瓶颈，那就用算力换带宽**。从原始投机 → Medusa → EAGLE 是一条沿"接受率 / draft 成本"权衡迭代的典型路线。开放前沿在**树形校验**（校验分支 draft 树而非线性链）、与 **prefill/decode 解耦部署**的配合、以及在超长上下文下（target forward 重新变贵时）如何组合。做 LLM serving，这是继 PagedAttention 之后的下一个必选优化——收益立竿见影，且能与其他优化叠加。

## 参考

- [1] Leviathan et al. _Fast Inference from Transformers via Speculative Decoding._ ICML '23 / arXiv:2211.17192.
- [2] Chen et al. _Accelerating Large Language Model Decoding with Speculative Sampling._ arXiv:2302.01318, 2023.
- [3] Cai et al. _Medusa._ arXiv:2401.10774, 2024.
- [4] Li et al. _EAGLE._ arXiv:2401.15077, 2024.
- [5] Fu et al. _Lookahead Decoding._ ICML '24.
