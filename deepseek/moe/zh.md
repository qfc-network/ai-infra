# DeepSeekMoE — 细粒度专家 + 共享专家隔离

- **作者 / 机构**：DeepSeek-AI
- **发表时间**：2024-01
- **链接**：[论文 (arXiv:2401.06066)](https://arxiv.org/abs/2401.06066)

## 一句话总结

DeepSeekMoE 对标准 top-k MoE 做了两处改造：**(1) 细粒度专家切分**——把每个专家切成 `m` 个更小的专家，同时把 top-k 放大 `m` 倍，保持激活算力不变但专家化粒度更细；**(2) 共享专家隔离**——设少量"始终激活"的共享专家承载通用知识，让路由专家得以真正专业化。在同等激活参数预算下，DeepSeekMoE 能匹敌算力显著更多的模型，这也是 V2、V3 沿用的架构基石。

## 背景与动机

经典 MoE（GShard、Switch）在 8–64 个专家中做 top-1 或 top-2 路由。观察到两种失效模式：

1. **知识冗余**——常见 token 迫使多个专家学出相似的通用特征，容量被浪费。
2. **专业化粒度粗**——专家少而大时，路由决策分辨率低，相似 token 常塌缩到同一个专家。

先前工作在均衡损失与路由算法上打转。DeepSeekMoE 的判断是：这是**结构性问题**——专家数太少且每个太厚，且没有机制去剥离共享知识。

## 核心方法

### 细粒度专家切分

取一个常规 MoE：`N` 个专家，中间维度 `d_ff`，top-`K` 路由。改为：

- `mN` 个专家，每个中间维度 `d_ff / m`。
- top-`mK` 路由。

激活参数与 FLOPs 不变。但不同专家组合的数量从 `C(N, K)` 爆炸到 `C(mN, mK)`，允许极细的专业化。V3 中 `m` 隐式地很大——256 路由专家，每 token top-8。

### 共享专家隔离

在路由专家之外，设 `K_s` 个**共享专家**，每个 token 都会经过。它们承担通用知识（语法、高频模式、跨域共享特征），从而让路由专家不必重复学这些。

形式化：

```
y_t = Σ_{i ∈ shared} E_i(x_t)  +  Σ_{j ∈ top-k routed} g_{t,j} · E_j(x_t)
```

共享专家本质上是一个小的 dense FFN，与稀疏路由并行运行。

### 负载均衡

原论文仍用标准的 aux load-balance loss；V3 后来改用无辅助损失的偏置方案。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 更多、更小的专家 | 更细的专业化，更多组合 | 路由开销随专家数上升；每 token 的 all-to-all 随 top-k 放大 |
| 共享专家 | 路由专家卸掉通用特征；稳定性更好 | 每 token 多一份 dense 计算；稀疏率下降一点 |
| top-mK 路由 | 更丰富的专家混合 | 分发成本上升；通信 kernel 压力更大 |
| 激活 FLOPs 不变 | 与 baseline MoE 公平对比 | 参数量增加（专家更小但更多） |

## 实验与结果

- 在 2B、16B、145B 总参数规模上，DeepSeekMoE 与同激活 FLOPs 的 GShard 式 MoE 持平或胜出。
- Ablation 显示"切分"和"共享专家"都独立贡献，去掉任一项都掉点。
- V2、V3 在生产规模沿用此配方；V3 每 MoE 层 256 路由 + 1 共享。

## 复现要点

- 架构可在任何 MoE 框架（Megatron-LM、DeepSpeed-MoE、Tutel）上直接加。
- 工程难点在 high-top-k、多专家场景下的高效 all-to-all——见 DeepEP 的实现。
- V2 / V3 开源权重可直接检视训练后的产物。

## 个人评注

这是一个**被低估的概念性贡献**：把 MoE 从"少数大而全的专家互相竞争"改造为"**多个小专家 + 一个小通才**"，是更合理的分解。共享专家尤其在事后看很显然，现已开始在业界普及。细粒度切分则是让 256+ 专家的 MoE 真正可用的关键——迫使架构依赖组合覆盖而非单专家容量，而组合恰是路由机制擅长的事。下一步值得关注：**条件性共享专家**（只在某个领域簇内共享）与**层次路由**（在超大 `N` 时驯服 top-k）。

## 参考

- [1] Dai et al. _DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models._ arXiv:2401.06066, 2024.
- [2] Fedus et al. _Switch Transformers._ JMLR, 2022.
- [3] Lepikhin et al. _GShard._ arXiv:2006.16668, 2020.
- [4] DeepSeek-AI. _DeepSeek-V2 / V3 Technical Reports._
