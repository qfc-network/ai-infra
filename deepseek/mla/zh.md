# MLA — Multi-head Latent Attention

- **作者 / 机构**：DeepSeek-AI（在 DeepSeek-V2 中提出）
- **发表时间**：2024-05
- **链接**：[DeepSeek-V2 论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) · [FlashMLA](https://github.com/deepseek-ai/FlashMLA)

## 一句话总结

MLA 把 MHA 中"每个 head 一份 KV cache"换成了**每个 token 只存一个低秩 latent 向量**，推理时再在线重建各 head 的 K、V。KV cache 缩到标准 MHA 的约 1/10，而质量**与 MHA 持平而非劣化**。不同于 MQA/GQA 那种"牺牲质量换显存"的权衡，MLA 通过把上投影矩阵吸收进注意力计算，在两个维度上同时压过 MHA。

## 背景与动机

长上下文推理的显存和带宽基本由 KV cache 主导：

- **MHA**：每序列 cache = `2 · n_h · d_h · L · n_layers`，很贵。
- **MQA**：所有 head 共享一组 K、V。cache 小 ~`n_h` 倍，但有明显质量损失。
- **GQA**（Llama 2 起）：折中——按组共享，质量接近 MHA。

MQA/GQA 都是"显存换质量"的单向交易。DeepSeek 的问题是：能不能**保留 MHA 的完整表达力**，同时把 cache 做到接近 MQA？

## 核心方法

### 低秩联合 KV 压缩

对每个 token，把隐状态 `h_t` 一次性投影到**压缩 latent** `c_t^{KV} ∈ R^{d_c}`，其中 `d_c ≪ n_h · d_h`：

```
c_t^{KV} = W_DKV · h_t
```

注意力计算时，再按 head 解压出 K、V：

```
k_t^{(i)} = W_UK^{(i)} · c_t^{KV}
v_t^{(i)} = W_UV^{(i)} · c_t^{KV}
```

**cache 只保存 `c_t^{KV}`**（外加下面一小段 RoPE 分量）。典型配置 `d_c ≈ 4·d_h` 时，每 token 的 cache 已与 MQA 相当，但各 head 的参数仍是独立的。

### 吸收技巧

表面看，每步都要重建 K、V 似乎很费算力。但 `W_UK` 可以**吸收进 W_Q**（因为 `Q · K^T = (h · W_Q) · (W_UK · c)^T = h · (W_Q · W_UK^T) · c^T`），`W_UV` 吸收进输出投影。推理时**无需显式上投影**——注意力直接跑在压缩 latent 上。

### 解耦 RoPE

RoPE 依赖位置，不能作用在已吸收的矩阵乘积上。MLA 的处理是把每个 key 分成两部分：

- **压缩部分**（不含 RoPE），通过吸收路径从 `c_t^{KV}` 重建。
- **解耦部分** `k_t^R`，小维度 `d_h^R`，**单独做 RoPE，单独缓存**。

Query 对称处理。两部分拼接得到最终 per-head K，RoPE 贡献正确流入。

### Query 压缩（可选）

同样的低秩技巧也用在 query 上，降低训练激活显存。这不影响 cache 大小。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 单个 latent 替代 per-head K,V | KV cache 缩小 ~10× | 多两个投影（推理时被吸收，训练时仍在） |
| 解耦 RoPE 维度 | 位置编码与吸收共存 | cache = latent + 一小段 RoPE 分量（开销可忽略） |
| 推理时吸收 | 省掉上投影算力 | kernel 必须融合复合权重，朴素实现吃不到性能 |
| 完整 per-head 表达力 | 质量对齐 MHA（无 GQA 税） | 参数量高于 MQA |

## 实验与结果

V2 报告中：

- 每 token 的 KV cache：MLA 约为 MHA 的 1/7 ~ 1/14（随配置变化）。
- 在同参数预算下，MLA 在主流 benchmark 上与 MHA 持平或略胜；在同 cache 大小下明显强于 GQA 变体。
- 训练吞吐也提升——attention 的激活显存变小。

FlashMLA（2025 开源）提供了调优过的 kernel，H800 上 decode 可接近 HBM 带宽上限。

## 复现要点

- V2 论文对架构描述足够详细，已有多个开源复现。
- FlashMLA 提供生产级 decode kernel；prefill kernel 生态尚在完善。
- 训练配方（初始化、稳定性）的公开信息少于架构本身。

## 个人评注

MLA 是少见的"两个维度上同时压过前作"的架构改动，而不是典型的权衡。它的洞察——K、V 投影可以被代数吸收，所以压缩在推理时不必付算力代价——正是 DeepSeek 风格的"系统感知式建模"。预期这个思路会扩散：**凡是 cache 可以重写成"存储 latent × 固定权重"的形式，都能用同样的技巧**。开放问题是 MLA 能否与未来的 attention 变体（sliding window、线性注意力混合）干净组合；从解耦 RoPE 的做法看，大概率可以，代价是实现复杂度。

## 参考

- [1] DeepSeek-AI. _DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model._ arXiv:2405.04434, 2024.
- [2] DeepSeek. _FlashMLA._ https://github.com/deepseek-ai/FlashMLA, 2025.
- [3] Shazeer. _Fast Transformer Decoding: One Write-Head is All You Need._ arXiv:1911.02150, 2019.（MQA）
- [4] Ainslie et al. _GQA: Training Generalized Multi-Query Transformer Models._ arXiv:2305.13245, 2023.
