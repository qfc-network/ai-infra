# 权重量化 —— GPTQ 与 AWQ

- **论文**：
  - **GPTQ** —— Frantar 等，IST Austria。ICLR '23。
  - **AWQ** —— Lin 等，MIT + NVIDIA + 上交 + UMass。MLSys '24。
- **链接**：[GPTQ (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) · [AWQ (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) · [GPTQ 代码](https://github.com/IST-DASLab/gptq) · [AWQ 代码](https://github.com/mit-han-lab/llm-awq)

## 一句话总结

两种把 LLM 权重从 FP16/BF16 压到 **INT4、精度损失极小**的训练后量化方法，让推理可以跑在消费级 GPU 上，serving 成本大降。都是**纯权重量化**：激活保持 FP16，只压权重。两者的分歧在于如何决定量什么、量多少：

- **GPTQ** 用**二阶信息（基于 OBQ 的 Hessian 逆）**贪心逐列选网格点并更新剩余列，最小化逐层重建误差。
- **AWQ** 观察到**权重中有一小撮"显著"通道**（其激活值很大），用**逐通道 scaling 技巧**在统一 INT4 量化下保护这些通道——无梯度、无反向、无二阶。

两者都是**训练后**（校准集上几小时，不重训），困惑度与 FP16 相差 ~0.1–0.5，已成 llama.cpp、vLLM、TGI 以及所有本地 LLM 部署的默认权重格式。现代栈通常偏好 AWQ——更简单、更硬件友好、有逐通道激活感知；GPTQ 在更多形状支持和成熟开源实现上仍常见。

## 背景与动机

70B 模型 FP16 是 140 GB 权重——单张消费卡装不下，H100 上跑也贵。训练时量化（QAT）需要重训，对已发布 checkpoint 不现实。INT8 训练后量化（PTQ）早已稀松平常（8 位几乎无损）；**真正的难题一直是不重训的 INT4**——4 位的误差会吞掉前向，朴素 round-to-nearest 直接把困惑度打崩。

问题：能否用一次几百样本的巧妙校准，在不动训练的情况下把 4 位丢失的精度找回来？

GPTQ 回答："能，用 Hessian 指导的逐列更新。"AWQ 回答："能，而且你不需要 Hessian——只需保护显著通道。"

## GPTQ —— Hessian 指导的逐列量化

基于 **Optimal Brain Quantization (OBQ)**，它把 Optimal Brain Surgeon 从剪枝推广到离散化。核心想法：量化一列权重时，可以用本层重建损失的**局部 Hessian 逆**算出*对剩余（未量）列的最优更新*，以补偿误差。

朴素 OBQ 每层 `O(d³)`，70B 模型扛不住。GPTQ 的贡献：

1. **任意顺序** —— OBQ 在每步贪心挑当前期望误差最小的列。GPTQ 证明 **固定从左到右的顺序** 几乎同样好，消掉 `O(d²)` 因子。
2. **延迟批量更新** —— 不在每次量化后立即更新所有剩余列，按组 batch 更新。显存友好、数值稳定。
3. **Cholesky 重写** —— 把 Hessian 逆的计算重写为 Cholesky 因子，主循环变成条件良好的三角求解。

结果：单张 A100 上约一小时量完 70B，INT4 困惑度损失通常 <0.3。

输出是**每组统一 INT4**（常见 group size 128——行方向上每 128 个连续权重共享一个 FP16 scale + zero-point）。group size 权衡压缩比与精度：128 是常见甜点。

## AWQ —— 激活感知的权重量化

起点是另一个观察：若你看权重幅值**按对应激活加权**后的分布，约 1% 的通道对输出的贡献比其他通道高几个数量级。这些是**显著**通道——保住它们，其他通道能激进量化；保不住，连 8 位都崩。

朴素保护：显著通道留 FP16、其他 INT4。能用，但混合精度张量对 kernel 很烦——破坏简单 matmul 布局、加分支。

AWQ 的技巧：**量化前把显著通道放大，推理时把对应激活缩小**。数学上，输入 `x`、权重列 `w`：

```
y = x · w = (x / s) · (w · s)    对任意 s > 0
```

选 `s` 使显著列的放大权重舒适地装进 INT4 区间，同时非显著列的缩小激活也还在范围内，乘积保持不变。**每个权重仍是 INT4**——唯一改动是一个逐通道 scale，这个可以**吸收进前一层的权重或归一化**，推理时零成本。

搜索：按通道在几个候选 scale 上网格搜索，校准集约 128 样本。无梯度、无 Hessian。整个过程对典型模型几分钟。

### 为什么实践中 AWQ 常胜 GPTQ

- **硬件友好**：逐通道 scale 是每个 GEMM 库的标准；GPTQ 的 group 布局不太标准。
- **无跨层校准耦合**：AWQ 的 scaling 逐层局部；GPTQ 的列更新连锁。
- **对 outlier 重的层更稳**：LLM 激活在特定层（第一个 attention、FFN 输出）有长尾——AWQ 的激活直接感知把这些清爽处理。
- **实现正确性简单**：AWQ 代码 ~1000 行；GPTQ 涉及小心的 Hessian 数值。

GPTQ 在某些特定层形状和权重分布异常的模型上仍更优。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 权重-only INT4（任一方法） | 模型压 ~4×；能塞进消费卡 | 激活仍 FP16——算力节省只对 decode（带宽瓶颈）有效 |
| group size 128 | 精度 / 压缩的好平衡 | 不是所有形状最优；64 更精，256 更压 |
| GPTQ Hessian 机制 | 理论有依据；某些形状上好 | 复杂；数值要小心；量化期间显存占用大 |
| AWQ 逐通道 scaling | 简单、快、硬件友好 | 形式化推理难些；校准样本选择重要 |
| 训练后（两者） | 不重训；能量化已发布 checkpoint | 小幅精度损失；激进位宽不如 QAT |
| 每组 FP16 scale | 开销可忽略、精度好 | kernel 必须支持该布局 |

## 实验与结果

- **GPTQ**：INT4 OPT-175B 困惑度与 FP16 差 0.1；端到端相对 FP16 快 3.25×；首篇在 100B+ 规模证明 PTQ INT4 可行。
- **AWQ**：INT4 Llama / Mistral / Qwen / Falcon 困惑度与 FP16 差 ~0.05–0.15；推理更快（布局更干净）；在指令微调模型上特别强（outlier 更明显）。
- **现实影响**：llama.cpp 的 `Q4_K_M` 与类似格式、vLLM 的 AWQ / GPTQ 支持、ExLlama、TGI 量化 kernel——几乎所有本地 LLM 部署都用这两者之一（或 **GGUF** 这样吸收两者设计的变体）。

## 复现要点

- 两个算法均完全开源，参考实现和测试良好的 wrapper 齐全。
- 校准数据重要：用 ~128 个领域内样本；WikiText-2 常用但对指令微调模型非最优。
- **AutoAWQ** 与 **AutoGPTQ** 提供 Python wrapper；HuggingFace Hub 上很多模型已预量化。
- 推理 kernel：AWQ 的来自 MIT Han Lab；GPTQ 的经 ExLlama / Marlin。**Marlin**（手调 INT4×FP16 GEMM）是常用共享后端。

## 个人评注

GPTQ + AWQ 留下的真正启示是**精度集中在少数通道里**。朴素量化心智模型（"每个权重失一点精度"）是错的：outlier 通道在干活，精度问题整体归约为"outlier 保住了吗？"这个框架正是 AWQ 比 GPTQ 更简单也常更好的原因——AWQ 显式处理 outlier，GPTQ 通过 Hessian 隐式捕获同一洞察，工作量更大。

看 2026 的格局：**FP8/FP4 硬件支持（Blackwell、MI300）让纯权重量化的重要性在下降**——原生格式就是 8 位甚至 4 位。但 AWQ 的**激活感知 scaling** 活下来了——它现在被用来在混合精度 FP8 / FP4 **训练**里决定哪些位置保留更高精度（V3 的细粒度 scaling 配方正是近亲）。算法想法的寿命超过了它最初的量化目标。

对实践者：新部署默认 AWQ；模型已知在 AWQ 下有问题才用 GPTQ（少见）。别自己发明量化——生态已成熟，kernel 已调优，2026 年重写不是找性能的地方。

## 参考

- [1] Frantar et al. _GPTQ._ ICLR '23 / arXiv:2210.17323.
- [2] Lin et al. _AWQ._ MLSys '24 / arXiv:2306.00978.
- [3] Frantar & Alistarh. _Optimal Brain Compression (OBC/OBQ)._ NeurIPS '22.（GPTQ 的基础）
- [4] Dettmers et al. _LLM.int8() 与 SmoothQuant_ 同期 W8 工作。
- [5] Marlin INT4 kernel：https://github.com/IST-DASLab/marlin
