# SmoothQuant —— 激活到权重的 outlier 迁移，通向 W8A8

- **作者 / 机构**：Xiao 等，MIT + NVIDIA
- **发表时间**：ICML '23
- **链接**：[论文 (arXiv:2211.10438)](https://arxiv.org/abs/2211.10438) · [代码](https://github.com/mit-han-lab/smoothquant)

## 一句话总结

与纯权重量化（GPTQ、AWQ）不同，SmoothQuant 把**激活与权重都量到 INT8**——获得 INT8 tensor core 带来的约 2× 吞吐，不只是省显存。核心障碍：激活在特定通道上有极端 outlier，这些通道上的 INT8 量化会毁掉精度。SmoothQuant 的洞察：outlier 存在于**一小批可预测的激活通道**上，可以**数学上把 outlier 从激活迁到权重**——用逐通道 scale 把激活变得好量化，同时让权重吃掉难度。反正权重本身没 outlier，能容得下多余动态范围。结果：OPT-175B、Llama 上实现无损 W8A8（INT8 权重 + INT8 激活），且获得真实算力加速（不只是显存）。

## 背景与动机

2022 年前后，LLM INT8 推理标准做法：

- **权重-only INT8**（容易）：权重分布良好，round-to-nearest 困惑度损失 <0.1。但激活留 FP16，算力仍走 FP16 tensor core。
- **W8A8**（难）：INT8×INT8 tensor core 在 Ampere 上比 FP16 快 2×，Hopper 上快 4×。serving 大赢——*如果* 能量化激活且精度不崩。

LLM.int8()（Dettmers 等，2022）指出激活有**持续性 outlier 通道**——特定通道的值比其他通道大 10–100×。若你对整个张量做 per-tensor INT8 量化，outlier 垄断了 scale，非 outlier 值失去全部分辨率。LLM.int8() 的解法：**混合精度**——运行时检测 outlier，留 FP16，其他 INT8。能用但慢（运行时分支）、笨重。

SmoothQuant 的目标：到 W8A8 且**纯 INT8 运算**，不混合精度、不分支。

## 核心方法

### 观察

LLM 激活 outlier 的性质：

- **通道持续** —— 特定激活通道在几乎所有 token 上都有 outlier，不是少数罕见 token。outlier 结构是层的属性，不是输入的。
- **层特异** —— 某些层（最早几层、FFN 输出）outlier 很强；其他层没有。
- **权重没有 outlier** —— 权重分布良好，能吸纳额外动态范围。

于是出现不对称：激活难量化，权重易量化。SmoothQuant 问：能不能把量化难度从激活转给权重？

### Scaling 变换

线性层 `y = x · W`，引入逐通道 scale `s`：

```
y = x · W = (x / s) · (s · W)    s 按输入通道
```

对任意正 `s` 等式成立。选 `s` 使：

- `x / s` 的 outlier 通道**被压**——更好 INT8 量化。
- `s · W` 的对应列**被放**——权重接过难度。

激活有 outlier 而权重没有，把难度挪到权重是划算的：权重从"极易量化"变到"略难量化"，激活从"不可能"变到"可行"。最终两边都 INT8。

### 如何选 `s`

论文给的公式很简单：

```
s_j = max(|x_j|)^α / max(|W_j|)^(1-α)
```

按输入通道 `j`，在"激活 outlier 多重"与"权重能承担多少范围"间平衡。`α ∈ [0, 1]` 是迁移强度超参；`α = 0.5` 是默认，`α = 0` 不迁移（标准量化），`α = 1` 把全部难度推给权重。

按模型（或按层）在校准集（~128 样本）上选 `α`。随后 scale `s` **被融进前一层的权重**（吸收进产生 `x` 的 FFN 或 LayerNorm）。**推理时零开销**——融合后模型就是 INT8 张量 + INT8 matmul。

### 融合细节

迁移 scale `s` 得落在某处。常见两种融合目标：

1. **LayerNorm**：LayerNorm 本来就有逐通道 scale（γ）；把 `γ` 乘 `1/s`，输出 `x/s` 天然出来。
2. **前一个线性层的权重**：前层的输出通道权重乘 `1/s`。

任一方式下，被变换的模型把 `x/s` 直接作为量化线性层的输入，运行时不需要除法。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| W8A8 而非权重-only | 全部 INT8 tensor core 加速（vs FP16 的 2–4×） | 需要解决激活 outlier 问题 |
| 通过 scaling 迁移 outlier | 纯 INT8 推理，无混合精度 | 每模型要选 `α`；权重少一点 headroom |
| scale 融入前一层 | 推理时零开销 | 需要能控制上游 op；并非总可以 |
| 逐通道迁移 | 按 outlier 自然粒度处理 | scale 比 per-tensor 多 |
| 静态（校准时） | 快；能部署 | 不适应漂移的激活分布 |

## 实验与结果

- **OPT-175B**：SmoothQuant W8A8 困惑度与 FP16 差 ~0.1；端到端延迟相对 FP16 1.56×。
- **Llama-1/2**：精度几乎相同；吞吐显著提升。
- 与 LLM.int8() 对比：精度相当或更好，**无混合精度分支**；实际快 2–3×。
- 与朴素 W8A8（无迁移）对比：精度崩（困惑度 +10 以上）——确认 outlier 问题才是真正的拦路虎。

## 复现要点

- 参考实现开源；通过 hook 式模型变换与 PyTorch 集成。
- 校准数据：基座模型用 ~128 个 Pile 样本就行；微调模型用领域内数据。
- 对 OPT、Llama、GPT-2 家族开箱即用；归一化层特殊的 transformer 可能要微调。
- **NVIDIA TensorRT-LLM** 等生产推理栈已把它作为标准 INT8 预处理选项集成。

## 个人评注

SmoothQuant 坐在纯权重量化（GPTQ、AWQ）与全量化（LLM.int8() 的混合精度）之间，给出最干净的工程权衡：**把问题推到能解的地方（权重）、接受微小精度代价、换来真实算力加速**。

更广的启示是**量化难度的不对称**：不是每个张量量化难度相同，巧妙的重参数化能在张量间移动难度而不改变结果。同一想法还出现在：

- **AWQ**（通过 scale 激活来保护显著权重通道）—— 数学近亲。
- **FP8 训练**（V3 细粒度 scaling 配方：逐 tile 激活 scale、逐 block 权重 scale）—— 训练时的 outlier 迁移。
- **对数线性量化**与**基于旋转的量化**（QuaRot、SpinQuant）—— 后续工作把"变换张量以减少 outlier"的框架推广。

到 2026 年，**SmoothQuant 的统治力在减弱**——FP8 硬件（Hopper、Blackwell）让 W8A8 级加速用原生 FP8 就能拿到，精度代价还更小。但**激活 outlier 迁移模式**已成量化工具箱的常规工具，在没有 FP8 硬件的 Ampere 时代部署上仍占优。

对实践者：有 FP8 支持的 H100/H200，用原生 FP8；A100 或更旧，或出于显存带宽考虑必须 INT8，SmoothQuant 是起点。INT4 权重压缩 + FP16 激活用 AWQ。三者（GPTQ、AWQ、SmoothQuant）合起来定义了 transformer 推理的量化默认工具箱。

## 参考

- [1] Xiao et al. _SmoothQuant._ ICML '23 / arXiv:2211.10438.
- [2] Dettmers et al. _LLM.int8()._ NeurIPS '22.（outlier 观察；混合精度基线）
- [3] Ashkboos et al. _QuaRot._ NeurIPS '24.
- [4] Liu et al. _SpinQuant._ 2024.（基于旋转的 outlier 消除）
- [5] NVIDIA TensorRT-LLM SmoothQuant 集成文档。
