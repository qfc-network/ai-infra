# DualPipe —— 源码走读

- **仓库**：[deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe)
- **发布**：2025-02（Open Source Week 第四天）
- **作者**：Jiashi Li、Chengqi Deng、Wenfeng Liang
- **配套笔记**：[`deepseek/v3-tech-report/`](../../v3-tech-report/) —— DualPipe 对应论文 §3.2.1
- **配套笔记**：[`foundational/megatron-lm/`](../../../foundational/megatron-lm/) —— 对照 1F1B / interleaved 1F1B

## 这个仓库是什么

DeepSeek-V3 训练中使用的**双向流水线调度**的 PyTorch 参考实现。DualPipe 让 micro-batch **在 PP 各 stage 上同时沿两个相反方向流动**，使每个 rank 总是在跑一个方向的算力，同时另一个方向的通信在后台。结果：**流水线 bubble 相对 1F1B 减半**，且**前向/反向与跨 stage 通信在调度层面就已重叠**，而不是靠 stream 魔法后补。

仓库本身很小（一个核心类，几百行 Python）——实际的 overlap 工作放在用户自己实现的 `overlapped_forward_backward` 方法里。DualPipe 给你调度；你给它一个"懂得自交织 F 与 B"的模块。这种分离是核心思想。

## Bubble / 显存对比（README）

| 方法        | Bubble                        | 参数/设备 | 激活/设备  | 设备数   |
|-------------|-------------------------------|-----------|------------|----------|
| 1F1B        | (PP−1)(F+B)                   | 1×        | PP         | PP       |
| ZB1P        | (PP−1)(F+B−2W)                | 1×        | PP         | PP       |
| DualPipe    | (PP/2−1)(F&B + B − 3W)        | 2×        | PP+1       | PP       |
| DualPipeV   | (PP/2−1)(F&B + B − 3W)        | 2×        | PP+1       | PP/2     |

其中 `F`、`B`、`W` = forward / 完整 backward / 仅权重梯度 的时长，`F&B` = 两个互相 overlap 的前后向一起跑的时长。

两点值得注意：

- **bubble 按 PP/2 而非 PP 增长** —— 这正是相对 1F1B 的 2× 改进。
- **每设备 2× 参数** —— 代价是真金白银。DualPipe 在每个设备上维护对称方向的第二份 PP 副本，这也是为什么 V3 能负担（MLA + FP8 已经省出足够显存来吸收这 2× 参数压力）。

DualPipeV 是随后"对半切"变体（Sea AI Lab 2024-10 博客首提），bubble 公式相同但**设备数变 PP/2**——通过 V 形布局把第二方向折叠到与第一方向相同的设备上。

## 目录结构

```
DualPipe/
├── dualpipe/
│   ├── __init__.py         # 导出 DualPipe、DualPipeV、WeightGradStore、set_p2p_*
│   ├── dualpipe.py         # 核心双向调度器
│   ├── dualpipev.py        # V 形 "cut-in-half" 变体
│   ├── comm.py             # P2P 收发原语；张量 shape/dtype 预注册
│   └── utils.py            # WeightGradStore（把 B 和 W 分开以填 bubble）
├── examples/
│   ├── example_dualpipe.py
│   └── example_dualpipev.py
├── images/                 # dualpipe.png、dualpipev.png
└── setup.py
```

整个包就这么小。两个导出类，几个工具。

## 核心想法：双向流水线

经典 1F1B：一条 micro-batch 流穿过 stage `0 → 1 → ... → PP-1`，再反向穿回来。流水线填充阶段前段空，排空阶段后段空。Interleaved 1F1B（Megatron）用非连续层组分配给 rank 改善了一点，但 bubble 仍是 `O(PP)`。

DualPipe **同时跑两条对称流水线**：

- 方向 A：micro-batch forward `0 → PP-1`，backward 回来。
- 方向 B：micro-batch forward `PP-1 → 0`，backward 回来。

每个 rank 同时持有两个方向的参数 —— 这就是 2× 参数代价的来源。任意时刻，每个 rank 在跑某种组合的 (A-forward、A-backward、B-forward、B-backward)，而关键是：**一个方向的 forward 可以与另一个方向的 backward 在同一 rank 上重叠**，因为它们用的是流水线算力预算的不同部分（MLP vs attention、计算 vs 通信）。

README 的配图清楚呈现：两个带粗黑边框共享的 cell 表示"互相重叠的计算+通信"。调度构造得让每个 rank 几乎时时都有可重叠对象 —— bubble 只剩在最边缘（`PP/2−1` 个，而不是 `PP−1` 个）。

## 真正的 overlap 在哪里：`overlapped_forward_backward`

调度器本身不生产加速，只生产 **overlap 机会**。用户负责实现热路径。README：

> 实际应用中，你需要针对具体 module 实现自定义的 `overlapped_forward_backward` 方法。

这个函数是 DualPipe 在调度说"同时跑 batch `i` 的 F 和 batch `j` 的 B"时调用的。它拿到两个任务，由你决定如何在 stream 与 kernel 上排布让它们真正重叠。在 V3 生产训练里：

- 一个方向的 forward 与另一方向的 backward 重叠（不同 CUDA stream、不同显存压力）。
- 两者都与 **DeepEP 的 all-to-all 通信重叠**（DeepEP 给出 SM 预算正是为了给 overlap 留空间——见 DeepEP 走读）。
- 权重梯度 `W` 从激活梯度 `B` 里剥出来，通过 `WeightGradStore` 延迟到更空闲的槽位。

库提供原语但**不规定 overlap 策略**——你按自己模型的实际 F/B/W 形状写。

## 通信层：`comm.py` 与 shape 预注册

流水线并行在 stage 边界 P2P 收发。`comm.py` 提供原语：

```python
set_p2p_tensor_shapes([list of shapes])
set_p2p_tensor_dtype(dtype)
```

**调度跑起来之前**把 shape 和 dtype 注册好。这样 P2P buffer 可以预分配、跨所有 micro-batch 复用，避免每步分配——在 V3 规模下每 step 有几万次 P2P 操作，这点很重要。

副作用：激活 shape 必须**静态可知**。流水线调度中间不能变序列长度，换 shape 需要重新注册。

## `WeightGradStore` —— 把 B 和 W 分开

bubble 公式里有 `W` 项：把反向拆成两阶段。

- **B（激活反向）**：算 `dX`，让上一 stage 能继续它的反向。
- **W（权重反向）**：算 `dW`，只需本 rank 的状态，可以在 optimizer step 前任意时刻跑。

零气泡流水线（ZB1P）就是靠这个把 `W` 塞进 bubble。DualPipe 沿用同样思路—— bubble 公式里的 `−3W` 代表滑进空闲槽位的 `W` 块。`WeightGradStore` 是暂存累积 `dW` 片段、在调度器说"该 flush 了"时释放的脚手架。

## 哪些能搬走

- **调度本身**是算法，与 V3 模型无关。任何 PP 训练只要 F/B 能有意义地重叠、且能承担 2× 参数代价就能用。Megatron-Core、DeepSpeed 已陆续采纳类 DualPipe 调度。
- **`WeightGradStore` 分离 B 与 W** 不只服务 DualPipe——ZB1P、interleaved 1F1B 以及任何未来做细粒度 bubble 填充的调度都需要。
- **P2P shape 预注册的规范**对任何流水线代码都是可移植优化。
- **DualPipeV** 特别值得看——它把设备数代价降到 PP/2，对 PP 成本敏感的团队尤其有价值。

## 哪些不能

- **每设备 2× 参数**在 vanilla DualPipe 下不可谈判。显存紧就用不了——这也是 Llama 3 405B 坚持 1F1B 的原因。
- **必须自己写 `overlapped_forward_backward`**。没有开箱即用默认实现，库只是脚手架。
- **需要静态 shape**。调度中途不能变 batch / 序列长度。
- **PyTorch only 的参考实现**。移到 JAX 或 Megatron-native 需要对那些栈的流水线原语重写。
- **PP 必须为偶数**。调度是对称的，奇数 PP 不接受。

## 个人评注

DualPipe 是这批走读里最干净的"论文即代码"案例——**insight 是一个调度，不是一个 kernel**。没有手调 CUDA、没有 PTX 花招，只是一个观察：两方向对跑 + 把 B 与 W 分开，bubble 大约砍一半。代价是 2× 参数，**只有因为 MLA 与 FP8 已经替 V3 留出巨量显存空间才能承担**。又是 DeepSeek 典型的 pattern：**infra 选择是复利的**——MLA 促成 FP8，FP8 促成 DualPipe，砍掉任何一环链条就断。对实践者真正的启示：**`overlapped_forward_backward` 方法才是活儿真正在的地方**，不是调度器。DualPipe 库只是 ~300 行调度逻辑；真正的 V3 训练用几百行精细调过 stream 的 kernel 去兑现每一个 overlap 机会。只抄调度不做这部分工作，只能拿到好看的图和 1.05× 的加速。

## 参考

- DualPipe 仓库：https://github.com/deepseek-ai/DualPipe
- DeepSeek-V3 技术报告 §3.2.1：arXiv:2412.19437
- Profile data（计算-通信重叠轨迹）：https://github.com/deepseek-ai/profile-data
- Zero Bubble Pipeline (ZB1P)：Qi et al.，arXiv:2401.10241
- Interleaved 1F1B：Narayanan et al. (Megatron-LM v2)，arXiv:2104.04473
- DualPipeV / "Cut-in-half"（Sea AI Lab 博客）：https://hackmd.io/@ufotalent/r1lVXsa9Jg
