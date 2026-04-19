# MoE 路由改进——Expert Choice 与无辅助损失负载均衡

- **关键论文**：
  - **Expert Choice** — Zhou 等，Google。NeurIPS 2022。
  - **DeepSeekMoE** — Dai 等，深度求索。2024-01。
  - **DeepSeek-V3** — 深度求索。2024-12。
- **链接**：[Expert Choice (arXiv:2202.09368)](https://arxiv.org/abs/2202.09368) · [DeepSeekMoE (arXiv:2401.06066)](https://arxiv.org/abs/2401.06066) · [DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)

## 一句话总结

MoE 路由的核心矛盾在于：要把语义相似的 token 送往同一专家（保证质量），同时让所有专家的负载大致均等（保证硬件效率）。以 Token Choice 为代表的主流方案——每个 token 通过 softmax 路由器自行选择 top-k 专家——贯穿了 Switch、GShard、Mixtral 等设计，但它依赖辅助负载均衡损失，不仅向梯度信号中注入噪声，还必须设置专家容量上限并丢弃溢出 token。两种改进方案相继成熟：**Expert Choice**（Zhou 等，2022）反转分配方向，由每个专家自行选择 token，负载完美均衡但与自回归解码不兼容；**无辅助损失负载均衡**（DeepSeek-V3）保留 Token Choice，以每专家偏置项替代辅助损失，消除梯度噪声的同时保留 token 完整路由。DeepSeekMoE 提出的细粒度加共享专家分解可叠加于任一路由策略之上，进一步提升专家专业化程度。路由方案的选择直接决定专家并行的调度内核、all-to-all 通信模式以及推理调度图的复杂度。

## 背景：路由问题的本质

MoE 层用 `E` 个专家 FFN 替换稠密 FFN，每个 token 只激活其中 `k` 个。路由器是一个可学习的线性投影加 softmax，负责分配路由权重。收益是真实存在的——在相同激活参数量下，总参数量可按 E/k 倍扩展，以相同计算量换取更大模型容量。然而路由器本身带来了一个基础设施问题：**专家负载不均衡**。

若路由器退化——把所有 token 都送往同两个专家——大部分专家闲置，有效容量大幅缩水，专家并行通信也会失衡：部分 GPU 承担全部工作，其余 GPU 空转。训练时可通过辅助损失恢复，但推理时不均衡直接转化为延迟：一步的时间由最慢的专家决定，调度失衡意味着专家并行流水线中出现大量空泡。

### Token Choice 路由（Switch / Mixtral 基线）

标准 Token Choice：token `t` 产生 logit 向量 `g_t = W_router · x_t ∈ R^E`，按 softmax 分数选择 top-k 专家。每个专家设有硬性容量上限 `C`：

```
C = floor((T / E) × capacity_factor)    # 每批次每专家可接收的最大 token 数
```

其中 `T` 是批次 token 总数，`capacity_factor` 是可调标量（通常 1.0–1.25）。被路由到已满专家的 token 会被**丢弃**——在该层完全跳过 FFN。丢弃损害质量；提高 `capacity_factor` 可减少丢弃，但会在空专家槽上浪费算力。

为抑制路由崩塌，训练目标中加入辅助均衡损失：

```
L_aux = α × E × Σ_i  f_i × p_i

其中：
  f_i = 批次中路由到专家 i 的 token 比例
  p_i = 路由器分配给专家 i 的平均概率
  α   = 损失权重（通常 0.01–0.1）
```

该损失有效，但制造了一个持续性矛盾：`α` 必须足够大以防止崩塌，又必须足够小以不主导语言模型损失。实践中，它在每一步向梯度中注入噪声，导致训练损失与验证损失的差距在大规模扩展时趋于扩大。

## Expert Choice 路由（Zhou 等，2022）

Expert Choice 反转了选择方向：不再由每个 token 选择专家，而是**由每个专家选择 token**。

给定批次大小 `T`、专家数 `E` 和容量乘数 `k`，专家 `i` 独立地从 logit 中选取 top-`B` 个 token，预算 `B` 固定不变：

```
B = floor(k × T / E)       # 分配给专家 i 的 token 数

对每个专家 i：
  scores_i = softmax(x · W_router^T)[:, i]   # 所有 T 个 token 对专家 i 的得分
  selected_i = argsort(scores_i, 降序)[:B]
  token_expert_matrix[i] = selected_i
```

负载均衡精确成立：每个专家恰好处理 `B` 个 token，无需辅助损失，且无 token 在专家层面被丢弃。每层的总 FLOPs 严格等于 `k × T × FFN_cost(d)`，与批次组成无关。

### 可变覆盖问题

这种反转带来了一个根本性后果：**单个 token 可能被 0、1、2 个乃至更多专家选中**。在 Token Choice top-2 下，每个 token 保证在每层恰好被 2 个专家处理。在 Expert Choice 下，覆盖数可变——热门 token 被多个专家处理，冷门 token 可能被完全跳过。期望值等于 `k`，但方差不可忽视。

这对**自回归（因果）解码**造成了根本性障碍。推理时，需要在运行各专家之前就知道哪些专家处理当前查询 token；Token Choice 下 token 自己决定，调度列表可在 O(1) 内从查询直接计算。Expert Choice 下，则需要先对整个序列上下文运行所有 E 个专家的选择过程，再进行调度——这是顺序执行且代价高昂的。实践中，Expert Choice **几乎专用于非自回归场景**：仅编码器模型（BERT 类）、分类任务以及能提前获得完整输入批次的全序列推理。在自回归 LLM 解码中鲜有采用。

### 专家并行调度差异

专家并行（EP）将 `E` 个专家分布到 `P` 张 GPU 上。路由需要执行 all-to-all：设备 `d` 上被分配到设备 `d'` 上专家 `e` 的 token 必须通过互连传送。Token Choice 下，调度内核为每个 token 构建专家列表，并将 token 按目标专家顺序打包进连续缓冲区。Expert Choice 下，内核构建的是每专家的 token 列表；token 可能以非原始序列顺序到达，必须重新排序以进行 gather 操作。这需要不同的重排/scatter 内核，也使计算与通信的流水线重叠更难实现。

## 无辅助损失负载均衡（DeepSeek-V3）

DeepSeek-V3 保留了 Token Choice（兼容自回归生成），但彻底去除辅助负载均衡损失，代之以**每专家偏置修正**机制。

每个专家 `i` 维护一个标量偏置 `b_i`（初始化为 0）。路由时，专家选择基于**带偏置的 logit** 进行：

```
g_t^biased = g_t + b            # b ∈ R^E，在 top-k 选择前加入
top-k 选择基于 g_t^biased       # 决定哪些专家接收该 token
路由权重计算基于 g_t（原始，无偏置）用于加权求和
```

每次前向传播（训练步或 micro-batch）后，偏置按简单符号规则更新：

```
load_i  = 当前步中路由到专家 i 的 token 比例
b_i ← b_i - γ × sign(load_i - target_load)

其中：
  target_load = k / E   （均匀目标）
  γ            = 偏置更新步长（小常数，如 0.001）
```

该更新**不是梯度步**——它是在自动微分图之外进行的基于符号的修正，仅作用于离散路由决策。过载专家的选择偏置被降低（后续步骤中被选中的概率降低）；负载不足的专家偏置被提升。系统收敛到近似均衡，训练损失中不含任何均衡相关项。

### 为何能提升训练稳定性

辅助损失 `L_aux` 是加入语言模型损失中的任务级信号，其梯度在每步流过路由器权重，推动权重趋向均衡，即使语言模型梯度希望它们进一步专业化。在大规模情况下（DeepSeek-V3 总参数量约 6710 亿，激活参数约 370 亿），这种矛盾是可测量的：使用辅助损失均衡的训练损失曲线方差更大，最终困惑度在均衡质量相当的条件下略高于无辅助损失均衡。DeepSeek-V3 报告显示，去除 `L_aux` 可获得更平滑的损失曲线，最终检查点质量有可量化的提升。

偏置更新规则也比各类替代方案（基于 EMA 的负载估计、二阶修正等）更简单：符号函数使其对异常批次具有鲁棒性，且 `γ` 对规模不敏感。

## 细粒度加共享专家分解（DeepSeekMoE）

DeepSeekMoE 对专家池本身引入了结构性改变，与路由算法的选择正交。

标准 MoE 层有 `E` 个路由专家，选择 top-`k`。DeepSeekMoE 将其拆分为：

- `Ks` 个**共享专家**：每个 token 均激活，处理通用的、广泛适用的知识。
- `N` 个**细粒度路由专家**：每个专家的隐藏维度缩小（比标准专家小）；路由器为每个 token 选择 top-`Kt` 个。

总激活参数量为：

```
active_params = Ks × d_model × d_expert_ffn          # 共享专家，始终激活
              + Kt × d_model × (d_expert_ffn / m)    # 路由专家，每个为标准专家的 1/m
```

其中 `m` 是细粒度系数（如 `m=4` → 每个路由专家大小为标准专家的 1/4，因此在相同激活 FLOPs 下可拥有 4 倍数量的专家）。

**知识冗余消减**：在标准 MoE 中，路由专家往往复制通用知识（如句法结构），因为每个专家都必须应对它接收到的任意 token。由共享专家承担通用知识后，路由专家可以更窄范围地专业化，减少冗余，提升路由容量的有效利用率。DeepSeekMoE-16B 仅用 28 亿激活参数即可达到 LLaMA-7B 相当的性能。

**显存影响**：共享专家始终需要驻留 HBM（因为始终被激活）。路由专家是专家卸载的候选对象——若 `E` 足够大，每步只需将 top-`Kt` 个专家的权重保留在设备上。共享专家阻断了这条卸载路径，必须在规划显存预算时单独考虑其开销。

## 端到端显存与延迟核算

以具体数字为参考：671B MoE 模型（DeepSeek-V3 规模），每层 256 个专家，top-8 路由，BF16 权重：

- **每 GPU 路由专家权重**（EP=64）：每张 GPU 托管约 4 个专家 × `d_model × d_ffn × 2` 字节。
- **每层 all-to-all 通信量**：每个 token 将其隐藏状态（BF16，`d_model=7168` → 14 KB）发送到 8 个专家，分布在 EP 组中；序列长度 4096、批次大小 1 时，每个 EP 对等节点每层每方向的调度量约为 4096 × 8 × 14 KB / 64 ≈ 7 MB。
- **不均衡开销**：20% 的负载不均衡（某专家接收平均 token 量的 1.2 倍）会使该 GPU 在下一次 all-to-all 屏障处等待，专家计算时间中有 17% 被浪费。

无辅助损失均衡在稳态下将不均衡控制在约 2–3%（按 DeepSeek-V3 消融实验），而使用较大 `α` 的辅助损失均衡通常在 5–8% 左右。

## 工程 Tradeoff

| 方案 | 负载均衡 | 训练稳定性 | 推理兼容性 | 辅助损失 | 硬件效率 |
|---|---|---|---|---|---|
| Token Choice + 辅助损失（Switch、Mixtral） | 近似均衡；由 `α` 和容量系数控制 | 中等；辅助损失引入梯度噪声 | 完整；每 token 恰好激活 k 个专家 | 有；`L_aux` 在训练目标中 | 中等；token 丢弃浪费容量 |
| Expert Choice（Zhou 2022） | 精确均衡；无溢出，无丢弃 | 高；无辅助损失 | 不兼容自回归；仅适用编码器/批量场景 | 无 | 训练时高；AR 解码时差 |
| 无辅助损失均衡（DeepSeek-V3） | 近似均衡；通过偏置收敛至近均匀 | 高；梯度信号更干净，损失曲线更平滑 | 完整；保留标准 top-k Token Choice | 无；偏置更新在自动微分图外 | 高；合理 γ 下无 token 丢弃 |
| 细粒度 + 共享专家（DeepSeekMoE） | 取决于所用路由算法 | 取决于所用路由算法 | 完整 | 取决于所用路由算法 | 高；减少专家冗余，提升利用率 |

## 实际部署要点

**路由方案选择**：任何需要自回归生成的模型，除非增加推理时近似（如回退到 Token Choice 进行解码），否则 Expert Choice 方案不可用。无辅助损失均衡是当前训练与推理兼顾场景下的最优方案。

**容量系数调优**：即使采用无辅助损失均衡，`capacity_factor` 的设置在推理时面对对抗性输入分布时仍然重要。计算预算匹配时设为 1.0；若预期分布偏移，可提升至 1.1–1.25。

**EP all-to-all 开销**：在节点内 NVLink 上，BF16 下 EP all-to-all 足够快，通常不是主要瓶颈。跨节点（InfiniBand）时则成为真实的延迟来源。Expert Choice 与 Token Choice 传输的字节总量相同；差异在于内核形态以及是否需要重排序。

**偏置初始化与预热**：DeepSeek-V3 将所有偏置初始化为 0，发现无需预热；符号更新从随机初始化出发在数百步内收敛到合理的均衡状态。

## 交叉参考

- `../switch-gshard/` — 使用辅助损失的 Token Choice 基础路由；前置阅读
- `../../deepseek/moe/` — DeepSeekMoE 细粒度与共享专家模式详解
- `../../deepseek/v3-tech-report/` — DeepSeek-V3 在 6710 亿参数规模下使用的无辅助损失均衡
- `../../mistral/mixtral/` — top-2 Token Choice 加辅助损失；开源 MoE 基线

## 参考文献

- [1] Zhou et al. _Mixture-of-Experts with Expert Choice Routing._ NeurIPS 2022 / arXiv:2202.09368.
- [2] Lepikhin et al. _GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding._ ICLR 2021 / arXiv:2006.16668.
- [3] Fedus et al. _Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity._ JMLR 2022 / arXiv:2101.03961.
- [4] Dai et al. _DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models._ arXiv:2401.06066, 2024.
- [5] DeepSeek-AI. _DeepSeek-V3 Technical Report._ arXiv:2412.19437, 2024.
- [6] Jiang et al. _Mixtral of Experts._ arXiv:2401.04088, 2024.
