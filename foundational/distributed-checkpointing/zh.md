# 异步检查点与 PyTorch DCP

- **相关系统**：PyTorch Distributed Checkpoint（`torch.distributed.checkpoint`）、DeepSpeed ZeRO 检查点、Megatron-Core 异步检查点、MegaScale
- **PyTorch DCP 稳定版本**：PyTorch 2.1（2023 年）
- **链接**：[PyTorch DCP 文档](https://pytorch.org/docs/stable/distributed.checkpoint.html) · [DCP 设计 RFC](https://github.com/pytorch/pytorch/issues/88083) · [DeepSpeed zero_to_fp32](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/zero_to_fp32.py)

## 一句话总结

在前沿规模下，检查点不是后台清理任务，而是每次对整个任务中所有 GPU 征收的周期性空闲税。一个 70B BF16 模型的检查点数据约为 420 GB（权重 + Adam 状态）；405B 模型约为 2.4 TB。以 10 GB/s 的聚合带宽同步写入对象存储，分别需要 42 秒和 240 秒的 GPU 死时间。异步内存检查点通过在后台线程将数据暂存到 CPU DRAM 的同时继续训练，彻底消除 GPU 停顿。PyTorch 的 `torch.distributed.checkpoint`（DCP）标准化了分片保存/加载 API，处理拓扑变化时的重新分片，并提供可插拔的存储后端。DeepSpeed ZeRO、Megatron-Core 和 MegaScale 均采用相同的异步模式，主要区别在于分片粒度和 CPU 内存管理方式。

## 为什么检查点是瓶颈

Adam 训练下 BF16/FP32 混合精度模型的检查点大小明细如下：

```
模型权重（BF16）：        2 字节 × P 参数
梯度（BF16）：            2 字节 × P    [瞬态，不检查点]
Adam m（FP32 主拷贝）：   4 字节 × P
Adam v（FP32 主拷贝）：   4 字节 × P
FP32 主权重：             4 字节 × P
────────────────────────────────────────────
检查点总量：              ≈ 10 字节 × P
  → 70B 模型：   10 × 70×10⁹  ≈ 700 GB（仅 BF16 权重为 140 GB）
  → 405B 模型：  10 × 405×10⁹ ≈ 4.0 TB
```

实际上，优化器状态是占用最大的项。对于使用 Adam 的 70B 模型，仅优化器状态（m + v + 主权重）便约为 560 GB。许多框架仅对 BF16 权重 + 优化器状态做检查点，最终约为 420 GB。405B 模型的总量约为 2.4 TB。

再来看空闲时间的算术。若对象存储的聚合写入带宽为 10 GB/s（并行写入负载下大型 NFS 或分布式对象存储的合理数字），同步检查点的代价为：

```
70B：   420 GB / 10 GB/s  = 42 秒/次检查点
405B：2400 GB / 10 GB/s  = 240 秒/次检查点
```

以每秒 1 次迭代（大型模型的典型值）计，每 500 次迭代做一次同步写入检查点：

```
空闲比例 = 42 s / 500 s = 8.4%   （70B）
空闲比例 = 240 s / 500 s = 48%   （405B——显然不可接受）
```

这就是为什么在超过约 100B 参数的规模下，异步检查点不是可选项。

## 同步与异步检查点

**同步**检查点是朴素基线：训练停止，所有 rank 写入各自的分片，训练恢复。整个写入过程中 GPU 处于空闲状态。逻辑正确、实现简单，但在数百亿参数以上无法承受。

**异步内存**检查点消除了 GPU 停顿：

1. 在检查点步骤 N，每个 rank 将其参数张量和优化器状态张量从 GPU HBM 复制到 CPU DRAM。此复制大约耗时 1–3 秒（H100 的 NVLink 主机带宽约为 900 GB/s；PCIe Gen5 约为 64 GB/s——对大多数系统而言，PCIe 是实际瓶颈）。
2. GPU 立即在步骤 N+1 恢复训练。
3. 一个后台线程将 CPU DRAM 中的张量序列化并写入对象存储。此写入与训练步骤 N+1、N+2、... 并行进行，并在下一个检查点步骤 N+K 之前完成。

GPU 停顿被替换为 1–3 秒的 CPU DRAM 复制延迟。这个代价只需支付一次；后台写入从 GPU 角度看是免费的。该模式被 DeepSpeed ZeRO（通过状态的内存内检查点）、Megatron-Core 的 `AsyncCheckpointingDataWriterWrapper` 以及 MegaScale（在 ByteDance 系统论文中描述为"带热备用节点的内存内检查点"）所采用。

**内存代价**：CPU DRAM 必须持有一个完整的检查点副本。对于 70B 模型，约为 420 GB 的 CPU 内存。前沿训练服务器（H100 DGX 节点）通常有 1–2 TB DRAM，因此可行。在 405B 规模下，约 2.4 TB 的需求变得紧张——一些系统只在 DRAM 中保留最新检查点，或在 CPU 端存储之前先进行压缩。

## PyTorch 分布式检查点（DCP）

`torch.distributed.checkpoint` 是 PyTorch 2.x 中用于分片、可重新分片的检查点保存和加载 API。其核心设计原则：**每个 rank 只写入自己的本地分片**。不存在 rank 0 的聚合瓶颈——正是这种朴素的 `torch.save` 方式使其在超过约 10B 参数后无法使用（rank 0 需要在内存中具体化整个模型）。

### 保存

```python
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemWriter

state = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
}
dist_cp.save(
    state_dict=state,
    storage_writer=FileSystemWriter("/checkpoint/step-1000"),
)
```

每个 rank 只写入其持有的参数分片。输出是一个目录，包含：

- `.metadata`：类 JSON 的清单文件，列出所有张量名称、形状、数据类型，以及每个分片在数据文件中所处的字节范围。
- `.distcp` 数据文件：每个 rank 每个张量组对应一个或多个文件，包含原始张量字节。文件总数按 `O(rank 数 × 张量组数)` 缩放。

### 加载与重新分片

DCP 相比每 rank 独立 `torch.save` 文件的决定性优势：`dist_cp.load` 可以将一种拓扑结构保存的检查点加载到不同拓扑结构中。以 TP=4 写入的检查点可以用 TP=8 恢复——DCP 读取 `.metadata` 清单，推算旧分片到新分片布局的映射关系，并只从存储中流式读取所需的字节范围：

```python
dist_cp.load(
    state_dict=state,
    storage_reader=FileSystemReader("/checkpoint/step-1000"),
)
```

重新分片逻辑由 DCP 的 `DefaultLoadPlanner` 处理，它从元数据计算张量块到 rank 的映射，并发出有针对性的读取请求。训练脚本中无需额外代码来处理拓扑变化。

### 存储后端

DCP 使用可插拔的存储写入器/读取器：

- `FileSystemWriter` / `FileSystemReader`：本地或 NFS 路径；使用 Python `os` 进行文件 I/O。
- `S3StorageWriter` / `S3StorageReader`：直接写入兼容 S3 的对象存储。
- `FSSpecStorageWriter`：使用 `fsspec` 支持该库所覆盖的任何文件系统（GCS、Azure Blob、HDFS）。

切换后端无需改动其他检查点代码——在从基于 NFS 的开发迁移到基于 S3 的生产环境时非常实用。

### 异步包装器

DCP 提供 `AsyncCheckpointingDataWriterWrapper`，它封装任意 `StorageWriter` 并实现异步模式：

```python
from torch.distributed.checkpoint import AsyncCheckpointingDataWriterWrapper

async_writer = AsyncCheckpointingDataWriterWrapper(
    FileSystemWriter("/checkpoint/step-1000"),
    async_snapshot_timeout=180,  # 秒
)
dist_cp.save(state_dict=state, storage_writer=async_writer)
# 立即返回；写入在后台线程中完成
```

`save` 返回后，训练立即继续。包装器在主调用中将张量暂存至 CPU，然后将文件写入委托给后台线程。`async_snapshot_timeout` 设置超时限制——若后台写入在下一个检查点之前尚未完成，框架将等待其结束再进行新快照，防止检查点重叠。

## ZeRO 分片检查点

在 ZeRO-3（或 FSDP 完全分片）下，每个 DP rank 只持有优化器状态和参数的 `1/N`。在此场景下的检查点保存意味着每个 rank 保存其自己的分片——与 DCP 的分片保存模型天然契合。恢复时，DCP 重新组装各分片。检查点一致且正确，无需任何 rank 具体化完整模型。

`zero_to_fp32.py` 工具（DeepSpeed）处理*推理*的特殊情况：给定一个 ZeRO-3 分片检查点目录，它拼接所有分片、剥离优化器状态，并写出一个适合不理解 ZeRO 分片的推理框架使用的连续 FP32 权重文件。这一转换仅需 CPU，对 70B 模型可能耗时 10–30 分钟，但这是一次性代价。

## 恢复带宽与热备用节点

故障后从检查点恢复的时间由读取带宽决定：

```
70B（420 GB）：  420 GB / 10 GB/s  = 42 秒（从对象存储）
405B（2.4 TB）：2400 GB / 10 GB/s  = 240 秒（从对象存储）
```

在一个拥有 16,000 张 H100 的集群上，四分钟的 GPU 空闲时间按当前云价格折算约耗费 5,000–10,000 美元的 GPU 算时。对于长期运行——Llama 3 在约 16,000 张 GPU 上历时 54 天的训练——最小化恢复时间至关重要。

**热备用节点**解决了这个问题：一小组待机节点通过可靠的分布式写入（如 RDMA）持续将最新检查点接收到 CPU DRAM 中。当某个训练节点故障时，热备用节点加入任务，此时已在 DRAM 中持有检查点。恢复只需数秒，而非数分钟。MegaScale 明确使用了此模式。

权衡之处在于：热备用节点持续消耗集群资源（通常为训练池的 1–5%）。对于足够大的任务和较高的故障率（H100 集群在万节点规模下，GPU 故障率可达每天约 1 次），从经济角度看热备用节点是合算的。

## 检查点频率优化

最优检查点间隔 K（以迭代次数计）平衡两个相互竞争的代价：

```
故障时期望浪费的算力  = (K/2) × (GPU 数 × 每次迭代 GPU 小时数)
检查点 I/O 开销      = 检查点大小 / (K × 迭代时间 × 写入带宽)
```

求总代价最小时：

```
K_optimal = sqrt(检查点大小 / (迭代时间 × 写入带宽 × 故障率 × GPU 成本))
```

对于 70B 模型，1 次迭代/秒，10 GB/s 写入带宽，整个集群日故障概率为 0.01（对 1000 张 GPU 而言偏保守），GPU 成本 3 美元/小时：

```
K_optimal ≈ sqrt(420×10⁹ / (1 × 10×10⁹ × 0.01 × 3/3600)) ≈ 800 次迭代
```

使用异步检查点后，I/O 项实际上消失（GPU 空闲时间接近零），因此 K 可以纯粹根据恢复代价来设定——通常为 200–1000 次迭代，取决于集群中观察到的故障率。

## 工程权衡

| 方案 | 每次检查点的 GPU 空闲时间 | CPU DRAM 需求 | 恢复时间 | 可重新分片 | 实现复杂度 |
|---|---|---|---|---|---|
| 同步全量写入 | 高：与检查点大小/写入带宽成正比（70B 约 42 秒） | 极低：张量留在 GPU 上 | 与检查点大小/读取带宽成正比 | 使用 DCP 可以 | 低 |
| 异步内存方案 | ~1–3 秒（仅 GPU 到 CPU 的复制） | 完整检查点大小需在 DRAM 中（70B 约 420 GB） | 若从存储恢复与同步相同；使用热备用节点则仅需数秒 | 使用 DCP 可以 | 中：需管理后台线程和超时 |
| ZeRO 分片（每 rank 分片） | 低：每个 rank 只写自己的分片，天然并行化 | 仅需 per-rank 分片（420 GB / N） | 需要所有分片齐全；DCP 处理重组 | 是，DCP 的核心特性 | 中：需追踪分片清单 |
| 流式检查点（训练中写入） | 接近零：与计算重叠 | 低：增量流式写出 | 与同步方案相同 | 取决于具体实现 | 高：需要可靠的有序流式传输 |

## 实践备注

在 70B 规模下使用异步 DCP：预期 GPU 停顿 1–2 秒用于 CPU 复制，30–60 秒后台写入 NFS，训练不受写入中断影响。`.metadata` 文件较小（< 1 MB）；数据文件线性增长。一个 16 rank、TP=8、PP=2 的任务每个检查点步骤将产生 16 个数据文件。

在 405B 规模下：CPU DRAM 压力是真实存在的。单个 DGX H100 节点有 2 TB DRAM；持有一份 2.4 TB 检查点要么需要在 DRAM 中压缩，要么将 CPU 端缓冲分散到多个节点。Megatron-Core 支持这两种方式。在超过约 200B 参数后，热备用节点从经济角度变得有吸引力。

## 交叉参考

- `../zero-fsdp/` — ZeRO-3 将优化器状态分片到各 DP rank；DCP 处理 ZeRO-3 检查点的分片保存/加载
- `../../deepseek/open-source-week/3fs/` — 3FS 是专为此类检查点 I/O 模式设计的存储底层：来自大量并发进程的高聚合带宽随机写入
- `../../meta/llama3/` — Llama 3 在 16,000 张 H100 上历时 54 天的训练需要健壮的检查点机制，以从 GPU 故障中快速恢复
- `../../deepseek/open-source-week/dualpipe/` — DualPipe 将异步检查点列为零 bubble 训练的协同需求
- `../torch-compile/` — `torch.compile` 和 DCP 同为 PyTorch 2.x 的生产特性，在 `torchtitan` 中共同使用

## 参考文献

- [1] Zhao et al. _PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel._ arXiv:2304.11277, 2023.
- [2] Rajbhandari et al. _ZeRO: Memory Optimizations Toward Training Trillion Parameter Models._ SC '20 / arXiv:1910.02054.
- [3] Jiang et al. _MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs._ arXiv:2402.15627, 2024.（第 4 节：容错与内存内检查点。）
- [4] Dubey et al. _The Llama 3 Herd of Models._ arXiv:2407.21783, 2024.（第 3.3.1 节：训练可靠性与检查点。）
- [5] PyTorch Distributed Checkpoint 文档：`pytorch.org/docs/stable/distributed.checkpoint.html`
- [6] DeepSpeed ZeRO 检查点工具：`github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/zero_to_fp32.py`
