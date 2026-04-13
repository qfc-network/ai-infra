# GPU 互联 —— NVLink、NVSwitch、RDMA、IBGDA

_参考 primer，不是论文走读。互联层次是本仓库几乎每篇（MoE all-to-all、ring attention、PD 解耦、Pathways、3FS、DeepEP）都默认假设的前提。这篇把各个零件点名。_

## 一句话总结

现代 LLM 训练与 serving 跑在**三层互联层次**上：

1. **节点内**：NVLink（以及聚合它的 NVSwitch）—— 同一机器里的 GPU 互连，百 GB/s 到低 TB/s 级。
2. **pod 内 / 机架内**：NVLink Switch System（NVL72）—— 把 NVLink 扩到一个机架，同级带宽。
3. **跨节点**：InfiniBand 或 RoCE Ethernet —— 跨机架的 GPU 通信，单 NIC 100–400 Gb/s，走 GPUDirect RDMA。

除了 fabric，一套软件栈（CUDA、NCCL、NVSHMEM、IBGDA）决定用户代码真正能用到多少物理能力。"fabric 上有 400 GB/s"与"应用拿到 400 GB/s"的差距常常是 10×，取决于你用哪层抽象。

这篇 primer 把各个零件列齐，并标出本仓库里每篇 infra 论文隐式做的选择。

## 节点内：NVLink 与 NVSwitch

### NVLink 世代（每 GPU 聚合带宽大致）

| 代 | GPU | 年 | 链路数 × 单链 GB/s | 单 GPU 聚合 |
|-----|-----|------|-------------------|---------------------|
| 1 | P100 | 2016 | 4 × 20 GB/s | 80 GB/s |
| 2 | V100 | 2017 | 6 × 25 GB/s | 150 GB/s |
| 3 | A100 | 2020 | 12 × 25 GB/s | 300 GB/s（双向 600） |
| 4 | H100 | 2022 | 18 × 25 GB/s | 450 GB/s（双向 900） |
| 4 | H800 | 2022 | 受限子集 | ~200 GB/s（双向 400） |
| 5 | Blackwell | 2024 | 18 × 50 GB/s | 900 GB/s（双向 1.8 TB/s） |

"双向" = bidirectional。H800 相对 H100 NVLink 约砍一半，这一具体硬件约束催生了 DeepSeek 的 DualPipe、DeepEP 的非对称转发、FlashMLA 的 seesaw 调度。同代芯片、不同 fabric，软件形态根本不同。

### NVSwitch

单纯 NVLink 是点到点：每个 GPU N 条链路、每条接一个对端。8 卡一节点时要么 (a) 环形拓扑（带宽按步降），要么 (b) 完全互联（端口不够）。NVSwitch 解法：一个 crossbar，让**每个 GPU 同时以完整 NVLink 带宽与每个其他 GPU 对话**。

H100 起：8 卡 DGX 4 颗 NVSwitch，任意流量模式都是双向满带宽。这就是"8 卡 TP 的通信可视为免费"的底层原因。

### NVL72（Blackwell）

把 NVSwitch 扩到**一个机架 72 GPU 以 NVLink 带宽互连**。创造一个"超级节点"——原本需要 InfiniBand（跨节点）的事如今跑在 NVLink 速率上。对训练：TP 与模型切分可扩到 72 而不只 8。对 serving：PD 解耦可跨更大池但仍以内部总线速度通信。

这也是 **2025 年起论文开始以"super-node"为单位而非"node"**的原因。

## 跨节点：RDMA

### InfiniBand（IB）与 RoCE

不同节点的 GPU 经网卡通信。两种主导 fabric：

- **InfiniBand** —— HPC 血统，专为 RDMA 设计，延迟最低，成本高，厂商（Mellanox / NVIDIA）锁定。CX7 400 Gb/s 是当前前沿集群标配。
- **RoCE（RDMA over Converged Ethernet）** —— 在标准以太网上实现 RDMA 语义。延迟略高、拥塞控制略弱，商品交换机。Meta Llama 3 16k GPU 集群用它；规模下有成本优势。

两者都给出 **RDMA（Remote Direct Memory Access）** —— 网卡直接读写对端内存，对端 CPU 不参与。微秒级通信的基础。

### GPUDirect RDMA

没有 GPUDirect：GPU → host CPU → NIC → 线路 → NIC → host CPU → GPU。CPU 跳每方向加 ~20 µs。

GPUDirect RDMA：NIC 通过 PCIe 直接读写 GPU 显存。**host CPU 不在路径上**。延迟降到 fabric 自身（IB 上 ~2–5 µs）。这是现代所有跨节点 GPU 通信的地基。

要求：PCIe 拓扑让 NIC 能看到 GPU 显存（常意味着 NIC 与 GPU 在同一 PCIe root complex / NUMA 节点）。

### IBGDA —— InfiniBand GPUDirect Async

GPUDirect RDMA 下 CPU 仍然*发起*传输（内核驱动给 NIC 投 work request）。对 kernel 内部的细粒度通信——MoE token 分发到专家、prefill 的 token 送给 decode——CPU 往返投 WR 本身就是瓶颈。

**IBGDA** 让 **GPU 自己往 NIC 投 work request**，通过内存映射的 queue pair。CPU 彻底退出通信回路。

DeepEP 的低延迟 kernel 就用 IBGDA。任何每 token 延迟关键的现代 MoE 推理栈都会用。fabric 需支持（CX6+ NIC，带正确固件的 IB 或 RoCE）。

## 软件抽象

fabric 只是一半，编程模型是另一半。

### NCCL

NVIDIA Collective Communications Library。PyTorch / Megatron / JAX 里 collective（`all-reduce`、`all-gather`、`reduce-scatter` 等）的默认。按 NVLink + NVSwitch + IB 拓扑调优。自动检测拓扑、按 collective + shape 选算法。

"PyTorch DDP 背后走 NCCL"说的就是它。NCCL 抹掉"这是 NVLink"与"这是 InfiniBand"的区别——一次 `all_reduce()` 调用自动映射到正确原语。

局限：NCCL 面向 collective。点对点模式如 MoE all-to-all 可表达但不总最优。基于 NVSHMEM 或 IBGDA 的自定义 kernel 在特定形状上可胜过 NCCL。

### NVSHMEM

PGAS 风格编程模型：把集群所有 GPU 当作逻辑上统一的内存，用 `get`/`put`/原子操作从任意 GPU 读写任意地址。底层 NVSHMEM 用 NVLink（节点内）+ GPUDirect RDMA 或 IBGDA（跨节点）。

何时选 NVSHMEM 而非 NCCL：

- **不规则或细粒度通信** —— MoE 分发、图神经网络、不规则稀疏注意力。
- **自定义 kernel** 中需要在 kernel 内部发通信。
- **非对称模式** —— 各 rank 数据量不均的 MoE all-to-all。

DeepEP 建立在 NVSHMEM 之上。很多自定义 MoE 与图 kernel 也是。

### PyTorch 分布式原语

NCCL 之上，PyTorch 暴露 `dist.all_reduce`、`dist.send`、`dist.recv`、`dist.all_to_all` 等。FSDP 和 DDP 都用这些。用法以 collective 为中心。

更新：`DTensor` 给 PyTorch 一套分片抽象（类似 JAX 的 shard_map），编译器能自动路由到合适的 collective。这是 PyTorch 向 GSPMD 收敛的路径。

## 拓扑：隐藏变量

任何严肃的训练或 serving 部署都有**拓扑图**：哪些 GPU 在同一 NVLink 岛、哪个 NUMA 持有哪块 NIC、哪些机架共享同一个 leaf 交换机。拓扑感知的 placement 与朴素 placement 在通信重 workload 上的性能差常常是 2–5×。

本仓库其他条目里的例子：

- **DeepEP 的转发模式** —— 把转发 rank 放在持有 RDMA NIC 的那个，其他人 NVLink 扇入。
- **V3 每 token ≤4 节点路由** —— 有界的 RDMA 出度，围绕 fabric 的对半带宽设计。
- **3FS 的链表** —— 链均衡使故障把负载分摊到*多数*对端，而不是直接 NVLink 邻居。
- **Ring Attention** —— 环序列选择与物理 NVLink/IB 拓扑匹配，每跳便宜。

## 论文里如何体现

本仓库其他条目的简易对照：

| 论文 / 系统 | 利用的互联 |
|---|---|
| Megatron TP | NVLink + NVSwitch（节点内） |
| Megatron PP | 跨节点 RDMA，小消息，容忍延迟 |
| FSDP / ZeRO | NCCL `all-gather` + `reduce-scatter`，NVLink + IB |
| Ring Attention / CP | 主要走 NVLink 环，IB 跨节点 |
| DeepEP normal | 非对称 NVLink（扇出）+ RDMA（跨节点） |
| DeepEP low-latency | 纯 IBGDA（后期再加节点内 NVLink） |
| Mooncake / DistServe | prefill 与 decode 池之间 RDMA |
| 3FS | 全栈 RDMA、局部性无关 |
| Pathways | 每 pod 资源管理器，抽象 fabric 细节 |

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 节点内 NVLink + NVSwitch | TB/s 级平带宽，8+ GPU 像一体 | 锁定 NVIDIA 拓扑；无出路 |
| InfiniBand 相对 RoCE | 最低延迟、成熟 RDMA | 成本、厂商锁定 |
| RoCE 相对 IB | 商品、大规模更便宜 | 拥塞控制要调、延迟略高 |
| GPUDirect RDMA | CPU 退出路径 | PCIe 拓扑约束 |
| IBGDA | GPU 发起 RDMA，CPU 不介入 | 固件/驱动要求；非普遍 |
| NCCL（高层） | 可移植、自动调优 | 只适合 collective 形状 |
| NVSHMEM | 细粒度、不规则模式 | 更难推理；调试是专家活 |
| 拓扑感知 placement | 通信重 workload 2–5× 收益 | 部署复杂度、每集群工程 |

## 个人评注

读一篇 LLM infra 论文却不在脑子里有个它所在 fabric 的粗略模型，就像读分布式系统论文却不思考延迟。具体选择——"为什么 V3 每 token 最多 4 节点？""为什么 DeepEP 有三个独立 kernel？""为什么 Llama 3 能用 RoCE 而 DeepSeek 要 IB？"——都可追到**互联形状 + 带宽 + 延迟 + 软件抽象**，而不是算法偏好。

向前看值得关注两件事：

1. **NVL72 及以后**。Scale-up（更大超级节点）正在接替 scale-out（更多更小节点）成为前沿 workload 的主流。72 GPU NVLink 域改变了可能性——TP 可扩到 72，MoE EP 可保持域内，PD 解耦可以以 NVLink 速度跨池。2025 年起的论文会默认这件事；2022–2024 的论文默认 8 卡节点。

2. **网络内计算与拥塞控制**。模型变大后，聚合对半带宽是瓶颈。未来工作会把 reduction 推进交换机（SHARP）、将拥塞控制与 collective 算法协同设计、把网络当作一等加速目标。

2026 年做 infra：**了解你的 fabric**。多数"这次训练为什么慢"的谜题在这一层解开。多数"这篇论文在我们这复现不了"的裂痕也在这里。

## 参考

- NVIDIA NVLink 技术概述：https://www.nvidia.com/en-us/data-center/nvlink/
- NVSwitch 架构：NVIDIA 白皮书（按代）
- GPUDirect RDMA：https://docs.nvidia.com/cuda/gpudirect-rdma/
- IBGDA：https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/
- NVSHMEM：https://developer.nvidia.com/nvshmem
- NCCL：https://github.com/NVIDIA/nccl
- Meta RoCE Llama 3 集群：https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/
