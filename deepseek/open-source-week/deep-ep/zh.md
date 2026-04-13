# DeepEP —— 源码走读

- **仓库**：[deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- **首次发布**：2025-02（Open Source Week 第二天）
- **配套笔记**：[`deepseek/v3-tech-report/`](../../v3-tech-report/) —— V3 论文的 all-to-all 与 group-limited gating
- **配套笔记**：[`deepseek/moe/`](../../moe/) —— DeepSeekMoE 架构

## 这个仓库是什么

GPU 端的**专家并行 all-to-all** 库——每次 MoE forward 的 "dispatch" 与 "combine" 两半。三套 kernel 针对三种场景，支持 FP8，带 hook 式 overlap：

| Kernel | Fabric | 目标场景 |
|---|---|---|
| **Intranode** | 仅 NVLink | 单节点 EP |
| **Internode（normal）** | NVLink + RDMA，非对称转发 | 训练 / prefill，吞吐优先 |
| **Internode low-latency (LL)** | 纯 RDMA（2025-06 起再加 NVLink），经 IBGDA | Decode，延迟敏感 |

H800 + CX7 400 Gb/s IB 官方数字（V3 预训练配置，4096 tokens/batch、7168 hidden、top-8 experts、FP8 dispatch + BF16 combine）：

- Intranode dispatch/combine：**~153–158 GB/s**（接近 NVLink 峰值）。
- Internode EP=32 dispatch/combine：**~58/57 GB/s**（RDMA 瓶颈；NIC 峰值 50 GB/s，靠内部 NVLink overlap 超过 NIC 单向上限）。
- EP=8 低延迟：**77 µs dispatch / 114 µs combine**（每 batch 128 tokens）。

## 为什么要专门写一个库

256 专家 + top-8 + 跨节点 EP 的 MoE，首先是通信问题、其次才是算力问题。NCCL 现成的 `all_to_all` 与 V3 论文报告的效率差得远。DeepEP 被两件 V3 特有的事情塑形：

1. **Group-limited gating**——每个 token 最多分发到 ≤4 个节点内的专家，不是整个 EP 组均匀散布。这意味着节点内（NVLink）与节点间（RDMA）本质不对称：token 先在 NVLink 上 fan out，只有跨节点那部分才走 RDMA。
2. **Decode 与训练目标相反**——训练要吞吐（愿意吃延迟换字节/秒），decode 要延迟（愿意吃利用率换微秒）。

所以 DeepEP **出了两个 normal kernel 加一个 LL kernel**，而非一个通用 `all_to_all`。

## 目录结构

```
DeepEP/
├── deep_ep/
│   ├── __init__.py
│   ├── buffer.py                     # 用户侧 Buffer API
│   └── utils.py
├── csrc/
│   ├── deep_ep.cpp / .hpp            # PyTorch binding
│   ├── config.hpp                    # 调优配置描述
│   ├── event.hpp                     # EventOverlap
│   └── kernels/
│       ├── intranode.cu              # NVLink dispatch/combine
│       ├── internode.cu              # NVLink+RDMA normal kernel
│       ├── internode_ll.cu           # 纯 RDMA 低延迟 kernel
│       ├── layout.cu                 # get_dispatch_layout（谁去哪）
│       ├── runtime.cu                # buffer / queue 初始化、handle 生命周期
│       ├── ibgda_device.cuh          # IBGDA 设备侧 RDMA 原语
│       ├── launch.cuh、buffer.cuh    # launch 机制、device buffer
│       ├── configs.cuh、utils.cuh    # 调优常量、warp helper
│       └── api.cuh
├── third-party/                      # NVSHMEM（必需）
├── tests/                            # test_intranode / test_internode / test_low_latency
└── figures/
```

三个 `.cu` 文件对应三种场景——刻意不合并到单个 kernel 后面。每个约几千行 CUDA，为各自的 fabric 独立调优。

## 用户接口（`deep_ep/buffer.py`）

围绕一个类：

```python
from deep_ep import Buffer, EventOverlap

Buffer.set_num_sms(24)                       # 全局 SM 预算
buf = Buffer(group, num_nvl_bytes, num_rdma_bytes)
```

四个主操作：

```python
num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, \
    is_token_in_rank, event = buf.get_dispatch_layout(topk_idx, num_experts, ...)

recv_x, recv_topk_idx, recv_topk_weights, nrecv_per_expert, handle, event = \
    buf.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, ...)

# ... 在 recv_x 上跑专家 ...

out_x, out_topk_weights, event = buf.combine(expert_out, handle, ...)

# dispatch 的反向 = combine（伴随）；combine 的反向 = dispatch。
```

两个值得点名的设计：

- **`handle`** 由 `dispatch` 返回，编码了按 rank / 按专家的计数与索引。`combine` 直接复用，无需重算路由，两个操作互为伴随（所以 `dispatch_backward` 字面上调用 `combine`）。
- **`EventOverlap`** 包装一个 CUDA event 与 allocator stream。`previous_event` 让 dispatch/combine 在通信 stream 上等这个事件，用户代码无需手工 stream 管理即可 comm-compute overlap。

## Normal kernel：非对称 NVLink + RDMA 转发

`csrc/kernels/internode.cu` 对应 V3 论文的快路径。思路：

1. **Layout 阶段**（`layout.cu::get_dispatch_layout`）扫描 `topk_idx`，按 token 计算它要去哪 ≤4 个 RDMA 对端、以及每个对端节点内的哪些本地 rank。
2. **Dispatch**：
   - 每个 token 先在节点内走 NVLink 送到**转发 rank**（与目标节点 RDMA NIC 同侧）。
   - 转发 rank 每目标节点发一次 RDMA write，批量打包所有要过去的 token。
   - 接收端从本地 RDMA buffer 按专家 NVLink 拉取自己需要的 token。
3. **Combine** 反向而行：专家产出 → 本地 NVLink 归集 → RDMA 回源节点 → NVLink 扇出给原 rank。

为什么非对称转发胜过对称 all-to-all：朴素设计会让每个 rank 对每个其他 rank 都写一次 RDMA，NIC 流量 8×。V3 的 ≤4 节点约束 + DeepEP 的转发结构把 RDMA 流量 压到 **每个 (token, 目标节点) 只一次写**，节点内 fan-out 全被 NVLink 免费吸收。

**FP8 dispatch、BF16 combine** 让入向字节减半（dispatch 的 token 是 FP8），回程留足梯度精度。这是 kernel 签名里写死的。

**SM 预算**：`Buffer.set_num_sms(N)` 限制通信 kernel 占用的 SM。这很重要——同一张 GPU 还要跑 MoE matmul，给通信多了 SM 就抢走了算力。

## 低延迟 kernel：IBGDA、纯 RDMA（再加 NVLink）

`csrc/kernels/internode_ll.cu` + `ibgda_device.cuh` 针对 decode：batch 小（128 tokens）、延迟为王、吞吐次要。

- 通过 NVSHMEM 用 **IBGDA**（InfiniBand GPUDirect Async）—— GPU 直接向 NIC 投递 WR，无需 CPU 介入。在微秒级延迟下至关重要；一次 CPU round-trip 就要吃掉 ~20 µs。
- **起初纯 RDMA**（无转发跳）：每个 rank 直接给所有目标 rank 写，decode 批量下额外 NIC 流量可接受。
- **2025-06 起重新引入 NVLink**（[#173](https://github.com/deepseek-ai/DeepEP/pull/173)）用于节点内路径，避免多余 NIC 往返——混合，但仍"延迟优先"。
- 反向 combine 拆成独立 kernel，可以与 decode 算力以不同方式融合。

hook 式 overlap 是这里最有意思的一块：

> "a hook-based communication-computation overlapping method that does not occupy any SM resource"

普通 CUDA stream overlap 下，comm 与 compute 用不同 stream，但都占 SM。DeepEP LL 暴露一个 **hook**，把通信挂到计算 stream 尾部，依赖 NIC 的异步完成，而非驻留 SM 的 kernel。**通信零 SM 预算**——当 SM 被专家 matmul 全占时，这极有用。

## Intranode kernel

`intranode.cu` 是三者中最简单的：仅 NVLink、无 RDMA、无 IBGDA。存在意义：(a) 单节点 dev / 测试；(b) 作为 internode kernel 里"RDMA 接收后的节点内 fan-out"阶段复用的原语。想读懂整个库，推荐先读它——概念可以向更难的 kernel 平滑迁移。

## "未定义 PTX" 提示

README 有 `DISABLE_AGGRESSIVE_PTX_INSTRS` 环境变量及警告：

> DeepEP 使用了一条未文档化的 PTX 指令 `ld.global.nc.L1::no_allocate.L2::256B` …… 官方文档没有，但实验观察到加速。若编译器/驱动不接受，关掉本 flag。

这是 DeepEP 调优方式的小小窗口：**尝试了 Hopper 理论上"不存在"的 PTX 指令**，因为硬件其实实现了、汇编器也接受。这种选择第三方库很难做——只有在自己集群、自己 PTX 编译路径上训练的人才承担得起风险。这也是 V3 成本故事的一部分。

## 配置与调优

`csrc/config.hpp` + `configs.cuh` 按 EP 大小给出调优：NVLink buffer size、RDMA buffer size、channel 数、队列深度。`Buffer.get_dispatch_config(ep_size)` 给默认值，但 README 明示：

> 你也可以用 tests 里的 autotune 结果替换 `get_*_config`

也就是**每个集群都该重调**。`tests/` 目录设计上就是 autotuner，不只是正确性测试。

README 的网络级建议：

- **IB 虚拟通道做流量隔离**：`NVSHMEM_IB_SL` 把 normal kernel / LL kernel / 其他 workload 分到不同 VL。否则一次训练 dispatch 能把延迟敏感的 decode 饿死。
- **自适应路由**重载开、轻载关——自适应会加延迟，LL kernel 偏好静态。
- **拥塞控制关**——DeepSeek 环境没观察到明显拥塞。不是普遍建议。

## 哪些能搬走

- **非对称 NVLink+RDMA 转发模式**：凡 top-k 路由在节点层面隐式/显式成簇的 MoE 都适用。不是所有 MoE 都这样——若路由均匀，对称 all-to-all 反而更省心。
- **IBGDA + NVSHMEM 的 decode EP**：任何愿意承担 NVSHMEM 部署成本的 MoE 推理系统都能用。
- **SM 预算可控的通信 kernel + hook 式 overlap**：这个模式不限 MoE——"让用户在 comm 与 compute 之间自行裁决 SM 预算"，强于假设默认 stream overlap 够用。
- **FP8 dispatch + BF16 combine**：路由能容忍量化噪声（多数可以）就是个直接收益。

## 哪些不能

- 硬假设：**group-limited gating**、为 7168 hidden 调优的参数、DeepSeek 的 IB 网络形状。其他 MoE（Mixtral top-2、非分组路由）用不到快路径。
- 依赖 **NVSHMEM** 与 fabric 的 IBGDA 支持。不是每个集群都启用。
- 用到文档 ISA 边缘的 PTX 指令。DeepSeek 内部可接受的风险，在受监管生产环境是问号。

## 个人评注

DeepEP 大概是 Open Source Week 里最"顶梁柱"的一个。V3 论文报告的 infra 数字离开 DeepEP 根本跑不出来；没有它，V3 在 H800 上的训练效率不可复现。代码最有意思的一点是**与 V3 架构选择的强耦合**：group-limited gating 不是 DeepEP "支持"的另一个选项，它就是让 DeepEP 快的那个前提。这也是 DeepEP 不追求成为 NCCL 通用替代的原因——**它是针对某种 MoE 形状协同设计的**，这正是它的价值。做 MoE infra 要记住两点：(a) **把路由形状与通信 kernel 当成一个协同设计对象**，而非分层；(b) **训练与 decode 需要不同的通信 kernel**——同一个函数调用、两套实现、启动时选择。预期这两条会在 2026 年成为开源 MoE 栈的标配。

## 参考

- DeepEP 仓库：https://github.com/deepseek-ai/DeepEP
- V3 技术报告（group-limited gating + ≤4 节点/token）：arXiv:2412.19437
- NVSHMEM：https://developer.nvidia.com/nvshmem
- IBGDA 概述：NVIDIA HPC-SDK 文档
- DeepSeek 推理系统综述：https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
