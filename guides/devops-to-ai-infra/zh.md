# DevOps 工程师如何切入 AI Infra

> 这是一篇职业路径指南，不是论文分析。目标读者：有 2–5 年 DevOps / SRE / 平台工程经验，想转向 AI 基础设施方向的工程师。

## 一句话结论

**DevOps → AI Infra 的核心动作是：把已有的集群、网络、存储、可观测性经验叠加到 GPU 和分布式训练 / 推理这个新的硬件软件栈上。** 不需要先成为 ML 算法专家，起点是 serving 运维，终点取决于你愿意往 ML 系统底层走多深。

---

## 一、优势盘点——你已经有的

DevOps 背景在 AI Infra 里的直接复用远超大多数人的预期：

| DevOps 技能 | AI Infra 对应场景 |
|---|---|
| Kubernetes / 容器编排 | GPU 集群调度（加 GPU Operator、MIG 分区、device plugin） |
| 网络工程（VLAN、BGP、ECMP） | RDMA、InfiniBand、RoCE 拓扑 —— 这里的高阶网络工程师极度稀缺 |
| Linux 系统 / NUMA 感知 | PCIe 拓扑、GPU-NIC 亲和性、驱动管理、hugepage |
| Prometheus / Grafana / 告警 | GPU 利用率、MFU、训练吞吐、serving 延迟监控 |
| 分布式存储 / NFS / Ceph | 高吞吐训练数据读取 + checkpoint 存储（对标 3FS、Lustre） |
| CI/CD 流水线 | 模型训练 pipeline、镜像构建、serving 灰度发布 |
| 故障处理 / oncall | 训练任务中断重启、节点故障隔离、NCCL 超时排查 |

**网络背景是最被低估的优势。** 大多数 ML 工程师不懂 RDMA，不懂 InfiniBand 调参，不懂 ECN 拥塞控制。如果你有网络工程背景，进入 AI Infra 是降维打击——参见本仓库 [GPU 互联 primer](../../foundational/gpu-interconnect/)，里面的绝大多数内容都是网络工程师的主场。

---

## 二、硬缺口——必须补的

不补这几块会在职业路径上卡住：

### 2.1 GPU 编程模型（概念层，不需要写 kernel）

需要理解：
- SM（Streaming Multiprocessor）/ warp / thread block 层次
- HBM（显存）vs SRAM（片上缓存）的带宽差距
- 为什么"把数据放在 SRAM 里"是几乎所有 kernel 优化的核心
- Tensor Core 是什么，FP16 / BF16 / FP8 的区别

不需要理解（短期）：CUDA C++ 语法、PTX 汇编、`__shared__` 内存声明。

**资源**：NVIDIA CUDA 编程模型文档前两章；Simon Boehm 的 "How to Optimize a CUDA Matmul Kernel"（只读思路，不需要跑代码）。

### 2.2 分布式训练基础

需要理解：
- Data Parallelism（DP）：每个 GPU 一份完整模型，梯度 all-reduce
- Tensor Parallelism（TP）：权重矩阵按列 / 行切分，在节点内 NVLink 上通信
- Pipeline Parallelism（PP）：按层切分，跨节点 RDMA 传激活
- ZeRO：把优化器状态 / 梯度 / 权重切分到各 GPU，reduce-scatter + all-gather

理解这四种并行的**通信模式和显存形状**，不需要实现它们。参见 [Megatron-LM](../../foundational/megatron-lm/) 和 [ZeRO / FSDP](../../foundational/zero-fsdp/)。

### 2.3 推理 serving 内部

需要理解：
- KV cache 是什么，为什么它是推理显存的主要消耗
- prefill（处理 prompt）和 decode（逐 token 生成）是两个完全不同的计算形状
- continuous batching 如何让 GPU 不空转
- 为什么 batch size 受 KV cache 大小约束

参见 [PagedAttention / vLLM](../../foundational/paged-attention/) 和 [DistServe](../../foundational/distserve/)。

### 2.4 PyTorch 够用级别

不需要精通，但需要能读懂：
- tensor、shape、dtype 操作
- `model.load_state_dict()`、`torch.save()` 是怎么做 checkpoint 的
- `DataLoader` 的 worker 和 pin_memory 为什么影响训练速度

---

## 三、切入路径——按阶段

### 第一步：拿到手感（1–2 个月）

**目标：在真实 GPU 上跑起来一个推理服务，用 ops 视角观察它。**

```bash
# 安装
pip install vllm

# 启动服务（需要 GPU）
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9

# 压测
pip install locust
# 或者用 vllm 自带的 benchmark_serving.py
```

观察点：
- `nvidia-smi` / `nvitop`：显存分配，GPU 利用率曲线
- 调低 `--max-model-len`，看 KV cache 减少后 batch size 如何变化
- 开两个并发请求，看 TTFT（Time to First Token）和 TPOT（Time per Output Token）如何分离

这一步不需要懂 ML，纯 ops 视角就够。

### 第二步：读本仓库的关键论文（2–3 个月，按顺序）

按 DevOps 背景最友好的顺序：

1. **[PagedAttention / vLLM](../../foundational/paged-attention/)** — KV cache 分页，类比 OS 虚拟内存管理，DevOps 看这篇最顺手
2. **[DistServe](../../foundational/distserve/)** — goodput 指标，prefill/decode 为什么要拆分
3. **[GPU 互联 primer](../../foundational/gpu-interconnect/)** — NVLink / IB / RDMA / IBGDA，网络工程师主场
4. **[ZeRO / FSDP](../../foundational/zero-fsdp/)** — 显存怎么被切分，all-reduce 在哪条链路上跑
5. **[SGLang](../../foundational/sglang/)** — RadixAttention 前缀缓存，类比 CDN 缓存策略
6. **[Megatron-LM](../../foundational/megatron-lm/)** — 三种并行的通信形状，选择性重算

选读（深入后）：
- [DeepEP](../../deepseek/open-source-week/deep-ep/) — RDMA + IBGDA 专家并行，网络背景直接能看懂动机
- [3FS](../../deepseek/open-source-week/3fs/) — RDMA 原生分布式存储，存储背景直接能看懂
- [Mooncake](../../moonshot/mooncake/) — KV cache 池化，类比 CDN 边缘缓存架构

### 第三步：选一个垂直方向深入

不要全面铺开。根据现有背景选最近的一个：

---

## 四、三条垂直路径

### 路径 A：GPU 集群运维（最快落地）

**适合**：有 Kubernetes、裸金属运维、数据中心网络背景的人。

**核心工作**：
- GPU 节点 provisioning：驱动版本、CUDA 版本、NCCL 版本矩阵管理
- Kubernetes GPU Operator：device plugin、MIG 分区、节点标签
- 健康检测：DCGM exporter + 自定义探针（GPU 温度、显存 ECC 错误、NVLink 错误计数）
- 拓扑感知调度：把通信量大的 GPU 调度到同一 NVLink 域 / 同一叶交换机
- 故障处理：识别 GPU 硬件故障 vs 驱动问题 vs NCCL 死锁

**技术栈**：Kubernetes + GPU Operator、SLURM（很多训练集群用）、DCGM、NCCL debug 工具。

**目标岗位**：GPU Platform Engineer、HPC Cluster Engineer、AI Infrastructure Engineer（Ops）

---

### 路径 B：推理平台（工程深度最好）

**适合**：有 SRE、微服务运维、API 网关经验的人。

**核心工作**：
- 部署和运营 vLLM / SGLang / TensorRT-LLM
- 自动扩缩容策略：基于队列深度 / GPU 利用率 / TTFT SLO
- 多模型调度：prefix cache 共享、routing、模型版本管理
- 成本优化：spot 实例 + checkpoint 恢复、KV cache offload 到 CPU / SSD
- 可观测性：TTFT p99、TPOT、token 吞吐、KV cache 命中率

**技术栈**：vLLM / SGLang、Kubernetes、Prometheus、自定义 load balancer。

**目标岗位**：Inference Platform Engineer、LLM Serving Engineer、ML Platform SRE

---

### 路径 C：训练基础设施（最硬，回报最高）

**适合**：有分布式系统、存储、网络全栈背景，愿意深入 PyTorch internals 的人。

**核心工作**：
- 作业调度与资源管理：SLURM / Volcano / 自研调度器
- Checkpoint 管理：异步存储、恢复速度优化、跨故障继续训练
- 通信性能分析：NCCL all-reduce 瓶颈、带宽利用率、拓扑感知 ring 构建
- 存储 I/O 优化：训练数据预取、多路并行读、checkpoint 写入带宽
- 故障恢复：节点失联后的自动检测 + 从最近 checkpoint 重启

**技术栈**：PyTorch、Megatron-LM / FSDP、NCCL、SLURM、高吞吐存储（Lustre / 3FS）。

**目标岗位**：Training Infrastructure Engineer、ML Systems Engineer

---

## 五、常见陷阱

**陷阱 1：过早陷入 CUDA 编程**

写 CUDA kernel 是另一个职业方向（ML Systems / Kernel Engineer），不是 DevOps 切入 AI Infra 的必经路。DevOps 的优势在系统层，不在 kernel 层。短期内知道 CUDA 模型是什么就够了。

**陷阱 2：从 ML 算法学起**

Attention 是什么、transformer 原理、loss function 推导——这些固然有用，但 DevOps 的优势在系统层。从算法学起是绕远路，反而拉平了你的差异化优势。

**陷阱 3：只看理论不动手**

AI Infra 的知识必须在真实 GPU 上验证才有感觉。"KV cache 会占满显存"这句话，在你亲眼看到 `nvidia-smi` 里显存被 KV cache 吃满之前，只是文字。

**陷阱 4：低估网络技能的价值**

多数 ML 工程师不懂 RDMA。他们知道"all-reduce 用 InfiniBand"，但不知道为什么 ECN 调参会影响 NCCL 吞吐，不知道 PFC 死锁是怎么产生的，不知道 IBGDA 省了哪一步。这是 DevOps / 网络工程师在 AI Infra 里的真实护城河。

---

## 六、本仓库作为学习地图

本仓库的所有条目都从工程视角写，不预设 ML 算法背景。建议的阅读顺序：

**Tier 1（DevOps 背景可直接看懂 80% 以上）**
- [PagedAttention / vLLM](../../foundational/paged-attention/)
- [GPU 互联 primer](../../foundational/gpu-interconnect/)
- [3FS 源码走读](../../deepseek/open-source-week/3fs/)
- [DistServe](../../foundational/distserve/)

**Tier 2（需要少量 GPU 模型背景）**
- [ZeRO / FSDP](../../foundational/zero-fsdp/)
- [SGLang](../../foundational/sglang/)
- [Megatron-LM](../../foundational/megatron-lm/)
- [Mooncake](../../moonshot/mooncake/)

**Tier 3（深入后读）**
- [FlashAttention 1/2/3](../../foundational/flash-attention/)
- [DeepEP 源码走读](../../deepseek/open-source-week/deep-ep/)
- [DeepSeek V3 技术报告](../../deepseek/v3-tech-report/)

---

## 七、6 个月里程碑

| 时间 | 目标 |
|---|---|
| 第 1 个月 | 本地 / 云上跑起 vLLM，能解释 KV cache 显存占用，能看 TTFT / TPOT 指标 |
| 第 2 个月 | 读完 Tier 1 论文，能解释 PagedAttention 和 goodput 是什么 |
| 第 3 个月 | 选定一条垂直路径，深入一个技术方向（集群 / serving / 训练） |
| 第 4–5 个月 | 读完 Tier 2 论文，能解释 TP / DP / ZeRO 的通信形状 |
| 第 6 个月 | 在目标方向有一个可以讲清楚的项目或贡献（哪怕是 benchmark、故障复盘、工具） |

---

## 参考资源

- [vLLM 文档](https://docs.vllm.ai/) — 最好的推理 serving 入手材料
- [Megatron-LM README](https://github.com/NVIDIA/Megatron-LM) — 分布式训练的事实标准
- [NCCL 调试指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/) — 排查集合通信问题
- [DCGM 文档](https://docs.nvidia.com/datacenter/dcgm/) — GPU 集群监控
- [Efficient Large Scale Language Modeling with Megatron (论文)](https://arxiv.org/abs/2104.04473) — Tier 2 的前置读物
- 本仓库 [GPU 互联 primer](../../foundational/gpu-interconnect/) — DevOps 切入 AI Infra 的网络基础
