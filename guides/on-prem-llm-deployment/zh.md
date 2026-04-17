# 本地大模型部署——从 Mac Studio 到 GPU 集群

> 这是一篇实战部署指南，不是论文分析。目标读者：正在评估如何在自有硬件上跑开源大模型的工程团队——从单机试用到生产级 GPU 集群。

## 一句话结论

**先用 Mac Studio + Ollama 一周内验证场景；并发上来后切 vLLM + GPU；全公司依赖时上 K8s GPU 集群。** 需要什么硬件完全取决于并发用户数和延迟要求。

---

## 一、决策框架——架构由什么驱动

选硬件之前先回答三个问题：

| 问题 | 决定了什么 |
|------|-----------|
| 并发用户多少？ | GPU 数量、内存带宽预算 |
| 模型多大？ | 显存容量（VRAM 或统一内存） |
| 延迟 / 吞吐目标？ | 量化级别、prefill/decode 是否分离 |

LLM 推理的 decode 阶段是**内存带宽瓶颈**——每生成一个 token 都要从显存读一遍完整的模型权重。所以对大多数 serving 场景来说，瓶颈是内存带宽，不是算力。

---

## 二、第一档——Mac Studio / Apple Silicon（1–5 人）

### 为什么能跑

Apple 的**统一内存架构（UMA）**让 GPU 能访问全部系统内存。一台 192 GB 的 Mac Studio 可以装下 70B INT4 模型——同样的模型在独立 GPU 方案里需要 2 张 A100 80 GB。不需要装驱动、不需要 CUDA、不需要容器运行时——`brew install ollama` 就能开始服务。

### 硬件矩阵

| 芯片 | 统一内存 | 内存带宽 | 70B INT4 速度 | 405B INT4 |
|------|---------|---------|--------------|-----------|
| M2 Ultra | 192 GB | ~800 GB/s | ~10–15 tok/s | 装不下 |
| M3/M4 Ultra | 192–512 GB | ~800–900 GB/s | ~12–18 tok/s | 512 GB 配置可装 |
| M4 Max | 128 GB | ~550 GB/s | ~8–12 tok/s | 装不下 |

### 顶配能跑什么模型（M4 Ultra 512 GB）

模型显存占用 ≈ 参数量 × 每参数字节数 + KV Cache 开销。512 GB 统一内存扣除系统和 KV Cache 后，可用约 **430–460 GB**。

**不同精度下的显存占用：**

| 精度 | 每参数字节 | 70B | 405B | 671B (DeepSeek V3) |
|------|-----------|-----|------|---------------------|
| FP16 | 2 | 140 GB | 810 GB | 1.34 TB |
| INT8 | 1 | 70 GB | 405 GB | 671 GB |
| INT4 | 0.5 | 35 GB | ~203 GB | ~336 GB |

**实际能跑的模型：**

| 模型 | 精度 | 能装？ | 速度 | 备注 |
|------|------|--------|------|------|
| Llama 3 70B | FP16 | 轻松（140 GB） | ~15–20 tok/s | 该尺寸下质量最好 |
| Llama 3 70B | INT4 | 轻松（35 GB） | ~20–25 tok/s | 速度与质量最佳平衡 |
| Qwen2.5 72B | FP16 | 轻松 | ~15–18 tok/s | |
| Llama 3.1 405B | INT4 | 可以（~203 GB） | ~3–5 tok/s | 能用但慢 |
| Llama 3.1 405B | INT8 | 勉强（~405 GB） | ~1.5–3 tok/s | 很慢 |
| DeepSeek V3 671B | INT4 | 可以（~336 GB） | ~2–4 tok/s | MoE 架构，每 token 只激活 37B，比 405B dense 更快 |

**核心结论：** 70B INT4 是甜点——又快质量又好。405B 能装但 ~800 GB/s 带宽读 200+ GB 权重意味着每个 token 约 250ms，体验偏慢。DeepSeek V3 是特殊情况：总参数 671B 但 MoE 每次只激活 37B，实际推理比 405B dense 更快。

**选购建议：** 如果主力就跑 70B，**192 GB 配置（~$7,000）就够了**——省下的 $5,000+ 以后升级 GPU 更划算。512 GB 只在跑 400B+ 模型时才值。

### 推荐软件栈

```
模型运行时:    llama.cpp（通过 Ollama）或 MLX
API 层:        Ollama（兼容 OpenAI /v1/chat/completions）
前端界面:      Open WebUI 或 LobeChat
```

启动：

```bash
brew install ollama
ollama serve &
ollama pull llama3:70b-instruct-q4_K_M
```

Ollama 在 `localhost:11434/v1/` 提供 OpenAI 兼容 API——现有调用 OpenAI 的代码只需改 base URL。

### 适用场景

- **原型验证**——投 GPU 之前先证明场景 work
- **隐私敏感**——数据完全不出本机
- **开发内循环**——本地编程助手、RAG 实验
- **< 5 并发用户**——单用户体验不错，并发上来会明显变慢

### 什么时候该升级

- 3 个人同时用就感觉卡了
- 需要持续 > 20 tok/s 的吞吐
- 模型要以内部 API 形式提供，带 SLA 保证

---

## 二-B、NVIDIA DGX 产品线——交钥匙方案

NVIDIA 在各个规模上都提供预集成的 AI 系统。相比 DIY 的核心优势：硬件软件栈已验证、企业级支持、DGX OS 预装优化驱动和容器运行时。

### DGX Spark（桌面级，~$3,000）

NVIDIA 版的 Mac Studio。搭载 **Grace Blackwell GB10 超级芯片**——ARM Grace CPU + Blackwell GPU 单芯片统一内存，类似 Apple 的 UMA 但支持 CUDA。

| 规格 | DGX Spark | Mac Studio M4 Ultra（顶配） |
|------|-----------|---------------------------|
| 内存 | 128 GB 统一内存 | 最高 512 GB 统一内存 |
| 内存带宽 | ~273 GB/s | ~800 GB/s |
| AI 算力（INT8） | 209 TOPS | ~74 TOPS |
| CUDA / Tensor Core | 有 | 无 |
| 操作系统 | DGX OS（Ubuntu 系） | macOS |
| 价格 | ~$3,000 | $4,000–14,000 |

**能跑什么：**

| 模型 | 精度 | 能装？ | 备注 |
|------|------|--------|------|
| Llama 3 7–14B | FP16 | 轻松 | 单用户速度不错 |
| Llama 3 70B | INT4 | 可以（~35 GB） | 带宽比 Mac Studio 低，tok/s 更慢 |
| Llama 3.1 405B | INT4 | 不行（~203 GB） | 只有 128 GB，装不下 |

**结论：** 比 Mac Studio 便宜，有 CUDA 生态（TensorRT-LLM、cuDNN），但 **128 GB 是硬上限**——200B+ 模型装不下。带宽更低（~273 vs ~800 GB/s）意味着同模型下 tok/s 比 Mac Studio 慢。最适合需要 CUDA 兼容性的低成本开发场景。两台 DGX Spark 可通过 ConnectX 互联，合并为 256 GB。

### DGX Station（工作站级，~$50–125K）

工作站形态的 GPU 服务器，当前代搭载 Blackwell GPU。

| 代次 | GPU | 显存 | 内存带宽 | 大致价格 |
|------|-----|------|---------|---------|
| DGX Station (A100) | 4× A100 80 GB | 320 GB | 8 TB/s | ~$50K（二手） |
| DGX Station (B200) | 1× B200 | 192 GB HBM3e | 8 TB/s | ~$125K |

放桌下，用普通市电（单相）。适合 10–30 人团队用真正的 GPU 吞吐跑 70B 模型。开箱即用 NVIDIA AI Enterprise 全套软件栈。

### DGX H100 / H200 / B200（数据中心级，$300K–$600K+）

旗舰数据中心一体机。每台 8 块 GPU 通过 NVSwitch 全互联。

| 系统 | GPU | 显存（总计） | GPU 间带宽 | FP8 算力 | 大致价格 |
|------|-----|------------|-----------|---------|---------|
| DGX H100 | 8× H100 SXM | 640 GB HBM3 | 900 GB/s NVLink | 16 PFLOPS | ~$300K |
| DGX H200 | 8× H200 SXM | 1.13 TB HBM3e | 900 GB/s NVLink | 16 PFLOPS | ~$400K |
| DGX B200 | 8× B200 SXM | 1.4 TB HBM3e | 1.8 TB/s NVLink | 36 PFLOPS | ~$500–600K |

**单台 DGX 能跑什么：**

| 模型 | 系统 | 精度 | 吞吐（batch serving） |
|------|------|------|----------------------|
| Llama 3 70B | DGX H100 | FP8, TP=4 | ~2,000+ tok/s |
| Llama 3.1 405B | DGX H100 | FP8, TP=8 | ~500–800 tok/s |
| Llama 3.1 405B | DGX H200 | FP16, TP=8 | ~400–600 tok/s（无需量化） |
| DeepSeek V3 671B | DGX B200 | FP8 | 1.4 TB 装得下；MoE 路由受益于 NVLink |

### DGX SuperPOD（机架/集群级）

多台 DGX 组成机架，通过 InfiniBand/NVLink 互联。NVIDIA 以整机架方案销售（8–32+ 台 DGX 节点）。这是多节点训练和大规模 serving 的路径。最小 SuperPOD 配置约 $2–5M 起。

### DGX vs. DIY——什么时候买哪个

| 维度 | DGX | DIY（白牌服务器 + GPU） |
|------|-----|----------------------|
| 部署时间 | 数天（预验证） | 数周到数月（驱动/固件/散热） |
| 支持 | NVIDIA AI Enterprise 企业支持 | 社区 + 硬件厂商 |
| 每 GPU 成本 | 更高（~1.5–2× 溢价） | 更低 |
| 灵活性 | 固定配置 | 任意 GPU 组合、自定义散热/网络 |
| 软件 | DGX OS、Base Command、预调优 NCCL | 手动调优 |
| 适合 | 重视可用性、没有专职 GPU 运维团队 | 有 GPU 集群运维经验、追求性价比 |

**经验法则：** 没有专职 GPU 集群运维团队的话，DGX 能省下数月的集成调试时间。如果有经验且想优化成本，OEM 服务器（Dell、Supermicro、Lambda）性价比更好。

---

## 二-C、消费级 NVIDIA GPU（RTX 4090 / 5090）

最低成本的 CUDA 入门方案。

| GPU | 显存 | 内存带宽 | FP16 算力 | 价格 |
|-----|------|---------|----------|------|
| RTX 4090 | 24 GB | 1 TB/s | 165 TFLOPS | ~$1,600 |
| RTX 5090 | 32 GB | 1.79 TB/s | 209 TFLOPS | ~$2,000 |

**能跑什么：** 单卡跑 7B–14B FP16。70B INT4 需要 2–3 卡，但消费卡没有 NVLink——PCIe 上做 tensor parallelism 比 NVLink 慢 5–10 倍，多卡 TP 效率差。

**优势：** 最便宜的 CUDA 硬件；好买；社区支持好（llama.cpp、vLLM 都能跑）。

**劣势：** 显存小（每卡 24–32 GB）；消费卡之间没有 NVLink；无 ECC 内存（训练不可靠）；功耗墙限制（每卡 450W TDP，普通工作站难堆多卡）。

**适合：** 预算极有限的小团队跑 7B–14B 模型，或个人开发者做本地 CUDA 开发。一台 2× RTX 5090 主机（~$5K 总价）可以跑 14B FP16 或 70B INT4（有 TP 效率损失）。

---

## 二-D、AMD Instinct GPU（MI300X / MI325X）

数据中心级推理的主要 NVIDIA 替代方案。核心差异化：**单卡 192–256 GB HBM**——一张 MI300X 就能装下需要 4–8 张 NVIDIA 卡的 405B INT4 模型。

| GPU | HBM | 内存带宽 | FP16 算力 | 价格 |
|-----|-----|---------|----------|------|
| MI300X | 192 GB HBM3 | 5.3 TB/s | 1.3 PFLOPS | ~$15–20K |
| MI325X | 256 GB HBM3e | 6 TB/s | 1.3 PFLOPS | ~$20–25K |

**与 NVIDIA 对比：**

| 维度 | MI300X（192 GB） | H100 SXM（80 GB） | H200 SXM（141 GB） |
|------|----------------|-------------------|-------------------|
| HBM 容量 | 192 GB | 80 GB | 141 GB |
| 内存带宽 | 5.3 TB/s | 3.35 TB/s | 4.8 TB/s |
| 405B INT4 | 单卡 | 4–8 卡（TP） | 2–4 卡（TP） |
| 软件生态 | ROCm + vLLM/SGLang | CUDA（完整生态） | CUDA（完整生态） |

**优势：** 单卡 HBM 巨大；内存带宽比 H100 高 50%+；每 GB HBM 价格更优；vLLM 和 SGLang 都有 ROCm 后端。

**劣势：** ROCm 生态不如 CUDA 成熟——FlashAttention AMD 移植版更新滞后，部分自定义 kernel 需要移植，调试工具较弱。NVIDIA 专属优化（TensorRT-LLM、FP8 方案）不能直接迁移。

**适合：** 愿意接受软件生态 tradeoff 换取更高单卡 HBM 和更低每 GB 成本的团队。在**大模型推理**（405B+）场景尤其有吸引力——替代方案是多卡 NVIDIA 组合。

---

## 二-E、Intel Gaudi 2 / 3

Intel 的 AI 加速器（收购 Habana Labs 后的产品线）。定位：NVIDIA 的性价比替代。

| 加速器 | HBM | 内存带宽 | 大致价格 |
|--------|-----|---------|---------|
| Gaudi 2 | 96 GB HBM2e | 2.45 TB/s | ~$6–8K |
| Gaudi 3 | 128 GB HBM2e | 3.7 TB/s | ~$12–15K |

Gaudi 3 的 LLM 推理性能接近 H100，但每卡价格约为一半。vLLM 有 Gaudi 后端。Intel 提供 8 卡 Gaudi 服务器（HL-325L）作为整机方案。

**优势：** 每卡价格显著更低；Gaudi 3 的 128 GB HBM 比 H100 的 80 GB 大；Intel 企业级支持。

**劣势：** 社区很小——遇到问题 Stack Overflow 帮不了你。软件栈（Synapse AI）成熟度不如 CUDA 和 ROCm。可用的优化 kernel 更少。

**适合：** Intel 生态绑定的企业；对每 token 成本敏感、对部署速度要求不高的推理场景。

---

## 二-F、OEM GPU 服务器——中间地带

介于"买 DGX"和"从零搭机"之间，OEM 厂商提供预验证配置的 GPU 服务器：

| 厂商 | 产品线 | 支持 GPU | 核心优势 |
|------|-------|---------|---------|
| Dell | PowerEdge XE9680 | 8× H100/H200/B200 SXM | 企业支持，比 DGX 便宜 10–20% |
| HPE | ProLiant DL380a | 8× H100 | GreenLake 云管理平台集成 |
| Supermicro | GPU SuperServer | 任意（最灵活） | 成本最低，配置选项最多 |
| Lambda | Hyperplane | 8× H100/H200 | 预装 ML 软件栈，面向 AI 团队 |

**典型成本：** 同 GPU 数量下比 DGX 便宜 10–30%。代价是没有 DGX OS 和 Base Command，但保留硬件保修和验证过的散热/供电方案。

**适合：** 有一定运维能力、想要硬件支持但不愿付 DGX 溢价的团队。

---

## 二-G、其他形态

### 多 Mac 集群

用 Thunderbolt 5 或万兆以太网将多台 Mac Studio 链接起来，使用 [Exo](https://github.com/exo-explore/exo) 等框架：

- 例如：4× Mac Studio 192 GB = 768 GB 总内存——可跑 405B FP16
- **问题：** 节点间带宽（Thunderbolt ~20–40 Gbps，以太网 ~10 Gbps）比 NVLink 慢约 100 倍。Tensor parallelism 不实际；只能用 pipeline parallelism，每 token 延迟高。
- **适合：** 已有多台 Mac 的团队，想不买 GPU 硬件就试试大模型。不适合生产 serving。

### 专用推理芯片

| 厂商 | 芯片 | 形态 | 备注 |
|------|------|------|------|
| SambaNova | SN40L（DataScale） | 本地一体机 | 企业 RAG 场景优化；支持本地部署 |
| Groq | LPU | 云 API 为主 | 超低延迟推理；本地部署选项有限 |
| Cerebras | WSE-3 | 云/本地 | 晶圆级芯片；极限 batch 吞吐；数百万美元级系统 |

这些属于小众方案——只在你的负载有特殊需求（超低延迟、极限 batch 吞吐）且通用 GPU 满足不了时才考虑。

---

## 二-H、硬件选型总结

| 预算 | 最优选择 | 能跑什么 |
|------|---------|---------|
| < $3K | DGX Spark / RTX 5090 单卡 | 7B–14B FP16 |
| $3–8K | Mac Studio 192 GB / 2× RTX 5090 | 70B INT4 |
| $8–20K | 1× A100（二手）或 1× MI300X | 70B FP8/INT4，真正的 GPU 吞吐 |
| $20–60K | 2× H100 / DGX Station（二手） | 70B FP16 或 405B INT4 |
| $60–150K | 4× MI300X 服务器 / DGX Station (B200) | 405B FP8；多模型 serving |
| $150–400K | DGX H100 / Dell XE9680 / 8× MI300X | 405B+ 规模化；100+ 用户 |
| $400K+ | DGX H200/B200 / SuperPOD | 671B 全精度；训练 + 推理 |

---

## 三、第二档——单节点 GPU 服务器（10–50 人）

### 硬件选型

| 配置 | 显存 | 内存带宽（总计） | 70B INT4 吞吐 | 大致价格 |
|-----|------|-----------------|--------------|---------|
| 2× RTX 5090 | 64 GB | 3.58 TB/s | ~30–50 tok/s（PCIe TP 损失） | ~$5K 整机 |
| 1× MI300X | 192 GB | 5.3 TB/s | ~100–150 tok/s | ~$15–20K |
| 1× A100 80 GB | 80 GB | 2 TB/s | ~40–60 tok/s | ~$15K（二手） |
| 2× A100 80 GB | 160 GB | 4 TB/s | ~80–120 tok/s | ~$30K |
| 1× H100 SXM | 80 GB | 3.35 TB/s | ~80–100 tok/s | ~$30K |
| 2× H100 SXM | 160 GB | 6.7 TB/s | ~150–200 tok/s | ~$60K |
| DGX Station (A100) | 320 GB | 8 TB/s | ~200+ tok/s | ~$50K（二手） |

从 Apple Silicon 到单张 A100 的提升大约是 **5–10 倍吞吐**，完全由 HBM 带宽差距驱动。

### 推荐软件栈

```
推理引擎:      vLLM 或 SGLang
API:           vLLM 内建的 OpenAI 兼容服务
反向代理:      Nginx（TLS、基础认证）
进程管理:      systemd 或 Docker Compose
监控:          Prometheus + Grafana（GPU 利用率、TTFT、TPS）
前端:          Open WebUI / LobeChat / 自研
```

启动：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-70B-Instruct-AWQ \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

### 这一档的关键决策

**量化策略：**

| GPU | 最佳选择 | 原因 |
|-----|---------|------|
| H100 | FP8（W8A8） | 硬件原生 FP8 Tensor Core；几乎无损 |
| A100 | INT4（AWQ/GPTQ） | 没有 FP8 硬件；INT4 减半显存，能装更大模型 |
| 都可以 | W8A8 SmoothQuant | INT4 质量不够时的折中 |

**Prefix Caching：** 如果业务有大量重复 system prompt（客服机器人、编程助手、RAG 固定检索模板），开启 vLLM/SGLang 的 prefix caching 可以节省 **30–60% prefill 计算**——参见 [Prefix Caching](../../foundational/prefix-caching/)。

**Chunked Prefill：** 长 prompt 场景（RAG 大上下文），开启 chunked prefill 防止长 prefill 阻塞 decode batch——参见 [Chunked Prefill](../../foundational/chunked-prefill/)。

---

## 四、第三档——多节点 GPU 集群（50–500+ 人）

### 架构

```
                    负载均衡（Nginx / Envoy）
                           │
                    API Gateway
              （认证、限流、用量计费）
                           │
              ┌────────────┼────────────┐
              │            │            │
         vLLM pod 0   vLLM pod 1   vLLM pod N
         (TP=4/8)     (TP=4/8)     (TP=4/8)
              │            │            │
              └────────────┼────────────┘
                           │
                    模型存储
                  （MinIO / NFS / S3）
```

### 基础设施选型

| 层级 | 建议方案 |
|------|---------|
| 编排 | Kubernetes + NVIDIA GPU Operator + MIG（如需共享 GPU） |
| 推理引擎 | **vLLM**（灵活、社区活跃）或 **TensorRT-LLM**（H100 极致性能） |
| 模型仓库 | 内网 HuggingFace Mirror 或 MinIO + 模型版本管理 |
| API 网关 | Kong / 自研——按团队限流、日志记录、用量回收 |
| 监控 | Prometheus + Grafana：TTFT p50/p99、tokens/sec、GPU 利用率、队列深度 |
| 前端 | Open WebUI / LobeChat / 公司自有 UI |

### 进阶优化

**Prefill / Decode 分离（DistServe 模式）：**

如果业务 prefill 压力大（长文档、RAG），考虑将 prefill 和 decode 分到不同的节点池。Prefill 是计算密集型，decode 是带宽密集型，混在同一张 GPU 上会导致次优 batching。参见 [DistServe](../../foundational/distserve/)。

**多模型路由：**

在路由层根据请求复杂度选择模型（简单任务走 7B，复杂任务走 70B）。大多数请求不需要最大的模型，这样能显著降低成本。

**扩缩容策略：**

| 信号 | 动作 |
|------|------|
| 队列深度 > 阈值 | 扩容推理 pod |
| GPU 利用率 < 30% 持续 10 分钟 | 缩容 |
| p99 TTFT > SLA | 增加 prefill 优化节点 |

---

## 五、第四档——加入微调能力

如果 prompting 和 RAG 无法满足领域适配需求：

| 需求 | 方案 | 硬件 |
|------|------|------|
| 轻量适配 | LoRA / QLoRA | 单张 A100 即可跑 70B QLoRA |
| 全参微调 | DeepSpeed ZeRO-3 或 FSDP | 8+ GPU |
| RLHF / GRPO 对齐 | verl（HybridFlow） | 8+ GPU，参见 [verl](../../foundational/verl/) |
| 数据标注 | Label Studio + 内部标注平台 | 仅需 CPU |
| 实验追踪 | W&B / MLflow | 仅需 CPU |

对大多数团队来说，**LoRA 微调就够了**——只训练 1% 的参数就能适配领域，单卡即可运行。全参微调和 RLHF 只在需要大幅改变模型行为时才值得投入。

---

## 六、成本参考

| 方案 | 硬件成本 | 月电费 | 可服务 |
|------|---------|-------|--------|
| RTX 5090 单卡主机 | ~$3–5K | ~$50 | 1–3 人（7B–14B） |
| DGX Spark（128 GB） | ~$3,000 | ~$10 | 1–3 人 |
| Mac Studio M4 Ultra 192 GB | ~$8,000 | ~$15 | 1–5 人 |
| 1× MI300X 服务器 | ~$15–20K | ~$250 | 10–30 人 |
| 1× A100 80 GB 服务器 | ~$15–20K（二手） | ~$200 | 10–30 人 |
| DGX Station (A100, 二手) | ~$50K | ~$300 | 10–50 人 |
| Dell XE9680（8× H100） | ~$220–270K | ~$1,500 | 100–300 人 |
| 8× MI300X 服务器 | ~$150–200K | ~$1,200 | 100–300 人 |
| DGX H100 | ~$300K | ~$1,500 | 100–300 人 |
| DGX H200 | ~$400K | ~$1,500 | 200–500 人 |
| 4× DGX H100（SuperPOD） | ~$1.5M+ | ~$6,000 | 500+ 人 |
| 云 H100（过渡） | $2–3/GPU/小时 | — | 弹性 |

**用云做过渡：** 先在云 GPU（Lambda、CoreWeave、RunPod）上验证规模，再决定买什么硬件。一个月 8× H100 租金（~$15K）远比买错配置便宜。

---

## 七、决策流程图

```
开始
  │
  ├─ "只是探索，< 5 人"
  │   ├─ 需要 CUDA？ → DGX Spark（$3K，128 GB）
  │   └─ 要最大内存？ → Mac Studio + Ollama（$5–14K，最高 512 GB）
  │
  ├─ "需要内部 API，10–50 人"
  │   ├─ 想省心？ → DGX Station（$50–125K）
  │   ├─ 要单卡最大 HBM？ → MI300X（$15–20K，单卡 192 GB）
  │   └─ 要灵活？ → 1–2× A100/H100 + vLLM（$15–60K）
  │
  ├─ "全公司服务，有 SLA 要求"
  │   ├─ 没有 GPU 运维团队？ → DGX H100/H200（$300–400K，企业支持）
  │   ├─ 要 NVIDIA 但更便宜？ → Dell XE9680 / Lambda（$220–270K）
  │   ├─ 考虑 AMD 替代？ → 8× MI300X 服务器（$150–200K）
  │   └─ 全 DIY？ → K8s 集群 + vLLM/TRT-LLM（$250K+）
  │
  └─ "需要针对领域做微调"
      → 加训练节点：LoRA（1 张 GPU）或全参（DGX 节点 / 8+ GPU）
```

---

## 参考

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention、continuous batching、OpenAI 兼容 API
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention prefix caching、结构化生成
- [Ollama](https://ollama.com) — llama.cpp 封装，一条命令启动
- [MLX](https://github.com/ml-explore/mlx) — Apple 官方 ML 框架，针对 Apple Silicon 优化
- [Open WebUI](https://github.com/open-webui/open-webui) — 自托管 ChatGPT 风格前端
- [Exo](https://github.com/exo-explore/exo) — 多 Mac 分布式推理集群
- [ROCm](https://rocm.docs.amd.com/) — AMD GPU 计算平台（CUDA 替代）
- 本仓库相关条目：[PagedAttention/vLLM](../../foundational/paged-attention/)、[Prefix Caching](../../foundational/prefix-caching/)、[DistServe](../../foundational/distserve/)、[Chunked Prefill](../../foundational/chunked-prefill/)、[Weight Quantization](../../foundational/weight-quantization/)、[verl](../../foundational/verl/)
