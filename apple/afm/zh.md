# Apple Foundation Models (AFM) 与 Private Cloud Compute

- **作者 / 机构**: Apple
- **发布时间**: 2024-06（Apple Intelligence 于 WWDC 2024 发布）；PCC 安全论文：2024-10
- **链接**: [Apple Intelligence 概览](https://www.apple.com/apple-intelligence/) · [Private Cloud Compute 安全论文](https://security.apple.com/blog/private-cloud-compute/) · [MLX（GitHub）](https://github.com/ml-explore/mlx) · [AFM 技术报告](https://machinelearning.apple.com/research/apple-foundation-models)

## 概述

Apple Foundation Models（AFM）是支撑 Apple Intelligence 的推理基础设施，于 2024 年 WWDC 正式发布。与当前所有主流前沿 AI 部署不同，AFM 的主要计算目标是终端用户设备——A17 Pro 或 M 系列芯片上的神经引擎——而非 GPU 数据中心。系统采用两层架构：简单请求路由至设备端约 3B 参数模型，复杂请求卸载至 Private Cloud Compute（PCC）——一套完全基于 Apple Silicon 构建的自研云基础设施，其核心约束是 Apple 工程师在结构上无法查看用户请求内容。

每一个系统层面的决策——量化方案、路由逻辑、PCC 的硬件选型、MLX 框架的设计——都围绕两个不可妥协的需求展开：设备端延迟低于 100 毫秒，以及云端可通过密码学手段验证的隐私保障。本文逐层追溯这些决策及其工程后果。

## 背景与动机

2024 年的 Apple 在战略位置上与 OpenAI、Google 或 Anthropic 截然不同。它没有公开的语言模型产品，但拥有以隐私为核心差异化的品牌积累，以及 20 亿台出货给消费者、内置神经引擎的活跃设备。在边缘侧运行推理——而非将其集中在 GPU 集群——是业务与工程的双重命题。这些约束从一开始就塑造了整个技术栈：

设备端内存预算极为有限。iPhone 16 的 8 GB DRAM 由操作系统、应用和神经引擎共享，模型必须在此约束内工作。网络延迟对于内联功能而言是不可接受的：邮件中的写作建议或通知摘要必须足够快，用户不应感知到网络往返。对大多数用户而言，通过云端往返实现 100 毫秒以内的总延迟在物理上不可行，本地推理是唯一出路。

Apple 的隐私承诺是产品层面的主张，不仅仅是政策层面的说辞。要使其具备可信度，基础设施本身必须可审计。"我们不记录请求"的政策声明远弱于在结构上阻止记录的硬件与软件设计。

AFM 的结果是：一个面向两个部署目标的模型家族，以及一个专为 Apple Silicon 设计的推理框架 MLX。

## 核心架构：两层服务体系

AFM 采用严格的两层服务模型。由设备端——而非服务端路由器——决定每个请求由哪一层处理。这一设计至关重要：客户端侧路由意味着服务端组件永远不会以明文形式接触请求内容，路由信号完全来自本地推理输出（请求复杂度估算、预期响应的 token 长度、任务类型）。

**第一层——设备端（AFM-on-device）**：约 3B 参数，量化为约 4-bit 调色板权重，占用大约 2 GB 设备 DRAM。处理绝大多数请求：写作改写、通知摘要、快速补全、Siri 短上下文查询。

**第二层——Private Cloud Compute（PCC）**：更大的模型（Apple 未公开确切规模；基于能力定位估计为 8B 至 70B 范围）运行在 Apple Silicon 服务器节点上。处理超出设备端能力的请求：长上下文摘要、复杂 Siri 请求、跨应用推理。

Apple 品牌功能不存在第三层——使用第三方 GPU 云的方案。PCC 完全由 Apple 自有自营，正是为了支撑下文所述的硬件认证模型。

## 设备端模型：量化与神经引擎

### 调色板化 4-bit 量化

标准 INT4 量化将每个权重独立映射到一个 4-bit 整数。Apple 公开的方案使用**调色板化（palettization）**：权重分组共享一个码本条目，码本通过训练过程中或训练后的向量量化步骤学习得到。其形式化表示如下：

对权重张量 W 按大小为 G 的块进行分区，每个块表示为大小为 2^b（b=4 时为 4-bit）的码本 C 中的一个索引：

    W_block ≈ C[k],   k ∈ {0, ..., 2^b − 1}

码本 C 与全精度模型联合或在训练后独立训练，通常通过 K-means 或可学习码本目标最小化重建误差。由于码本是学习得到的而非线性均匀分布，它能比朴素 INT4 更准确地捕捉每个权重块的实际分布，尤其对离群值表现更好。Apple Core ML Tools 的 `coremltools.optimize.torch.palettization` API 对外公开了这一能力。

内存影响：3B 参数模型在 BF16 精度下需要约 6 GB；在 4-bit 调色板化后，加上码本存储开销大约需要 1.5 至 2 GB。这是使 iPhone 16 设备端部署成为可能的内存预算。

### 神经引擎吞吐量

A17 Pro 神经引擎提供 35 TOPS（INT8），M4 神经引擎提供 38 TOPS。在实际推理中：

| 芯片 | 神经引擎 | 内存带宽 | 目标延迟（3B，4-bit） |
|------|----------|----------|----------------------|
| A17 Pro（iPhone 15 Pro） | 35 TOPS（INT8） | ~68 GB/s | ~80–120 毫秒/请求 |
| A18（iPhone 16） | 35+ TOPS | ~68 GB/s | ~80–100 毫秒/请求 |
| M4（Mac、iPad） | 38 TOPS | 120 GB/s | ~40–60 毫秒/请求 |
| M4 Max（Mac Studio） | 38 TOPS/集群 | 546 GB/s | <30 毫秒/请求 |

自回归解码的瓶颈是内存带宽而非算力——每生成一个 token 都需要从 DRAM 读取全部权重。较高的 TOPS 指标主要影响预填充阶段；实际吞吐量主要由上表中的带宽数字决定。

### 统一内存架构的优势

Apple Silicon 在推理方面最具决定性的基础设施优势是统一内存模型：CPU、GPU 和神经引擎共享同一个 DRAM 池，通过高带宽内部总线互连，CPU 与加速器之间不存在 PCIe 总线。其工程后果如下：

模型权重一旦加载到 DRAM，即可被神经引擎直接读取，无需显式复制或 DMA 传输。在离散 GPU 系统中，权重驻留在显存中，必须独立管理；任何需要权重数据的 CPU 侧操作都会产生 PCIe 复制开销。

给定相同 DRAM 容量时，有效模型容量更高。搭载 192 GB DRAM 的 M2 Ultra 可以完整容纳 70B BF16 模型；而配备 192 GB 系统内存和 H100（80 GB 显存）的服务器无法将模型完全载入 GPU 内存，必须进行卸载，产生 NVMe 或 PCIe 往返开销。

KV 缓存管理的逻辑也有所不同。设备端的 KV 缓存与权重共处同一 DRAM 池，缓存压力直接表现为对模型本身内存空间的竞争，而非独立的显存预算问题。

### 设备端推测性解码

Apple 在设备端采用推测性解码来提升神经引擎的 token 生成吞吐量。草稿模型先生成 K 个 token 提案，目标模型（完整大小的 AFM-on-device）通过一次批处理前向传播对所有 K 个 token 进行验证。草稿模型和目标模型均在设备端运行。约束在于：两个模型必须同时驻留在设备 DRAM 中，进一步收紧了内存预算。推测性解码算法详见 [`../../foundational/speculative-decoding/`](../../foundational/speculative-decoding/)。

## MLX：推理框架

MLX 由 Apple 于 2023 年 12 月开源，是 Apple Silicon 的推理（及训练）框架，功能定位类似 JAX 或 PyTorch，但从设计之初便围绕统一内存构建。

### 关键设计选择

**惰性求值与计算图编译。** 操作在调用时不立即执行，而是在计算图中累积，直到 `.eval()` 被调用或 Python 原语强制物化时才触发执行。这使得自动内核融合成为可能：相邻的逐元素操作、layernorm + linear 序列等模式被融合为单次 Metal 内核调度，效果类似于 CUDA 上带有 Inductor 后端的 `torch.compile`。CUDA 侧的类比参见 [`../../foundational/torch-compile/`](../../foundational/torch-compile/)。

**无需显式 `.to(device)` 调用。** 数组存在于单一地址空间中，将计算从 CPU 侧 Python 迁移至 GPU 侧 Metal 是调度决策而非数据搬移决策。这消除了一整类 bug，大幅简化了模型移植工作。

**一等公民量化支持。** `mlx.core.quantize` 实现了 4-bit 量化，包括分组量化。AFM-on-device 使用的调色板化方案是其生产演化版本——开源 MLX API 对外暴露了相关基础构建块。

**MLX-LM。** 构建于 MLX 之上的 LLM 推理高层库，支持 Llama、Mistral、Qwen、Phi 等开放权重模型家族，处理 KV 缓存、推测性解码和量化权重加载。在 Mac Studio 上部署 Llama 3 70B 的开发者使用的正是 MLX-LM。

### Mac 作为本地推理硬件

统一内存架构使高端 Mac 硬件在严肃的本地推理场景中具备实用价值：

| 硬件 | 统一内存 | 带宽 | Llama 3 70B INT4 吞吐量 |
|------|---------|------|------------------------|
| M2 Ultra（Mac Studio） | 192 GB | 800 GB/s | ~20–35 tokens/s |
| M4 Max（MacBook Pro、Mac Studio） | 128 GB | 546 GB/s | ~25–40 tokens/s |
| M4 Ultra（Mac Pro） | 192 GB | 800 GB/s | ~35–50 tokens/s |
| H100 SXM（单卡，对比参考） | 80 GB HBM3 | 3350 GB/s | ~80–120 tokens/s（batch=1） |

H100 在吞吐量上占优，但需要三万美元以上的服务器硬件、CUDA 工具链，且不适合没有数据中心的部署场景。对于无需数据中心运维开销的单用户本地部署，M2/M4 Ultra 在技术上具备竞争力。运维指南参见 [`../../guides/on-prem-llm-deployment/`](../../guides/on-prem-llm-deployment/)。

## Private Cloud Compute：隐私架构

PCC 是 AFM 技术栈中差异化程度最高的部分。其设计目标不仅是强隐私保护，而是*可验证*的隐私保护——一种即使面对被攻陷或怀有恶意的 Apple 基础设施团队仍然有效的结构性保证。

### 硬件基础

PCC 节点运行 Apple Silicon（基于时间节点和能力定位判断应为 M2 Ultra 或 M4 Ultra）。选用 Apple Silicon 并非偶然：它使 PCC 能够使用与 iPhone 相同的 Secure Enclave、硬件认证和安全启动链。NVIDIA GPU 节点无法提供同等的认证保证，原因在于 NVIDIA 的固件和硬件信任根不在 Apple 的控制范围内。

### 无状态处理

PCC 处理的每个请求均独立进行，请求生命周期结束后不持久化任何会话状态。运行于 PCC 节点的操作系统去除了所有可能积累用户数据的持久存储机制：请求内容不存在写入非易失性存储的路径。请求内容日志在结构上就不存在，而非仅靠管理制度禁止。

### 无特权访问

Apple 工程师无法 SSH 登录 PCC 节点并查看流量，这一限制在硬件层面强制执行：运行在 PCC 节点上的操作系统镜像经过签名，若镜像哈希与已发布的可审计值不符，Secure Enclave 会拒绝启动。允许临时代码执行的机制——调试接口、特权 shell 访问——均从生产镜像中移除。这一约束不是政策层面的"工程师不被允许这样做"，而是能力层面的"系统根本不支持这样做"。

### 密码学认证与可验证透明度

在用户设备向 PCC 发送任何数据之前，它会执行一次密码学认证检查，流程如下：

1. PCC 节点出示一份以其 Secure Enclave 硬件密钥为根的认证证书。
2. 证书将节点的硬件身份绑定到当前运行的操作系统镜像哈希。
3. 用户设备对照由 Apple 维护、可供独立安全研究人员审计的透明日志，验证操作系统镜像哈希。
4. 仅当认证有效且操作系统镜像存在于已发布的透明日志中，设备才会加密并发送请求数据。

这意味着 Apple 自身网络基础设施发起的中间人攻击在结构上被阻断：拦截流量只能得到密文，而在特定认证节点的 Secure Enclave 密钥之外无法解密。用被篡改的操作系统镜像替换 PCC 节点会破坏认证，设备将拒绝发送数据。

Apple 在透明日志中公开发布 PCC 软件镜像，使外部安全研究人员能够审计运行的代码。这是"信任但验证"模型的工程化落地：隐私主张不是"Apple 这样说"，而是"代码在这里，自行验证"。

### 路由逻辑的深层含义

由于设备侧路由确保服务端永远不会接触未加密的请求内容来做路由决策，PCC 无法基于请求语义实现服务质量分层。路由决策（设备端还是 PCC）完全在本地由设备根据请求长度、任务类型和预估模型能力需求做出。这是一个实质性约束：它阻止了基于请求内容的服务端 A/B 测试，也阻止了 Apple 基于哪些请求路由至 PCC 来构建用户行为画像。

## AFM 服务端：PCC 上的推理

运行在 PCC 上的服务端模型使用相同的 Apple Silicon 推理栈，但操作参数有所不同：

**适配 M 系列 GPU 的 Flash Attention 变体。** 标准 Flash Attention（FA2/FA3）假设 NVIDIA 硬件上的 HBM-SRAM 分块架构。Apple Silicon 的内存层次结构不同：统一 DRAM 配以较大的 L2 缓存和 GPU tile 内存（功能上类似 SRAM）。Apple 针对此层次结构对 Flash Attention 进行了适配。

**KV 缓存管理。** 在 PCC 节点上，KV 缓存与模型权重共处同一 DRAM 池。与 H100 上的 vLLM（KV 缓存在独立显存池中由 PagedAttention 管理）不同，Apple 的 KV 管理必须考虑统一内存压力。缺少 HBM 意味着在大批量场景下 KV 缓存读取的峰值带宽更低，这是 PCC 相对于 H100 部署方案的吞吐量上限。

**基于适配器的任务专项微调。** AFM 对写作工具、摘要、通知分类及其他 Apple Intelligence 功能采用适配器专项化方案（LoRA 风格）。基础模型权重共享，适配器体积小，可按请求动态切换。详见 [`../../foundational/lora/`](../../foundational/lora/)。

## 工程权衡

| 维度 | 设备端（AFM ~3B） | Private Cloud Compute | Cloud GPU（vLLM / TGI on H100） |
|------|-------------------|----------------------|--------------------------------|
| 隐私 | 最强——数据不离设备，无网络暴露 | 强——无状态，密码学认证，工程师无访问权限 | 标准——依赖服务商信任；可能存在日志；仅有 GDPR/合同层面控制 |
| 延迟 | <100 毫秒（无网络往返） | ~200–500 毫秒（等效 Apple 数据中心局域网） | 500 毫秒–5 秒（广域网往返 + 排队 + 预填充） |
| 模型规模 | ~3B 参数，4-bit 调色板化，~2 GB | 估计 ~8B–70B | 无上限（水平扩展） |
| 硬件 | 神经引擎（35–38 TOPS），68–546 GB/s 带宽 | Apple Silicon M 系列（与设备同 ISA） | H100/A100，HBM3 带宽 3.35 TB/s |
| 吞吐量 | 单用户，串行请求 | 每节点估计 ~10–100 并发 | 连续批处理支持数千并发 |
| 成本模型 | 零边际成本（设备已购入） | 打包在 Apple 设备成本中，无按 token 计费 | 按 token 付费，规模化后运营成本高 |

## 可复现性说明

Apple 尚未公开完整的 AFM 训练方案、模型架构细节或 PCC 集群规模。已公开的产出包括：

- Apple Intelligence 功能集（通过 iPhone 16 / macOS Sequoia 可公开观察）。
- PCC 安全研究论文（Apple Security Research 博客，2024 年 10 月），详细描述了隐私架构。
- MLX（完全开源，Apache 2.0 许可证）；生产端 AFM 推理路径使用内部版本，包含额外优化。
- Core ML Tools 及调色板化 API（开源）。
- Apple Machine Learning Research 上的 AFM 技术报告（2024 年 6 月）：描述了模型能力和部分架构选择，但未涉及训练数据或集群规格。

外部研究人员可验证的内容：PCC 透明日志（操作系统镜像哈希）、认证协议，以及已发布 PCC 镜像中特权访问机制的缺失。

## 评述

AFM 的工程意义不在于模型本身——3B 设备端模型并非前沿能力水平。其意义在于将 AI 功能部署到 20 亿台设备所依赖的系统架构，且须满足一个没有其他前沿实验室尝试过在结构上强制执行的隐私约束。

以下三点值得特别指出：

**PCC 认证模型在生产规模上真正具有创新性。** 硬件信任根认证、已发布软件透明性与无特权访问强制执行的组合，在云端 AI 领域没有直接先例。Google 和 Microsoft 做出隐私政策承诺；Apple 则设计了一个破坏这些承诺需要攻破硬件认证的系统。

**统一内存是设备端推理的真实架构优势，不是营销话术。** 在手机上以 4-bit 量化运行 3B 模型并实现 100 毫秒以内的延迟，是离散 GPU 系统在同等 TDP 下根本无法复制的带宽与内存架构优势。MLX 使这种优势可编程化。

**吞吐量上限是真实存在的。** PCC 的 Apple Silicon 节点峰值内存带宽低于 H100 集群。对于大规模批量推理，Apple Silicon 与 NVIDIA 不具竞争力。Apple 的赌注在于：个人 AI 功能的隐私和延迟需求使批量吞吐量成为错误的衡量指标——对于消费设备功能而言，这个赌注大概率是正确的。对于更高计算强度的工作负载（长上下文推理、多模态生成），这个赌注能否继续成立，将是下一代硬件面临的开放性问题。

## 参考文献

- [1] Apple，"Private Cloud Compute: A new frontier for AI privacy in the cloud"，Apple Security Research 博客，2024 年 10 月。
- [2] Apple，"Introducing Apple's On-Device and Server Foundation Models"，Apple Machine Learning Research，2024 年 6 月。
- [3] Apple，MLX: An array framework for machine learning on Apple Silicon，GitHub（2023 年 12 月开源）。https://github.com/ml-explore/mlx
- [4] Leviathan 等，"Fast Inference from Transformers via Speculative Decoding"，ICML 2023。arXiv:2211.17192。
- [5] Hu 等，"LoRA: Low-Rank Adaptation of Large Language Models"，ICLR 2022。arXiv:2106.09685。
- [6] Apple，Core ML Tools — Palettization API。https://coremltools.readme.io/docs/palettization-overview
- [7] Dao 等，"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"，ICLR 2024。arXiv:2307.08691。
