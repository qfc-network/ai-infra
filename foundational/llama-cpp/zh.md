# llama.cpp 与 GGUF

- **作者 / 机构**：Georgi Gerganov（ggerganov）及社区贡献者
- **代码仓库**：[ggml-org/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **首次提交**：2023 年 3 月（Llama 1 发布当周）
- **相关资料**：[GGUF 格式规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## 一句话总结

llama.cpp 是一个纯 C/C++ 编写的 Transformer LLM 推理引擎，零运行时依赖。它可以只跑在 CPU 上，只跑在 GPU 上（CUDA、Metal、Vulkan、OpenCL），也可以在 CPU+GPU 混合模式下运行——把模型各层分配到显存和系统内存。Ollama、LM Studio、Jan 以及数十款本地推理工具都以 llama.cpp 为底层引擎。配套文件格式 GGUF 将模型权重、分词器和所有配置打包进一个自描述二进制文件——加载时无需外部 `config.json`，无需 Python 导入链。llama.cpp 与 GGUF 共同定义了当今边缘/本地 LLM 推理的主流技术栈。

## 背景与动机

2023 年初 Meta 发布 Llama 1 时，最小可用模型是 70 亿参数，FP16 约 14 GB。在笔记本上运行需要大显存 GPU 或者可用于 CPU 的 INT4 量化实现，而这样的独立二进制当时并不存在。现有推理代码（PyTorch、Transformers）要拖入数 GB 的 Python 依赖，且从未针对 CPU 执行路径优化。

Gerganov 当时已有 GGML——一个面向 CPU 计算的 C 张量库。Llama 1 权重泄露后数日内，他就基于 GGML 产出了一个自包含 C 实现，加载量化权重、完全在 CPU 上运行推理。让它跑通的两个工程决策是：

1. **推理时零分配器开销**：所有权重张量通过 mmap 从磁盘映射进来，而非拷贝到堆内存；只有激活缓冲区在启动时 malloc。
2. **从一开始就有 Q4_0 量化**：4 位整数权重把 7B 模型从 14 GB 压到约 4 GB，能装进 2019 款 MacBook Pro 的内存。

整个项目一条 `make && ./main` 即可构建。社区立即大量采用。随后 18 个月内，它逐步增加了 CUDA、Metal（Apple GPU）、Vulkan、OpenCL 后端，支持所有主流开源模型，并加入了生产级 HTTP 服务器。到 2025 年，llama.cpp 已成为本地 LLM 安装量最大的推理引擎。

## GGUF 格式

GGUF（GPT-Generated Unified Format）于 2023 年 8 月取代早期的 GGML 格式，解决了向后兼容问题并扩展了元数据容量。二进制布局如下：

```
[magic: 4 字节 "GGUF"] [version: u32] [tensor_count: u64] [metadata_kv_count: u64]
[metadata: key-value 键值对，变长]
[tensor info 数组: 每个张量的名称、维度、数据类型、文件偏移]
[填充至对齐边界]
[tensor data: 原始字节，可 mmap]
```

元数据键值段是 GGUF 的决定性特性。标准字段包括：

- `general.architecture` — 字符串，如 `"llama"`、`"mistral"`、`"qwen2"`
- `general.name` — 人类可读的模型名称
- `llama.context_length` — 模型训练时的最大序列长度
- `llama.embedding_length` — 隐藏层维度
- `llama.attention.head_count`、`llama.attention.head_count_kv` — 注意力头配置
- `llama.rope.freq_base` — RoPE 基础频率
- `tokenizer.ggml.model` — 分词器类型，如 `"llama"`、`"gpt2"`、`"bpe"`
- `tokenizer.ggml.tokens` — 完整词表（字符串数组）
- `tokenizer.ggml.token_type` — 词元类型标志（normal、BOS、EOS、unknown、control）
- `tokenizer.ggml.scores` — SentencePiece 分数（如适用）

由于架构、分词器和全部超参数都存于文件内，GGUF 消费端无需任何外部配置。`llama_load_model_from_file()` 读取一个文件即返回完整配置的模型对象。

**加载时的内存映射**：tensor data 段通过 `mmap(MAP_SHARED)` 映射而非拷贝到堆内存。操作系统在首次访问时按需换入权重数据，内存压力大时可换出冷页。一个 Q4_K_M 版 70B 模型（磁盘约 40 GB），只有当前正在计算的层需要驻留物理内存。在 macOS 上，这与统一内存架构集成：Metal 后端可以直接在 GPU 地址空间中引用 mmap 区域，避免二次拷贝。

**文件命名规约**：社区约定在文件名中编码量化类型，例如 `Meta-Llama-3-70B-Instruct-Q4_K_M.gguf`。量化类型同样记录在每个张量的元数据中，运行时不依赖文件名解析。

## 量化类型

llama.cpp 定义了一套存于 GGUF 的量化类型体系，各类型在块结构、位数和 scale 精度上有所不同。

**Q4_0**：4 位均匀量化。权重分成 32 元素的块，每块有一个 FP16 scale 和 32 个 INT4 值。存储：`(16 + 32×0.5) = 32 字节` / 32 个权重 = 8 bpw（每权重比特数）。所有后端最快；现代方案中质量最低。

**Q4_K_M**：4 位 k-quant。"K"系列使用 256 元素超块，再细分为 32 元素子块。两级 scale：每 128 元素一个 FP16 精度的 scale（"M"代表"中等"scale 精度），每 32 元素子块一个量化 scale。额外的 scale 元数据通过适应超块内权重分布减少量化误差。同等 4 位平均位宽下，Q4_K_M 在 Llama 系列模型上通常比 Q4_0 低 0.1–0.3 困惑度点。这是一般用途的默认推荐。

```
Q4_K_M 超块（256 个权重）:
  2x FP16 scales（两个 128 元素半段各一个）  =   4 字节
  8x 量化子块 scales（每个 4 位）            =   4 字节
  256x 4 位权重                              = 128 字节
  合计：136 字节 / 256 个权重 ≈ 4.25 bpw
```

**Q5_K_M、Q6_K**：使用相同 k-quant 超块结构的 5 位和 6 位变体。Q6_K 以约 30% 更优的压缩比接近 Q8_0 精度，适合对质量敏感的任务。

**Q8_0**：8 位均匀量化。每 32 元素块一个 FP16 scale，存储 8.5 bpw。几乎无损——困惑度与 FP16 通常差距 < 0.05。模型文件比 Q4_K_M 大一倍，适用于精度不可妥协且内存/显存充裕的场景。

**IQ2_XXS、IQ3_S（重要性矩阵量化）**：一套独立的量化流程，使用校准数据集（"imatrix"，即重要性矩阵）。对每个权重，校准阶段计算其在样本数据上对层激活的贡献量。贡献大的权重分配更高精度，贡献小的分配更低精度。IQ2_XXS 在约 2.06 bpw 下大幅超越朴素 Q2，在部分基准测试上与 Q3_K_M 相当。代价是需要几小时的离线校准，换来极端压缩下的大幅精度提升。

**经验法则**：日常使用选 Q4_K_M（质量与速度的最佳平衡，通用性最强）；质量敏感场景选 Q6_K；近无损要求选 Q8_0；设备受限、Q4 过大时选 IQ3_S 或 IQ4_XS。

## CPU+GPU 混合 Offload

`--n-gpu-layers N`（简写 `-ngl N`）控制有多少个 Transformer 层被 offload 到 GPU。剩余层在 CPU 上使用系统内存运行。

**工作原理**：llama.cpp 为模型构建静态计算图。初始化时，最后 `N` 层（按惯例是 Transformer 栈中靠近输出的层）分配给 GPU 后端，前 `L - N` 层分配给 CPU 后端。推理时，前向传播先跑 CPU 层，再将激活张量传输到 GPU 显存，跑完 GPU 层后将结果传回。

**70B Q4_K_M 模型的内存算术**（量化后约 40 GB）：
- 每个 Transformer 层约 40 GB ÷ 80 层 = 500 MB
- 8 GB 显存可容纳约 14 个 GPU 层（其中 6–7 GB 可用于权重，其余留给 KV cache 和激活缓冲区）
- 其余 66 层跑在系统 RAM 中

**吞吐后果**：每秒生成的 token 数取决于两条路径中较慢的那条。现代笔记本（Apple M2 Pro，12 核）上 CPU decode 的典型速度是 70B 模型 2–6 tokens/s；同款硬件上 Metal/CUDA offload 层的速度是 20–80 tokens/s。瓶颈在 CPU 侧。实际测量中，配备 64 GB 统一内存的 M2 MacBook Pro 运行 70B 混合模式约达 5–12 tokens/s——满足交互使用，无法与服务器级 GPU 部署竞争。

Apple Silicon 的情况特殊：统一内存意味着 CPU 和 Metal GPU 共享同一物理内存。层传输没有 PCIe 拷贝开销。配备 128 GB 内存的 MacBook Pro M3 Max 可以把整个 70B Q4_K_M 模型（约 40 GB）保持在统一内存中，全部 80 层跑在 Metal 上，达到 20–40 tokens/s。

## Metal 后端（Apple Silicon）

Metal 后端通过 `MPSCommandBuffer` 和自定义 Metal 量化矩阵向量乘法 compute shader，将 GGUF 量化张量映射到 Apple GPU 计算单元上。decode 阶段（批大小为 1）的关键操作是矩阵向量积：单个 token 的激活向量乘以大型权重矩阵。Q4_K_M 下，每个权重在乘积累加前需要反量化。

Apple Silicon 上的性能受内存带宽而非 FLOPs 支配。M3 Pro 统一内存带宽约 150 GB/s。7B Q4_K_M 模型约 4.3 GB；以约 150 GB/s 的速度，一次前向传播读取模型需约 29 ms，理论上限约为 34 tokens/s。实测 M3 Pro 上 7B 的性能通常为 28–50 tokens/s，取决于上下文长度和 KV cache 大小。

Metal 后端避免了对 mmap 区域的显式显存管理：Metal 框架可以引用与 CPU mmap 相同的物理页，模型权重从不被复制。

## CUDA 后端

CUDA 后端在 prefill（批大小 > 1）阶段使用 cuBLAS 进行 GEMM，在 decode 阶段使用自定义 CUDA kernel 进行量化矩阵向量乘法。关键 kernel：

- `mul_mat_vec_q4_0_q8_1_cuda`：在单一 kernel 内对 Q4_0 权重即时反量化并与 Q8_1 激活做点积。每个线程块处理一行；warp 级规约累加部分和。
- `mul_mat_q4_K_cuda`：k-quant 变体；从独立缓冲区读取超块 scale 元数据，反量化后累加。

Prefill 使用 cuBLAS `cublasGemmEx` 进行 FP16 或 BF16 运算（纯 CUDA 模式下使用显存中的反量化副本），或在使用 flash-style 实现时使用量化 kernel。

RTX 4090（24 GB 显存）的近似吞吐量：
- 7B Q4_K_M（全量入显存，约 4.3 GB）：decode 约 130–180 tokens/s，prefill 约 2,500–4,000 tokens/s
- 13B Q4_K_M（约 7.6 GB）：decode 约 80–120 tokens/s
- 70B 需多 GPU 或与系统 RAM 混合

## KV Cache 内存与量化

KV cache 存储所有历史 token 在所有层的 key 和 value 张量。FP16 下，70B 模型上下文长度 8192 的 KV cache 约为：

```
KV cache 大小 = 2 (K+V) × context_len × n_layers × n_kv_heads × head_dim × 2 字节
= 2 × 8192 × 80 × 8 × 128 × 2 字节 ≈ 26.8 GB
```

这与权重内存直接竞争。长上下文下，KV cache 可能超过模型权重本身。`--cache-type-k q8_0 --cache-type-v q8_0` 将 KV cache 量化到 INT8，在同等上下文下将 KV 内存减半至约 13.4 GB。对多数任务精度影响极小；部分长上下文推理任务有轻微退化。`--cache-type-k q4_0` 可进一步压缩到约 7 GB，但精度风险更高。

## 用 llama-server 提供服务

`llama-server`（原名 `server`）在 `/v1/chat/completions`、`/v1/completions` 和 `/v1/embeddings` 上暴露 OpenAI 兼容 HTTP API。这是 Ollama 处理请求时调用的后端。

连续批处理（continuous batching）于 2024 年中期加入：服务器维护一个 slot 池；新请求进入空闲 slot，无需等待当前请求完成。在这种模式下，单个 llama-server 实例可以处理多个同时进行的用户请求，代价是内存增加（每个 slot 维护自己的 KV cache 片段）。

`llama-server` 没有实现动态块级 KV 共享（vLLM 通过 PagedAttention 实现）。每个 slot 获得静态分配的 KV 缓冲区。这简化了实现，但意味着多用户吞吐在高并发下的扩展效率不如 vLLM。

## 工程 Tradeoff

| 维度 | llama.cpp | vLLM | Ollama | TGI |
|---|---|---|---|---|
| 硬件要求 | CPU、任意 GPU 或混合；不需要 CUDA | 主要 CUDA（ROCm 尚属实验性） | 封装 llama.cpp 或 llama 后端；支持 Mac/Linux/Win | 主要 CUDA |
| 量化支持 | GGUF 原生：Q4_0、Q4_K_M、Q6_K、Q8_0、IQ2_XXS、FP16 | GPTQ、AWQ、FP8、BF16；不支持 GGUF | 继承 llama.cpp 量化方案 | GPTQ、AWQ、BNB；不支持 GGUF |
| 单用户 decode 吞吐 | 在本地硬件上较高；受内存带宽限制 | 单流下低于 llama.cpp | 与 llama.cpp 相同 | 与 vLLM 相当 |
| 多用户 / 高并发 | 2024 年加入连续批处理；无分页 KV 分配 | PagedAttention；业界最优的多用户方案 | 将请求队列到 llama-server | PagedAttention v2；多用户表现良好 |
| 部署复杂度 | 单个二进制 + 一个文件；`./llama-server -m model.gguf` 即可启动 | `pip install vllm`；需要 CUDA 驱动 | Docker 或安装包；UI 最简单 | Docker；依赖链较重 |
| 模型覆盖 | 覆盖所有主流开源模型；支持新架构最快 | 良好；在最新模型上落后 llama.cpp | 与 llama.cpp 相同（通过后端） | 良好；在最新模型上有滞后 |

## 交叉参考

- `../../guides/on-prem-llm-deployment/` — Ollama 封装 llama.cpp，是本地部署的推荐入口；该指南涵盖运维事项。
- `../weight-quantization/` — GPTQ 和 AWQ 是 GPU 服务器量化方法；GGUF k-quant 是 CPU/混合端的对应方案。AWQ 中保护显著通道的思路影响了 IQ 系列量化。
- `../../apple/afm/` — MLX 是 Apple 为 Apple Silicon 开发的替代推理框架；llama.cpp Metal 也面向同一硬件。MLX 侧重 Python 人体工程学，llama.cpp 侧重零依赖可移植性。
- `../paged-attention/` — vLLM 实现 PagedAttention 面向服务器级多用户推理。llama.cpp 面向本地/边缘的单用户或低并发场景；超过 10 个并发用户且有 GPU 资源时，vLLM 是更好的选择。

## 个人评注

llama.cpp 之所以成功，在于它在正确的时间以正确的约束解决了正确的问题：自包含、零依赖、能跑在手头实际有的硬件上。GGUF 格式把这一理念推进了一步——单个文件，无需任何生态即可打开。这种设计哲学是 llama.cpp 成为多数人私密体验 LLM 方式的根本原因。

技术天花板现在清晰可见：没有分页 KV 分配，llama.cpp 无法扩展到高并发服务。Metal 后端在 Apple Silicon 上已接近理论内存带宽上限，算法改进的空间有限。CPU 路径在大模型尺寸下饱和 DDR 带宽。这些不是缺陷——而是为本地执行而非服务器吞吐优化所付出的自然代价。

llama.cpp 给这个领域的启示是：**正确的抽象边界**（单个二进制、单个文件、硬件自适应量化）在采用率上可以胜过技术上更优越的系统。对构建本地或边缘推理的工程师而言，结论是：量化格式和部署摩擦远比数据中心基准测试里的峰值吞吐数字重要。

## 参考

- [1] Gerganov 等. llama.cpp 代码仓库. https://github.com/ggerganov/llama.cpp
- [2] GGUF 格式规范. https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- [3] Gerganov. GGML 张量库. https://github.com/ggerganov/ggml
- [4] Frantar 等. _GPTQ._ ICLR '23 / arXiv:2210.17323.（量化背景）
- [5] Lin 等. _AWQ._ MLSys '24 / arXiv:2306.00978.（激活感知 scaling；影响了 IQ 系列量化）
- [6] llama.cpp k-quants PR. https://github.com/ggerganov/llama.cpp/pull/1684（由 ikawrakow 引入的原始 k-quant）
- [7] Ollama. https://github.com/ollama/ollama（基于 llama.cpp 的本地服务方案）
