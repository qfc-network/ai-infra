# TensorRT-LLM — NVIDIA 生产级大语言模型推理引擎

- **作者 / 机构**: NVIDIA TensorRT 团队
- **发布时间**: 2023-09（开源发布）
- **链接**: [代码](https://github.com/NVIDIA/TensorRT-LLM) · [文档](https://nvidia.github.io/TensorRT-LLM/) · [博客](https://developer.nvidia.com/blog/accelerating-inference-with-tensorrt-llm/)

## TL;DR

TensorRT-LLM 是 NVIDIA 开源的大语言模型推理优化库，运行于 NVIDIA GPU 之上。其核心思路是将模型定义编译为 TensorRT 引擎——融合算子内核、in-flight 批处理（连续批处理）、分页 KV 缓存、多 GPU 张量/流水线并行，以及 INT4/INT8/FP8 量化——所有功能通过封装 CUTLASS 和自定义 CUDA 内核的 Python API 对外暴露。它是 NVIDIA Triton 推理服务器的推理后端，也是 H100 LLM 吞吐量的官方参考实现。

## 背景与动机

2023 年中期，LLM 推理生态高度碎片化，各工具各有短板：

- **vLLM**：通过 PagedAttention 实现了出色的吞吐量，但调度器以 Python 实现，量化支持有限，缺乏引擎编译能力。
- **FasterTransformer**（NVIDIA 自己的前身）：彼时性能领先，但属于研究级实现，难以扩展，且与特定模型架构紧耦合。
- **HuggingFace Transformers + Accelerate**：灵活性最强，但因缺乏算子融合、基于 PyTorch eager 执行，比手工优化引擎慢 3–5 倍。

规模化运营的场景——云厂商、NVIDIA 合作伙伴、前沿模型部署——同时需要四件事：(1) 在 A100/H100 上达到硬件峰值利用率；(2) 生产级可靠性，尾延迟可预测；(3) 涵盖 INT4、INT8、FP8、SmoothQuant 的全面量化支持；(4) 从单机到多节点的便捷多 GPU 扩展。当时没有任何一款工具能同时满足这四点。

FasterTransformer 已在 NVIDIA 内部使用，但它是围绕具体模型架构有机生长出来的，而非面向可组合推理平台的系统设计。与其持续打补丁，NVIDIA 决定从零构建 TensorRT-LLM，并于 2023 年 9 月开源，将其定位为 NVIDIA 硬件上的生产推理参考栈。它同时也是 NVIDIA NIM（前身为"AI Foundations"）服务层和 Triton 推理服务器 LLM 场景的推荐后端。

## 核心方法

### 编译流水线

TensorRT-LLM 的工作分为三个明确阶段：

**1. 模型定义。** 用户用 Python 以 `tensorrt_llm.Module` 描述模型——这是一套类似 PyTorch `nn.Module` 的 TensorRT 专属模块 API，但构建的是 TensorRT 函数式算子图，而非 eager 执行。各模型提供从 HuggingFace checkpoint 加载权重的转换脚本；在图的层面，权重被视为嵌入引擎的常量。

**2. 引擎构建。** `trtllm-build` 命令接收模型图，经 TensorRT 优化器处理（层融合、精度选择、CUTLASS tile 配置搜索），序列化为 `.engine` 文件。所有代价高昂的优化均在此阶段完成：TensorRT 对每个候选内核在各相关输入形状下逐一 profile，为每个 GEMM 选出最快的 tile 配置，将相邻逐元素算子融合进单个 CUDA 内核，并将量化/反量化内联进周边的矩阵乘内核。构建时间从 7B 模型的数分钟到四卡 70B 模型的 30–60 分钟不等。

**3. 运行时。** C++ 运行时加载序列化引擎并驱动推理，负责调度器、KV 缓存分配器和 token 流式输出逻辑。通过 `tensorrt_llm.runtime` 提供 Python 绑定，暴露 `GenerationSession` 接口。生产部署通常通过 `tritonserver` + `tensorrtllm_backend` 插件访问 C++ 运行时。

### In-Flight 批处理（连续批处理）

TensorRT-LLM 通过自定义 C++ 调度器实现连续批处理，而非采用 Orca 的 Python 迭代级调度器。每个解码步骤，调度器检查所有活跃序列的完成状态，驱逐已完成的序列，并将等待队列中的预填充请求提升进活跃 batch——全部在 C++ 运行时的同步调用中完成。因此 TensorRT 引擎每步看到的 batch 形状会发生变化，这通过引擎构建时配置的动态形状（`--max_batch_size`、`--max_input_len`、`--max_output_len`）来处理。

### 分页 KV 缓存

运行时维护一个 KV 缓存块池。每个块存储固定数量（可配置，通常 16–64）的 token 对应的所有层、所有头的 key 和 value 张量。新请求到来时按需从池中分配块；请求完成后立即归还。注意力内核接收块指针表而非连续 KV 张量，实现与 vLLM 相当的 PagedAttention 内存效率。跨请求 KV 共享（前缀缓存）同样支持：两个请求若共享相同系统提示，其前缀块可以别名共享。

### 融合注意力内核

核心注意力实现是基于 CUTLASS MMA（矩阵乘累加）原语的自定义 CUDA 内核，或可选用 FMHA（来自 APEX 的融合多头注意力）。单次内核调用通过 `cuSeqLens` 描述每序列长度，处理完整的变长 batch——短序列无 padding 浪费。在 Hopper（H100）上，TensorRT-LLM 切换为使用 WGMMA 和 TMA 的 Hopper 原生内核，接近 FlashAttention-3 的吞吐量。

单头注意力的有效计算遵循标准缩放点积形式，但内核将 softmax 缩放、掩码及 dropout 随机掩码（若启用）融合进对 KV tile 的同一遍历中，永远不将完整注意力权重矩阵写入 HBM：

```
# 伪代码：对每个 Q tile，遍历 K/V tile
m_i = -inf; l_i = 0; O_i = 0
for 每个 K/V tile j:
    S_ij = Q_i @ K_j^T / sqrt(d)    # 片上计算，tile 大小
    m_ij = rowmax(S_ij)
    P_ij = exp(S_ij - m_ij)
    # 在线 softmax 修正
    m_new = max(m_i, m_ij)
    l_i = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * rowsum(P_ij)
    O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
    m_i = m_new
O_i = O_i / l_i                     # 最终归一化
```

结构与 FlashAttention 完全相同；CUTLASS 实现达到同等 IO 感知效果，但使用 NVIDIA 自己的内核基础设施而非 Dao-AILab 的实现。

### 量化

量化在引擎构建时通过 `trtllm-build` 参数和独立校准步骤进行配置：

- **INT4 纯权重量化**（GPTQ 风格或 AWQ 风格）：权重以 INT4 存储，在矩阵乘内核中通过 CUTLASS 混合精度 GEMM 内联反量化为 FP16/BF16。分组量化（group size 128）是保证精度的标准配置。无激活量化；计算仍为 FP16/BF16。
- **INT8 权重 + 激活（SmoothQuant）**：权重和激活均量化为 INT8，矩阵乘在 INT8 张量核心（IMMA）上执行。SmoothQuant 的迁移因子将逐通道激活缩放吸收进权重，以最小精度损失实现逐张量或逐通道激活量化。适用于 Ampere 及以后架构。
- **FP8（仅限 Hopper）**：权重和激活以 E4M3 FP8 存储和计算。TensorRT-LLM 是 Hopper FP8 LLM 推理的参考实现。缩放因子为逐张量或逐通道；在构建时吸收进引擎。

反量化和缩放均融合进周边矩阵乘内核——不存在独立的"反量化"步骤将权重回写至 HBM。

### 张量并行与流水线并行

张量并行沿头维度或隐藏维度将 QKV 投影和 FFN 权重矩阵分片至多 GPU。每次矩阵乘后通过 NCCL AllReduce 同步偏结果。张量并行度在构建时指定；每种 TP 配置需单独构建引擎。流水线并行将连续的层区间分配给不同 GPU（或节点），通过微批次流水线隐藏跨阶段激活传输延迟。TP 与 PP 可组合使用：4 节点 × 8 GPU 集群可在节点内运行 TP=8，跨节点运行 PP=4，构成 32 GPU 的逻辑设备。

### 推测解码

TensorRT-LLM 内置推测解码（draft+verify）支持。草稿模型（如 7B 模型）作为独立 TensorRT-LLM 引擎运行，每步生成 k 个候选 token。目标模型在单次前向中验证全部 k 个 token（对草稿候选的批量操作）。拒绝采样步骤在 CPU 或小型 CUDA 内核中执行。最终效果是墙钟延迟随每步接受 token 数而非 k 扩展，对草稿接受率高的任务（代码生成、结构化输出）实现 2–3× 加速。

### 插件系统

自定义 CUDA 操作——注意力、LayerNorm、量化矩阵乘、旋转位置编码——以 TensorRT 插件形式注册：实现 `IPluginV2DynamicExt` 接口的 C++ 类。该接口使 TensorRT 图优化器将自定义算子视为图中的一等节点，在有利时与周边标准算子融合。插件注册模式如下：

```cpp
class FMHAPlugin : public nvinfer1::IPluginV2DynamicExt {
    nvinfer1::DimsExprs getOutputDimensions(...) override;
    void enqueue(const PluginTensorDesc* inputDesc,
                 const PluginTensorDesc* outputDesc,
                 const void* const* inputs,
                 void* const* outputs,
                 void* workspace,
                 cudaStream_t stream) override;
    // ... 序列化 / 反序列化 / 克隆
};
```

插件模型是 TensorRT 的可扩展机制，但代价真实存在：插件 API 版本随 TensorRT 发版而变，跨主版本向后不兼容。

## 工程权衡

| 决策 | 获得 | 放弃 |
|---|---|---|
| 预编译引擎（AOT compilation） | 通过穷举 profile 实现峰值内核选择；服务时零 JIT 开销；延迟可复现 | 大模型构建耗时 10–60 分钟；引擎不可跨 GPU 世代移植（A100 引擎无法在 H100 运行）；任何模型变更需完整重建 |
| C++ 运行时（vs. Python，如 vLLM） | 更低的单 token 延迟；更好的尾延迟；无 GIL 争用的生产级调度 | 定制困难；Python API 是薄封装；调试调度逻辑或内存管理需要 C++ 能力 |
| TensorRT 插件系统 | 自定义算子作为图节点参与一等融合；可与周边逐元素层完整融合 | API 冗长；TensorRT 主版本间存在 breaking change；跨版本移植插件需手动操作 |
| 编译时量化（INT4/INT8/FP8） | 单一框架覆盖全栈；运行时量化零开销；吞吐量数字稳定可预测 | 校准必须离线提前完成；切换量化方案需重新构建引擎；灵活性不如 vLLM 的运行时量化 API |
| 仅支持 NVIDIA 硬件 | 直接使用 WGMMA、TMA、FP8 张量核心、NCCL；H100 参考性能；驱动与编译器紧密协同 | 不支持 AMD/CPU/Intel GPU；完全供应商锁定；非 NVIDIA 硬件用户必须使用其他栈 |
| 动态形状引擎（vs. 静态） | 单引擎在配置上限内处理变长 batch/序列 | 构建时需对预期形状范围进行 profile；偏离 profile 最优点的形状可能性能次优 |

## 实验与结果

**LLaMA-2-70B，4× A100-80GB，FP16（batch 64）：** TensorRT-LLM 达到约 2,100 tokens/sec 生成吞吐量，vLLM 约 1,400 tokens/sec，优势约 1.5×。差异主要源于更紧密的算子融合、预编译 GEMM 调优，以及调度器 Python 开销更低。

**LLaMA-2-70B，4× H100-80GB，FP8（batch 64）：** 约 4,500 tokens/sec——接近 H100 FP8 张量核心理论峰值。这是 TensorRT-LLM 优势最明显之处：它是目前唯一端到端充分利用 Hopper FP8 的开源栈。

**INT4 纯权重量化（AWQ）：** 内存占用降低 1.9×，在标准基准（MMLU、HellaSwag）上质量损失低于 1%，使 LLaMA-2-70B 可在 2× A100（而非 4×）上运行。这是成本敏感部署中最常见的生产配置。

**推测解码（7B 草稿 + 70B 目标）：** 在 HumanEval 代码生成任务上实现 2.2× 延迟降低——该任务草稿接受率高（60–75%），因为代码具有局部可预测性。对开放式创意生成等接受率低的任务，提升优雅降级至接近 1.0×。

**batch=1 时的延迟：** TensorRT-LLM 编译引擎在 A100 上处理 LLaMA-2-7B 的首 token 时间约为 30–35 ms，同配置 vLLM 约为 45–50 ms——对单请求延迟比聚合吞吐量更重要的交互式应用场景尤为关键。

## 可复现性说明

**代码**：github.com/NVIDIA/TensorRT-LLM（Apache 2.0 许可）。

**Docker**：从 NVIDIA NGC 获取 `nvcr.io/nvidia/tensorrt-llm`。这是推荐起点；从源码构建需要仔细匹配 CUDA/TensorRT 版本。

**构建流程**：
```bash
# 转换 HuggingFace checkpoint
python convert_checkpoint.py --model_dir <hf_dir> --output_dir <ckpt_dir> \
    --tp_size 4 --dtype float16

# 构建 TensorRT 引擎
trtllm-build --checkpoint_dir <ckpt_dir> \
    --output_dir <engine_dir> \
    --max_batch_size 64 --max_input_len 2048 --max_output_len 512 \
    --gemm_plugin float16 --gpt_attention_plugin float16

# 通过 Python 运行时推理
python run.py --engine_dir <engine_dir> --tokenizer_dir <hf_dir> \
    --input_text "你好，世界"
```

**Triton 部署**：通过 `tensorrtllm_backend` 仓库与 `tritonserver` 集成，后端负责 gRPC 协议、服务端动态批处理和健康检查。

**环境要求**：TensorRT 9.x 或 10.x，CUDA 12.x，Python 3.10+。INT8 SmoothQuant 需要 Ampere 或更新架构。FP8 需要 Hopper（H100、H200）或 Ada Lovelace（消费端 RTX 40 系列）。

**模型支持**（截至 2024 年）：LLaMA 1/2/3、Mistral、Mixtral（MoE）、Falcon、GPT-NeoX、GPT-J、Gemma、Phi-1/2/3、Qwen 1/2、Baichuan、ChatGLM、BLOOM、OPT、Whisper。

**已知摩擦点**：引擎构建对 TensorRT 版本敏感；TRT 9.x 构建的引擎无法被 TRT 10.x 运行时加载。`examples/` 中的模型转换脚本是规范参考且迭代快速——生产构建建议固定到特定 commit。

## 评述

TensorRT-LLM 是 NVIDIA 对碎片化推理生态的主动回应：将编译、量化、批处理和多 GPU 服务统一在一个框架下，形成单一、优化、开源的生产推理栈。其核心设计赌注在于：预编译——以较长的构建时间换取峰值运行时性能——是生产服务的正确权衡，因为引擎会运行数周，构建成本在数百万次请求上被充分摊销。

这与 vLLM 的思路形成鲜明对比：保持全 Python 实现，接受适度开销，最大化迭代速度和生态宽度。实践中两套系统在生产中共存：超大规模服务商在高 QPS 服务层使用 TensorRT-LLM，在研究集群和低流量端点使用 vLLM。这种分工映射了长期存在的 CUTLASS vs. Triton 格局——CUTLASS/TRT-LLM 用于生产峰值性能；Triton/vLLM 用于研究灵活性。

TensorRT-LLM 优势最清晰、最持久的领域是 H100 上的 FP8 路径。FP8 需要 Hopper 专属硬件（FP8 GEMM 单元与 Ampere 上的 INT8 张量核心不同），而 TensorRT-LLM 拥有最成熟的校准工具链、最完整的模型覆盖和最紧密的驱动协同设计。任何在 H100 上规模化部署 LLM 的团队，都应将 TensorRT-LLM 视为其他方案必须超越的性能基线。

长期来看，主要问题是随着模型多样性和更新节奏的增加，预编译能否保持可行性。仅权重不同的微调变体可以复用同一引擎（权重与编译图分离加载），这在很大程度上缓解了该问题。但架构变化——新注意力变体、MoE 路由、SSM 层——每种都需要新的插件实现和引擎重建。NVIDIA 团队推进迅速，但"每架构一插件"的税负是真实存在的，并将随模型园的扩展持续增长。

## 参考文献

- [1] NVIDIA 2023. _TensorRT-LLM: A TensorRT Toolbox for Optimized Large Language Model Inference._ github.com/NVIDIA/TensorRT-LLM.
- [2] NVIDIA 2022. _FasterTransformer._ github.com/NVIDIA/FasterTransformer（前身，现已归档）。
- [3] Kwon et al. 2023. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ arXiv:2309.06180.
- [4] Yu et al. 2022. _Orca: A Distributed Serving System for Transformer-Based Generative Models._ OSDI 2022.
- [5] Lin et al. 2023. _AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration._ arXiv:2306.00978.
- [6] Xiao et al. 2023. _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models._ ICML 2023.
- [7] Dao et al. 2022. _FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness._ arXiv:2205.14135.
- [8] NVIDIA 2024. _TensorRT-LLM 文档._ nvidia.github.io/TensorRT-LLM.
