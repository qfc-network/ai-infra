# LMDeploy / TurboMind

- **作者 / 机构**：上海人工智能实验室（OpenMMLab）
- **代码仓库**：[InternLM/lmdeploy](https://github.com/InternLM/lmdeploy)
- **首次公开发布**：2023 年 7 月
- **后端选项**：TurboMind（C++/CUDA，默认）、PyTorch（Python）

## 一句话总结

LMDeploy 是上海人工智能实验室在研发 InternLM 和 Qwen 的同时打造的生产级 LLM 服务引擎。其默认的 TurboMind 后端是一个自定义 C++/CUDA 引擎，内含针对 GQA 注意力、W4A16 AWQ 量化、分页 KV cache 手工调优的 CUDA kernel，以及——在开源服务引擎中独树一帜地——对 DeepSeek MLA（多头潜在注意力）及其压缩 KV cache 布局的高效原生支持。TurboMind 上的 W4A16 AWQ 相较 BF16 在单流 decode 吞吐上提升约 2 倍；H100 基准测试显示，Llama 3 70B 在批大小 32 下 AWQ 可达约 5,500 tokens/s，BF16 约 3,500 tokens/s，优于同硬件上的 vLLM。对于国产实验室模型（InternLM、Qwen、DeepSeek），LMDeploy 是第一优先级的部署方案。

## 背景与动机

2023 年中，开源服务领域已有 TGI（HuggingFace）、初生的 vLLM（UC Berkeley）和 FasterTransformer（NVIDIA）。三者各有短板：TGI 缺乏量化优化 kernel；vLLM 有 PagedAttention 但 kernel 层面无权重量化；FasterTransformer 需要模型转换且起源自 NVIDIA 内部。

上海人工智能实验室面临具体的约束。InternLM 模型在内部训练和部署；Qwen（阿里巴巴，同城且有研究交集）正在成为重要的模型系列。实验室需要一个能够满足以下条件的服务引擎：

1. 对 W4A16 AWQ kernel 有第一优先级支持——这是他们部署目标的主要量化方式。
2. 在 80 GB H100 集群（内部服务）和更小显卡（用户外部部署）上均有良好表现。
3. 能快速支持多样化架构——InternLM、Qwen、后来的 DeepSeek——而非每个模型耗费数月的工程投入。
4. 最终需要支持 MLA，其 KV cache 布局在结构上与标准多头注意力不兼容。

TurboMind 正是为满足这些约束而设计的。PyTorch 后端则用于覆盖尚未完成自定义 CUDA kernel 开发的模型系列。

## TurboMind 后端架构

TurboMind 是一个带有自定义 CUDA kernel、分页 KV cache 管理器和连续批处理调度器的 C++ 推理运行时。一个典型请求的执行路径如下：

1. **请求进入调度器**：调度器从物理块池中分配可用 KV 块，创建序列槽位（slot）。
2. **Prefill 阶段**：token ID 被 embedding；前向传播通过所有层。W4A16 模型的权重以 INT4 存储，在 GEMM kernel 内即时反量化；BF16 模型由 cuBLAS 处理密集矩阵乘法。
3. **Decode 循环**：每一步，单 token 的激活穿过整个模型。关键操作是量化矩阵向量乘（GEMV）。TurboMind 的 W4A16 GEMV kernel 将反量化与点积融合；激活全程保持 FP16/BF16。
4. **KV cache 更新**：注意力计算后，新的 K 和 V 张量写入已分配的物理块。
5. **输出**：采样得到的 token 返回给调用方；该槽位的 block table 为下一步扩展。

调度器是迭代级的（连续批处理）：每个 decode 步都检查已完成的序列和新到来的请求，而非在 batch 边界处处理。

## W4A16 AWQ 的吞吐优势

权重-only INT4 量化（W4A16）显著提升单流 decode 速度，因为 decode 是内存带宽受限的，而非算力受限的。

在批大小为 1 时，每个 decode 步需加载整个模型权重一次以计算一个输出 token。性能方程如下：

```
tokens/s ≈ HBM 带宽 / 模型大小（字节）

BF16 70B：140 GB / （H100 SXM 3.35 TB/s）≈ 24 ms/token → 约 42 tokens/s
W4A16 70B：约 37 GB / （3.35 TB/s）≈ 11 ms/token → 约 90 tokens/s
```

实测中，TurboMind 在批大小 1 下 H100 上 70B W4A16 decode 约达 80–100 tokens/s，BF16 约 35–45 tokens/s——大约 2 倍的比率，与理论预测一致。W4A16 能击败朴素 W4A16 实现的原因在于自定义 GEMV kernel：标准 CUDA INT4 矩阵乘法是为批量 GEMM（prefill）设计的，而非 decode 的流式 GEMV 访问模式。TurboMind 的 kernel 读取 128 位对齐的 INT4 瓦片，每次用位操作解包 8 个值，与 FP16 激活做融合乘加。Warp 级规约累加部分和，无需全局显存往返。

随着批大小增大，优势收窄。批大小 32 时，BF16 和 W4A16 都越来越受算力约束（矩阵乘法不再是带宽瓶颈），比率缩小至约 1.3–1.5 倍。TurboMind 在 H100 上对 Llama 3 70B 的基准数据：

- **BF16，批大小 32**：约 3,500 tokens/s 总吞吐
- **W4A16 AWQ，批大小 32**：约 5,500 tokens/s 总吞吐
- **同款 H100 上 vLLM BF16，批大小 32**：约 3,200 tokens/s

vLLM 与 LMDeploy 的对比并非完全在同等条件下（调度和开销特征不同），但这一方向在 2025 年初的多项第三方基准测试中保持一致。

## MLA 支持

DeepSeek V2 引入了多头潜在注意力（MLA），通过将 key 和 value 投影到更低维度的潜空间来压缩 KV cache。标准多头注意力每个 token 每层存储 `n_heads × head_dim` 的 K 和 V 张量；MLA 存储一个维度为 `kv_lora_rank`（例如 DeepSeek V2 为 512）的单一压缩潜向量，外加一个小的 rope 分量 key。

对服务引擎的影响：物理 KV 块布局必须改变。以 `[n_heads, head_dim]` 为每 token 单位存储 KV 的系统，无法直接存储 MLA 的 `[kv_lora_rank]` 压缩潜向量。vLLM 最初通过在注意力 kernel 边界将 MLA 展开回完整 K/V 来处理，保留了大体量 KV cache。TurboMind 从一开始就实现了原生 MLA KV 布局，直接将潜向量存入 KV 块。

内存节省相当可观。对于 DeepSeek V2（236B MoE），其中 `kv_lora_rank = 512`，`n_kv_heads × head_dim = 128 × 128 = 16384`：

```
标准 KV，每 token 每层：16384 × 2（K+V）× 2 字节 = 65,536 字节
MLA KV，每 token 每层：  512 × 2（K+V）× 2 字节 =  2,048 字节
压缩比：约 32 倍
```

在上下文长度 8192、61 个注意力层（DeepSeek V2）的条件下，单流 KV cache 从约 32 GB 降至约 1 GB。这正是在合理的 GPU 预算内提供 MoE 模型服务成为可能的原因。

LMDeploy 是最早实现这一 MLA 原生路径的生产级服务引擎之一，这使其在 DeepSeek 部署上具有早期优势。该实现将潜向量向 key-head 和 value-head 空间的上投影融合在注意力 kernel 内部，避免了 Python 侧的独立投影步骤。

## 量化流水线

LMDeploy 提供了一套集成的量化命令行工具：

```bash
# W4A16 AWQ 校准与量化
lmdeploy lite auto_awq \
    model_path \
    --calib-dataset ptb \
    --calib-samples 128 \
    --calib-seqlen 2048 \
    --work-dir quantized_model_path

# W8A8 SmoothQuant
lmdeploy lite smooth_quant \
    model_path \
    --calib-dataset ptb \
    --calib-samples 128 \
    --work-dir quantized_model_path
```

AWQ 路径使用与 AutoAWQ 相同的通道 scaling 校准方法（在量化前放大显著权重通道，将逆 scale 吸收进前一层），但量化后的权重以 LMDeploy 自有格式存储，由 TurboMind 的 W4A16 kernel 消费，而非 AutoAWQ 的 kernel。两者的 kernel 实现不同：TurboMind 的 GEMV kernel 针对服务侧 GEMV 访问模式优化，AutoAWQ 面向 GEMM。

FP8 KV cache 同样可用：`--quant-policy 8` 启用 FP8 KV 量化，与 BF16 相比 KV 内存减半，在标准基准测试上精度影响极小。结合 W4A16 权重，70B 模型的 KV cache 占用大约等同于 BF16 35B 模型，在相同 GPU 预算内可支持更长的有效上下文。

## 模型覆盖范围

TurboMind 为以下模型提供优化 kernel 路径：

- **Llama 系列**（Llama 2/3、Mistral、Mixtral MoE）：标准 GQA 注意力、密集 FFN 或 MoE 路由
- **InternLM 1/2/3**：第一优先级支持，源于本实验室的原生模型；架构与 TurboMind 协同演进
- **Qwen 1.5/2/2.5/3**：第一优先级支持；Qwen 与上海人工智能实验室有研究交集；LMDeploy 是 Qwen 官方文档中的参考部署方案
- **DeepSeek V1/V2/V3/R1**：MLA 原生路径；MoE 路由支持；最早高效支持 DeepSeek V2 的开源引擎之一
- **视觉语言模型**：InternVL（InternLM + ViT）、LLaVA（CLIP + Llama）；视觉编码器在 PyTorch 中运行，文本解码器在 TurboMind 中运行

PyTorch 后端覆盖尚未移植到 TurboMind kernel 的模型系列，吞吐较低。

## 部署

启动 OpenAI 兼容服务器：

```bash
lmdeploy serve api_server \
    model_path \
    --server-port 23333 \
    --tp 2 \
    --cache-max-entry-count 0.8 \
    --quant-policy 4
```

`--tp N` 在 N 个 GPU 上启用张量并行。`--cache-max-entry-count` 设置为 KV cache 页预留的 GPU 内存比例。`--quant-policy 4` 在模型已完成 AWQ 量化的情况下选择 W4A16 路径。

Docker 镜像：`openmmlab/lmdeploy:latest` 提供 CUDA 12.1 + TurboMind 二进制文件。Kubernetes Helm chart 在代码仓库的 `kubernetes/` 目录下。LMDeploy 与 OpenCompass 集成，可对量化后的服务端点（而非本地模型文件）运行评测流水线。

服务器级近似吞吐量（H100 80GB SXM，2025 年初数据）：
- Llama 3 8B BF16：批大小 32 约 12,000 tokens/s
- Llama 3 8B W4A16：批大小 32 约 18,000 tokens/s
- Llama 3 70B BF16：批大小 32 约 3,500 tokens/s
- Llama 3 70B W4A16：批大小 32 约 5,500 tokens/s
- DeepSeek V2 236B（2× H100，MLA 原生）：批大小 32 约 2,800 tokens/s

## 工程 Tradeoff

| 维度 | LMDeploy / TurboMind | vLLM | TGI | SGLang |
|---|---|---|---|---|
| W4A16 AWQ kernel 质量 | 自定义 GEMV kernel；单流 W4A16 吞吐最优 | AWQ 通过 Marlin / 自定义 kernel；大批量下有竞争力 | 支持 AWQ；kernel 质量随版本不同 | AWQ 通过 Marlin；有竞争力 |
| MLA 支持 | 原生 MLA KV 布局；完整压缩收益 | 早期版本展开 K/V；后续加入原生 MLA | 截至 2024 年无原生 MLA | 为 DeepSeek 加入了 MLA 支持 |
| 分页 KV cache | 有；基于物理块的池化 | 有；PagedAttention；最成熟的实现 | 有（v2）；效果相当 | 有；顶层加 radix 树实现前缀共享 |
| 吞吐（BF16，大批量） | 70B H100 约 3,500 tokens/s | 70B H100 约 3,200 tokens/s | 与 vLLM 相当 | 利用 RadixAttention 略有超越 |
| 国产实验室模型支持 | 第一优先级：InternLM、Qwen、DeepSeek | 良好；在最新中文模型上有滞后 | 部分支持 | 增长中；DeepSeek 支持良好 |
| 社区规模 | 中等；贡献者以中文社区为主 | 最大的开源 LLM 服务社区 | 较大；HuggingFace 背书 | 中等；UC Berkeley 起源 |
| 部署复杂度 | Docker + 单条命令行；中英文文档完善 | 良好的 Python 工具链；文档详尽 | 良好；HuggingFace Hub 集成 | Python 优先；配置略多 |

## 交叉参考

- `../paged-attention/` — LMDeploy 实现了与 vLLM 相同的物理块 KV cache 概念。block table、空闲块池和 copy-on-write 机制在概念上完全一致；具体实现各自独立。
- `../weight-quantization/` — AWQ 是 LMDeploy 生产流水线的主要量化方法。校准算法与 AutoAWQ 相同；服务 kernel 为 LMDeploy 独有。
- `../smoothquant/` — W8A8 SmoothQuant 通过 `lmdeploy lite smooth_quant` 同样可用，适合 INT4 精度损失不可接受而内存预算允许 8 位权重的场景。
- `../../deepseek/mla/` — MLA 原生 KV 布局是 DeepSeek 部署的关键差异点；压缩机制及其对服务的影响在该条目中详述。
- `../tgi/` — TGI 是欧洲实验室一侧的同级服务引擎；对比见该条目。
- `../../qwen/qwen3/` — Qwen 模型在 LMDeploy 中享有第一优先级支持；Qwen 官方部署文档以 LMDeploy 为主要参考方案。

## 个人评注

LMDeploy 的技术身份由两个选择定义：W4A16 AWQ kernel 质量和 MLA 原生支持。二者都是同一组织事实的结果——这个引擎是由交付模型的团队为交付模型而构建的。当你的用户就是写下模型架构的工程师时，你能在第一天就拿到 MLA 支持，拿到针对自家模型形状访问模式调优的 AWQ kernel。

更宏观的启示是**训练与服务垂直整合的价值**。vLLM 在某些维度上是更好的通用服务引擎（社区、生态、多模态覆盖、工具链）。LMDeploy 在高效运行国产实验室模型这一具体问题上，尤其是在权重量化下，更具优势。两者都没有绝对的统治地位——权衡确实存在，取决于你的模型系列。

对实践者的建议：如果你在规模化部署 InternLM、Qwen 或 DeepSeek，LMDeploy 是正确的起点。仅 W4A16 吞吐优势（单流约 2 倍，批大小 32 约 1.5 倍）就足以抵消其相对于 vLLM 略小的社区规模。如果你运营的是混合模型系列的集群，有复杂调度需求和高并发要求，vLLM 更成熟的分页分配器和更广泛的工具链可能是更好的选择。这个决策不是信仰问题——用你自己的流量模式对两者做基准测试。

## 参考

- [1] InternLM 团队. LMDeploy 代码仓库. https://github.com/InternLM/lmdeploy
- [2] Lin 等. _AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration._ MLSys '24 / arXiv:2306.00978.
- [3] Xiao 等. _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models._ ICML '23 / arXiv:2211.10438.
- [4] Kwon 等. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180.（分页 KV cache 概念）
- [5] DeepSeek-AI. _DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model._ arXiv:2405.04434.（MLA 规范）
- [6] Zheng 等. _SGLang: Efficient Execution of Structured Language Model Programs._ arXiv:2312.07104.
- [7] OpenCompass. https://github.com/open-compass/opencompass（评测集成）
