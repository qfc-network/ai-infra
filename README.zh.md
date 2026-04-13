# ai-infra

> 前沿 AI 实验室基础设施论文与开源系统的深度解析。
> Deep dives into AI infrastructure papers and open-source systems from frontier labs.

[English README](./README.md)

## 范围

本仓库收集以工程视角切入的论文与开源项目分析，覆盖：

- **训练系统** — 并行策略、混合精度、通信原语
- **推理系统** — KV cache、投机解码、serving 架构
- **对 infra 有显著影响的模型架构** — MoE 路由、注意力变体、长上下文
- **开源 infra 组件** — 算子库、调度器、文件系统

每篇论文都会写中文（`zh.md`）与英文（`en.md`）两份笔记，遵循统一模板。见 [`_template/`](./_template)。

## 索引

### Foundational（通用基础）
- [FlashAttention 1 / 2 / 3](./foundational/flash-attention/) — IO-aware 精确注意力，Hopper 异步 + FP8
- [Triton](./foundational/triton/) — 块级 GPU kernel DSL + MLIR 编译器；生产力放大器
- [PagedAttention / vLLM](./foundational/paged-attention/) — OS 分页式 KV cache、连续批处理
- [Megatron-LM (TP / PP / SP)](./foundational/megatron-lm/) — 张量、流水线、序列并行 + 选择性重算
- [ZeRO / FSDP](./foundational/zero-fsdp/) — 切分式数据并行；与 Megatron 正交
- [Speculative Decoding](./foundational/speculative-decoding/) — draft + verify；无损 2–4× 解码加速
- [Ring Attention / Context Parallelism](./foundational/ring-attention/) — 序列切分，1M+ 精确 attention
- [DistServe](./foundational/distserve/) — prefill/decode 解耦；goodput 作为正确度量
- [SGLang](./foundational/sglang/) — RadixAttention 前缀缓存；多 call LLM 程序前端 DSL
- [权重量化 — GPTQ & AWQ](./foundational/weight-quantization/) — INT4 训练后；Hessian vs 激活感知
- [SmoothQuant](./foundational/smoothquant/) — W8A8；激活到权重的 outlier 迁移

### DeepSeek
- [V3 Technical Report](./deepseek/v3-tech-report/) — FP8 训练、DualPipe、671B MoE
- [MLA — Multi-head Latent Attention](./deepseek/mla/) — KV cache 压缩
- [DeepSeekMoE](./deepseek/moe/) — 细粒度 + 共享专家
- [R1](./deepseek/r1/) — 规则奖励 RL 驱动推理；GRPO；R1-Zero 涌现
- [Open Source Week — FlashMLA 源码走读](./deepseek/open-source-week/flash-mla/) — seesaw 调度、FP8 稀疏 decode
- [Open Source Week — DeepEP 源码走读](./deepseek/open-source-week/deep-ep/) — 专家并行 all-to-all、IBGDA 低延迟
- [Open Source Week — DeepGEMM 源码走读](./deepseek/open-source-week/deep-gemm/) — JIT FP8/BF16 GEMM、MoE 三种布局、V3.2 indexer
- [Open Source Week — DualPipe 源码走读](./deepseek/open-source-week/dualpipe/) — 双向流水线调度，bubble 减半
- [Open Source Week — 3FS 源码走读](./deepseek/open-source-week/3fs/) — RDMA 原生分布式 FS；CRAQ + FDB + USRBIO

### Meta
- [Llama 3 Herd of Models](./meta/llama3/) — 405B dense、16k H100、4D 并行

### Mistral
- [Mixtral of Experts](./mistral/mixtral/) — 8×7B、top-2 路由、开源 MoE 基线

### Moonshot
- [Mooncake](./moonshot/mooncake/) — KVCache 为中心的解耦推理；PD 解耦；缓存池

### Google
- [Pathways](./google/pathways/) — 异步分布式数据流运行时；TPU pod 规模单控制器
- [GSPMD](./google/gspmd/) — XLA 自动并行 pass；sharding 作为类型

### Anthropic
- _公开 infra 材料_ — 待写

## 贡献

新增论文：复制 [`_template/`](./_template) 到对应厂商目录，同时填写 `zh.md` 与 `en.md`，更新本索引。

## 许可证

文档采用 [CC BY 4.0](./LICENSE)。
