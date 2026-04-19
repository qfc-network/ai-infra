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
- [Scaling Laws — Kaplan 2020 + Chinchilla 2022](./foundational/scaling-laws/) — 算力/数据/参数的幂律关系；20 token/参数法则
- [FlashAttention 1 / 2 / 3](./foundational/flash-attention/) — IO-aware 精确注意力，Hopper 异步 + FP8
- [Triton](./foundational/triton/) — 块级 GPU kernel DSL + MLIR 编译器；生产力放大器
- [PagedAttention / vLLM](./foundational/paged-attention/) — OS 分页式 KV cache、连续批处理
- [Orca — 连续批处理](./foundational/orca/) — 迭代级调度；goodput 作为正确度量；vLLM/SGLang 的基础
- [Megatron-LM (TP / PP / SP)](./foundational/megatron-lm/) — 张量、流水线、序列并行 + 选择性重算
- [ZeRO / FSDP](./foundational/zero-fsdp/) — 切分式数据并行；与 Megatron 正交
- [Speculative Decoding](./foundational/speculative-decoding/) — draft + verify；无损 2–4× 解码加速
- [Ring Attention / Context Parallelism](./foundational/ring-attention/) — 序列切分，1M+ 精确 attention
- [分组查询注意力（GQA）](./foundational/gqa/) — H/G 倍 KV cache 压缩；Llama 3、Mistral、Gemma 默认注意力方案
- [旋转位置编码（RoPE）](./foundational/rope/) — 旋转编码位置；相对位置、零参数、Flash 友好；长上下文扩展
- [DistServe](./foundational/distserve/) — prefill/decode 解耦；goodput 作为正确度量
- [SGLang](./foundational/sglang/) — RadixAttention 前缀缓存；多 call LLM 程序前端 DSL
- [前缀缓存](./foundational/prefix-caching/) — 哈希 vs 基数字典树匹配；淘汰策略；命中率经济学；GPU→CPU→磁盘多级缓存；解耦路由
- [权重量化 — GPTQ & AWQ](./foundational/weight-quantization/) — INT4 训练后；Hessian vs 激活感知
- [SmoothQuant](./foundational/smoothquant/) — W8A8；激活到权重的 outlier 迁移
- [RLHF / InstructGPT](./foundational/rlhf/) — 三阶段 SFT+RM+PPO；后训练的地基
- [DPO](./foundational/dpo/) — RLHF 折叠为单次监督步；更简单的默认
- [GRPO — 组相对策略优化](./foundational/grpo/) — 无 critic 的 RL；组优势估计；DeepSeek-R1 背后的算法
- [SimPO — 简单偏好优化](./foundational/simpo/) — 无参考模型的 DPO；长度归一化奖励 + 目标奖励间隔
- [LoRA / QLoRA](./foundational/lora/) — 低秩权重适配；单卡 4-bit 微调
- [混合精度训练](./foundational/mixed-precision/) — FP16 → BF16 → FP8；AMP、损失缩放、per-tensor/block FP8 缩放
- [Chunked Prefill — Sarathi-Serve](./foundational/chunked-prefill/) — prefill 分块与 decode 交织；单卡消除首 token 延迟抖动
- [KV Cache 量化 — KIVI & KVQuant](./foundational/kv-cache-quantization/) — INT2 key + INT4 value；完成量化闭环
- [verl — HybridFlow](./foundational/verl/) — 每模型独立并行策略 + CPU 卸载；70B+ 规模 PPO/GRPO 的训练基础设施
- [GPU 互联 primer](./foundational/gpu-interconnect/) — NVLink、NVSwitch、RDMA、IBGDA；其他条目默认的 fabric
- [Hopper / H100 架构 Primer](./foundational/hopper-h100/) — wgmma、TMA、FP8 Tensor Core、线程块集群；FlashAttention-3、DeepGEMM、DualPipe 背后的计算原语
- [Mamba 与状态空间模型](./foundational/mamba-ssm/) — 线性时间、常量显存 decode；与 attention 的混合
- [推理时扩展 — 测试时计算量](./foundational/inference-time-scaling/) — 并行搜索（best-of-N、PRM 束搜索）vs 顺序细化（思考 token）；按问题难度的计算最优策略
- [过程奖励模型（PRM）](./foundational/process-reward-models/) — 步骤级验证；PRM800K 人工标注 vs MC 回滚自动标注；束搜索与 MCTS 整合
- [投机解码变体 — Medusa / EAGLE](./foundational/speculative-decoding-variants/) — 树形自草稿；Medusa 并行多头；EAGLE 特征级自回归草稿；EAGLE-2 自适应树
- [Blackwell / B200 架构 Primer](./foundational/blackwell-b200/) — FP4 张量核心、HBM3e 192 GB、NVLink 5 1.8 TB/s、GB200 NVL72 机架级互联
- [分词 — BPE、SentencePiece、Tiktoken](./foundational/tokenization/) — 子词算法；多语言 token 预算经济学；不同语言的词汇率
- [数据管线 — FineWeb / MinHash / 质量过滤](./foundational/data-pipeline/) — Common Crawl → 预训练语料；MinHash 去重；万亿 token 规模质量分类器
- [位置内插 — YaRN / LongRoPE / NTK-aware](./foundational/position-interpolation/) — 将预训练 RoPE 模型扩展至 128k–2M 上下文；NTK-aware 基频变换；YaRN 非均匀缩放
- [Streaming LLM 与注意力汇聚点](./foundational/streaming-llm/) — 有界 KV cache 实现无限长生成；注意力汇聚点现象；K_sink + 滑动窗口
- [Switch Transformer & GShard](./foundational/switch-gshard/) — top-1 路由；辅助负载均衡损失；专家容量因子；奠基性 MoE 论文
- [TGI — Text Generation Inference](./foundational/tgi/) — HuggingFace 的 Rust + Python Serving 栈；连续批处理；量化格式；与 vLLM 对比
- [多租户 LoRA Serving — SLoRA / Punica](./foundational/multi-tenant-lora/) — 分页适配器池（SLoRA）；SGMV 批量 kernel（Punica）；单基模型服务数千适配器
- [扩散基础 — DDPM / DDIM / CFG](./foundational/diffusion-fundamentals/) — 前向/反向扩散过程；DDIM 确定性采样；无分类器引导；NFE 是主要延迟驱动因素
- [序列并行变体 — Ulysses 与 Megatron-CP](./foundational/sequence-parallelism/) — head 维度 all-to-all（Ulysses）；因果奇偶交错（Megatron-CP）；与 Ring Attention 共同完成序列轴并行全景
- [torch.compile / Inductor](./foundational/torch-compile/) — TorchDynamo 字节码追踪；AOTAutograd 联合图；Inductor Triton 代码生成；算子融合；CUDA Graphs
- [大规模知识蒸馏](./foundational/knowledge-distillation/) — 带温度的软标签损失；白盒/黑盒/序列级蒸馏；R1 蒸馏流程；Gemma 2 logit 软截断
- [工具调用基础设施](./foundational/tool-use-infra/) — JSON Schema 工具定义；并行工具调用分发；流式 delta 解析；安全层；上下文窗口增长分析
- [Agent 框架全景](./foundational/agent-frameworks/) — LangGraph 有状态图；AutoGen 多 agent 对话；OpenAI Swarm 移交模式；DSPy 提示编译；状态持久化与 p99 延迟权衡
- [Agent 评测基础设施](./foundational/agent-evaluation/) — SWE-bench、TAU-bench、WebArena；超越二元通过率的轨迹指标；沙箱评测环境；LLM-as-judge 校准

### 多模态
- [CLIP — 对比语言-图像预训练](./multimodal/clip/) — 双编码器对比目标；N² 对上的 InfoNCE 损失；零样本分类；网络规模训练
- [Vision Transformer（ViT）](./multimodal/vit/) — patch 嵌入；[CLS] token；ViT-L/14 → 576 token；FlashAttention 兼容；DeiT 蒸馏
- [LLaVA / 视觉语言模型](./multimodal/llava/) — MLP 投影器；两阶段训练；视觉 token 数量（256→576→2880）；图像前缀缓存
- [Whisper — 语音识别](./multimodal/whisper/) — log-mel 频谱图；30 秒分块；特殊 token 多任务；68 万小时弱监督训练
- [DiT — 扩散 Transformer](./multimodal/dit/) — 潜在扩散 + Transformer 骨干；adaLN 条件调制；Sora / SD3 / FLUX 谱系；计算密集型推理
- [VLM Serving](./multimodal/vlm-serving/) — 视觉 token prefill 经济学；变分辨率分块；图像前缀缓存；异构批处理

### DeepSeek
- [V2 — 经济高效的 236B MoE](./deepseek/v2/) — MLA + DeepSeekMoE 作为系统；21B 激活 / 236B 总参数；吞吐较稠密前代提升 5.76×
- [V3 Technical Report](./deepseek/v3-tech-report/) — FP8 训练、DualPipe、671B MoE
- [MLA — Multi-head Latent Attention](./deepseek/mla/) — KV cache 压缩
- [DeepSeekMoE](./deepseek/moe/) — 细粒度 + 共享专家
- [R1](./deepseek/r1/) — 规则奖励 RL 驱动推理；GRPO；R1-Zero 涌现
- [Open Source Week — FlashMLA 源码走读](./deepseek/open-source-week/flash-mla/) — seesaw 调度、FP8 稀疏 decode
- [Open Source Week — DeepEP 源码走读](./deepseek/open-source-week/deep-ep/) — 专家并行 all-to-all、IBGDA 低延迟
- [Open Source Week — DeepGEMM 源码走读](./deepseek/open-source-week/deep-gemm/) — JIT FP8/BF16 GEMM、MoE 三种布局、V3.2 indexer
- [Open Source Week — DualPipe 源码走读](./deepseek/open-source-week/dualpipe/) — 双向流水线调度，bubble 减半
- [Open Source Week — 3FS 源码走读](./deepseek/open-source-week/3fs/) — RDMA 原生分布式 FS；CRAQ + FDB + USRBIO
- [DeepSeek-Prover](./deepseek/prover/) — Lean 4 定理证明；RL + RMaxTS；编译器作为完美 PRM；自动形式化流程
- [V3.2 / 原生稀疏注意力（NSA）](./deepseek/v3-2-nsa/) — 压缩 + 选择 + 窗口三路径稀疏注意力；V3.2 indexer kernel；无需 RoPE 插值的原生 128k 上下文

### Meta
- [Llama 3 Herd of Models](./meta/llama3/) — 405B dense、16k H100、4D 并行
- [Llama 4](./meta/llama4/) — 首个 MoE 系列；Scout 17B×16E / Maverick 17B×128E；iRoPE 交替注意力；原生多模态

### Qwen
- [Qwen3](./qwen/qwen3/) — 思考/非思考切换；MoE 235B/22A + 稠密 0.6B–32B；RL 后训练；最强开放权重系列

### Mistral
- [Mixtral of Experts](./mistral/mixtral/) — 8×7B、top-2 路由、开源 MoE 基线

### Moonshot
- [Mooncake](./moonshot/mooncake/) — KVCache 为中心的解耦推理；PD 解耦；缓存池

### Google
- [Pathways](./google/pathways/) — 异步分布式数据流运行时；TPU pod 规模单控制器
- [GSPMD](./google/gspmd/) — XLA 自动并行 pass；sharding 作为类型
- [Gemma 2](./google/gemma2/) — 27B 教师模型蒸馏；logit 软截断；交替局部/全局注意力

### Microsoft
- [DeepSpeed — MoE、Chat 与推理引擎](./microsoft/deepspeed/) — 专家并行、混合 RLHF 引擎、融合 INT8 推理

### NVIDIA
- [CUTLASS](./nvidia/cutlass/) — C++ GEMM 模板层级；FlashAttention、DeepGEMM 和 cuBLAS 的底层算子库
- [TensorRT-LLM](./nvidia/tensorrt-llm/) — AOT 编译 LLM 推理引擎；分页 KV cache、FP8、H100 峰值性能连续批处理
- [Dynamo](./nvidia/dynamo/) — 解耦推理编排；KV 感知路由；prefill/decode 资源池管理；TensorRT-LLM 集成

### Anthropic
- [Building Effective Agents](./anthropic/building-effective-agents/) — workflow 与 agent 的区分、五种 workflow 模式
- [Model Context Protocol (MCP)](./anthropic/mcp/) — LLM 的 LSP；tools / resources / prompts over JSON-RPC
- [Constitutional AI](./anthropic/constitutional-ai/) — 批判-修订循环 + RLAIF；基于 AI 反馈的无害性对齐
- [Computer Use 与浏览器自动化](./anthropic/computer-use/) — 基于截图的像素级动作空间；视觉定位；单步延迟分解；容器沙箱隔离

## 指南

职业路径与入门指南——工程视角优先，对 ML 算法背景要求低。

- [DevOps 工程师如何切入 AI Infra](./guides/devops-to-ai-infra/) — 可迁移技能、需补缺口、三条垂直路径（集群运维 / 推理平台 / 训练基础设施）、6 个月里程碑
- [本地大模型部署指南](./guides/on-prem-llm-deployment/) — 从 Mac Studio + Ollama 到多节点 GPU 集群；硬件选型、软件栈、成本参考、决策流程图
- [安全私有化 Agent 部署指南](./guides/secure-agent-deployment/) — 工具调用沙箱（gVisor、NetworkPolicy、seccomp）；MCP 权限边界；Agent 循环中的 secret 管理；最小安全栈参考

## 贡献

新增论文：复制 [`_template/`](./_template) 到对应厂商目录，同时填写 `zh.md` 与 `en.md`，更新本索引。

## 许可证

文档采用 [CC BY 4.0](./LICENSE)。
