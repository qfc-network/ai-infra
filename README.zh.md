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

### DeepSeek
- [V3 Technical Report](./deepseek/v3-tech-report/) — FP8 训练、DualPipe、671B MoE
- [MLA — Multi-head Latent Attention](./deepseek/mla/) — KV cache 压缩
- [DeepSeekMoE](./deepseek/moe/) — 细粒度 + 共享专家

### Meta
- _Llama 3 Herd of Models_ — 待写

### Mistral
- _Mixtral / 稀疏 MoE_ — 待写

### Google
- _Gemini / Pathways / TPU 系统_ — 待写

### Anthropic
- _公开 infra 材料_ — 待写

## 贡献

新增论文：复制 [`_template/`](./_template) 到对应厂商目录，同时填写 `zh.md` 与 `en.md`，更新本索引。

## 许可证

文档采用 [CC BY 4.0](./LICENSE)。
