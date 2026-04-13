# ai-infra

> Deep dives into AI infrastructure papers and open-source systems from frontier labs.
> 前沿 AI 实验室基础设施论文与开源系统的深度解析。

[中文版 README](./README.zh.md)

## Scope

This repo collects engineering-focused analyses of papers and open-source releases covering:

- **Training systems** — parallelism strategies, mixed precision, communication primitives
- **Inference systems** — KV cache, speculative decoding, serving architectures
- **Model architectures with infra implications** — MoE routing, attention variants, long context
- **Open-source infra components** — kernels, schedulers, file systems

Every paper gets both an English (`en.md`) and Chinese (`zh.md`) write-up using the same template. See [`_template/`](./_template).

## Index

### DeepSeek
- [V3 Technical Report](./deepseek/v3-tech-report/) — FP8 training, DualPipe, MoE at 671B
- [MLA — Multi-head Latent Attention](./deepseek/mla/) — KV cache compression
- [DeepSeekMoE](./deepseek/moe/) — fine-grained + shared experts

### Meta
- _Llama 3 Herd of Models_ — planned

### Mistral
- _Mixtral / sparse MoE_ — planned

### Google
- _Gemini / Pathways / TPU systems_ — planned

### Anthropic
- _Public infra writings_ — planned

## Contributing

New papers: copy [`_template/`](./_template) into the appropriate vendor directory, fill in both `zh.md` and `en.md`, update this index.

## License

Documentation: [CC BY 4.0](./LICENSE).
