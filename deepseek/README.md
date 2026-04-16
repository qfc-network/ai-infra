# DeepSeek

Papers and open-source infra from DeepSeek.
DeepSeek 的论文与开源 infra。

## Papers

| Topic | EN | ZH |
|---|---|---|
| V2 — Economical MoE at 236B | [en](./v2/en.md) | [zh](./v2/zh.md) |
| V3 Technical Report | [en](./v3-tech-report/en.md) | [zh](./v3-tech-report/zh.md) |
| MLA — Multi-head Latent Attention | [en](./mla/en.md) | [zh](./mla/zh.md) |
| DeepSeekMoE | [en](./moe/en.md) | [zh](./moe/zh.md) |
| R1 — RL-driven reasoning + GRPO | [en](./r1/en.md) | [zh](./r1/zh.md) |

## Open-Source Week (2025) — source walkthroughs

See [`open-source-week/`](./open-source-week/README.md).

- [FlashMLA](./open-source-week/flash-mla/) — MLA attention kernels; seesaw schedule; FP8 sparse decode
- [DeepEP](./open-source-week/deep-ep/) — expert-parallel all-to-all; asymmetric NVLink+RDMA; IBGDA low-latency
- [DeepGEMM](./open-source-week/deep-gemm/) — JIT FP8/BF16 GEMM; 1d1d vs 1d2d scaling; masked grouped for MoE decode
- [DualPipe](./open-source-week/dualpipe/) — bidirectional pipeline schedule; halves bubbles at 2× param cost
- [3FS](./open-source-week/3fs/) — distributed FS; CRAQ + FoundationDB + USRBIO `io_uring`-style API
