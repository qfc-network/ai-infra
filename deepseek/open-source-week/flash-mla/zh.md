# FlashMLA —— 源码走读

- **仓库**：[deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- **首次发布**：2025-02（Open Source Week 第一天）
- **里程碑**：2025-04 新 kernel（"seesaw" 调度）、2025-09 FP8 稀疏 decode（DSA）
- **配套笔记**：[`deepseek/mla/`](../../mla/) 是 MLA 论文笔记

## 这个仓库是什么

Hopper（SM90）和 Blackwell（SM100）上的生产级 **MLA attention CUDA kernel**，直接为 DeepSeek-V3 / V3.2 推理服务。四类 kernel：

| Kernel | 阶段 | 模式 | KV 格式 | 架构 |
|---|---|---|---|---|
| Dense Decoding | decode | MQA-absorbed | BF16 | SM90 |
| Sparse Decoding | decode | MQA-absorbed | FP8 | SM90 / SM100 |
| Dense Prefill | prefill | MHA | — | SM100 |
| Sparse Prefill | prefill | MQA-absorbed | — | SM90 / SM100 |

H800 SXM5（CUDA 12.8）上官方数字：dense decode **3000 GB/s** 带宽、**660 TFLOPs** 算力；FP8 稀疏 decode **410 TFLOPs**。

仓库内附两篇 deep-dive 博客：
- [`docs/20250422-new-kernel-deep-dive.md`](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md) —— "seesaw" 调度
- [`docs/20250929-hopper-fp8-sparse-deep-dive.md`](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md) —— FP8 稀疏

下文多处为官方博客的转述，以它们为准。

## 目录结构

```
FlashMLA/
├── flash_mla/
│   ├── __init__.py
│   └── flash_mla_interface.py     # Python API：get_mla_metadata、flash_mla_with_kvcache
├── csrc/
│   ├── api/                       # C++/CUDA Python 绑定入口
│   ├── kerutils/                  # 共享 kernel 工具
│   ├── sm90/
│   │   ├── decode/dense/          # SM90 dense MLA decode（BF16）
│   │   ├── decode/sparse_fp8/     # SM90 稀疏 FP8 decode
│   │   ├── prefill/               # SM90 稀疏 prefill
│   │   └── helpers.h
│   ├── sm100/                     # Blackwell kernel
│   ├── smxx/                      # 架构无关部分
│   ├── cutlass                    # submodule
│   ├── params.h、defines.h、utils.h
├── benchmark/
├── docs/                          # 博客 + 配图
├── tests/
└── setup.py
```

一个细节：kernel **按架构在目录级切分**（`sm90/`、`sm100/`），不走 `#ifdef`。每类 kernel 是独立的 translation unit，按 shape 模板化但不按架构。

## Python 接口

`flash_mla/flash_mla_interface.py` 暴露两个函数：

```python
tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, is_fp8, topk,
)

o_i, lse_i = flash_mla_with_kvcache(
    q_i, kvcache_i, block_table, cache_seqlens, dv,
    tile_scheduler_metadata, num_splits,
    is_causal, is_fp8_kvcache, indices,
)
```

`get_mla_metadata` **每个 decode step 调一次**（不是每层），给出 tile scheduler 的计划，所有层复用。`flash_mla_with_kvcache` 真正跑 kernel。metadata 把请求级负载均衡从热路径里解耦出来。

## 为什么 MLA decode 是算力瓶颈

反直觉——decode 阶段 attention 一般是带宽瓶颈。官方博客的算式（转述）：

- 每请求 FLOPs ≈ `2 · h_q · s_q · s_k · (d_k + d_v)`
- HBM 字节 ≈ `2 · s_k · d_k`（`s_k ≫ h_q · s_q`，KV cache 主导）
- FLOPs/byte ≈ `h_q · s_q · (d_k + d_v) / d_k ≈ 2 · h_q · s_q`

H800 降频后算力峰值 ~865 TFLOPs、HBM 3.35 TB/s，交叉点 `h_q · s_q ≈ 128`。DeepSeek 推理系统 **decode 实例不开 TP**（见 DeepSeek 推理系统综述），所以即便关掉 MTP，`h_q = 128`，MLA decode 稳落在算力瓶颈区。

这改变了 kernel 设计问题：不是再跟带宽较劲，而是**死磕 Tensor Core 不断流**。

## Seesaw 调度（2025-04 新 kernel）

FlashAttention-3 的 **ping-pong 调度**让两个 warpgroup 各自持有一个输出 tile 交替角色，从而让 softmax（CUDA core）与 matmul（Tensor core）重叠。MLA decode 做不到：每 warpgroup 的输出 `[64, 512]` 要 32,768 寄存器——恰好是 SM 寄存器文件 65,536 的一半。**一个 SM 放两份完整输出不可能。**

替代方案：**保留单份输出**，但**垂直切**成 `O_L` 与 `O_R`（各 `[64, 256]`），每 warpgroup 持一半。每 step 处理两对 K/V 块（`K_0, V_0` 与 `K_1, V_1`），两个 warpgroup 在交替更新中协作：

```
0. m ← -inf, O_L ← 0, O_R ← 0   （m 共享，输出各 WG 一份）
1. [WG0]  p_0 = q · K_0^T / scale
2. [WG1]  p_1 = q · K_1^T / scale
3. [WG0]  scale_0 = exp(m_new - m); m ← max(m, max(p_0))
4. [WG0]  p_0 = exp(p_0 - m_new)
5. [WG0]  O_L = O_L · scale_0 + p_0 · V_{0L}
6. [WG1]  scale_1 = exp(m_new - m); m ← max(m, max(p_1))
7. [WG1]  p_1 = exp(p_1 - m_new)
8. [WG1]  O_R = O_R · (scale_0 · scale_1) + p_1 · V_{1R}
9. [WG0]  p_0 = p_0 · scale_1
10. [WG1] O_R = O_R + p_0 · V_{0R}
11. [WG0] O_L = O_L · scale_1 + p_1 · V_{1L}
```

数学上与 FlashAttention 在线 softmax 等价。操作上，一个 warpgroup 的 CUDA core 操作（softmax、scaling）与另一个的 Tensor core 操作（WGMMA）重叠；当前块用完立刻发起下一块 K/V 的 TMA 拷贝。

结果：持续 **~80% 降频后 Tensor Core 峰值**、~3 TB/s HBM、H800 上 ~660 TFLOPs。完整调度图见 2025-04 的博客。

## 显存流水线技巧

即便进入算力瓶颈区，延迟仍要管。博客给的两个技巧：

1. **细粒度 TMA / GEMM pipeline。** `[64, 576]` 的 K 块用 **9 次 `[64, 64]` TMA 拷贝**搬运，而不是一次大拷贝。第一个子块 TMA 一完成，第一个 GEMM 就能开跑。
2. **缓存驱逐提示。** TMA 拷贝用 `cute::TMA::CacheHintSm90::EVICT_FIRST` 反而提升 L2 命中率——标记为"早驱逐"能让 L2 工作集收紧。

## Tile scheduler 与 split-KV

`get_mla_metadata` 返回 tile scheduler metadata 和 `num_splits`。设计上借鉴 **Flash-Decoding** 的 split-KV：长 KV 序列跨 SM 切分，每个 SM 算局部 attention，另一个 `combine` kernel 通过 LSE 合并。

`splitkv_mla` → `combine` 的重叠靠 **Programmatic Dependent Launch**（CUDA 12 特性）——combine 可在 split 完成时立刻起飞，不用等整场同步。

Tile scheduler 负责把 (request, block) 对均衡到各 SM——decode batch 的 `cache_seqlens` 方差大，朴素轮询会让 SM 闲置。metadata 是请求级、层无关的，成本可在 V3 的 61 层 MLA 上摊销。

## FP8 稀疏 decode（2025-09）

支撑 V3.2 引入的 **DeepSeek Sparse Attention (DSA)**。两个关键点：

- **KV cache 以 FP8 E4M3 存储**，matmul 用 **BF16**。kernel 在进入 shared memory 的路径上，按 128 元素分组用 FP32 scale 反量化。
- **每 token 656 字节**：512 字节 FP8 NoPE + 16 字节（4 × FP32）分组 scale + RoPE 与元数据，整块连续。
- **每个 query 传入 top-k 索引**以选定要关注的 KV 位置，kernel 从分页 KV cache 中稀疏 gather。

FP8 KV 让有效 cache 大致减半（相对 BF16 MLA），再加稀疏，长上下文 decode 的显存墙被显著推后。

## 哪些能搬走

- **MQA-absorbed MLA decode** 对任何 MLA 风格（`d_k = 576, d_v = 512`）的压缩 KV 都能用。
- **Seesaw 调度**是对"输出过大无法复制"这类算力瓶颈 decode kernel 的一份可照搬配方。模式不限于 MLA——任何 `d_v ≥ 256` 的 attention 变体都会撞同一堵寄存器墙。
- **Tile scheduler + split-KV + PDL** 的组合可作为变长 decode batch 的模板。
- **分组 scale 的 FP8 KV** 格式可移植到能接受该格式的其他模型。

## 哪些不能

kernel **专门针对 MLA 的 head 维度**（`d_k = 576 / 192 / 128`、`d_v = 512 / 128`）和 V3/V3.2 的推理模式（decode 无 TP、`h_q = 128`）。要换到 MLA 维度不同或 decode 开 TP=8 的模型，需要重调调度参数，甚至重构 warpgroup 切分。

## 个人评注

FlashMLA 是我见过最干净的"**与推理系统协同设计的 kernel**"的例子。"decode 不开 TP → `h_q = 128` → 算力瓶颈"这条链，是 infra 推理在驱动 kernel 设计，而不是反过来。seesaw 调度也很漂亮：寄存器压力看起来是死路，解法却是切**输出**而不是切操作数——既保住了 FlashAttention 的在线 softmax 不变量，又拿到了 ping-pong 的 overlap。把这份代码和 MLA 论文、V3 技术报告一起读，才看得到整栈逻辑：**架构（MLA）为算力瓶颈 decode 铺路 → 系统决策（decode 不开 TP）把你推进那个区间 → kernel（FlashMLA）压出接近峰值利用率**。每一环都不显然，组合在一起才是 V3 推理快的原因。

## 参考

- FlashMLA 仓库：https://github.com/deepseek-ai/FlashMLA
- 新 kernel 深度解析：`docs/20250422-new-kernel-deep-dive.md`
- FP8 稀疏深度解析：`docs/20250929-hopper-fp8-sparse-deep-dive.md`
- FlashAttention-3（ping-pong）：arXiv:2407.08608
- Flash-Decoding：https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- DeepSeek V3.2-Exp（DSA）：https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- DeepSeek 推理系统综述：https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
