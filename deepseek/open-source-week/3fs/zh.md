# 3FS —— 源码走读

- **仓库**：[deepseek-ai/3FS](https://github.com/deepseek-ai/3FS)（Fire-Flyer File System）
- **发布**：2025-02（Open Source Week 第五天）
- **配套笔记**：[`deepseek/v3-tech-report/`](../../v3-tech-report/) —— 3FS 是 V3 训练与 R1/V3 推理 KV cache 的数据面

## 这个仓库是什么

一个**面向 AI workload、原生 RDMA、解耦架构的分布式文件系统**。在数百个存储节点的 NVMe SSD 上提供类 POSIX 文件接口，**强一致性**通过 CRAQ 链式复制实现，**元数据无状态**靠 FoundationDB 承载，**性能敏感客户端用 `io_uring` 风格的异步零拷贝 API（USRBIO）**。180 节点集群上聚合**读峰值 6.6 TiB/s**；25 storage + 50 compute 节点上 GraySort **3.66 TiB/min**；KVCache 单集群读峰值 **40 GiB/s**（生产）。

3FS 是让 V3 训练 dataloader 和 R1/V3 KVCache offload 不成为瓶颈的存储层。少了它，V3 论文里的成本故事就闭不上环。

## 四种目标 workload（README）

| Workload | 3FS 提供什么 |
|---|---|
| **数据预处理** | 层级目录、超大目录原子 rename、递归 delete |
| **Dataloader** | 跨计算节点对训练样本随机访问——无需 prefetch、无需 shuffle |
| **Checkpoint** | 高吞吐并行写、`fsync` 语义 |
| **推理 KVCache** | 高吞吐、大容量的 DRAM 缓存替代 |

最后一个最有特色：3FS 被用作 **LLM 推理的 DRAM 替代 KV cache 层**——HBM/DRAM 装不下的 KV 通过网络速率溢写到 NVMe。

## 架构

```
                   ┌──────────────────┐
                   │ Cluster Manager  │ ← 心跳、成员、链表
                   │ （主 + 备）       │ ← 元数据存 FoundationDB
                   └──────────────────┘
                            ▲
            ┌───────────────┼───────────────┐
            │               │               │
   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
   │ Metadata Svc   │ │ Storage Svc    │ │ Client         │
   │ （无状态，      │ │ （每 SSD 一个，│ │ （FUSE + 原生  │
   │  on FoundationDB）│ │  CRAQ 链复制）│ │  USRBIO）      │
   └────────────────┘ └────────────────┘ └────────────────┘
                            ▲
                     RDMA（IB / RoCE）
```

四个组件，全 RDMA。架构上几个关键选择：

- **元数据服务无状态** —— 文件元数据放 FoundationDB（带可串行化快照隔离的事务 KV）。meta 服务可不停机重启/升级；客户端可故障切换到任一 meta 服务。
- **存储用 CRAQ** —— 链式复制带分摊读：写沿链传播，读可发任一副本。强一致性，且读吞吐随副本数线性增长。
- **不强调局部性** —— 应用以 "locality-oblivious" 方式访问存储，不存在 "client 应读哪个节点" 的概念。RDMA fabric 够快，局部性优化不必要。

## 目录结构

```
3FS/
├── src/
│   ├── client/          # 客户端 RPC、连接管理
│   ├── core/、common/   # 通用工具、RPC、序列化
│   ├── meta/            # 元数据服务（文件系统语义）
│   ├── storage/         # 存储服务（SSD 上的 chunk store）
│   ├── mgmtd/           # cluster manager（成员、链表）
│   ├── fdb/             # FoundationDB 集成
│   ├── kv/              # KV 抽象
│   ├── fuse/            # FUSE 守护进程
│   ├── lib/             # USRBIO 原生客户端 API（见 UsrbIo.md）
│   ├── memory/          # RDMA 内存管理
│   ├── monitor_collector/、analytics/
│   ├── migration/、tools/
│   ├── fbs/             # FlatBuffers schema
│   └── stubs/、simple_example/
├── hf3fs/、hf3fs_fuse/、hf3fs_utils/   # FUSE 二进制与 Python 工具
├── benchmarks/          # 基于 USRBIO 的 fio engine
├── deploy/              # 集群部署
├── docs/
│   ├── design_notes.md  # 经典架构文档——先读它
│   ├── metrics.md
│   └── images/
├── specs/               # P/TLA+ 规格
├── configs/、dockerfile/、patches/、third_party/
├── Cargo.toml + Cargo.lock   # Rust 组件
├── CMakeLists.txt            # 主 C++ 构建
└── setup.py
```

OSW 系列里规模最大的一个——一个真正的分布式系统，不是 kernel 库。热路径 C++、部分组件 Rust，硬依赖 FoundationDB、libfuse 3.16+、RDMA verbs。

## 存储服务里的 CRAQ

链式复制带分摊读（Chain Replication with Apportioned Queries）：

- 写发到链的**头节点**，节点接力到**尾节点**。尾节点 ack 后写才算 durable。
- 读可以发到链上**任何**节点。每个节点跟踪本地副本版本与"已 clean"（被尾节点确认）的版本。被请求一个尚未 clean 的版本时，节点转发到尾节点确认。

读吞吐随副本数扩展；写吞吐受链长限制——但 V3 训练里写（checkpoint、log）相对读（dataloader）只占少量字节。

### 链表与优雅恢复

每个文件切 chunk，每个 chunk 落在一条**链**（跨节点、互斥的 SSD 集合）上。链按 **chain table** 组织，一种数据放置策略一张表（如批处理一张、在线一张）。

朴素链布局有问题：SSD `A` 挂掉时，所有 `A` 的链的读流量重定向到同一小撮伙伴 SSD（`B`、`C`），瞬间打爆它们。design notes 给出**整数规划导出的布局**：让 `A` 在自己的多条链里**与所有其他 SSD 各配一次**，故障时每张盘只多扛 `1/N`，而不是 `1/2`。SSD 替换 + 同步要数小时；这套布局保证集群在这段时间内仍可用。

这是把"研究型"分布式 FS 与"生产型"区分开的运维细节。

## 基于 FoundationDB 的无状态元数据

`src/meta/` + `src/fdb/`。元数据是两个 key 空间：

- **`INOD<inode_id>`** → inode 值（属性、文件长度、chunk size、chain table id、shuffle seed）。inode ID 用 little-endian 编码以散布到 FDB 各分片。
- **`DENT<parent_inode><name>`** → 子 inode id + 类型。range scan 一个 `DENT` 前缀就是 `readdir`。

所有元数据操作都是 FDB 事务，可串行化快照隔离。只读操作（`stat`、`lookup`）走只读事务；写（`create`、`link`、`unlink`、`rename`）走读写事务；FDB 处理冲突检测，meta 服务在冲突时自动重试。

效果：**多个 meta 服务副本可无协调并行服务请求**——FDB 是唯一真理源。

### 动态文件长度

一个巧妙优化。文件长度存在 inode 里，但活跃写时是**陈旧的**——客户端定期（默认 5 秒）把自己的最大写位置上报给 meta 服务，无并发 truncate 时被采纳为新长度。避免了每次写都要走一次 meta op。

`close`/`fsync` 时 meta 服务向存储服务问最后一块 chunk 的**精确**长度。为了不查 stripe 里的全部 200 条链，每个 inode 还跟踪**目前写过多少条链**（初始 16，按需翻倍）。小文件只为它实际用到的链付费。

## USRBIO —— 原生客户端 API

`src/lib/api/UsrbIo.md`（API 参考）。问题动机（design notes 转述）：

> FUSE 在共享自旋锁队列饱和前能处理约 40 万次 4KiB 读/秒；`perf` 显示 kernel spin lock 占大量 CPU。SSD 与 RDMA 带宽用不满。

对随机小读（dataloader 抽训练样本的典型场景），FUSE 在网络和 SSD 还远没饱和时就先成了瓶颈。把客户端写成 kernel module 能解决性能问题但带来其他坑（kernel panic、崩溃无 log、升级要重启机器）。

3FS 的答案：**把原生客户端嵌进 FUSE daemon**，暴露 `io_uring` 风格 API：

- **`Iov`** —— 用户进程与原生客户端共享的大内存区。客户端只注册一次为 RDMA pinned memory。所有读落在这里、所有写从这里发。**零拷贝。**
- **`Ior`** —— 用户与客户端共享的小环形缓冲（请求队列 + 完成队列）。用户 enqueue 读写请求；原生客户端线程 dequeue 后批量打到存储服务的 RDMA RPC。`io_depth` 控批大小；多线程应用建议用多个 ring 避免单 ring 锁竞争。

`open()`/`close()`/`stat()` 仍走 FUSE（POSIX 语义，代码迁移友好）。注册 fd 后的 I/O 走 USRBIO——无 kernel/user 拷贝、无 per-op 系统调用、无 FUSE 队列锁。

这是仓库里**对 AI workload 最关键的一个组件**。标准 FUSE 部署的 3FS 远达不到 6.6 TiB/s；USRBIO 才让那个数字成为可能。

## 性能（设计自己的视角）

README 选的三个 benchmark 刻意对应四种 workload：

- **180 节点上聚合读 6.6 TiB/s** —— SSD 与 RDMA fabric 同时打满，证明**忽略局部性的前提下存储可随节点数线性扩展**。
- **3.66 TiB/min GraySort** 经 [smallpond](https://github.com/deepseek-ai/smallpond) —— 数据预处理 workload，3FS 作为分析的 shuffle 基底。
- **KVCache 单集群读峰值 40 GiB/s** —— 推理 KV cache 层，附带 GC IOPS 公布（朴素的"磁盘 cache"被 GC 杀死，所以这个数字必须公开）。

## 哪些能搬走

- **CRAQ + 带均衡恢复的链表**模式可移植到任何 RDMA fabric 上的分布式 FS。整数规划导出的链布局是多数学院系统会跳过的细节。
- **基于 FoundationDB 的无状态元数据**是任何想做 HA 但又不想自己实现共识的 FS 的强模板。FDB 把难的部分做了。
- **USRBIO 的 `io_uring` 风格设计**对所有想突破 FUSE 性能上限的用户态服务都是可复用模式——远超 DeepSeek 自身用例。
- **文件系统作为 KV cache** 这个架构本身就有创新性。做大规模 LLM 推理的都该看看自己的 KV cache 能不能溢写到网络 NVMe，而不是被 HBM/DRAM 卡死。

## 哪些不能

- **硬依赖**：FoundationDB、libfuse 3.16+、RDMA fabric（IB 或 RoCE）、特定 Linux 发行版/clang 版本。立起来成本不低。
- **C++ 构建复杂度** + Rust 工具链。不适合周末项目。
- **某些边缘场景下 POSIX 不保证** —— 如并发写时文件长度只是最终一致（刻意设计，但要警觉）。
- **Linux 5.x 上 FUSE 不支持单文件并发写**。3FS 鼓励多文件写来绕开；如果你的应用必须单文件并发写，FUSE 仍咬你。
- **KVCache 用例需要自己集成**。仓库给你一个快文件系统；把 LLM KV 块映射上去得自己写（参考 [smallpond](https://github.com/deepseek-ai/smallpond) 与 DeepSeek 推理文档）。

## 个人评注

3FS 是 OSW 里**最难在小规模上欣赏、也最被公开讨论低估**的一个。它不是 kernel、不是巧调度、不是模型——而是一个 10 万+ 行的分布式系统，悄悄回答了 V3 规模 workload 下"训练样本和 KV cache 到底放哪"。两个设计选择有广泛适用性：

1. **CRAQ + 链表，而非 erasure coding**。现代分布式 FS 多用 EC 提升存储效率。3FS 选复制是因为 (a) 读扩展更好，(b) 链表让你能工程化恢复行为（整数规划布局），(c) 故障恢复更快。对 AI workload——存储便宜、网络贵、读带宽是一切——复制赢。这是正确选择，但常规选择会是 EC。

2. **USRBIO，作为 "FUSE 太慢"的非内核解法。** 多数团队要么接受 FUSE 的 ~40 万 IOPS 上限，要么写 kernel module 接收 kernel module 痛苦。3FS 找到第三条路：**用户态 `io_uring`**。任何被 FUSE 卡住的数据面 infra 团队都该照抄这个设计。

第三条 meta 启示：**DeepSeek 把存储当成一等 infra 问题**。多数 ML 栈把训练硬装到现有集群里（Lustre、NFS、S3-compatible）。DeepSeek 自己写了文件系统，因为现有方案在卡 V3 速率。这部分投入是 V3 论文没有充分计入的成本故事。

## 参考

- 3FS 仓库：https://github.com/deepseek-ai/3FS
- 设计文档：`docs/design_notes.md`（架构经典文档）
- USRBIO API 参考：`src/lib/api/UsrbIo.md`
- smallpond（基于 3FS 的数据预处理框架）：https://github.com/deepseek-ai/smallpond
- CRAQ 原论文：Terrace & Freedman，_Object Storage on CRAQ_，USENIX ATC '09
- FoundationDB：https://apple.github.io/foundationdb/
