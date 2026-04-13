# 3FS — Source Walkthrough

- **Repo**: [deepseek-ai/3FS](https://github.com/deepseek-ai/3FS) (Fire-Flyer File System)
- **Released**: 2025-02 (Open Source Week, Day 5)
- **Companion**: [`deepseek/v3-tech-report/`](../../v3-tech-report/) — 3FS is the data plane behind V3 training and R1/V3 inference KV cache

## What this repo is

A **disaggregated, RDMA-native distributed file system** purpose-built for AI workloads. Delivers a POSIX-ish file interface on top of NVMe SSDs spread across hundreds of storage nodes, with **strong consistency** via CRAQ chain replication, **stateless metadata** backed by FoundationDB, and a **`io_uring`-style async zero-copy API (USRBIO)** for performance-critical clients. Reported peak **6.6 TiB/s aggregate read** on a 180-storage-node cluster; **3.66 TiB/min** GraySort throughput on 25 storage + 50 compute nodes; KVCache reads at **40 GiB/s peak** per cluster in production.

3FS is the storage layer that makes V3 training data loaders and R1/V3 KVCache offload work without becoming the bottleneck. Without it, the cost story in the V3 paper does not close.

## Four target workloads (stated in README)

| Workload | What 3FS gives it |
|---|---|
| **Data preparation** | Hierarchical directories, atomic rename of large dirs, recursive delete |
| **Dataloaders** | Random access to training samples across compute nodes — no prefetch, no shuffle |
| **Checkpointing** | High-throughput parallel writes, `fsync` semantics |
| **KVCache for inference** | High-throughput, large-capacity alternative to DRAM caching |

That last one is the most distinctive: 3FS is being used as a **DRAM-substitute KV cache tier** for LLM inference, where caches that don't fit in HBM or DRAM spill to NVMe via 3FS at network speed.

## Architecture

```
                   ┌──────────────────┐
                   │ Cluster Manager  │ ← heartbeats, membership, chain tables
                   │ (primary + repl) │ ← metadata in FoundationDB
                   └──────────────────┘
                            ▲
            ┌───────────────┼───────────────┐
            │               │               │
   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
   │ Metadata Svc   │ │ Storage Svc    │ │ Client         │
   │ (stateless,    │ │ (per-SSD, CRAQ │ │ (FUSE + native │
   │  on FoundationDB)│ │  chain repl)  │ │  USRBIO)       │
   └────────────────┘ └────────────────┘ └────────────────┘
                            ▲
                     RDMA (IB / RoCE)
```

Four components, all on RDMA. Notable architectural choices:

- **Stateless metadata service** — file metadata lives in FoundationDB (transactional KV with serializable snapshot isolation). Meta services are restartable / upgradable without downtime; clients can fail over to any meta service.
- **CRAQ for storage** — chain replication with apportioned queries: writes propagate down the chain, but reads can hit any replica. Keeps strong consistency while letting read traffic scale linearly with replica count.
- **No special locality** — applications access storage "in a locality-oblivious manner." There is no concept of "this client should read from that node." The RDMA fabric is fast enough to make locality optimization unnecessary.

## Repository layout

```
3FS/
├── src/
│   ├── client/          # client-side RPC and connection management
│   ├── core/, common/   # shared utilities, RPC, serialization
│   ├── meta/            # metadata service (file system semantics)
│   ├── storage/         # storage service (chunk store on SSDs)
│   ├── mgmtd/           # cluster manager (membership, chain tables)
│   ├── fdb/             # FoundationDB integration
│   ├── kv/              # key-value abstraction
│   ├── fuse/            # FUSE daemon
│   ├── lib/             # USRBIO native client API (see UsrbIo.md)
│   ├── memory/          # RDMA memory management
│   ├── monitor_collector/, analytics/
│   ├── migration/, tools/
│   ├── fbs/             # FlatBuffers schemas
│   └── stubs/, simple_example/
├── hf3fs/, hf3fs_fuse/, hf3fs_utils/   # FUSE binaries and Python utils
├── benchmarks/          # fio engine for USRBIO benchmarking
├── deploy/              # cluster setup
├── docs/
│   ├── design_notes.md  # the canonical architecture doc — read this first
│   ├── metrics.md
│   └── images/
├── specs/               # P/TLA+ specifications
├── configs/, dockerfile/, patches/, third_party/
├── Cargo.toml + Cargo.lock   # Rust components
├── CMakeLists.txt            # main C++ build
└── setup.py
```

This is by far the largest of the OSW repos — a real distributed system, not a kernel library. C++ for the hot paths plus Rust for some components, FoundationDB as a hard dependency, libfuse 3.16+, RDMA verbs.

## CRAQ in the storage service

Chain Replication with Apportioned Queries works as follows:

- Writes go to the **head** of a chain, propagate node-by-node to the **tail**. Tail acks back; only then is the write durable.
- Reads can go to **any** node in the chain. Each node tracks the version of its local copy and which versions are "clean" (acknowledged by tail). If asked for a not-yet-clean version, the node forwards to the tail to confirm.

Read throughput scales with chain replicas; write throughput is bounded by chain length but writes for V3-style training (checkpoint, log) are a small fraction of bytes vs reads (dataloader).

### Chain tables and graceful recovery

Each file is split into chunks; each chunk lives on a **chain** of storage targets (= disjoint SSDs across nodes). Chains are organized into **chain tables**, one per data-placement policy (e.g. one for batch jobs, one for online services).

A naive chain layout has a problem: when SSD `A` dies, all of `A`'s chains' read traffic redirects to the same handful of partner SSDs (`B`, `C`), which immediately saturate. The design notes give an integer-programming-derived layout where `A` is paired with **every other SSD across its chains**, so a failure spreads to `1/N` extra load each, not `1/2`. Recovery from a failed disk takes hours; this layout keeps the cluster usable that whole time.

This is the kind of operational detail that distinguishes a "research" distributed FS from a production one.

## Stateless metadata on FoundationDB

`src/meta/` plus `src/fdb/`. Metadata is two key spaces:

- **`INOD<inode_id>`** → inode value (attrs, file length, chunk size, chain table id, shuffle seed). Inode IDs are little-endian-encoded so they spread across FDB shards.
- **`DENT<parent_inode><name>`** → child inode id + type. Range-scan a `DENT` prefix = `readdir`.

All metadata operations are FDB transactions with serializable snapshot isolation. Read-only ops (`stat`, `lookup`) use read-only txns; writes (`create`, `link`, `unlink`, `rename`) use read-write txns; FDB handles conflict detection and the meta service retries on conflict.

The result: **multiple meta service replicas can serve in parallel** without coordination, because FDB is the single source of truth.

### Dynamic file length

A clever optimization. File length lives in the inode but is **stale** during active writes — clients periodically (5s default) report their max write position to the meta service, which adopts it as the new length if no concurrent truncate. This avoids a per-write meta op.

On `close`/`fsync` the meta service queries the storage service for the **exact** length of the last chunk. To avoid querying all 200 chains in the stripe, each inode tracks **how many chains have been written so far** (starts at 16, doubles when needed). Small files only pay for chains they actually use.

## USRBIO — the native client API

`src/lib/api/UsrbIo.md` (the API reference). The motivating problem (paraphrased from the design notes):

> FUSE handles ~400K 4KiB reads/s before its shared spin-locked queue saturates. `perf` shows the kernel spin lock dominates CPU time. Bandwidth of SSDs and RDMA is not fully utilized.

For random small reads (typical of dataloaders sampling individual training examples), FUSE is the bottleneck before the network or SSDs are. Implementing the client as a kernel module would fix this but introduces other problems (kernel panics, no log on crash, restart-the-machine upgrades).

3FS's answer: **a native client embedded in the FUSE daemon**, exposing an `io_uring`-style API:

- **`Iov`** — a large memory region shared between the user process and the native client. The client registers it as RDMA pinned memory once. All reads land here, all writes are sourced from here. **Zero copy.**
- **`Ior`** — a small ring buffer (request queue + completion queue) shared between user and client. The user enqueues read/write requests; native client threads dequeue and batch them into RDMA RPCs to storage services. `io_depth` controls batch size; multiple rings allow multi-threaded apps to avoid lock contention on a single ring.

`open()`/`close()`/`stat()` still go through FUSE (POSIX semantics, easy to migrate code). I/O on a registered fd uses USRBIO — no kernel/user copy, no syscall per op, no FUSE queue lock.

This is the **single most important component for AI workloads** in the repo. A standard FUSE deployment of 3FS gets nowhere near the 6.6 TiB/s number; USRBIO is what makes it possible.

## Performance, in the design's own framing

The README's three benchmarks are picked deliberately to map to the four workloads:

- **6.6 TiB/s aggregate read** on 180 storage nodes — saturates SSDs and the RDMA fabric simultaneously, demonstrating that **storage scales linearly with node count when locality is ignored**.
- **3.66 TiB/min GraySort** via the [smallpond](https://github.com/deepseek-ai/smallpond) framework — the data-prep workload, showing 3FS as a shuffle substrate for analytics.
- **40 GiB/s KVCache reads** per cluster — the inference KVCache tier, with garbage-collection IOPS published alongside (because GC is what kills naive cache-on-disk designs).

## What's reusable outside DeepSeek

- **The CRAQ + chain-tables-with-balanced-recovery pattern** is portable to any distributed FS targeting RDMA fabrics. The integer-programming-derived chain layout is the kind of detail most academic systems skip.
- **Stateless metadata on FoundationDB** is a strong template for any FS where you want HA without rolling your own consensus. FDB does the hard part.
- **The USRBIO `io_uring`-style design** is reusable as a pattern for any user-space service trying to escape FUSE's perf ceiling — useful far beyond DeepSeek's setup.
- **KVCache-on-FS** as an architecture is genuinely novel at this scale. Anyone running large-scale LLM inference should look at whether their KV cache can spill to network NVMe instead of being capped at HBM/DRAM.

## What's not

- **Hard dependencies**: FoundationDB, libfuse 3.16+, RDMA fabric (IB or RoCE), specific Linux distros / clang versions. Stand-up cost is non-trivial.
- **C++ build complexity** plus Rust toolchain. Not weekend-project friendly.
- **No POSIX guarantees in some edge cases** — e.g. file length under concurrent writes is eventually consistent (intentional, but watch for it).
- **FUSE on Linux 5.x doesn't support concurrent writes to a single file**. 3FS works around this by encouraging multi-file write patterns; if your app insists on single-file concurrent writes, FUSE bites you regardless.
- **The KVCache use case requires custom integration**. The repo gives you a fast FS; mapping LLM KV blocks onto it is your job (see [smallpond](https://github.com/deepseek-ai/smallpond) and DeepSeek's inference docs for hints).

## Commentary

3FS is the OSW release that's hardest to appreciate without operating at scale, and **the most overlooked one in the open discussion**. It's not a kernel, not a clever schedule, not a model — it's a 100k+ LOC distributed system that quietly answers "where do training samples and KV cache actually live" for V3-scale workloads. Two design choices stand out as broadly applicable:

1. **The CRAQ + chain-table choice over erasure coding**. Most modern distributed FS use erasure coding for storage efficiency. 3FS chose replication because (a) read scaling is better, (b) chain tables let you engineer recovery behavior (the integer-programming layout), and (c) failure recovery is faster. For AI workloads — where storage is cheap, network is expensive, and read bandwidth is everything — replication wins. This is the right call but the conventional one would have been erasure coding.

2. **USRBIO as the answer to "FUSE is too slow" without going kernel.** Most teams either accept FUSE's ~400K IOPS ceiling or write a kernel module and inherit kernel-module pain. 3FS found a third path: a userspace `io_uring`. For any team building data-plane infra that bumps into FUSE, this is the design to copy.

The third lesson is meta: **DeepSeek treats storage as a first-class infra question, not an afterthought.** Most ML stacks bolt training onto whatever the cluster has (Lustre, NFS, S3-compatible). DeepSeek built a file system because the existing options were rate-limiting V3. That investment is part of the cost story the V3 paper doesn't fully credit.

## References

- 3FS repo: https://github.com/deepseek-ai/3FS
- Design notes: `docs/design_notes.md` (the canonical architecture doc)
- USRBIO API reference: `src/lib/api/UsrbIo.md`
- smallpond (data-prep framework on 3FS): https://github.com/deepseek-ai/smallpond
- CRAQ original paper: Terrace & Freedman, _Object Storage on CRAQ_, USENIX ATC '09
- FoundationDB: https://apple.github.io/foundationdb/
