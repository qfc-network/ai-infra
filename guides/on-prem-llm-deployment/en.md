# On-Premise LLM Deployment — From Mac Studio to GPU Cluster

> A practical deployment guide, not a paper analysis. Target audience: engineering teams evaluating how to run open-weight LLMs on their own hardware — from single-machine prototyping to production GPU clusters.

## One-line conclusion

**Start with a Mac Studio + Ollama to validate the use case in one week; graduate to vLLM on GPU when you need concurrency; scale to a K8s GPU cluster when the whole company depends on it.** The hardware you need depends entirely on how many concurrent users you serve and what latency you can tolerate.

---

## I. Decision framework — what drives the architecture

Before picking hardware, answer three questions:

| Question | Drives |
|----------|--------|
| How many concurrent users? | GPU count, memory bandwidth budget |
| What model size? | Memory capacity (VRAM or unified memory) |
| What latency / throughput target? | Quantization level, prefill/decode separation |

LLM inference is **memory-bandwidth-bound** during the decode phase — every generated token reads the full model weights from memory once. This is why memory bandwidth, not FLOPS, is the bottleneck for most serving workloads.

---

## II. Tier 1 — Mac Studio / Apple Silicon (1–5 users)

### Why it works

Apple's **Unified Memory Architecture (UMA)** lets the GPU access the full system memory pool. A Mac Studio with 192 GB unified memory can load a 70B INT4 model that would require 2× A100 80 GB in a discrete GPU setup. No driver installation, no CUDA, no container runtime — `brew install ollama` and you're serving.

### Hardware matrix

| Chip | Unified Memory | Memory Bandwidth | 70B INT4 speed | 405B INT4 |
|------|---------------|-------------------|-----------------|-----------|
| M2 Ultra | 192 GB | ~800 GB/s | ~10–15 tok/s | Won't fit |
| M3/M4 Ultra | 192–512 GB | ~800–900 GB/s | ~12–18 tok/s | Fits in 512 GB config |
| M4 Max | 128 GB | ~550 GB/s | ~8–12 tok/s | Won't fit |

### Max-config model compatibility (M4 Ultra 512 GB)

Model memory ≈ parameters × bytes-per-parameter + KV cache overhead. With 512 GB unified memory, ~430–460 GB is available after OS and KV cache reservation.

**Memory footprint by precision:**

| Precision | Bytes/param | 70B | 405B | 671B (DeepSeek V3) |
|-----------|------------|-----|------|---------------------|
| FP16 | 2 | 140 GB | 810 GB | 1.34 TB |
| INT8 | 1 | 70 GB | 405 GB | 671 GB |
| INT4 | 0.5 | 35 GB | ~203 GB | ~336 GB |

**What actually runs:**

| Model | Precision | Fits? | Speed | Notes |
|-------|-----------|-------|-------|-------|
| Llama 3 70B | FP16 | Easy (140 GB) | ~15–20 tok/s | Best quality at this size |
| Llama 3 70B | INT4 | Easy (35 GB) | ~20–25 tok/s | Best speed/quality tradeoff |
| Qwen2.5 72B | FP16 | Easy | ~15–18 tok/s | |
| Llama 3.1 405B | INT4 | Yes (~203 GB) | ~3–5 tok/s | Usable but slow |
| Llama 3.1 405B | INT8 | Tight (~405 GB) | ~1.5–3 tok/s | Very slow |
| DeepSeek V3 671B | INT4 | Yes (~336 GB) | ~2–4 tok/s | MoE — only 37B active per token, faster than 405B dense |

**Key takeaway:** 70B INT4 is the sweet spot — fast and high quality. 405B fits but ~800 GB/s bandwidth reading 200+ GB weights means ~250 ms per token. DeepSeek V3 is a special case: 671B total but MoE activates only 37B per token, so it runs faster than 405B dense despite being "larger."

**Buying advice:** If you'll mainly run 70B models, the **192 GB config (~$7,000) is sufficient** — save the $5,000+ premium for a future GPU upgrade. The 512 GB config only pays off for 400B+ models.

### Recommended stack

```
Model runtime:    llama.cpp (via Ollama) or MLX
API layer:        Ollama (OpenAI-compatible /v1/chat/completions)
Frontend:         Open WebUI or LobeChat
```

Startup:

```bash
brew install ollama
ollama serve &
ollama pull llama3:70b-instruct-q4_K_M
```

Ollama exposes an OpenAI-compatible API at `localhost:11434/v1/` — existing code that calls OpenAI just needs a base URL change.

### When to use

- **Prototyping & validation** — prove the use case before investing in GPUs
- **Privacy-sensitive workloads** — data never leaves the machine
- **Developer inner loop** — local coding assistant, RAG experiments
- **< 5 concurrent users** — single-user experience is good; degrades quickly under load

### When to move on

- Response time doubles under 3+ concurrent requests
- You need > 20 tok/s sustained throughput
- The model must serve an internal API with SLA guarantees

---

## II-B. NVIDIA DGX family — turnkey appliances

NVIDIA sells pre-integrated AI systems at every scale. The key advantage over DIY builds: validated hardware/software stack, enterprise support, and DGX OS with optimized drivers and container runtimes pre-installed.

### DGX Spark (desktop, ~$3,000)

NVIDIA's answer to Mac Studio. Ships with the **Grace Blackwell GB10 Superchip** — an ARM Grace CPU + Blackwell GPU on a single chip with unified memory, similar to Apple's UMA approach but with CUDA support.

| Spec | DGX Spark | Mac Studio M4 Ultra (max) |
|------|-----------|---------------------------|
| Memory | 128 GB unified | up to 512 GB unified |
| Memory bandwidth | ~273 GB/s | ~800 GB/s |
| AI compute (INT8) | 209 TOPS | ~74 TOPS |
| CUDA / Tensor Cores | Yes | No |
| OS | DGX OS (Ubuntu-based) | macOS |
| Price | ~$3,000 | $4,000–14,000 |

**What it runs:**

| Model | Precision | Fits? | Notes |
|-------|-----------|-------|-------|
| Llama 3 7–14B | FP16 | Easy | Good single-user speed |
| Llama 3 70B | INT4 | Yes (~35 GB) | Slower than Mac Studio due to lower bandwidth |
| Llama 3.1 405B | INT4 | No (~203 GB) | Only 128 GB — won't fit |

**Verdict:** Cheaper than Mac Studio, has CUDA ecosystem access (TensorRT-LLM, cuDNN), but **128 GB memory is the hard ceiling** — can't load 200B+ models. The lower memory bandwidth (~273 vs ~800 GB/s) means slower tok/s than a Mac Studio for the same model. Best for developers who need CUDA compatibility at a low price point. Two DGX Spark units can be linked via ConnectX for 256 GB combined memory.

### DGX Station (workstation, ~$50–125K)

A workstation-form-factor GPU server. The current generation ships with Blackwell GPUs.

| Generation | GPUs | VRAM | Memory BW | Approx. price |
|-----------|------|------|-----------|--------------|
| DGX Station (A100) | 4× A100 80 GB | 320 GB | 8 TB/s | ~$50K (used) |
| DGX Station (B200) | 1× B200 | 192 GB HBM3e | 8 TB/s | ~$125K |

Sits under a desk, runs on standard power (single-phase). Good for a team of 10–30 serving 70B models with real GPU throughput. Runs the full NVIDIA AI Enterprise stack out of the box.

### DGX H100 / H200 / B200 (data center, $300K–$600K+)

The flagship data center appliances. Each is a single server node with 8 GPUs interconnected via NVSwitch.

| System | GPUs | VRAM (total) | GPU-GPU BW | FP8 compute | Approx. price |
|--------|------|-------------|------------|-------------|--------------|
| DGX H100 | 8× H100 SXM | 640 GB HBM3 | 900 GB/s NVLink | 16 PFLOPS | ~$300K |
| DGX H200 | 8× H200 SXM | 1.13 TB HBM3e | 900 GB/s NVLink | 16 PFLOPS | ~$400K |
| DGX B200 | 8× B200 SXM | 1.4 TB HBM3e | 1.8 TB/s NVLink | 36 PFLOPS | ~$500–600K |

**What a single DGX node handles:**

| Model | System | Precision | Throughput (batch serving) |
|-------|--------|-----------|---------------------------|
| Llama 3 70B | DGX H100 | FP8, TP=4 | ~2,000+ tok/s |
| Llama 3.1 405B | DGX H100 | FP8, TP=8 | ~500–800 tok/s |
| Llama 3.1 405B | DGX H200 | FP16, TP=8 | ~400–600 tok/s (no quant needed) |
| DeepSeek V3 671B | DGX B200 | FP8 | Fits in 1.4 TB; MoE routing benefits from NVLink |

### DGX SuperPOD (rack/cluster scale)

Multiple DGX nodes in a rack with InfiniBand/NVLink networking. NVIDIA sells these as integrated rack-scale systems (DGX SuperPOD with 8–32+ DGX nodes). This is the path to multi-node training and large-scale serving. Starts at ~$2–5M for a minimal SuperPOD configuration.

### DGX vs. DIY — when to buy which

| Factor | DGX | DIY (whitebox + GPUs) |
|--------|-----|----------------------|
| Time to deploy | Days (pre-validated) | Weeks–months (driver/firmware/cooling) |
| Support | NVIDIA AI Enterprise support | Community + vendor |
| Cost per GPU | Higher (~1.5–2× premium) | Lower |
| Flexibility | Fixed configs | Any GPU mix, custom cooling/networking |
| Software | DGX OS, Base Command, pre-tuned NCCL | Manual tuning |
| Best for | Teams that value uptime over savings | Teams with GPU cluster ops experience |

**Rule of thumb:** If you don't have a dedicated GPU cluster ops team, DGX saves you months of integration pain. If you have the expertise and want to optimize cost, DIY with OEM servers (Dell, Supermicro, Lambda) gives better price/performance.

---

## II-C. Consumer NVIDIA GPUs (RTX 4090 / 5090)

The budget entry point — real CUDA GPUs at consumer prices.

| GPU | VRAM | Memory BW | FP16 compute | Price |
|-----|------|-----------|-------------|-------|
| RTX 4090 | 24 GB | 1 TB/s | 165 TFLOPS | ~$1,600 |
| RTX 5090 | 32 GB | 1.79 TB/s | 209 TFLOPS | ~$2,000 |

**What they run:** 7B–14B FP16 on a single card. 70B INT4 requires 2–3 cards but lacks NVLink — tensor parallelism over PCIe is 5–10× slower than NVLink, so multi-card TP is inefficient.

**Advantages:** Cheapest CUDA hardware; widely available; strong community support (llama.cpp, vLLM all work).

**Disadvantages:** Small VRAM (24–32 GB per card); no NVLink between consumer cards; no ECC memory (unreliable for training); power-limited (450 W TDP each, can't scale in standard workstations).

**Best for:** Budget-constrained teams running 7B–14B models, or individual developers who want local CUDA access. A 2× RTX 5090 rig (~$5K total build) is a reasonable dev box for 14B FP16 or 70B INT4 (with TP efficiency penalty).

---

## II-D. AMD Instinct GPUs (MI300X / MI325X)

The main NVIDIA alternative for datacenter inference. The headline advantage: **192–256 GB HBM per card** — a single MI300X can load a 405B INT4 model that requires 4–8 NVIDIA GPUs.

| GPU | HBM | Memory BW | FP16 compute | Price |
|-----|-----|-----------|-------------|-------|
| MI300X | 192 GB HBM3 | 5.3 TB/s | 1.3 PFLOPS | ~$15–20K |
| MI325X | 256 GB HBM3e | 6 TB/s | 1.3 PFLOPS | ~$20–25K |

**Comparison with NVIDIA:**

| Dimension | MI300X (192 GB) | H100 SXM (80 GB) | H200 SXM (141 GB) |
|-----------|----------------|-------------------|-------------------|
| HBM capacity | 192 GB | 80 GB | 141 GB |
| Memory BW | 5.3 TB/s | 3.35 TB/s | 4.8 TB/s |
| 405B INT4 | Single card | 4–8 cards (TP) | 2–4 cards (TP) |
| Software ecosystem | ROCm + vLLM/SGLang | CUDA (full ecosystem) | CUDA (full ecosystem) |

**Advantages:** Massive per-card HBM; higher memory bandwidth than H100; competitive price per GB of HBM; vLLM and SGLang both have ROCm backends.

**Disadvantages:** ROCm ecosystem is less mature than CUDA — FlashAttention AMD ports lag behind, some custom kernels need porting, debugging tools are weaker. NVIDIA-specific optimizations (TensorRT-LLM, FP8 recipes) don't transfer.

**Best for:** Teams willing to accept the software ecosystem tradeoff in exchange for more HBM per card and lower cost per GB. Particularly compelling for **large-model inference** (405B+) where the alternative is multi-card NVIDIA setups.

---

## II-E. Intel Gaudi 2 / 3

Intel's AI accelerator (from the Habana Labs acquisition). Positioned as the cost-competitive alternative to NVIDIA.

| Accelerator | HBM | Memory BW | Approx. price |
|-------------|-----|-----------|--------------|
| Gaudi 2 | 96 GB HBM2e | 2.45 TB/s | ~$6–8K |
| Gaudi 3 | 128 GB HBM2e | 3.7 TB/s | ~$12–15K |

Gaudi 3 performance approaches H100 for LLM inference at roughly half the per-card cost. vLLM has a Gaudi backend. Intel offers 8-card Gaudi servers (HL-325L) as integrated systems.

**Advantages:** Significantly lower price per card; 128 GB HBM on Gaudi 3 is larger than H100's 80 GB; Intel enterprise support.

**Disadvantages:** Small community — when you hit an issue, Stack Overflow won't help. Software stack (Synapse AI) is less mature than both CUDA and ROCm. Fewer optimized kernels available.

**Best for:** Intel-aligned enterprises; inference workloads where cost per token matters more than time-to-deploy.

---

## II-F. OEM GPU servers — the middle ground

Between "buy a DGX" and "build from parts," OEM vendors sell pre-built GPU servers with validated configurations:

| Vendor | Product line | GPUs supported | Key advantage |
|--------|-------------|---------------|---------------|
| Dell | PowerEdge XE9680 | 8× H100/H200/B200 SXM | Enterprise support, 10–20% cheaper than DGX |
| HPE | ProLiant DL380a | 8× H100 | GreenLake cloud management integration |
| Supermicro | GPU SuperServer | Any (most flexible) | Lowest cost, most config options |
| Lambda | Hyperplane | 8× H100/H200 | Pre-installed ML stack, AI-team-friendly |

**Typical cost:** 10–30% less than equivalent DGX for the same GPU count. You give up DGX OS and Base Command but keep hardware warranty and validated cooling/power.

**Best for:** Teams with some ops capability that want hardware support without the DGX premium.

---

## II-G. Other form factors

### Multi-Mac cluster

Link multiple Mac Studios via Thunderbolt 5 or 10GbE using frameworks like [Exo](https://github.com/exo-explore/exo):

- Example: 4× Mac Studio 192 GB = 768 GB total memory — can run 405B FP16
- **Catch:** Inter-node bandwidth (~20–40 Gbps Thunderbolt, ~10 Gbps Ethernet) is ~100× slower than NVLink. Tensor parallelism is impractical; only pipeline parallelism works, with high latency per token.
- **Best for:** Teams that already own multiple Macs and want to experiment with large models without buying GPU hardware. Not viable for production serving.

### Specialized inference accelerators

| Vendor | Chip | Form factor | Notes |
|--------|------|-------------|-------|
| SambaNova | SN40L (DataScale) | On-prem appliance | Enterprise RAG-optimized; available for on-prem deployment |
| Groq | LPU | Cloud API (primary) | Ultra-low latency inference; limited on-prem options |
| Cerebras | WSE-3 | Cloud / on-prem | Wafer-scale chip; extreme batch throughput; multi-million dollar systems |

These are niche — consider them only if your workload has specific requirements (ultra-low latency, extreme batch throughput) that general-purpose GPUs don't meet cost-effectively.

---

## II-H. Hardware selection summary

| Budget | Best option | What it runs |
|--------|-----------|-------------|
| < $3K | DGX Spark / RTX 5090 single card | 7B–14B FP16 |
| $3–8K | Mac Studio 192 GB / 2× RTX 5090 | 70B INT4 |
| $8–20K | 1× A100 (used) or 1× MI300X | 70B FP8/INT4 with real throughput |
| $20–60K | 2× H100 / DGX Station (used) | 70B FP16 or 405B INT4 |
| $60–150K | 4× MI300X server / DGX Station (B200) | 405B FP8; multi-model serving |
| $150–400K | DGX H100 / Dell XE9680 / 8× MI300X | 405B+ at scale; 100+ users |
| $400K+ | DGX H200/B200 / SuperPOD | 671B full-precision; training + inference |

---

## III. Tier 2 — Single-node GPU server (10–50 users)

### Hardware options

| Config | VRAM | Memory BW (total) | 70B INT4 throughput | Approx. cost |
|--------|------|--------------------|---------------------|-------------|
| 2× RTX 5090 | 64 GB | 3.58 TB/s | ~30–50 tok/s (PCIe TP penalty) | ~$5K build |
| 1× MI300X | 192 GB | 5.3 TB/s | ~100–150 tok/s | ~$15–20K |
| 1× A100 80 GB | 80 GB | 2 TB/s | ~40–60 tok/s | ~$15K (used) |
| 2× A100 80 GB | 160 GB | 4 TB/s | ~80–120 tok/s | ~$30K |
| 1× H100 SXM | 80 GB | 3.35 TB/s | ~80–100 tok/s | ~$30K |
| 2× H100 SXM | 160 GB | 6.7 TB/s | ~150–200 tok/s | ~$60K |
| DGX Station (A100) | 320 GB | 8 TB/s | ~200+ tok/s | ~$50K (used) |

The jump from Apple Silicon to a single A100 is roughly **5–10× in throughput** for the same model, driven entirely by HBM bandwidth.

### Recommended stack

```
Inference engine:  vLLM or SGLang
API:               vLLM's built-in OpenAI-compatible server
Reverse proxy:     Nginx (TLS termination, basic auth)
Process manager:   systemd or Docker Compose
Monitoring:        Prometheus + Grafana (GPU util, TTFT, TPS)
Frontend:          Open WebUI / LobeChat / custom
```

Launch:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-70B-Instruct-AWQ \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

### Key decisions at this tier

**Quantization strategy:**

| GPU | Sweet spot | Why |
|-----|-----------|-----|
| H100 | FP8 (W8A8) | Native FP8 Tensor Core support; near-lossless |
| A100 | INT4 (AWQ/GPTQ) | No FP8 hardware; INT4 halves memory, fits larger models |
| Either | W8A8 SmoothQuant | Good middle ground if INT4 quality isn't acceptable |

**Prefix caching:** If your workload has repeated system prompts (customer service bots, coding assistants, RAG with fixed retrieval templates), enable prefix caching in vLLM/SGLang. This can save **30–60% of prefill compute** — see the [Prefix Caching](../../foundational/prefix-caching/) entry.

**Chunked prefill:** For long-prompt workloads (RAG with large context), enable chunked prefill to prevent long prefills from stalling decode batches — see [Chunked Prefill](../../foundational/chunked-prefill/).

---

## IV. Tier 3 — Multi-node GPU cluster (50–500+ users)

### Architecture

```
                    Load Balancer (Nginx / Envoy)
                           │
                    API Gateway
              (auth, rate limiting, usage tracking)
                           │
              ┌────────────┼────────────┐
              │            │            │
         vLLM pod 0   vLLM pod 1   vLLM pod N
         (TP=4/8)     (TP=4/8)     (TP=4/8)
              │            │            │
              └────────────┼────────────┘
                           │
                    Model Storage
                  (MinIO / NFS / S3)
```

### Infrastructure stack

| Layer | Recommendation |
|-------|---------------|
| Orchestration | Kubernetes + NVIDIA GPU Operator + MIG (if sharing GPUs) |
| Inference engine | **vLLM** (flexibility, community) or **TensorRT-LLM** (peak H100 perf) |
| Model registry | Internal HuggingFace Mirror or MinIO with model versioning |
| API gateway | Kong / custom — enforce per-team rate limits, log usage for chargeback |
| Monitoring | Prometheus + Grafana: TTFT p50/p99, tokens/sec, GPU utilization, queue depth |
| Frontend | Open WebUI / LobeChat / company-specific UI |

### Advanced optimizations

**Prefill / decode disaggregation (DistServe pattern):**

If your workload is prefill-heavy (long documents, RAG), consider separating prefill and decode onto different node pools. Prefill is compute-bound; decode is memory-bandwidth-bound. Mixing them on the same GPU forces suboptimal batching. See [DistServe](../../foundational/distserve/).

**Multi-model serving:**

Run different models (e.g., 7B for simple tasks, 70B for complex ones) behind a router that selects based on request complexity. This improves cost efficiency — most requests don't need the largest model.

**Scaling policy:**

| Signal | Action |
|--------|--------|
| Queue depth > threshold | Scale up inference pods |
| GPU utilization < 30% for 10 min | Scale down |
| p99 TTFT > SLA | Add prefill-optimized nodes |

---

## V. Tier 4 — Adding fine-tuning capability

If you need domain adaptation beyond prompting and RAG:

| Need | Approach | Hardware |
|------|----------|----------|
| Lightweight adaptation | LoRA / QLoRA | Single A100 for 70B QLoRA |
| Full fine-tuning | DeepSpeed ZeRO-3 or FSDP | 8+ GPUs |
| RLHF / GRPO alignment | verl (HybridFlow) | 8+ GPUs, see [verl](../../foundational/verl/) |
| Data labeling | Label Studio + internal annotation | CPU-only |
| Experiment tracking | W&B / MLflow | CPU-only |

For most teams, **LoRA fine-tuning is sufficient** — it adapts the model to your domain with 1% of the parameters and can run on a single GPU. Full fine-tuning and RLHF are only needed for significant behavior changes.

---

## VI. Cost reference

| Setup | Hardware cost | Power (monthly) | Serves |
|-------|-------------|-----------------|--------|
| RTX 5090 single card build | ~$3–5K | ~$50 | 1–3 users (7B–14B) |
| DGX Spark (128 GB) | ~$3,000 | ~$10 | 1–3 users |
| Mac Studio M4 Ultra 192 GB | ~$8,000 | ~$15 | 1–5 users |
| 1× MI300X server | ~$15–20K | ~$250 | 10–30 users |
| 1× A100 80 GB server | ~$15–20K (used) | ~$200 | 10–30 users |
| DGX Station (A100, used) | ~$50K | ~$300 | 10–50 users |
| Dell XE9680 (8× H100) | ~$220–270K | ~$1,500 | 100–300 users |
| 8× MI300X server | ~$150–200K | ~$1,200 | 100–300 users |
| DGX H100 | ~$300K | ~$1,500 | 100–300 users |
| DGX H200 | ~$400K | ~$1,500 | 200–500 users |
| 4× DGX H100 (SuperPOD) | ~$1.5M+ | ~$6,000 | 500+ users |
| Cloud H100 (bridge) | $2–3/GPU/hr | — | Flexible |

**Cloud as a bridge:** Rent GPU instances (Lambda, CoreWeave, RunPod) to validate at scale before committing to hardware purchase. A month of 8× H100 rental (~$15K) is much cheaper than buying the wrong config.

---

## VII. Decision flowchart

```
Start
  │
  ├─ "Just exploring, < 5 people"
  │   ├─ Need CUDA? → DGX Spark ($3K, 128 GB)
  │   └─ Max memory? → Mac Studio + Ollama ($5–14K, up to 512 GB)
  │
  ├─ "Need an internal API, 10–50 users"
  │   ├─ Want turnkey? → DGX Station ($50–125K)
  │   ├─ Max HBM per card? → MI300X ($15–20K, 192 GB single card)
  │   └─ Want flexibility? → 1–2× A100/H100 + vLLM ($15–60K)
  │
  ├─ "Company-wide service, SLA required"
  │   ├─ No GPU ops team? → DGX H100/H200 ($300–400K, enterprise support)
  │   ├─ Want NVIDIA but cheaper? → Dell XE9680 / Lambda ($220–270K)
  │   ├─ Want AMD alternative? → 8× MI300X server ($150–200K)
  │   └─ Full DIY? → K8s cluster + vLLM/TRT-LLM ($250K+)
  │
  └─ "Need to fine-tune for our domain"
      → Add training nodes: LoRA (1 GPU) or full FT (DGX node / 8+ GPUs)
```

---

## References

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention, continuous batching, OpenAI-compatible API
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention prefix caching, structured generation
- [Ollama](https://ollama.com) — llama.cpp wrapper with one-command setup
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework optimized for Apple Silicon
- [Open WebUI](https://github.com/open-webui/open-webui) — Self-hosted ChatGPT-style frontend
- [Exo](https://github.com/exo-explore/exo) — Multi-Mac distributed inference cluster
- [ROCm](https://rocm.docs.amd.com/) — AMD's GPU compute platform (CUDA alternative)
- Related entries in this repo: [PagedAttention/vLLM](../../foundational/paged-attention/), [Prefix Caching](../../foundational/prefix-caching/), [DistServe](../../foundational/distserve/), [Chunked Prefill](../../foundational/chunked-prefill/), [Weight Quantization](../../foundational/weight-quantization/), [verl](../../foundational/verl/)
