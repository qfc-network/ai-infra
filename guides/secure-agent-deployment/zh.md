# 私有化与本地部署环境中的 AI 智能体安全部署指南

**目标读者**：已负责管理 Kubernetes/Linux 基础设施的平台工程师和 DevOps 工程师，目前正在受监管或安全敏感环境中部署基于大语言模型的智能体系统。

**前置要求**：熟悉 Kubernetes、Linux 网络和容器安全。不要求具备大语言模型智能体的先验知识。

---

## 1. 本地部署智能体与云端智能体的本质区别

部署云端大语言模型智能体时，每次推理请求的 LLM API 调用都会离开你的网络边界。智能体发起的工具调用——网络搜索、API 查询、代码执行——同样可能触达外部服务。你的数据流经供应商基础设施，由此带来合规性和数据主权方面的全部隐患。

本地部署完全颠覆了这一模式。大语言模型在本地运行（使用 Ollama、vLLM 或 TensorRT-LLM），工具执行在本地完成，数据从不离开你的安全边界。这一架构满足 GDPR 数据驻留要求、HIPAA 受控实体义务，以及国防、金融和医疗健康领域常见的气隙隔离规定。代价是你必须独自承担整个安全攻击面——没有供应商来吸收配置失误的爆炸半径。

### 三类风险面

明确暴露面所在是控制风险的第一步。

**（a）大语言模型本身。** 提示注入是主要威胁：嵌入在检索文档、工具结果或用户消息中的恶意字符串可以指示模型忽略系统提示或执行未经授权的操作。越狱攻击是次要关注点——在受监管环境中，模型可能被要求输出从工具结果中摄取的个人身份信息。开源模型在对抗这些攻击方面的鲁棒性差异显著；模型选型和系统提示加固都至关重要。

**（b）工具执行环境。** 这是本地部署为基础设施工程师带来最新颖风险的领域。配备了 `bash_exec` 工具的智能体，在极端情况下具备与其运行身份在宿主机上相同的能力。通过提示注入劫持工具调用可以升级为任意代码执行。工具执行沙箱不是可选项——它是主要防御层。

**（c）智能体循环状态（上下文窗口）。** 智能体的上下文窗口会持续积累所有内容：系统提示、所有用户轮次、所有工具调用的输入输出以及所有中间推理过程。读取配置文件或日志的工具可能无意中将 API 密钥或数据库密码存入上下文。这个秘密随后对后续的 LLM 调用可见，可能在智能体响应中被逐字复述，或被日志管道捕获。这是一条标准网络控制无法覆盖的数据泄露路径。

---

## 2. 本地大语言模型作为智能体核心

### 模型选型

对于本地智能体部署，当前实用的两个开源权重选择是 **Llama 3.1 70B** 和 **Qwen2.5 32B**（Qwen3 32B 发布后亦可考虑）。两者都能可靠地支持结构化工具调用输出格式——这是智能体应用的硬性要求。并非所有开源模型都能在对抗性提示条件下输出格式正确的 JSON 工具调用。

其他在函数调用方面表现稳定的选择：Mistral-Nemo、Llama 3.1 8B（适用于低延迟低风险任务）以及面向代码密集型智能体的 Qwen2.5-Coder 系列。

**能力与硬件的权衡：**

| 模型 | VRAM（BF16） | VRAM（INT4） | 推荐硬件 |
|---|---|---|---|
| Llama 3.1 70B | 约 140 GB | 约 40 GB | 2×H100 或 4×A100-40GB（BF16）；1×A100-80GB（INT4） |
| Qwen2.5 32B | 约 64 GB | 约 20 GB | 1×H100-80GB 或 2×A100-40GB（BF16）；1×A100-40GB（INT4） |
| Llama 3.1 8B | 约 16 GB | 约 5 GB | 1×A100-40GB 或 1×RTX 4090（INT4） |

对于大多数受监管环境的部署——审计和可靠性比原始吞吐量更重要——**单张 A100-80GB 上的 Llama 3.1 70B INT4** 是实际可行的起点。

### 推理栈：vLLM 与 Ollama

**vLLM** 是多智能体或高并发部署的生产级选择。它支持连续批处理、PagedAttention 内存效率优化、多 GPU 张量并行以及兼容 OpenAI 的 API。安装和启动命令：

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --quantization awq \
  --dtype float16 \
  --max-model-len 32768 \
  --port 8000 \
  --host 0.0.0.0
```

**Ollama** 适用于单人开发场景、低流量内部工具，或运维简单性优先于吞吐量的环境。它将模型管理和推理服务封装在单个二进制文件中：

```bash
ollama pull llama3.1:70b
ollama serve  # 默认监听 localhost:11434
```

两者均在 `/v1/chat/completions` 暴露兼容 OpenAI 的端点。LangGraph 和 AutoGen 等智能体框架通过 `base_url` 参数覆盖即可接入本地推理栈：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used",          # vLLM 默认不需要密钥
    model="meta-llama/Llama-3.1-70B-Instruct",
)
```

### 验证函数调用能力

在将模型部署为智能体核心之前，请验证工具调用格式的合规性：

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_file_contents",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    }],
    messages=[{"role": "user", "content": "读取 /etc/hostname"}],
    tool_choice="auto"
)
print(response.choices[0].message.tool_calls)
```

如果 `tool_calls` 返回 `None` 或格式错误的 JSON，说明该模型在当前量化精度或上下文长度下无法可靠用作智能体核心。可尝试提高模型精度或缩短上下文长度后再重新评估。

---

## 3. 工具调用沙箱

### 威胁模型

配备了不受限制的 `bash_exec` 工具的智能体，实质上是一个等待足够巧妙的提示注入来触发的远程代码执行漏洞。工具沙箱是你的主要防御层——用对待公网暴露服务的同等严格态度来对待它。

### 网络隔离

工具执行器应运行在独立的网络命名空间中，出站防火墙规则只允许访问明确许可的目标地址。在 Kubernetes 中，通过 NetworkPolicy 来强制执行：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tool-executor-egress
  namespace: agent-tools
spec:
  podSelector:
    matchLabels:
      role: tool-executor
  policyTypes:
  - Egress
  egress:
  - to:
    - ipBlock:
        cidr: 10.0.0.0/8          # 仅允许访问内部 API
    ports:
    - protocol: TCP
      port: 443
  # 不存在针对 0.0.0.0/0 的规则——默认拒绝所有其他出站流量
```

对于非 Kubernetes 部署，使用 iptables 配置 OUTPUT 链，丢弃除特定内部 CIDR 和 DNS 之外的所有出站流量：

```bash
iptables -P OUTPUT DROP
iptables -A OUTPUT -d 10.0.0.0/8 -j ACCEPT
iptables -A OUTPUT -d 127.0.0.0/8 -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT  # 仅允许内部 DNS
```

### 系统调用过滤与 Linux 安全模块

为工具执行器容器应用 seccomp 配置文件。以下配置文件阻断最高风险的系统调用：

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "syscalls": [
    {
      "names": ["read", "write", "open", "close", "stat", "fstat", "mmap",
                "mprotect", "munmap", "brk", "rt_sigaction", "rt_sigprocmask",
                "ioctl", "access", "pipe", "select", "dup", "dup2", "nanosleep",
                "getpid", "sendfile", "socket", "connect", "recvfrom", "sendto",
                "recvmsg", "sendmsg", "bind", "listen", "accept", "getsockname",
                "getpeername", "socketpair", "shutdown", "getsockopt", "setsockopt",
                "clone", "fork", "vfork", "execve", "exit", "wait4", "kill",
                "getdents", "getcwd", "chdir", "rename", "mkdir", "rmdir",
                "unlink", "readlink", "chmod", "getrlimit", "getrusage",
                "sysinfo", "times", "getuid", "getgid", "geteuid", "getegid",
                "futex", "openat", "getdents64", "set_tid_address",
                "pread64", "pwrite64", "readv", "writev", "fcntl", "flock",
                "fsync", "fdatasync", "ftruncate", "lseek", "pipe2",
                "epoll_create1", "epoll_ctl", "epoll_wait", "lstat",
                "newfstatat", "exit_group", "tgkill", "arch_prctl",
                "prlimit64", "getrandom", "clock_gettime", "clock_nanosleep"],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

明确缺失的系统调用（即被拦截的）包括：`ptrace`、`mount`、`setuid`、`setgid`、`setns`、`unshare`、`keyctl`、`add_key`、`request_key`、`bpf`、`perf_event_open`。

配合 AppArmor 或 SELinux 使用。执行器的最小化 AppArmor 配置文件示例：

```
profile tool-executor flags=(attach_disconnected) {
  include <abstractions/base>
  /proc/*/status r,
  /tmp/** rw,
  /workspace/** rw,
  deny /etc/shadow r,
  deny /etc/passwd w,
  deny /root/** rw,
  deny /home/** rw,
  deny network raw,
}
```

### Kubernetes Pod 规格：综合配置示例

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: tool-executor
  namespace: agent-tools
spec:
  runtimeClassName: gvisor          # 参见代码执行章节
  securityContext:
    runAsNonRoot: true
    runAsUser: 65534                 # nobody 用户
    runAsGroup: 65534
    fsGroup: 65534
    seccompProfile:
      type: Localhost
      localhostProfile: profiles/tool-executor.json
  containers:
  - name: executor
    image: your-registry/tool-executor:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: workspace
      mountPath: /workspace
    - name: tmp
      mountPath: /tmp
    resources:
      limits:
        memory: "512Mi"
        cpu: "500m"
  volumes:
  - name: workspace
    emptyDir: {}                     # Pod 终止后自动清除
  - name: tmp
    emptyDir: {}
```

### 代码执行：gVisor 与 Firecracker

对于执行任意 Python 代码或 Shell 脚本的智能体，单独的 Docker 提供的隔离强度不足。容器逃逸漏洞确实存在，对抗性 payload 可能以此为目标。请使用更强的隔离原语：

- **gVisor（runsc）**：用户态内核，在所有系统调用到达宿主内核之前将其拦截。安装 gVisor 运行时并在 Pod 规格中设置 `runtimeClassName: gvisor`（如上所示）。大多数工作负载的性能开销为 5%–15%，对工具执行而言完全可接受。
- **Firecracker microVM**：完整的虚拟机隔离，启动时间低于一秒。AWS Lambda 的底层技术。运维复杂度更高，但提供最强的隔离边界。适用于具备计算机操作能力的智能体——该威胁模型详见 `../../anthropic/computer-use/`。

仅使用 Docker（不配合 gVisor 或 Firecracker）仅在工具集完全确定性且不涉及执行不可信代码路径的情况下可以接受。

---

## 4. MCP 服务器权限边界

MCP（模型上下文协议）通过 JSON-RPC 向智能体暴露工具和资源。在本地部署中，你自行托管 MCP 服务器——这意味着你来设计授权模型。

### 每个服务器遵循最小权限原则

不要部署单个暴露所有工具的 MCP 服务器。按工具域和智能体角色进行拆分：

```
智能体角色：代码审查员
  → mcp-server-git（只读：clone、diff、log）
  → mcp-server-static-analysis（只读：lint、scan）

智能体角色：发布管理员
  → mcp-server-git（读写：tag、push）
  → mcp-server-ci（触发流水线）
  → mcp-server-registry（推送镜像）
```

代码审查员智能体不应该有通往 `mcp-server-registry` 的路由。在网络层（NetworkPolicy）强制执行这一约束，而不仅仅依赖配置层面的限制。

### 身份认证

MCP 服务器必须要求身份认证。两种实用方案：

**Bearer token**（更简单，适用于大多数场景）：
```python
# 在 MCP 服务器处理程序中
def verify_request(request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if token != os.environ["MCP_TOKEN"]:
        raise PermissionError("无效的令牌")
```

**mTLS**（高安全环境的首选）：从内部 CA 为每个智能体会话签发客户端证书。MCP 服务器对照授权智能体身份白名单验证客户端证书的 CN 字段。每个会话轮换证书。

```bash
# 为一个智能体会话签发短期证书
vault write pki/issue/agent-role \
  common_name="agent-session-$(uuidgen)" \
  ttl="1h"
```

### 工具级授权与速率限制

在 MCP 服务器前面添加一个薄授权网关。对于每个传入的工具调用，检查：

1. 此智能体身份是否被授权调用此工具？
2. 此智能体是否超过了该工具类别的速率限制？

对破坏性工具实施速率限制：

```python
# 基于 Redis 的速率限制：每个智能体会话每分钟最多写入 10 次文件
def check_rate_limit(agent_session_id: str, tool_name: str) -> bool:
    key = f"ratelimit:{agent_session_id}:{tool_name}"
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, 60)
    return count <= 10
```

### 审计日志

每次工具调用都必须生成结构化日志条目：

```json
{
  "timestamp": "2025-01-15T14:23:01.847Z",
  "agent_session_id": "sess_abc123",
  "agent_role": "code-reviewer",
  "mcp_server": "mcp-server-git",
  "tool_name": "git_diff",
  "input": {"repo": "internal/payments-service", "ref": "main..HEAD"},
  "output_size_bytes": 4821,
  "duration_ms": 234,
  "status": "success"
}
```

将日志发送至 SIEM 系统。需要告警的场景：单个会话在 60 秒内发起超过 10 次文件写入；对智能体声明角色之外的服务器发起工具调用；非自动化智能体在异常时间段发起工具调用。

### 人工审批检查点

对于高风险工具类别——带有副作用的外部 API 调用、文件删除、数据库变更——在执行前需要人工明确批准。LangGraph 的中断/恢复模式是实用的实现方式：

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

def tool_execution_node(state):
    tool_call = state["pending_tool_call"]
    if tool_call["name"] in HIGH_RISK_TOOLS:
        # 暂停执行；人工通过界面审核
        return {"status": "awaiting_approval", "tool_call": tool_call}
    return execute_tool(tool_call)

# 人工批准后恢复执行
graph.update_state(thread_id, {"status": "approved"})
graph.invoke(None, config={"configurable": {"thread_id": thread_id}})
```

---

## 5. 智能体循环中的密钥管理

### 详细风险说明

设想一个智能体的任务是诊断失败的服务。它的工具调用链：检查日志、读取配置文件、查询服务状态 API。配置文件读取返回了如下内容：

```
DATABASE_URL=postgresql://app:s3cr3tpassw0rd@db.internal:5432/prod
AWS_ACCESS_KEY_ID=AKIA...
```

这个字符串现在已进入 LLM 上下文窗口。如果智能体对其发现进行总结，密码可能逐字出现在输出中。如果工具输出被记录用于调试，密钥就进入了你的日志管道。标准的 DLP 和网络控制对此无能为力。

### 在摄取阶段进行密钥脱敏

在将工具输出注入上下文之前扫描并脱敏：

```python
import re

SECRET_PATTERNS = [
    r'AKIA[0-9A-Z]{16}',                          # AWS 访问密钥
    r'(?i)password\s*[=:]\s*\S+',                  # password= 或 password:
    r'(?i)secret\s*[=:]\s*\S+',                    # secret= 或 secret:
    r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',  # JWT
    r'postgresql://[^@]+:[^@]+@',                  # 含凭证的数据库连接串
    r'(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*',        # Bearer token
]

def redact_secrets(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, '[REDACTED]', text)
    return text

# 在将工具结果插入智能体上下文之前应用
tool_output = redact_secrets(raw_tool_output)
```

使用 `detect-secrets` 或 `trufflehog` 等库可以获得更全面的覆盖。将其作为流水线阶段运行，而不是临时调用。

### Vault 集成

智能体不应持有长期凭证。在工具调用时获取动态短期凭证：

```python
import hvac

def get_db_credentials(agent_session_id: str) -> dict:
    client = hvac.Client(url="http://vault.internal:8200", token=VAULT_TOKEN)
    # 动态凭证有效期 1 小时，与本次会话绑定
    creds = client.secrets.database.generate_credentials(
        name="agent-readonly-role"
    )
    return creds["data"]  # {"username": "...", "password": "..."}
```

智能体在调用时获取凭证；凭证自动过期；凭证的权限范围限定为智能体角色所需的只读访问。即使会话被攻陷，爆炸半径也受到约束。

### 上下文窗口卫生

已知输出冗长的工具结果应在进入上下文之前进行摘要或截断。5 万行的日志转储包含一条相关错误和 4.9 万行噪音——其中还可能包含密钥和内部主机名：

```python
def sanitize_log_output(raw_logs: str, max_lines: int = 100) -> str:
    lines = raw_logs.splitlines()
    if len(lines) > max_lines:
        # 保留最后 N 行，通常包含最近的错误信息
        lines = lines[-max_lines:]
        lines.insert(0, f"[日志已截断：显示最后 {max_lines} 行，共 {len(raw_logs.splitlines())} 行]")
    return redact_secrets("\n".join(lines))
```

**系统提示卫生**：系统提示不得以字面量形式包含 API 密钥、密码、内部主机名或任何凭证。在推理层通过环境变量注入；系统提示模板中引用 `{service_base_url}`，值在运行时从密钥管理器注入——它永远不会出现在源代码或版本控制的配置文件中。

---

## 6. 决策流程图：选择云端 LLM API 还是本地部署 LLM

```
开始：评估智能体的 LLM 部署模式
         |
         v
数据是否包含个人身份信息、受保护健康信息或保密信息？
         |
    是   |  否
         |   \
         v    v
是否存在合规要求          是否希望利用托管弹性
（GDPR、HIPAA、          扩展和零 GPU 运维
SOC2、等保合规）？               负担？
         |                         |
    是   |  否                是   |  否
         |   \                     |   \
         v    v                    v    v
是否具备        延迟要求        选择云端       继续
GPU 预算        是否允许        LLM API        评估
（≥1×A100）？   >500ms？
         |           |
    是   |  否   是  |  否
         |   \       |   \
         v    v      v    v
本地部署      考虑更小  本地部署     云端 API
（vLLM +     的模型 + 可接受      仍可用于
 Llama 3.1  量化，或   （延迟      非敏感数
 70B INT4） 使用含 DPA  可控）     据，但需
            的云端 API  但需验证    审查数据
                       数据不经    处理协议
                       中转直接
                       在本地处理
         |
         v
任务是否需要接近 GPT-4 级别的能力？
（复杂多步推理、代码生成、长上下文）
         |
    是   |  否
         |   \
         v    v
考虑微调或使用      Llama 3.1 8B
Llama 3.1 70B      或 Qwen2.5 7B
BF16（2×H100）     即可满足需求；
                   硬件成本更低
```

**汇总对照表：**

| 影响因素 | 倾向本地部署 | 倾向云端 API |
|---|---|---|
| 数据敏感性 | 个人身份信息/受保护健康信息/保密数据 | 公开数据或低敏感度数据 |
| 合规要求 | GDPR 数据驻留、HIPAA、气隔 | 无严格驻留要求 |
| GPU 预算 | 有专用 GPU 算力 | 无 GPU；按调用计费可接受 |
| 延迟 | 可接受本地推理延迟 | 需要弹性亚百毫秒响应 |
| 运维能力 | 具备 MLOps 团队管理推理服务 | 无机器学习基础设施团队 |
| 模型能力 | 70B 模型足以完成任务 | 任务需要前沿闭源模型 |

---

## 7. 最小化安全栈参考

以下是受监管环境智能体部署的具体推荐栈。这不是理想化设计——每个组件都经过生产验证。

```
┌─────────────────────────────────────────────────────────────────┐
│                    智能体服务层                                  │
│  LangGraph 智能体（SQLite 检查点、高风险操作触发中断）           │
│         │ base_url=http://vllm-service:8000/v1                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM 推理层                                    │
│  vLLM · Llama 3.1 70B INT4 · 1×A100-80GB                       │
│  仅通过集群内部 ClusterIP 服务暴露                               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼（工具调用通过 MCP JSON-RPC 分发）
┌─────────────────────────────────────────────────────────────────┐
│                    MCP 服务器（每个域独立部署）                  │
│  mcp-git | mcp-filesystem | mcp-ci | mcp-db-readonly            │
│  每个服务器：mTLS 客户端证书认证 · 工具级授权网关               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼（工具执行器）
┌─────────────────────────────────────────────────────────────────┐
│                    工具执行层（Kubernetes Jobs）                 │
│  运行时：gVisor（runsc）                                        │
│  NetworkPolicy：出站全拒绝，仅允许内部 CIDR                     │
│  securityContext：非 root、只读根文件系统、删除所有 capabilities │
│  存储卷：emptyDir（临时，任务完成后自动清除）                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    密钥管理层                                    │
│  HashiCorp Vault · 动态短期凭证                                 │
│  智能体按工具调用获取凭证；TTL 1 小时；审计日志写入 Vault        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    可观测性层                                    │
│  结构化 JSON 日志 → Promtail → Loki → Grafana                   │
│  告警：异常写入速率、越权工具调用、OOM Kill                      │
└─────────────────────────────────────────────────────────────────┘
```

**组件选型理由：**

- **vLLM 而非 Ollama**：支持多智能体并发、PagedAttention 内存优化，以及面向未来规模扩展的张量并行能力。
- **LangGraph 而非裸提示循环**：内置检查点（故障后可恢复）、人工审批的中断/恢复机制、可审计的状态图。
- **SQLite 检查点**：简单，无需额外的状态存储基础设施。需要并发写入时可切换至 PostgreSQL。
- **gVisor 而非单独 Docker**：用户态内核拦截为对抗性代码执行场景中的容器逃逸提供纵深防御。
- **Vault 而非环境变量**：动态凭证、自动轮换、每次凭证访问的审计追踪。
- **Loki 而非 ELK**：运维负担更轻；Grafana 对大多数平台团队而言已经熟悉。

---

## 8. 对照表：本地智能体栈选项

| 维度 | 方案 A | 方案 B | 方案 C | 方案 D |
|---|---|---|---|---|
| **LLM 推理** | vLLM | Ollama | TGI（Text Generation Inference） | llama.cpp server |
| 适用场景 | 多智能体、高并发 | 开发测试、单用户 | HuggingFace 原生模型 | CPU 推理、边缘部署 |
| 并发能力 | 高（连续批处理） | 低（顺序执行） | 中等 | 低 |
| 运维复杂度 | 中等 | 低 | 中等 | 低 |
| **工具沙箱** | 仅 Docker | gVisor（runsc） | Firecracker microVM | Kata Containers |
| 隔离强度 | 命名空间/cgroup | 用户态内核 | 完整虚拟机 | 轻量级虚拟机 |
| 性能开销 | 约 0% | 5%–15% | 启动时间 <1 秒 | 启动时间 1–2 秒 |
| 使用时机 | 仅限确定性工具 | 通用智能体场景 | 计算机操作、对抗性代码 | 受监管环境，需 VM 级隔离 |
| **密钥管理** | Pod 规格中的环境变量 | HashiCorp Vault | AWS Secrets Manager（通过 Vault Agent 本地化） | Kubernetes Secrets |
| 动态凭证 | 不支持 | 支持 | 支持 | 不支持 |
| 审计追踪 | 无 | 有 | 有 | 无 |
| 凭证轮换 | 手动 | 自动 | 自动 | 手动 |
| **MCP 认证** | 无认证 | Bearer token | mTLS | OAuth2 / OIDC |
| 安全强度 | 无 | 适用于内部场景 | 强（双向认证） | 强（联合身份） |
| 运维开销 | 无 | 低 | 中等（证书管理） | 高（需要 IdP） |
| 推荐适用范围 | 仅开发环境 | 内部低风险场景 | 受监管生产环境 | 企业级 SSO 集成 |

**推荐生产基线**：vLLM + gVisor + Vault + mTLS。仅在开发者沙箱环境中降级使用 Ollama + Docker + Bearer token。

---

## 交叉参考

- `../on-prem-llm-deployment/` — 硬件规格详解及 vLLM/Ollama 配置实操
- `../../anthropic/mcp/` — MCP 协议规范、传输选项、资源与工具的区别
- `../../foundational/tool-use-infra/` — 工具调用模式、并行分发机制、错误处理
- `../../foundational/agent-frameworks/` — LangGraph 中断/恢复机制、状态持久化模式
- `../../anthropic/computer-use/` — 最高风险操作面；本指南所有沙箱建议在此场景下同样适用，且须更加严格执行
