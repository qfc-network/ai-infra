# Secure Agent Deployment in Private and On-Premise Environments

**Audience**: Platform engineers and DevOps engineers managing Kubernetes/Linux infrastructure who are now deploying LLM-based agent systems in regulated or security-sensitive environments.

**Prerequisites**: Familiarity with Kubernetes, Linux networking, and container security. No prior LLM agent experience assumed.

---

## 1. Why On-Prem Agents Are Different

When you deploy a cloud-based LLM agent, the LLM API call exits your network on every inference request. Tool calls made by the agent — web searches, API lookups, code execution — may also reach external services. Your data traverses vendor infrastructure with all the compliance and residency implications that entails.

On-prem agents invert this entirely. The LLM runs locally (Ollama, vLLM, TensorRT-LLM), tool execution is local, and data never leaves your perimeter. That satisfies GDPR data residency requirements, HIPAA covered-entity obligations, and the kind of air-gap mandates common in defense, finance, and healthcare. The tradeoff is that you now own the entire security surface — there is no vendor absorbing the blast radius of a misconfiguration.

### The Three Risk Surfaces

Understanding where your exposure sits is the first step to controlling it.

**(a) The LLM itself.** Prompt injection is the principal threat: a malicious string embedded in a retrieved document, a tool result, or a user message instructs the model to ignore its system prompt or take unauthorized actions. Jailbreak attempts are a secondary concern — in a regulated environment, the model may be asked to output PII it ingested from a tool result. Open-weight models vary significantly in their robustness to these attacks; model selection and system prompt hardening both matter.

**(b) The tool execution environment.** This is where on-prem deployment introduces the most novel risk for infrastructure engineers. An agent with a `bash_exec` tool has, in the limit, the same capabilities as the identity it runs under on your host. Prompt injection that hijacks a tool call can escalate to arbitrary code execution. The tool execution sandbox is not a nice-to-have — it is the primary defense layer.

**(c) The agent loop state (context window).** An agent's context window accumulates everything: the system prompt, all user turns, all tool call inputs and outputs, all intermediate reasoning. A tool that reads a config file or a log may inadvertently deposit an API key or database password into the context. That secret is then visible to subsequent LLM calls and may be echoed verbatim in agent responses or captured in log pipelines. This is a data exfiltration path that standard network controls do not cover.

---

## 2. Local LLM as Agent Backbone

### Model Selection

For on-prem agent deployments, the two practical open-weight choices at the time of writing are **Llama 3.1 70B** and **Qwen2.5 32B** (or Qwen3 32B when available). Both reliably support structured tool-call output format, which is a hard requirement for agentic use — not all open-weight models produce well-formed JSON tool calls under adversarial prompt conditions.

Additional solid choices for function calling: Mistral-Nemo, Llama 3.1 8B (for low-latency low-stakes tasks), and Qwen2.5-Coder variants for code-heavy agents.

**Capability vs. hardware tradeoffs:**

| Model | VRAM (BF16) | VRAM (INT4) | Recommended Hardware |
|---|---|---|---|
| Llama 3.1 70B | ~140 GB | ~40 GB | 2×H100 or 4×A100-40GB (BF16); 1×A100-80GB (INT4) |
| Qwen2.5 32B | ~64 GB | ~20 GB | 1×H100-80GB or 2×A100-40GB (BF16); 1×A100-40GB (INT4) |
| Llama 3.1 8B | ~16 GB | ~5 GB | 1×A100-40GB or 1×RTX 4090 (INT4) |

For most regulated-environment deployments where audit and reliability matter more than raw throughput, **Llama 3.1 70B INT4 on a single A100-80GB** is the practical starting point.

### Serving Stack: vLLM vs. Ollama

**vLLM** is the production choice for multi-agent or high-concurrency deployments. It supports continuous batching, PagedAttention for memory efficiency, tensor parallelism across multiple GPUs, and an OpenAI-compatible API. Install and serve:

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

**Ollama** is appropriate for single-developer use, low-traffic internal tools, or environments where simplicity of operation outweighs throughput. It wraps model management and serving behind a single binary:

```bash
ollama pull llama3.1:70b
ollama serve  # listens on localhost:11434 by default
```

Both expose an OpenAI-compatible endpoint at `/v1/chat/completions`. Agent frameworks such as LangGraph and AutoGen accept a `base_url` override that points at your local serving stack:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used",          # vLLM does not require a key by default
    model="meta-llama/Llama-3.1-70B-Instruct",
)
```

### Verifying Function Calling

Before deploying a model as an agent backbone, verify tool-call format compliance:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_file_contents",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        }
    }],
    messages=[{"role": "user", "content": "Read /etc/hostname"}],
    tool_choice="auto"
)
print(response.choices[0].message.tool_calls)
```

If `tool_calls` is `None` or malformed JSON, the model is not reliably usable as an agent backbone at this quantization level or context length. Increase model precision or reduce context length before proceeding.

---

## 3. Tool Call Sandboxing

### The Threat Model

An agent with an unrestricted `bash_exec` tool is, in practice, a remote code execution vulnerability waiting for a sufficiently clever prompt injection in a retrieved document. The tool sandbox is your primary defense — treat it with the same rigor you would a public-facing service.

### Network Isolation

Tool executors should run in a separate network namespace with egress firewall rules that allow only explicitly permitted destinations. In Kubernetes, enforce this with NetworkPolicy:

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
        cidr: 10.0.0.0/8          # internal APIs only
    ports:
    - protocol: TCP
      port: 443
  # No rule for 0.0.0.0/0 — default deny all other egress
```

For non-Kubernetes deployments, use iptables with an OUTPUT chain that drops everything except specific internal CIDRs and DNS:

```bash
iptables -P OUTPUT DROP
iptables -A OUTPUT -d 10.0.0.0/8 -j ACCEPT
iptables -A OUTPUT -d 127.0.0.0/8 -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT  # internal DNS only
```

### Syscall Filtering and Linux Security Modules

Apply a seccomp profile to the tool executor container. The following profile blocks the highest-risk syscalls:

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
                "futex", "openat", "getdents64", "set_tid_address", "set_robust_list",
                "pread64", "pwrite64", "readv", "writev", "fcntl", "flock",
                "fsync", "fdatasync", "ftruncate", "lseek", "pipe2", "epoll_create1",
                "epoll_ctl", "epoll_wait", "lstat", "newfstatat", "exit_group",
                "tgkill", "arch_prctl", "prlimit64", "getrandom", "clock_gettime",
                "clock_nanosleep", "rseq"],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

Explicitly absent: `ptrace`, `mount`, `setuid`, `setgid`, `setns`, `unshare`, `keyctl`, `add_key`, `request_key`, `bpf`, `perf_event_open`. These are blocked.

Pair with AppArmor or SELinux. A minimal AppArmor profile for the executor:

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

### Kubernetes Pod Spec: Putting It Together

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: tool-executor
  namespace: agent-tools
spec:
  runtimeClassName: gvisor          # see Code Execution section
  securityContext:
    runAsNonRoot: true
    runAsUser: 65534                 # nobody
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
    emptyDir: {}                     # wiped after pod termination
  - name: tmp
    emptyDir: {}
```

### Code Execution: gVisor and Firecracker

For agents that execute arbitrary Python or shell code, Docker alone provides insufficient isolation. Container escape vulnerabilities exist and an adversarial payload may target them. Use a stronger isolation primitive:

- **gVisor (runsc)**: A userspace kernel that intercepts all syscalls before they reach the host kernel. Install the gVisor runtime and set `runtimeClassName: gvisor` in your pod spec (shown above). Overhead is 5–15% for most workloads, acceptable for tool execution.
- **Firecracker microVMs**: Full VM isolation with sub-second startup. Used by AWS Lambda. More complex to operate but provides the strongest isolation boundary. Appropriate for agents with computer-use capabilities — see `../../anthropic/computer-use/` for that threat model.

Docker-only (without gVisor or Firecracker) is acceptable only if the tool set is fully deterministic and does not involve executing untrusted code paths.

---

## 4. MCP Server Permission Boundaries

MCP (Model Context Protocol) exposes tools and resources to the agent over JSON-RPC. In on-prem deployments you self-host MCP servers — which means you design the authorization model.

### Least Privilege Per Server

Do not deploy a single MCP server that exposes all tools. Decompose by tool domain and agent role:

```
agent-role: code-reviewer
  → mcp-server-git (read-only: clone, diff, log)
  → mcp-server-static-analysis (read-only: lint, scan)

agent-role: release-manager
  → mcp-server-git (read-write: tag, push)
  → mcp-server-ci (trigger pipelines)
  → mcp-server-registry (push images)
```

A code-reviewer agent should never have a route to `mcp-server-registry`. Enforce this at the network layer (NetworkPolicy), not just by configuration.

### Authentication

MCP servers must require authentication. The two practical options:

**Bearer token** (simpler, adequate for most cases):
```python
# In your MCP server handler
def verify_request(request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if token != os.environ["MCP_TOKEN"]:
        raise PermissionError("invalid token")
```

**mTLS** (preferred for high-security environments): Issue per-agent-session client certificates from an internal CA. The MCP server validates the client cert CN against an allowlist of authorized agent identities. Rotate certificates per session.

```bash
# Issue a short-lived cert for an agent session
vault write pki/issue/agent-role \
  common_name="agent-session-$(uuidgen)" \
  ttl="1h"
```

### Tool-Level Authorization and Rate Limiting

Add a thin authorization gateway in front of your MCP servers. For each incoming tool call, check:

1. Is this agent identity authorized to call this tool?
2. Has this agent exceeded the rate limit for this tool category?

Implement rate limiting on destructive tools:

```python
# Redis-backed rate limit: max 10 file writes per agent session per minute
def check_rate_limit(agent_session_id: str, tool_name: str) -> bool:
    key = f"ratelimit:{agent_session_id}:{tool_name}"
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, 60)
    return count <= 10
```

### Audit Logging

Every tool call must produce a structured log entry:

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

Ship to your SIEM. Alert on: >10 file-write calls from a single session in 60 seconds; tool calls to servers outside the agent's declared role; tool calls at unusual hours for non-automated agents.

### Human-in-the-Loop Checkpoints

For high-risk tool categories — external API calls with side effects, file deletion, database mutations — require explicit human approval before execution. LangGraph's interrupt/resume pattern is the practical implementation:

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

def tool_execution_node(state):
    tool_call = state["pending_tool_call"]
    if tool_call["name"] in HIGH_RISK_TOOLS:
        # Suspend execution; human reviews via UI
        return {"status": "awaiting_approval", "tool_call": tool_call}
    return execute_tool(tool_call)

# Resume after human approval
graph.update_state(thread_id, {"status": "approved"})
graph.invoke(None, config={"configurable": {"thread_id": thread_id}})
```

---

## 5. Secret Management in Agent Loops

### The Risk in Detail

Consider an agent whose task is to diagnose a failing service. Its tool calls: check logs, read config files, query the service status API. A config file read returns something like:

```
DATABASE_URL=postgresql://app:s3cr3tpassw0rd@db.internal:5432/prod
AWS_ACCESS_KEY_ID=AKIA...
```

That string is now in the LLM context window. If the agent summarizes its findings, the password may appear verbatim in the output. If tool outputs are logged for debugging, the key is in your log pipeline. Standard DLP and network controls do not catch this.

### Secret Redaction at Ingestion

Scan all tool outputs before injecting them into the context:

```python
import re

SECRET_PATTERNS = [
    r'AKIA[0-9A-Z]{16}',                          # AWS access key
    r'(?i)password\s*[=:]\s*\S+',                  # password= or password:
    r'(?i)secret\s*[=:]\s*\S+',                    # secret= or secret:
    r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',  # JWT
    r'postgresql://[^@]+:[^@]+@',                  # DB connection string with creds
    r'(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*',        # Bearer token
]

def redact_secrets(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, '[REDACTED]', text)
    return text

# Apply before inserting tool result into agent context
tool_output = redact_secrets(raw_tool_output)
```

Use a library like `detect-secrets` or `trufflehog` for more comprehensive coverage. Run it as a pipeline stage, not ad hoc.

### Vault Integration

Agents should never hold long-lived credentials. Use dynamic short-lived credentials fetched at tool-call time:

```python
import hvac

def get_db_credentials(agent_session_id: str) -> dict:
    client = hvac.Client(url="http://vault.internal:8200", token=VAULT_TOKEN)
    # Dynamic credentials valid for 1 hour, tied to this session
    creds = client.secrets.database.generate_credentials(
        name="agent-readonly-role"
    )
    return creds["data"]  # {"username": "...", "password": "..."}
```

The agent receives credentials at call time; they expire automatically; they are scoped to read-only access for the agent's role. If the session is compromised, the blast radius is bounded.

### Context Window Hygiene

Tool outputs that are known to be verbose should be summarized or truncated before entering the context. A log dump of 50,000 lines contains one relevant error and 49,999 lines of noise — including potential secrets and internal hostnames:

```python
def sanitize_log_output(raw_logs: str, max_lines: int = 100) -> str:
    lines = raw_logs.splitlines()
    if len(lines) > max_lines:
        # Keep last N lines, which typically contain the most recent errors
        lines = lines[-max_lines:]
        lines.insert(0, f"[Log truncated: showing last {max_lines} of {len(raw_logs.splitlines())} lines]")
    return redact_secrets("\n".join(lines))
```

**System prompt hygiene**: The system prompt must not contain API keys, passwords, internal hostnames as literals, or any credential. Use environment variable injection at the serving layer; the system prompt template references `{service_base_url}` and the value is injected from a secrets manager at runtime — it never appears in source code or configuration files checked into version control.

---

## 6. Decision Flowchart: Cloud LLM API vs. On-Prem LLM

```
START: Evaluating LLM deployment model for agent
         |
         v
Does your data include PII, PHI, or classified information?
         |
    YES  |  NO
         |   \
         v    v
Is there a compliance          Would you benefit from
mandate (GDPR, HIPAA,          managed scaling and
SOC2, FedRAMP)?                zero GPU ops burden?
         |                              |
    YES  |  NO                     YES  |  NO
         |   \                          |   \
         v    v                         v    v
Do you have        Is latency      Use Cloud      Continue
GPU budget         tolerance       LLM API        evaluating
(≥1×A100)?         >500ms?
         |               |
    YES  |  NO      YES  |  NO
         |   \           |   \
         v    v          v    v
ON-PREM     Consider   ON-PREM   Cloud API
(vLLM +     smaller    acceptable  may still
 Llama 3.1  model +     (latency   work for
 70B INT4)  quantize    manageable) non-sensitive
            or cloud    but verify  data, but
            API with    data stays  review data
            DPA         on-prem     handling
                        in transit  agreements
         |
         v
Does the task require capability near GPT-4 level?
(complex multi-step reasoning, code generation, long context)
         |
    YES  |  NO
         |   \
         v    v
Consider      Llama 3.1 8B
fine-tuning   or Qwen2.5 7B
or Llama 3.1  sufficient;
70B at BF16   lower hardware
(2×H100)      cost
```

**Summary table:**

| Factor | Favors On-Prem | Favors Cloud API |
|---|---|---|
| Data sensitivity | PII/PHI/classified | Public or low-sensitivity |
| Compliance | GDPR residency, HIPAA, air-gap | No strict residency req |
| GPU budget | Have dedicated GPU capacity | No GPU; cost per call acceptable |
| Latency | Can tolerate local serving latency | Need elastic sub-100ms |
| Ops capacity | Have MLOps to manage serving | No ML infrastructure team |
| Capability | 70B sufficient for task | Task requires frontier model |

---

## 7. Minimal Secure Stack Reference

A concrete recommended stack for a regulated-environment agent deployment. This is not aspirational — each component is production-tested.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT SERVING LAYER                          │
│  LangGraph agent  (SQLite checkpointer, interrupt on high-risk) │
│         │ base_url=http://vllm-service:8000/v1                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM SERVING                                  │
│  vLLM  ·  Llama 3.1 70B INT4  ·  1×A100-80GB                  │
│  Exposed only on cluster-internal ClusterIP service             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼ (tool calls dispatched via MCP JSON-RPC)
┌─────────────────────────────────────────────────────────────────┐
│                    MCP SERVERS (one per domain)                 │
│  mcp-git  |  mcp-filesystem  |  mcp-ci  |  mcp-db-readonly     │
│  Each: mTLS client cert auth · per-tool authz gateway           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼ (tool executor)
┌─────────────────────────────────────────────────────────────────┐
│                    TOOL EXECUTION (Kubernetes Jobs)             │
│  Runtime: gVisor (runsc)                                        │
│  NetworkPolicy: egress deny-all except internal CIDRs           │
│  securityContext: nonRoot, readOnlyRootFilesystem, cap-drop ALL │
│  Volumes: emptyDir (ephemeral, wiped on completion)             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SECRET MANAGEMENT                            │
│  HashiCorp Vault  ·  Dynamic short-lived credentials            │
│  Agent fetches creds per tool-call; TTL 1h; audit log to Vault  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY                                 │
│  Structured JSON logs → Promtail → Loki → Grafana               │
│  Alerts: anomalous write rate, off-role tool calls, OOM kills   │
└─────────────────────────────────────────────────────────────────┘
```

**Component rationale:**

- **vLLM over Ollama**: Multi-agent concurrency, PagedAttention, tensor parallelism support for future scale-up.
- **LangGraph over raw prompt loop**: Built-in checkpointing (resumable after failure), interrupt/resume for human-in-the-loop, auditable state graph.
- **SQLite checkpointer**: Simple, no additional infrastructure for the state store. Swap to PostgreSQL when you need concurrent writers.
- **gVisor over Docker-only**: Userspace kernel intercepts provide defense-in-depth against container escape in adversarial code execution scenarios.
- **Vault over env vars**: Dynamic credentials, automatic rotation, audit trail per credential access.
- **Loki over ELK**: Lighter operational footprint; Grafana already familiar to most platform teams.

---

## 8. Comparison Table: On-Prem Agent Stack Options

| Dimension | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| **LLM Serving** | vLLM | Ollama | TGI (Text Generation Inference) | llama.cpp server |
| Best for | Multi-agent, high concurrency | Dev/test, single user | HuggingFace-native models | CPU inference, edge |
| Concurrency | High (continuous batching) | Low (sequential) | Medium | Low |
| Ops complexity | Medium | Low | Medium | Low |
| **Tool Sandbox** | Docker only | gVisor (runsc) | Firecracker microVM | Kata Containers |
| Isolation | Namespace/cgroup | Userspace kernel | Full VM | Lightweight VM |
| Overhead | ~0% | 5–15% | <1s startup | 1–2s startup |
| Use when | Deterministic tools only | General agent use | Computer-use, adversarial code | Regulated, need VM-level |
| **Secret Mgmt** | Env vars in pod spec | HashiCorp Vault | AWS Secrets Manager (on-prem via Vault Agent) | Kubernetes Secrets |
| Dynamic creds | No | Yes | Yes | No |
| Audit trail | No | Yes | Yes | No |
| Rotation | Manual | Automatic | Automatic | Manual |
| **MCP Auth** | None | Bearer token | mTLS | OAuth2 / OIDC |
| Strength | None | Adequate for internal | Strong (mutual auth) | Strong (federation) |
| Ops overhead | None | Low | Medium (cert mgmt) | High (IdP required) |
| Recommended for | Dev only | Internal low-risk | Regulated environments | Enterprise SSO |

**Recommended production baseline**: vLLM + gVisor + Vault + mTLS. Downgrade to Ollama + Docker + bearer token only for developer sandboxes.

---

## Cross-References

- `../on-prem-llm-deployment/` — hardware sizing detail and vLLM/Ollama setup walkthrough
- `../../anthropic/mcp/` — MCP protocol specification, transport options, resource vs tool distinction
- `../../foundational/tool-use-infra/` — tool call schema, parallel dispatch mechanics, error handling
- `../../foundational/agent-frameworks/` — LangGraph interrupt/resume, state persistence patterns
- `../../anthropic/computer-use/` — highest-risk action surface; all sandboxing recommendations here apply with extra force
