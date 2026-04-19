# 机密 LLM 推理

- **涵盖技术**：NVIDIA H100 机密计算（SPM）、AWS Nitro Enclaves、Azure 机密虚拟机（AMD SEV-SNP）、Apple Private Cloud Compute
- **相关标准**：TCG TPM 2.0、IETF RATS（远程证明过程）、AMD SEV-SNP ABI 规范
- **链接**：[H100 CC 白皮书](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/h100-security-whitepaper.pdf) · [AWS Nitro Enclaves](https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclaves-concepts.html) · [Apple PCC 安全性](https://security.apple.com/documentation/private-cloud-compute)

## 一句话总结

标准云端推理在运行时将模型权重、用户提示词和模型输出完整暴露于云服务商的 Hypervisor 层。对于受监管行业——医疗（HIPAA）、金融（SOX/GDPR）、法律、国防——以及任何处理敏感专有数据的企业而言，这种暴露在政策层面不可接受。机密计算硬件通过硬件信任根来应对这一问题：内存加密、密码学证明以及加密通信信道，使 Hypervisor 的窥视在计算上不可行，即便是对恶意的云运营商也如此。H100 安全性能模式（CC 模式）、AWS Nitro Enclaves 和 Azure AMD SEV-SNP 虚拟机各自在不同层次、以不同的信任模型实现了这一目标。性能开销相对温和——延迟增加 5–10%——但基础设施复杂度和云端成本溢价（约 20–30%）较为显著。在决定部署机密推理之前，必须深刻理解威胁模型、证明链和密钥管理架构。

## 威胁模型

机密计算针对的是一种特定对手：被入侵或存在恶意的基础设施运营商。这与应用层加密所应对的常见威胁模型截然不同。

在标准云端推理中：

1. 用户通过 HTTPS 提交提示词。TLS 在应用负载均衡器——即云服务商的基础设施——处终止。明文提示词在数据中心网络内部可见。
2. 提示词被发送至运行在 GPU 实例上的推理服务。GPU 驱动和 Hypervisor 对 GPU 内存拥有无限制访问权限，而 GPU 内存中存储着模型权重、KV 缓存以及当前正在处理的提示词。
3. 响应被返回。在整个推理过程中，被入侵的 Hypervisor 可以记录每一条提示词和响应。

这一威胁模型实际适用于哪些场景？不适用于初创公司，也不适用于大多数企业 SaaS。它适用于：

- **医疗 AI**：患者记录、临床笔记、由 LLM 处理的诊断查询——受 HIPAA 约束；覆盖实体无法向监管机构证明云服务商未访问受保护健康信息（PHI）。
- **法律 AI**：受特权保护的律师-当事人通信被发送至 LLM 进行分析。
- **金融 AI**：未公开的重大信息（MNPI）被处理用于交易或合规分析。
- **政府/国防**：机密或受控非机密信息（CUI）。
- **模型知识产权保护**：不希望其权重对运行服务的云基础设施可见的模型提供商。

标准 TLS 加 IAM 策略处理的是网络层和访问控制层，无法防范对 GPU 主机拥有物理或特权软件访问权限的被入侵 Hypervisor。机密计算填补了这一空白。

## H100 机密计算（安全性能模式）

NVIDIA 在 Hopper（H100）中引入了基于硬件的机密计算能力。生产名称为**安全性能模式（SPM）**；更宽泛的功能集也称为 H100 机密计算。

### 硬件信任根

每颗 H100 芯片在出厂时由 NVIDIA 熔断了一对独一无二的证明密钥。私钥存储在制造后无法读取的片上熔丝中；NVIDIA 保存了对应公钥的记录，并用其签发证明证书。这一每芯片密钥是信任链的根。

### CC 模式架构

当 GPU 在 CC 模式下启动时：

1. GPU 固件被度量（固件镜像的 SHA-256 哈希），度量值由芯片证明密钥签名。
2. GPU 与主机侧的 CPU 可信执行环境（TEE——通常是 AMD SEV 或 Intel TDX）建立加密信道。
3. 所有 PCIe DMA 传输使用从 GPU-CPU 握手中派生的会话密钥加密。Hypervisor 位于 PCIe 总线上，但无法读取明文 DMA 流量。
4. GPU 内存（HBM3）使用每会话密钥加密。GPU 内存控制器透明地处理加解密；CUDA 内核在 GPU 内部对明文数据运行，但离开 GPU 的数据始终是密文。

完整的证明与密钥建立流程：

```
1. 客户端向 GPU 服务请求证明
2. GPU 生成证明报告：
   - 固件度量值（SHA-256）
   - 启动配置
   - 由芯片证明密钥签名
3. 客户端对照 NVIDIA 证明服务验证报告签名
   （NVIDIA 的类 OCSP 证明服务返回签名证书链）
4. 客户端验证固件度量值与预期值匹配
5. 客户端与 GPU 协商会话密钥（在已验证信道上进行 ECDH）
6. 模型权重通过加密 PCIe DMA 传输至 GPU 内存
7. 推理在加密 GPU 内存内运行；提示词通过加密会话密钥信道进入，
   响应通过同一信道返回
```

这一流程提供**可验证的隔离**：客户端可以密码学证明推理运行在搭载已知固件的真实 H100 上，而无需信任云服务商的任何声明。

### 性能开销

H100 CC 模式带来两类开销：

- **PCIe 加密开销**：加解密 DMA 传输。对于推理，这主要影响初始权重加载阶段。在 PCIe Gen5 带宽（约 128 GB/s）下，添加 AES-256-GCM 加密给数据传输时间增加约 5–8% 的开销。推理本身（在 GPU 内存内进行）不受影响。
- **内存屏障和同步开销**：确保操作相对于内存加密以正确顺序进行，为每次 CUDA 内核启动增加了小而非零的同步开销。实测结果：LLM 推理工作负载的端到端延迟增加约 5–10%。

对于在标准模式下以 2,000 token/秒服务的 H100 上的 70B 模型：CC 模式将吞吐量降至约 1,800–1,900 token/秒。这一开销对受监管工作负载是可接受的；付出的代价是合规成本而非推理预算。

## AWS Nitro Enclaves

AWS Nitro Enclaves 是 EC2 实例内的隔离计算环境，专为在 AWS 基础设施上处理敏感数据而设计。

### 架构

Nitro Enclave 是从父 EC2 实例的 vCPU 和内存资源中划分出的一部分创建的精简虚拟机。它运行最小化 Linux 内核（或 Amazon Linux 2），没有持久化存储，没有交互访问权限（无 SSH），只有一个有意限制的网络接口——唯一的通信渠道是连接到父 EC2 实例的本地 VSOCK 连接。

安全保证来自 **AWS Nitro Hypervisor**：Enclave 的内存和 CPU 状态与父实例以及 AWS 运营者相互隔离。即便是 EC2 实例所有者也无法检查 Enclave 内存。AWS 证明提供了对 Enclave 内运行代码的密码学证明。

### 证明流程

1. Enclave 镜像（EIF——Enclave 镜像文件）通过 `nitro-cli` 构建。构建过程产生对 Enclave 内容的密码学度量值（PCR 值——平台配置寄存器）。
2. 运行时，Enclave 生成包含其 PCR 度量值的证明文档，由 **AWS Nitro TPM**（嵌入 Nitro 卡的硬件支持 TPM）签名。
3. 父 EC2 实例（或外部 KMS）对照预期 PCR 值验证证明文档。
4. 仅在证明成功后，解密密钥才被释放给 Enclave。敏感数据（用户提示词或模型解密密钥）被密封至 PCR 值——只有经过验证的该 Enclave 镜像的确切实例才能解密它。

### 局限性：CPU 专属与 GPU 空白

Nitro Enclaves 用于 LLM 推理的关键局限：**在标准 EC2 实例上，不支持将完整 GPU 直通给 Nitro Enclaves**。Enclave 运行于 CPU 上或 vGPU 上，而 vGPU 驱动由 Nitro Hypervisor 管理，GPU 内存隔离与 H100 CC 模式不等价。

在 AWS 上实现完整 GPU 机密计算需要**裸金属 EC2 实例**（例如搭载 A100 的 `p4de.24xlarge` 裸金属实例），在 H100 裸金属实例可用时配合 H100 SPM 使用。在裸金属上，访客操作系统不经过介入性 Hypervisor 运行，H100 CC 证明链直接从 GPU 到租户应用，无需经过 AWS 控制的 Hypervisor 层。

这一架构分割意味着 AWS Nitro Enclaves 适用于：提示词加解密、密钥管理操作，以及在 CPU 上对小型模型做推理。对于需要强机密性的大型模型 GPU 推理，H100 裸金属加 H100 CC 才是正确架构。

## Azure 机密虚拟机（AMD SEV-SNP）

Azure 的机密推理方案建立在 **AMD SEV-SNP**（安全加密虚拟化——安全嵌套分页）之上。

### AMD SEV-SNP

SEV-SNP 使用每虚拟机加密密钥对访客虚拟机内存进行加密，该密钥仅存在于 AMD 安全处理器（AMD SP）中——AMD SP 是嵌入 CPU 芯片的独立安全处理器。Hypervisor 管理内存地址转换，但无法访问虚拟机内存页面的明文内容。

SEV-SNP 增加了嵌套页表完整性保护（"SNP"部分）：每个访客内存页都有一个完整性受 AMD SP 保护的反向映射表条目。这防止了 Hypervisor 将访客内存重映射到不同物理页面的攻击（重放或别名攻击），这是早期 SEV 版本存在漏洞的地方。

AMD SEV-SNP 在 CPU 侧计算的性能开销：典型工作负载约 3–7%。开销来自内存控制器的加解密以及每次页面访问的额外完整性检查。

### Azure 集成

Azure 的机密 GPU 方案将 AMD SEV-SNP 机密虚拟机与 H100 CC 模式配对：

- 访客虚拟机在 AMD SEV-SNP 内运行；Hypervisor 无法读取访客内存。
- 从机密虚拟机访问 H100；H100 到虚拟机的通信信道按 H100 CC 部分所述进行加密。
- 证明链将 AMD SEV-SNP 证明（证明虚拟机环境）与 H100 证明（证明 GPU 环境）组合为复合证明，租户可对其进行验证。

**Azure 证明服务（MAA）**验证两个证明报告，并签发一个签名令牌，租户应用可将其用作推理在已验证机密环境中运行的证明。

### 密钥管理集成

Azure Key Vault 可以配置为仅向经过证明的机密虚拟机释放密钥。流程如下：

1. 模型权重以加密形式存储在 Azure Blob Storage 中。
2. 解密密钥存放在 Azure Key Vault 中，策略为："仅向出示与预期度量值匹配的有效 MAA 证明令牌的虚拟机释放"。
3. 启动时，机密虚拟机进行证明，获得 MAA 令牌，将其提交给 Key Vault，并接收解密密钥。
4. 虚拟机将模型权重解密到 SEV-SNP 保护的内存中，然后通过加密 CC 信道将其加载到 H100。
5. 在整个过程中，解密密钥对 Hypervisor 和 Azure 运营者始终不可见。

## Apple Private Cloud Compute（交叉参考）

Apple 的 **Private Cloud Compute（PCC）**代表了一种截然不同的机密推理架构方案。详细内容见 `../../apple/afm/`；以下是关键比较要点：

- PCC 运行在 **Apple Silicon（M2/M4 Ultra）**服务器上，而非商用 x86 硬件。内存加密和证明在 Apple 定制芯片中实现。
- 信任模型差异：在 H100 CC 和 SEV-SNP 中，第三方证明服务（NVIDIA、AMD、Azure、AWS）位于信任链中。在 PCC 中，用户的 **iPhone/Mac 设备通过 Apple 透明日志直接验证 PCC 节点**——透明日志是一个类似证书透明度的仅追加日志，记录了所有已部署的 PCC 软件镜像。没有运行时第三方证明调用；设备在发送任何请求之前检查日志。
- 软件执行：PCC 强制要求模型推理代码不得记录提示词，不得写入持久存储，不得发起除特定 Apple 服务之外的出站网络调用。这些约束由 PCC 节点内运行的操作系统执行，并由该软件镜像的透明日志条目加以验证。
- Apple 掌控完整技术栈：硬件、固件、操作系统、模型、服务栈。这使信任模型更简单（参与方更少），但灵活性更低（只有 Apple 能部署到 PCC；企业无法部署自己的模型）。

## TEE 门控服务模式

面向企业的生产级机密推理系统通常实现**TEE 门控服务模式**，在安全性、成本和吞吐量之间取得平衡：

```
传入请求
      |
      v
请求分类器
  - 包含 PII / PHI / MNPI？  → 路由至 CC 节点
  - 标准内容？               → 路由至标准节点
      |                              |
      v                              v
  CC 节点集群                  标准节点集群
  （H100 SPM，成本较高）        （H100 标准，基准成本）
  约 20–30% 成本溢价           基准成本
      |                              |
      +----------+  +----------------+
                 |  |
                 v  v
           响应合并
           返回客户端
```

该模式中的密钥管理：

- **硬件安全模块（HSM）**存储模型权重的主密钥。
- 通过 HKDF 从主密钥和会话特定熵派生每会话密钥。
- CC 节点集群在启动时进行证明；HSM 仅向经验证的 TEE 环境释放主密钥。
- 会话密钥不离开 TEE；HSM 不接触单个提示词/响应内容。

审计日志：每个 CC 推理会话与一份证明报告关联。出于合规目的（HIPAA 审计轨迹、SOC 2），证明报告连同元数据（时间戳、模型版本、会话 ID）一起记录，但不记录提示词内容本身。这在不记录敏感数据的同时提供了可信度证明。

## 延迟与成本开销汇总

| 平台 | CPU/内存开销 | GPU 开销 | 云端成本溢价 | 备注 |
|---|---|---|---|---|
| H100 SPM（CC 模式） | 不适用（仅 GPU） | 端到端延迟约 5–10% | GPU 实例约溢价 20–25% | PCIe DMA 加密；GPU HBM 加密 |
| AWS Nitro Enclaves | 约 3–5%（VSOCK 开销） | 标准实例不可用；GPU 需裸金属 | 裸金属约溢价 15–20% | 大多数部署为纯 CPU |
| Azure AMD SEV-SNP | CPU 侧约 3–7% | 需配合 H100 CC 使用 | 机密 GPU SKU 约溢价 20–30% | AMD SP 处理内存加密 |
| Apple PCC | 不适用 | Apple Silicon；开销未公开 | Apple 托管；无公开定价 | 垂直技术栈；不支持企业自托管 |
| 纯软件（无 TEE） | 约 0% | 约 0% | 基准 | 无法防范被入侵的 Hypervisor |

## 工程权衡

| 维度 | 方案 | 获得 | 牺牲 |
|---|---|---|---|
| 威胁覆盖 | H100 CC + SEV-SNP | 全栈加密：CPU 内存 + PCIe + GPU 内存 | 约 10–15% 综合延迟开销；部署复杂度提升 |
| 威胁覆盖 | 仅应用层加密 | 部署简单；无硬件依赖 | 推理期间提示词以明文形式在主机 RAM 和 GPU 中可见 |
| 成本管理 | 全量流量路由至 CC 节点 | 统一安全态势 | 全部推理支出增加 20–30% |
| 成本管理 | TEE 门控路由（仅敏感流量） | 成本效率高；CC 节点只处理敏感流量 | 分类器必须可信；误分类会将敏感数据暴露至标准节点 |
| 证明模型 | 第三方（NVIDIA/AMD/Azure） | 可审计；外部验证者 | 需信任证明服务；类 OCSP 的可用性依赖 |
| 证明模型 | 第一方（Apple PCC） | 受信任方更少；设备直连验证 | 只有 Apple 能部署；企业无法定制模型 |
| 密钥管理 | HSM + 每会话 HKDF | 主密钥从不暴露；每会话前向保密 | HSM 可用性成为关键路径；每次会话建立增加 1–2ms |

## 参考文献

- [1] NVIDIA. _H100 Tensor Core GPU Architecture: Confidential Computing._ NVIDIA 技术白皮书, 2022.
- [2] Kaplan 等. _AMD SEV-SNP: Strengthening VM Isolation with Integrity Protection and More._ AMD 技术白皮书, 2020.
- [3] AWS. _AWS Nitro Enclaves: Isolated Compute Environments._ AWS 文档, 2023.
- [4] Apple. _Private Cloud Compute: A New Frontier for AI Privacy._ Apple Security Research Blog, 2024.
- [5] Coker 等. _Principles of Remote Attestation._ International Journal of Information Security, 2011.
- [6] TCG. _TPM 2.0 Library Specification._ Trusted Computing Group, 2019.

## 交叉参考

- `../../guides/secure-agent-deployment/` —— Agent 沙箱化；机密推理是 Agent 隔离之上的计算层保障
- `../../foundational/hopper-h100/` —— H100 硬件架构，包括 CC 模式内存子系统和 PCIe 加密
- `../../apple/afm/` —— Apple Foundation Models 与 PCC：商用 TEE 基础设施的垂直技术栈替代方案
- `../../anthropic/mcp/` —— MCP 传输安全保护协议层；机密推理保护其下方的计算层
