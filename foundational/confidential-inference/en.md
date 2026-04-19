# Confidential LLM Inference

- **Technologies covered**: NVIDIA H100 Confidential Computing (SPM), AWS Nitro Enclaves, Azure Confidential VMs (AMD SEV-SNP), Apple Private Cloud Compute
- **Standards**: TCG TPM 2.0, IETF RATS (Remote ATtestation procedureS), AMD SEV-SNP ABI specification
- **Links**: [H100 CC Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/h100-security-whitepaper.pdf) · [AWS Nitro Enclaves](https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclaves-concepts.html) · [Apple PCC Security](https://security.apple.com/documentation/private-cloud-compute)

## TL;DR

Standard cloud inference exposes model weights, user prompts, and model outputs to the cloud provider's hypervisor layer at runtime. For regulated industries — healthcare (HIPAA), finance (SOX/GDPR), legal, defense — and any enterprise handling sensitive proprietary data, this exposure is unacceptable by policy. Confidential computing hardware addresses this with a hardware root of trust: encrypted memory, cryptographic attestation, and encrypted communication channels that make hypervisor inspection computationally infeasible even for a malicious cloud operator. H100's Secure Performance Mode (CC mode), AWS Nitro Enclaves, and Azure AMD SEV-SNP VMs each implement this at different layers and with different trust models. The performance overhead is modest — 5–10% latency increase — but the infrastructure complexity and cost premium (~20–30% on cloud) are substantial. Understanding the threat model, attestation chain, and key management architecture is necessary before committing to a confidential inference deployment.

## The Threat Model

Confidential computing is designed against a specific adversary: a compromised or malicious infrastructure operator. This is distinct from the more common threat models addressed by application-layer encryption.

In standard cloud inference:

1. User submits a prompt over HTTPS. The TLS termination happens at the application load balancer — the cloud provider's infrastructure. The plaintext prompt is visible inside the data center network.
2. The prompt is sent to the inference service running on a GPU instance. The GPU driver and hypervisor have unrestricted access to GPU memory, which contains model weights, KV cache, and the current prompt being processed.
3. The response is returned. During the entire inference, a compromised hypervisor could log every prompt and response.

Who does this threat model actually apply to? Not startups; not most enterprise SaaS. It applies to:

- **Healthcare AI**: patient records, clinical notes, diagnostic queries processed by an LLM — covered under HIPAA; the covered entity cannot prove to a regulator that the cloud provider did not access PHI.
- **Legal AI**: privilege-protected attorney-client communications sent to an LLM for analysis.
- **Financial AI**: non-public material information (MNPI) processed for trading or compliance analysis.
- **Government/defense**: classified or controlled unclassified information (CUI).
- **Model IP protection**: a model provider who does not want their weights visible to the cloud infrastructure on which they serve.

Standard TLS + IAM policies address the network and access control layer. They do not protect against a compromised hypervisor with physical or privileged software access to the GPU host. Confidential computing closes this gap.

## H100 Confidential Computing (Secure Performance Mode)

NVIDIA introduced hardware-based confidential computing in Hopper (H100). The production name is **Secure Performance Mode (SPM)**; the broader feature set is also called H100 Confidential Computing.

### Hardware Root of Trust

Each H100 chip has a unique attestation key pair fused at manufacturing time by NVIDIA. The private key is stored in on-chip fuses that cannot be read after manufacturing; NVIDIA retains a record of the corresponding public key and uses it to issue attestation certificates. This per-chip key is the root of the trust chain.

### CC Mode Architecture

When the GPU boots in CC mode:

1. The GPU firmware is measured (SHA-256 hash of the firmware image) and the measurement is signed by the chip's attestation key.
2. An encrypted channel is established between the GPU and the CPU Trusted Execution Environment (TEE — typically AMD SEV or Intel TDX on the host side).
3. All PCIe DMA transfers are encrypted using session keys derived from the GPU-CPU handshake. The hypervisor sits on the PCIe bus but cannot read plaintext DMA traffic.
4. GPU memory (HBM3) is encrypted using per-session keys. The GPU memory controller handles encryption/decryption transparently; CUDA kernels run on plaintext data inside the GPU, but data leaving the GPU is always ciphertext.

The complete attestation and key establishment flow:

```
1. Client requests attestation from the GPU service
2. GPU generates attestation report:
   - Firmware measurement (SHA-256)
   - Boot configuration
   - Signed by chip attestation key
3. Client verifies report signature against NVIDIA Attestation Service
   (NVIDIA's OCSP-like attestation service returns the signing cert chain)
4. Client verifies firmware measurement matches expected values
5. Client and GPU negotiate a session key (ECDH over the verified channel)
6. Model weights transferred to GPU memory via encrypted PCIe DMA
7. Inference runs inside encrypted GPU memory; prompts enter and responses
   exit via the encrypted session key channel
```

This flow provides **verifiable isolation**: the client can cryptographically prove that inference is running on a genuine H100 with known firmware, without trusting the cloud provider's claims.

### Performance Overhead

H100 CC mode imposes two categories of overhead:

- **PCIe encryption overhead**: encrypting/decrypting DMA transfers. For inference, this primarily affects the initial weight loading phase. At PCIe Gen5 bandwidth (~128 GB/s), adding AES-256-GCM encryption adds approximately 5–8% overhead to data transfer time. Inference itself (which happens inside GPU memory) is unaffected.
- **Memory fence and sync overhead**: ensuring operations happen in the correct order relative to memory encryption adds small but non-zero synchronization overhead per CUDA kernel launch. Empirically: ~5–10% end-to-end latency increase for LLM inference workloads.

For a 70B model on H100 serving at 2,000 tokens/second in standard mode: CC mode brings this to approximately 1,800–1,900 tokens/second. The overhead is acceptable for regulated workloads; the cost is policy compliance rather than inference budget.

## AWS Nitro Enclaves

AWS Nitro Enclaves are isolated compute environments within EC2 instances, designed for processing sensitive data on AWS infrastructure.

### Architecture

A Nitro Enclave is a stripped-down VM created from a subset of the parent EC2 instance's vCPU and memory resources. It runs a minimal Linux kernel (or Amazon Linux 2) with no persistent storage, no interactive access (no SSH), and a deliberately restricted network interface — the only communication channel is a local VSOCK connection to the parent EC2 instance.

The security guarantees come from the **AWS Nitro Hypervisor**: the enclave's memory and CPU state are isolated from the parent instance and from AWS operators. Even the EC2 instance owner cannot inspect enclave memory. AWS attestation provides cryptographic proof of what code is running inside the enclave.

### Attestation Flow

1. The enclave image (EIF — Enclave Image File) is built with `nitro-cli`. The build process produces a cryptographic measurement (PCR values — Platform Configuration Registers) of the enclave content.
2. At runtime, the enclave generates an attestation document containing its PCR measurements, signed by the **AWS Nitro TPM** (a hardware-backed TPM embedded in the Nitro card).
3. The parent EC2 instance (or an external KMS) verifies the attestation document against the expected PCR values.
4. Only after successful attestation is the decryption key released to the enclave. Sensitive data (the user prompt, or a model decryption key) is sealed against the PCR values — it can only be decrypted by a verified instance of that exact enclave image.

### Limitation: CPU-Only and vGPU Gap

The critical limitation of Nitro Enclaves for LLM inference: **full GPU passthrough to Nitro Enclaves is not available on standard EC2 instances**. Enclaves run on CPU or on vGPU, where the vGPU driver is managed by the Nitro hypervisor but GPU memory isolation is not equivalent to H100 CC mode.

Full GPU confidential computing on AWS requires **bare-metal EC2 instances** (e.g., `p4de.24xlarge` bare metal with A100s) paired with H100 SPM when H100 bare-metal instances are available. On bare metal, the guest OS runs without an intervening hypervisor, so the H100 CC attestation chain goes directly from the GPU to the tenant's application without passing through an AWS-controlled hypervisor layer.

This architecture split means AWS Nitro Enclaves are useful for: prompt encryption/decryption, key management operations, and inference on CPU for small models. For large model GPU inference with strong confidentiality, H100 bare-metal plus H100 CC is the correct architecture.

## Azure Confidential VMs (AMD SEV-SNP)

Azure's confidential inference offering is built on **AMD SEV-SNP** (Secure Encrypted Virtualization — Secure Nested Paging).

### AMD SEV-SNP

SEV-SNP encrypts guest VM memory with a per-VM encryption key that resides only in the AMD Secure Processor (AMD SP) — a separate security processor embedded in the CPU die. The hypervisor manages memory address translation but cannot access the plaintext content of the VM's memory pages.

SEV-SNP adds nested page table integrity protection (the "SNP" part): each guest memory page has a reverse mapping table entry whose integrity is protected by the AMD SP. This prevents the hypervisor from remapping guest memory to a different physical page (a replay or aliasing attack), which earlier SEV versions were vulnerable to.

Performance overhead for AMD SEV-SNP on CPU-side computation: approximately 3–7% across typical workloads. The overhead is from memory controller encryption/decryption and from the additional integrity check on every page access.

### Azure Integration

Azure's confidential GPU offering pairs an AMD SEV-SNP confidential VM with H100 CC mode:

- The guest VM runs inside AMD SEV-SNP; the hypervisor cannot read guest memory.
- The H100 is accessed from the confidential VM; the H100-to-VM communication channel is encrypted as described in the H100 CC section above.
- The attestation chain combines AMD SEV-SNP attestation (proves the VM environment) with H100 attestation (proves the GPU environment) into a compound attestation that a tenant can verify.

**Azure Attestation Service (MAA)** verifies both attestation reports and issues a signed token that the tenant's application can use as proof that inference ran in a verified confidential environment.

### Key Management Integration

Azure Key Vault can be configured to release keys only to attested confidential VMs. The flow:

1. Model weights are stored encrypted in Azure Blob Storage.
2. The decryption key is held in Azure Key Vault under a policy: "release only to VMs that present a valid MAA attestation token matching the expected measurements."
3. At startup, the confidential VM attests, receives the MAA token, presents it to Key Vault, and receives the decryption key.
4. The VM decrypts the model weights into SEV-SNP protected memory, then loads them to the H100 via the encrypted CC channel.
5. At no point is the decryption key accessible to the hypervisor or Azure operators.

## Apple Private Cloud Compute (Cross-Reference)

Apple's **Private Cloud Compute (PCC)** represents a distinct architectural approach to confidential inference. It is covered in detail in `../../apple/afm/`; the key points for comparison are:

- PCC runs on **Apple Silicon (M2/M4 Ultra)** servers, not commodity x86 hardware. The memory encryption and attestation are implemented in Apple's custom silicon.
- Trust model difference: in H100 CC and SEV-SNP, a third-party attestation service (NVIDIA, AMD, Azure, AWS) is in the chain of trust. In PCC, the user's **iPhone/Mac device verifies the PCC node directly** via Apple's Transparency Log — a Certificate-Transparency-style append-only log of all PCC software images that have ever been deployed. There is no runtime third-party attestation call; the device checks the log before sending any request.
- Software enforcement: PCC enforces that the model inference code cannot log prompts, cannot write to persistent storage, and cannot make outbound network calls except to specific Apple services. These constraints are enforced by the OS running inside the PCC node and verified by the Transparency Log entry for that software image.
- Apple controls the full stack: hardware, firmware, OS, model, serving stack. This makes the trust model simpler (fewer parties) but less flexible (only Apple can deploy to PCC; enterprises cannot deploy their own models).

## TEE-Gated Serving Pattern

Production confidential inference systems for enterprises typically implement a **TEE-gated serving pattern** that balances security, cost, and throughput:

```
Incoming requests
        |
        v
Request classifier
   - Contains PII / PHI / MNPI?  → Route to CC nodes
   - Standard content?           → Route to standard nodes
        |                                |
        v                                v
  CC-enabled cluster             Standard cluster
  (H100 SPM, higher cost)        (H100 standard, lower cost)
  ~20-30% cost premium           baseline cost
        |                                |
        +----------+   +----------------+
                   |   |
                   v   v
             Response merge
             Return to client
```

Key management in this pattern:

- A **Hardware Security Module (HSM)** holds the master key for the model weights.
- Per-session derived keys are established via HKDF from the master key and session-specific entropy.
- The CC cluster attests at startup; the HSM releases the master key only to verified TEE environments.
- Session keys never leave the TEE; the HSM does not see individual prompt/response content.

Audit logging: each CC inference session is associated with an attestation report. For compliance purposes (HIPAA audit trail, SOC 2), the attestation report is logged alongside metadata (timestamp, model version, session ID) but not the prompt content itself. This provides proof-of-trustworthiness without logging sensitive data.

## Latency and Cost Overhead Summary

| Platform | CPU/memory overhead | GPU overhead | Cloud cost premium | Notes |
|---|---|---|---|---|
| H100 SPM (CC mode) | N/A (GPU only) | ~5–10% end-to-end latency | ~20–25% on GPU instances | Encryption on PCIe DMA; GPU HBM encrypted |
| AWS Nitro Enclaves | ~3–5% (VSOCK overhead) | Not available on standard; requires bare metal for GPU | ~15–20% on bare metal | CPU-only for most deployments |
| Azure AMD SEV-SNP | ~3–7% CPU | Requires H100 CC pairing | ~20–30% on confidential GPU SKUs | AMD SP handles memory encryption |
| Apple PCC | Not applicable | Apple Silicon; overhead not disclosed | Apple-managed; no public pricing | Vertical stack; not available for enterprise self-hosting |
| Software-only (no TEE) | ~0% | ~0% | Baseline | No protection against compromised hypervisor |

## Engineering Tradeoffs

| Dimension | Option | Gained | Given up |
|---|---|---|---|
| Threat coverage | H100 CC + SEV-SNP | Full stack encryption: CPU memory + PCIe + GPU memory | ~10–15% combined latency overhead; deployment complexity |
| Threat coverage | Application-layer encryption only | Simple deployment; no hardware dependency | Prompts visible in plaintext inside host RAM and GPU during inference |
| Cost management | Route all traffic to CC nodes | Uniform security posture | 20–30% cost increase across all inference spend |
| Cost management | TEE-gated routing (sensitive only) | Cost-efficient; CC nodes only handle sensitive traffic | Classifier must be trusted; mis-classification exposes sensitive data to standard nodes |
| Attestation model | Third-party (NVIDIA/AMD/Azure) | Auditable; external verifier | Trust in attestation service; OCSP-like availability dependency |
| Attestation model | First-party (Apple PCC) | Fewer trusted parties; device-direct verification | Only Apple can deploy; no enterprise model customization |
| Key management | HSM + per-session HKDF | Master key never exposed; per-session forward secrecy | HSM availability becomes critical path; adds 1–2ms per session establishment |

## References

- [1] NVIDIA. _H100 Tensor Core GPU Architecture: Confidential Computing._ NVIDIA Technical Whitepaper, 2022.
- [2] Kaplan et al. _AMD SEV-SNP: Strengthening VM Isolation with Integrity Protection and More._ AMD Technical Whitepaper, 2020.
- [3] AWS. _AWS Nitro Enclaves: Isolated Compute Environments._ AWS Documentation, 2023.
- [4] Apple. _Private Cloud Compute: A New Frontier for AI Privacy._ Apple Security Research Blog, 2024.
- [5] Coker et al. _Principles of Remote Attestation._ International Journal of Information Security, 2011.
- [6] TCG. _TPM 2.0 Library Specification._ Trusted Computing Group, 2019.

## Cross-References

- `../../guides/secure-agent-deployment/` — agent sandboxing; confidential inference is the compute-layer complement to agent isolation
- `../../foundational/hopper-h100/` — H100 hardware architecture including the CC mode memory subsystem and PCIe encryption
- `../../apple/afm/` — Apple Foundation Models and PCC: the vertical-stack alternative to commodity TEE infrastructure
- `../../anthropic/mcp/` — MCP transport security secures the protocol layer; confidential inference secures the compute layer beneath it
