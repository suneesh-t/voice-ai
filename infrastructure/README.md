# Infrastructure — EKS Cluster & Ray Serve Platform

Complete Infrastructure-as-Code to provision and configure the **model-management** EKS cluster for GPU-accelerated ML model serving with Ray Serve.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EKS Cluster: model-management                │
│                    Region: us-east-1 │ K8s: 1.32               │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │   System Node Group  │  │     Karpenter GPU NodePools      │ │
│  │   (m5.xlarge × 2-4)  │  │                                  │ │
│  │                       │  │  ┌─────────────────────────────┐ │ │
│  │  • CoreDNS            │  │  │ gpu-g6-transcribe           │ │ │
│  │  • kube-proxy         │  │  │ g6e/g6/g5 (L40S/L4/A10G)   │ │ │
│  │  • Karpenter          │  │  │ On-demand, 1-4 GPUs         │ │ │
│  │  • KubeRay operator   │  │  └─────────────────────────────┘ │ │
│  │  • NVIDIA plugin      │  │  ┌─────────────────────────────┐ │ │
│  │  • CloudWatch agent   │  │  │ gpu-g6e-tts                 │ │ │
│  │                       │  │  │ g6e/g6/g5 (L40S/L4/A10G)   │ │ │
│  │                       │  │  │ On-demand, 1-4 GPUs         │ │ │
│  └──────────────────────┘  │  └─────────────────────────────┘ │ │
│                             └──────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    RayService Deployments                  │  │
│  │                                                            │  │
│  │  cohere-transcribe (ASR)       qwen3-tts (TTS)            │  │
│  │  ┌──────┐ ┌──────────┐       ┌──────┐ ┌──────────┐       │  │
│  │  │ Head │→│ GPU Worker│×1-4   │ Head │→│ GPU Worker│×1-4  │  │
│  │  │ CPU  │ │ + vLLM   │       │ CPU  │ │ + vllm-  │       │  │
│  │  │ 4C/8G│ │ 3C/14G/1G│       │ 4C/8G│ │   omni   │       │  │
│  │  └──────┘ └──────────┘       └──────┘ └──────────┘       │  │
│  │                                                            │  │
│  │  whisper-v3-turbo (ASR)       omni-voice (TTS)            │  │
│  │  ┌──────┐ ┌──────────┐       ┌──────┐ ┌──────────┐       │  │
│  │  │ Head │→│ GPU Worker│×1-4   │ Head │→│ GPU Worker│×1-4  │  │
│  │  │ CPU  │ │ + vLLM   │       │ CPU  │ │ + Omni-  │       │  │
│  │  │ 4C/8G│ │ 3C/14G/1G│       │ 4C/8G│ │   Voice  │       │  │
│  │  └──────┘ └──────────┘       └──────┘ └──────────┘       │  │
│  │                                                            │  │
│  │  kokoro-tts (TTS, 82M params, num_gpus=0.5)                │  │
│  │  ┌──────┐ ┌───────────────┐                                │  │
│  │  │ Head │→│ GPU Worker×1-3│ (up to 6 Serve replicas)       │  │
│  │  │ CPU  │ │ + Kokoro PT   │                                │  │
│  │  │ 4C/8G│ │ 3C/8G/1G      │                                │  │
│  │  └──────┘ └───────────────┘                                │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Components

| Component | Version | Purpose |
|---|---|---|
| **EKS** | 1.32 | Managed Kubernetes cluster |
| **eksctl** | >= 0.200.0 | Cluster provisioning |
| **Karpenter** | 1.3.0 | GPU node auto-provisioning (replaces Cluster Autoscaler) |
| **KubeRay** | 1.3.0 | Ray Serve lifecycle management on Kubernetes |
| **NVIDIA Device Plugin** | latest | Exposes GPUs as schedulable K8s resources |
| **CloudWatch Container Insights** | EKS add-on | Cluster metrics, logs, and observability |
| **Ray** | 2.52.0 | Distributed model serving framework |

---

## File Structure

```
infrastructure/
├── setup-cluster.sh                  # One-command full setup (runs all steps)
├── teardown-cluster.sh               # One-command full teardown
├── README.md
│
├── eks/
│   └── cluster-config.yaml           # eksctl ClusterConfig (EKS + system nodes + add-ons)
│
├── karpenter/
│   ├── install-karpenter.sh          # IAM roles, IRSA, Helm install
│   └── ec2-nodeclass.yaml            # EC2NodeClass (AMI, subnets, security groups, EBS)
│
├── kuberay/
│   └── install-kuberay.sh            # KubeRay operator via Helm
│
├── nvidia/
│   └── install-nvidia-plugin.sh      # NVIDIA device plugin via Helm
│
└── monitoring/
    ├── container-insights.sh          # CloudWatch Container Insights EKS add-on
    └── prometheus-servicemonitor.yaml # ServiceMonitor for Ray Serve metrics
```

GPU NodePool definitions live with their respective services:
- `cohere-transcribe/karpenter-gpu-nodepool.yaml` — pool `gpu-g6-transcribe`
- `qwen3-tts/karpenter-gpu-nodepool.yaml` — pool `gpu-g6e-tts`
- `whisper-large-v3-turbo/karpenter-gpu-nodepool.yaml` — pool `gpu-g6-whisper`
- `omni-voice/karpenter-gpu-nodepool.yaml` — pool `gpu-g6e-omnivoice`
- `kokoro-tts/karpenter-gpu-nodepool.yaml` — pool `gpu-kokoro-tts` (L4/A10G only; 3-GPU cap)

Only the cohere and qwen3 NodePools are applied by `setup-cluster.sh`. The whisper, omni-voice, and kokoro-tts NodePools are applied by each service's own `deploy.sh`.

---

## Prerequisites

| Tool | Min Version | Install |
|---|---|---|
| AWS CLI | v2 | [Install guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| eksctl | 0.200.0 | `brew install eksctl` |
| kubectl | 1.32 | `brew install kubectl` |
| helm | 3.16 | `brew install helm` |
| docker | latest | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |

**AWS IAM permissions required:**
- EKS full access
- EC2 (instances, security groups, subnets, ENIs)
- IAM (roles, policies, instance profiles, OIDC providers)
- CloudFormation (Karpenter stack)
- ECR (repository creation, image push)
- CloudWatch (Container Insights, logs)

---

## Quick Start — Full Setup

```bash
cd infrastructure
bash setup-cluster.sh
```

This runs all 5 steps in order:

| Step | What It Does | Time |
|---|---|---|
| 1 | Create EKS cluster with system node group and add-ons | ~15-20 min |
| 2 | Install NVIDIA device plugin for GPU scheduling | ~1 min |
| 3 | Install Karpenter + EC2NodeClass + GPU NodePools | ~3 min |
| 4 | Install KubeRay operator | ~1 min |
| 5 | Enable CloudWatch Container Insights | ~1 min |

Total: **~20 minutes** for a fresh cluster.

After setup, deploy the ML services:

```bash
export HF_TOKEN=hf_xxxxx

cd ../cohere-transcribe && bash deploy.sh
cd ../whisper-large-v3-turbo && bash deploy.sh
cd ../qwen3-tts && bash deploy.sh
cd ../omni-voice && bash deploy.sh
cd ../kokoro-tts && bash deploy.sh
```

---

## Step-by-Step Manual Setup

If you prefer to run each step individually:

### Step 1: Create EKS Cluster

```bash
eksctl create cluster -f eks/cluster-config.yaml
```

Creates:
- EKS 1.32 cluster with OIDC provider
- System node group: 2× `m5.xlarge` (managed, auto-scaling 2-4)
- Add-ons: VPC-CNI (with network policy), CoreDNS, kube-proxy, EBS-CSI
- CloudWatch cluster logging (API, audit, authenticator, controller, scheduler)

### Step 2: NVIDIA Device Plugin

```bash
bash nvidia/install-nvidia-plugin.sh
```

Required for any pod requesting `nvidia.com/gpu` resources. Installs with GPU Feature Discovery (GFD) enabled.

### Step 3: Karpenter

```bash
bash karpenter/install-karpenter.sh
kubectl apply -f karpenter/ec2-nodeclass.yaml
kubectl apply -f ../cohere-transcribe/karpenter-gpu-nodepool.yaml
kubectl apply -f ../qwen3-tts/karpenter-gpu-nodepool.yaml
kubectl apply -f ../whisper-large-v3-turbo/karpenter-gpu-nodepool.yaml
kubectl apply -f ../omni-voice/karpenter-gpu-nodepool.yaml
```

Creates:
- Karpenter IAM roles (CloudFormation stack)
- IRSA for Karpenter controller
- EC2 instance profile for Karpenter-provisioned nodes
- `aws-auth` ConfigMap mapping for new nodes
- Karpenter Helm release
- `EC2NodeClass` — AMI, subnet, security group, and EBS config
- Two GPU `NodePool` resources (one per service)

### Step 4: KubeRay Operator

```bash
bash kuberay/install-kuberay.sh
```

Installs the KubeRay operator in `kuberay-system` namespace. Watches all namespaces for `RayService` and `RayCluster` CRDs.

### Step 5: CloudWatch Container Insights (optional)

```bash
bash monitoring/container-insights.sh
```

Enables application and performance logs + metrics in CloudWatch.

---

## GPU Node Provisioning

Karpenter provisions GPU nodes **on demand** when Ray worker pods are scheduled. No GPU nodes exist until a service is deployed.

### NodePool: `gpu-g6-transcribe` (Cohere STT)

| Property | Value |
|---|---|
| Instance Types | `g6e.xlarge`, `g6e.2xlarge`, `g6.xlarge`, `g5.xlarge` |
| GPUs | L40S, L4, A10G |
| Capacity | On-Demand |
| Max Resources | 32 CPU, 128Gi memory, 4 GPUs |
| Taint | `nvidia.com/gpu=true:NoSchedule` |
| Consolidation | When empty or underutilized (5 min delay) |
| Node TTL | 14 days |

### NodePool: `gpu-g6e-tts` (Qwen3 TTS)

Same instance types and limits as above, with label `workload: qwen3-tts`.

### NodePool: `gpu-g6-whisper` (Whisper Large-v3-Turbo)

Same instance types and limits as above, with label `workload: whisper-v3-turbo`.

### NodePool: `gpu-g6e-omnivoice` (OmniVoice TTS)

Same instance types and limits as above, with label `workload: omni-voice`.

---

## Cluster Verification

After setup, verify all components:

```bash
# Cluster and nodes
kubectl get nodes
kubectl cluster-info

# System components
kubectl get pods -n kube-system | grep -E "karpenter|nvidia|coredns|aws-node"

# KubeRay operator
kubectl get pods -n kuberay-system

# Karpenter resources
kubectl get nodepools
kubectl get ec2nodeclasses

# After deploying services
kubectl get rayservice -n default
kubectl get pods -l ray.io/cluster
```

---

## Teardown

To delete the entire cluster and all associated AWS resources:

```bash
cd infrastructure
bash teardown-cluster.sh
```

Deletes in reverse order: RayServices → KubeRay → Karpenter (including IAM) → NVIDIA plugin → CloudWatch → ECR repos → EKS cluster.

**This is destructive and irreversible.** You will be prompted to type `yes` to confirm.

---

## Cost Considerations

| Resource | Instance | On-Demand Price (us-east-1) | Notes |
|---|---|---|---|
| System nodes | 2× m5.xlarge | ~$0.384/hr | Always running |
| GPU (STT) | g6.xlarge (L4) | ~$0.98/hr | Karpenter scales to 0 when idle |
| GPU (STT) | g5.xlarge (A10G) | ~$1.006/hr | Alternative |
| GPU (TTS) | g6e.xlarge (L40S) | ~$1.86/hr | Karpenter scales to 0 when idle |
| EKS control plane | — | $0.10/hr | Fixed cost |

Karpenter consolidation shuts down GPU nodes after 5 minutes of no workload, minimizing idle GPU costs.

---

## Known Issues

| Issue | Cause | Fix |
|---|---|---|
| `MetricsHead` 30s timeout crash | Ray 2.52 dashboard startup | Already set in RayService YAMLs: `RAY_DASHBOARD_SUBPROCESS_MODULE_WAIT_READY_TIMEOUT=300` |
| Worker `maxReplicas` vs Serve `max_replicas` | Worker group caps at 2, Serve expects 4 | Set worker `maxReplicas: 4` in ray-service-*.yaml |
| NodePool instance type mismatch | Existing nodes may not match NodePool spec | Delete stale nodes: `kubectl delete node <name>` |
| Karpenter nodes stuck `NotReady` | NVIDIA plugin not installed | Ensure Step 2 ran before Step 3 |
