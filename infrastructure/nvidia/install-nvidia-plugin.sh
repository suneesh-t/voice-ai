#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing NVIDIA Device Plugin for Kubernetes ==="

# The NVIDIA device plugin exposes GPUs as schedulable resources.
# Required for any pod requesting nvidia.com/gpu resources.

helm repo add nvdp https://nvidia.github.io/k8s-device-plugin 2>/dev/null || true
helm repo update

helm upgrade --install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system \
  --set gfd.enabled=true \
  --set tolerations[0].key=nvidia.com/gpu \
  --set tolerations[0].operator=Exists \
  --set tolerations[0].effect=NoSchedule \
  --wait

echo ""
echo "=== NVIDIA Device Plugin installed ==="
echo "GPU nodes will now expose nvidia.com/gpu as a schedulable resource."
echo "Verify: kubectl get nodes -l nvidia.com/gpu.present=true"
