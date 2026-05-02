#!/usr/bin/env bash
set -euo pipefail

KUBERAY_VERSION="1.3.0"

echo "=== Installing KubeRay Operator v${KUBERAY_VERSION} ==="

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>/dev/null || true
helm repo update

helm upgrade --install kuberay-operator kuberay/kuberay-operator \
  --version "${KUBERAY_VERSION}" \
  --namespace kuberay-system \
  --create-namespace \
  --set watchNamespace="" \
  --wait

echo ""
echo "=== KubeRay Operator installed ==="
echo "Verify: kubectl get pods -n kuberay-system"
echo ""
echo "The operator watches all namespaces for RayService/RayCluster resources."
echo "Deploy services with:"
echo "  cd cohere-transcribe && bash deploy.sh"
echo "  cd qwen3-tts && bash deploy.sh"
