#!/bin/bash
# Deploy EAGLE3 Producer-Consumer Phase 3
# Usage: bash deploy_p3.sh [--dry-run]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo '=== Creating ConfigMap from script files ==='
kubectl create configmap eagle3-pc-demo-p3-scripts   --from-file=producer_p3.py="${SCRIPT_DIR}/producer_p3.py"   --from-file=consumer_p3.py="${SCRIPT_DIR}/consumer_p3.py"   --dry-run=client -o yaml | kubectl apply -f -

echo '=== Applying LWS ==='
kubectl apply -f "${SCRIPT_DIR}/eagle3-pc-demo-lws-p3.yaml" --dry-run=client -o yaml | grep -E '(name|kind|nodeName)' | head -20

if [ "$1" != '--dry-run' ]; then
  kubectl apply -f "${SCRIPT_DIR}/eagle3-pc-demo-lws-p3.yaml"
  echo '=== Waiting for pods ==='
  sleep 5
  kubectl get pods -l app=eagle3-pc-demo-p3
fi
