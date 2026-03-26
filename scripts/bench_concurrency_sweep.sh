#!/bin/bash
# Benchmark concurrency sweep for vLLM speculative decoding.
# Requires vLLM server running on port 8200.
#
# Usage:
#   scripts/bench_concurrency_sweep.sh <label> [concurrency_list]
#
# Example:
#   scripts/bench_concurrency_sweep.sh mtp_v6_ep9 "1 4 8 16 32"
#
# Output: tab-separated results to stdout + saved JSON to /tmp/vllm_bench_<label>/

set -e

LABEL="${1:?Usage: $0 <label> [concurrency_list]}"
CONCURRENCIES="${2:-1 4 8 16 32}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_BIN="${SCRIPT_DIR}/../.venv/bin/vllm"
DATASET="${SCRIPT_DIR}/../data/bench_prompts_500_sharegpt.json"
NUM_PROMPTS=200

# Fallback dataset locations
if [ ! -f "$DATASET" ]; then
    DATASET="/tmp/bench_prompts_500_sharegpt.json"
fi
if [ ! -f "$DATASET" ]; then
    echo "ERROR: benchmark dataset not found" >&2
    exit 1
fi

# Verify server is running
if ! curl -s http://localhost:8200/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server not responding on port 8200" >&2
    exit 1
fi

# Check GPU sw_power_cap throttling (indicates degraded performance, need reboot)
check_power_cap() {
    local throttled
    throttled=$(nvidia-smi --query-gpu=index,clocks_throttle_reasons.sw_power_cap --format=csv,noheader 2>/dev/null \
        | grep -i "active" | grep -v "Not Active" || true)
    if [ -n "$throttled" ]; then
        echo "ERROR: GPU sw_power_cap throttling detected! Performance will be degraded (~2x)." >&2
        echo "Affected GPUs:" >&2
        echo "$throttled" >&2
        echo "Fix: reboot the machine before benchmarking." >&2
        exit 1
    fi
}
check_power_cap

RESULT_DIR="/tmp/vllm_bench_${LABEL}"
mkdir -p "$RESULT_DIR"

echo "=== Benchmark: $LABEL ==="
echo "Dataset: $DATASET ($NUM_PROMPTS prompts)"
echo "Concurrencies: $CONCURRENCIES"
echo ""

for C in $CONCURRENCIES; do
    echo "--- Concurrency=$C ---"
    $VLLM_BIN bench serve \
        --backend openai-chat \
        --base-url http://localhost:8200 \
        --endpoint /v1/chat/completions \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --num-prompts $NUM_PROMPTS \
        --request-rate inf \
        --max-concurrency $C \
        --trust-remote-code \
        --temperature 0 \
        --save-result \
        --result-dir "${RESULT_DIR}/c${C}/" \
        2>&1 | grep -E "^(Successful|Benchmark duration|Request throughput|Output token throughput|Peak output|Total token throughput|Mean TTFT|Median TTFT|Mean TPOT|P99 TPOT|Acceptance rate|Acceptance length|Per-position acceptance)"
    echo ""
done

echo "=== Done: $LABEL ==="
echo "Results saved to: $RESULT_DIR"
