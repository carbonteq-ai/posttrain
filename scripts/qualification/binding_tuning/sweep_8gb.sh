#!/usr/bin/env bash
# Tune the local-cuda-8gb binding family on a real 8 GB GPU (RTX 3070 Ti) inside
# the published serve job-kind image, so the vLLM build matches production.
#   scripts/qualification/binding_tuning/sweep_8gb.sh <exported-bindings-dir> <results-dir>
set -u
BINDINGS=$(realpath "$1"); RESULTS=$(realpath -m "$2"); mkdir -p "$RESULTS"
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
IMAGE=registry.lan/carbonteq/posttrain-kind-serve@sha256:7e0e5ef0166424a85f9525cf3891100e5aea93cad1e590c87096c55b048c9801

tune() {  # binding target_gb input output variants_json
  docker run --rm --gpus all --ipc=host \
    -v "$REPO/scripts/qualification/binding_tuning:/tuning:ro" -v "$BINDINGS:/bindings:ro" -v "$RESULTS:/results" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -e HF_HOME=/root/.cache/huggingface \
    --entrypoint /opt/posttrain/venv/bin/python "$IMAGE" /tuning/tune.py \
    --binding "/bindings/$1.json" --target-gb "$2" --input-tokens "$3" --output-tokens "$4" \
    --variants "$5" --output "/results/$1.json" 2>&1 | grep --line-buffered -E "^\[|Traceback|Error" | tail -20
}

GRAPHS='"graphs": {"enforce_eager": false, "enable_prefix_caching": true}'
BF16='"graphs_bf16": {"enforce_eager": false, "enable_prefix_caching": true, "dtype": "bfloat16"}'
BTOK='"graphs_bf16_btok8k": {"enforce_eager": false, "enable_prefix_caching": true, "dtype": "bfloat16", "max_num_batched_tokens": 8192}'
SEQS='"graphs_bf16_seqs8": {"enforce_eager": false, "enable_prefix_caching": true, "dtype": "bfloat16", "max_num_seqs": 8, "max_num_batched_tokens": 8192}'
# TurboQuant KV requires float16 compute; keep the binding's dtype.
TQ_BTOK='"graphs_btok8k": {"enforce_eager": false, "enable_prefix_caching": true, "max_num_batched_tokens": 8192}'
TQ_SEQS='"graphs_seqs8": {"enforce_eager": false, "enable_prefix_caching": true, "max_num_seqs": 8, "max_num_batched_tokens": 8192}'

tune qwen3.5-0.8b-vllm-screen-standard@1 8 1024 256 "{\"binding\": {}, $GRAPHS, $BF16, $BTOK, $SEQS}"
tune qwen3.5-0.8b-vllm-screen-mtp@1 8 1024 256 "{\"binding\": {}, $GRAPHS, $BF16, $BTOK}"
tune qwen3.5-0.8b-vllm-screen-turboquant@1 8 1024 256 "{\"binding\": {}, $GRAPHS, $TQ_BTOK, $TQ_SEQS}"
tune qwen3.5-2b-vllm-screen@1 8 1024 256 "{\"binding\": {}, $GRAPHS, $BF16, $BTOK, $SEQS}"
tune qwen3.5-2b-vllm-eval@2 8 2048 1024 "{\"binding\": {}, $GRAPHS, $TQ_BTOK, $TQ_SEQS}"
tune lfm2.5-1.2b-vllm-screen@1 8 1024 256 "{\"binding\": {}, $GRAPHS, $BF16, $BTOK, $SEQS}"
tune lfm2.5-1.2b-vllm-eval@1 8 2048 1024 "{\"binding\": {}, $GRAPHS, $TQ_BTOK, $TQ_SEQS}"
echo "SWEEP_8GB DONE"
