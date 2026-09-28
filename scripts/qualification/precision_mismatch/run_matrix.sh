#!/bin/bash
# The whole offline precision check for one model, one GPU job at a time:
# prompts -> vLLM samples (bf16, fp16, fp32 or the Transformers fp32 fallback) ->
# Transformers scoring (bf16, fp16, fp32) -> vLLM fp16 activation hooks -> report.
#
# Required: IMAGE, OUT, MODEL, REVISION, PREP_ARGS (prep.py source arguments).
# Optional: TEMP (1.0), TOP_P (1.0), VLLM_ARGS, HOOK_ARGS, SKIP_FP32 (set to skip fp32 rows).
# Example (LFM2.5-2.6B, AutomationBench traces, the training sampling):
#   IMAGE=... OUT=$HOME/precision/lfm26 TRACES=$HOME/precision/traces \
#   MODEL=LiquidAI/LFM2.5-2.6B REVISION=654f9463ce32b05d0429d76fe1f580b27d4c1ac0 \
#   PREP_ARGS="--source traces --num-prompts 64 --max-tokens 4096 --min-prompt-tokens 2048 --max-prompt-tokens 20480" \
#   TEMP=0.5 TOP_P=0.95 VLLM_ARGS="--max-model-len 24576 --max-num-seqs 64 --max-num-batched-tokens 8192 \
#   --gpu-memory-utilization 0.85" HOOK_ARGS="--max-num-batched-tokens 4096 --gpu-memory-utilization 0.6" run_matrix.sh
set -euo pipefail
: "${IMAGE:?}" "${OUT:?}" "${MODEL:?}" "${REVISION:?}" "${PREP_ARGS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
RUN="$HERE/run.sh"
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
SAMPLING=(--temperature "$TEMP" --top-p "$TOP_P")
M=(--model "$MODEL" --revision "$REVISION")
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/matrix.log"; }
busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
if busy; then echo "the GPU is in use; refusing to start (one GPU job at a time)" >&2; exit 2; fi

[ -s "$OUT/prompts.json" ] || { log prep; "$RUN" prep.py "${M[@]}" $PREP_ARGS --out /work/prompts.json; }
for dtype in bfloat16 float16 float32; do
  [ "$dtype" = float32 ] && [ -n "${SKIP_FP32:-}" ] && continue
  [ -s "$OUT/samples_$dtype.json" ] && continue
  log "vllm $dtype"
  if ! "$RUN" gen_vllm.py "${M[@]}" --dtype "$dtype" "${SAMPLING[@]}" ${VLLM_ARGS:-} >"$OUT/gen_$dtype.log" 2>&1; then
    log "vllm $dtype FAILED (see gen_$dtype.log, failed_$dtype.json)"
    if [ "$dtype" = float32 ]; then
      log "fp32 sampler fallback: Transformers generate (slow, sequential)"
      "$RUN" score_hf.py "${M[@]}" --dtype float32 --generate "${SAMPLING[@]}" --tag float32 >"$OUT/gen_float32_hf.log" 2>&1
    fi
  fi
done
for dtype in bfloat16 float16 float32; do
  [ "$dtype" = float32 ] && [ -n "${SKIP_FP32:-}" ] && continue
  [ -s "$OUT/scores_$dtype.json" ] && continue
  log "score $dtype"
  "$RUN" score_hf.py "${M[@]}" --dtype "$dtype" >"$OUT/score_$dtype.log" 2>&1
done
if [ ! -s "$OUT/vllm_hooks_float16.json" ]; then
  log "vllm float16 hooks (eager decode + prefill replay of the bf16 samples)"
  "$RUN" gen_vllm.py "${M[@]}" --dtype float16 --hooks --tag float16 --prefill-samples /work/samples_bfloat16.json \
    "${SAMPLING[@]}" ${VLLM_ARGS:-} ${HOOK_ARGS:-} >"$OUT/hooks_float16.log" 2>&1
fi
log report
"$RUN" analyze.py /work | tee "$OUT/report.txt"
