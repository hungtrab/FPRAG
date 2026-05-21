#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source venv/bin/activate

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-fprag_rtn_w3g128}"
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}"
SLUG="${SLUG:-Meta_Llama_3_8B}"
BITS="${BITS:-3}"
GROUP_SIZE="${GROUP_SIZE:-128}"

OUT="./quantized_models/${SLUG}_rtn_w${BITS}g${GROUP_SIZE}"
LOG="logs/${SLUG}_rtn_w${BITS}g${GROUP_SIZE}.log"

mkdir -p quantized_models logs

echo "START_RAW_RTN model=${MODEL_PATH} bits=${BITS} group_size=${GROUP_SIZE} out=${OUT} $(date -Is)"

python rtn.py \
  --model-path "$MODEL_PATH" \
  --bits "$BITS" \
  --group-size "$GROUP_SIZE" \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

status="${PIPESTATUS[0]}"

if [[ "$status" == "0" ]]; then
  python wandb_log_rtn.py \
    --kind quantize \
    --name "${SLUG}_rtn_w${BITS}g${GROUP_SIZE}" \
    --model-slug "$SLUG" \
    --bits "$BITS" \
    --group-size "$GROUP_SIZE" \
    --log-path "$LOG" \
    --project "$WANDB_PROJECT" || true
fi

echo "END_RAW_RTN status=${status} $(date -Is)"
exit "$status"
