#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

set -a
source .env
set +a

PYTHON_BIN="${MLLM_PRUNE_ENV_PATH}/bin/python"
RUN_ROOT="${EXPERIMENT_005_RUN_ROOT:-/data2/wenhaozhang/runs/mllm_prune/experiment_005}"
CONFIG="configs/experiments/005_learned_utility_scorer_v2.yaml"
read -r -a GPUS <<< "${EXP005_GPUS:-6 7}"
NUM_SHARDS="${#GPUS[@]}"

if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "EXP005_GPUS must contain at least one physical GPU index" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"/{aligned_cache,labels,checkpoints,logs,pids,predictions}
echo "$$" > "$RUN_ROOT/pids/full_pipeline.pid"
echo "run_root=$RUN_ROOT gpus=${GPUS[*]} shards=$NUM_SHARDS started=$(date --iso-8601=seconds)"

if [[ ! -f "$EXPERIMENT_003_CALIBRATION_PATH" ]]; then
  echo "phase=calibration gpu=${GPUS[0]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -u scripts/fit_score_calibration.py \
    --device cuda:0 \
    --attn-implementation flash_attention_2
else
  echo "phase=calibration reuse=$EXPERIMENT_003_CALIBRATION_PATH"
fi

echo "phase=aligned_cache"
cache_worker() {
  local shard="$1"
  local gpu="$2"
  local split
  for split in train validation test; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/cache_aligned_scores.py \
      --config "$CONFIG" \
      --split "$split" \
      --output-root "$RUN_ROOT/aligned_cache" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --device cuda:0 \
      --attn-implementation flash_attention_2
  done
}
worker_pids=()
for shard in "${!GPUS[@]}"; do
  cache_worker "$shard" "${GPUS[$shard]}" &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do wait "$pid"; done

echo "phase=conditional_labels"
label_worker() {
  local shard="$1"
  local gpu="$2"
  local split
  for split in train validation; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/generate_conditional_utility_labels.py \
      --config "$CONFIG" \
      --split "$split" \
      --aligned-cache-root "$RUN_ROOT/aligned_cache" \
      --output-root "$RUN_ROOT/labels" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --device cuda:0 \
      --attn-implementation flash_attention_2
  done
}
worker_pids=()
for shard in "${!GPUS[@]}"; do
  label_worker "$shard" "${GPUS[$shard]}" &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do wait "$pid"; done

echo "phase=scorer_training"
worker_pids=()
training_seeds=(0 1 2)
for position in "${!training_seeds[@]}"; do
  seed="${training_seeds[$position]}"
  gpu="${GPUS[$((position % NUM_SHARDS))]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/train_utility_scorer_v2.py \
    --config "$CONFIG" \
    --label-root "$RUN_ROOT/labels" \
    --checkpoint-dir "$RUN_ROOT/checkpoints" \
    --seed "$seed" \
    --device cuda:0 &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do wait "$pid"; done

echo "phase=test_evaluation"
worker_pids=()
prediction_paths=()
for shard in "${!GPUS[@]}"; do
  output="$RUN_ROOT/predictions/shard_${shard}.jsonl"
  prediction_paths+=("$output")
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PYTHON_BIN" -u scripts/run_learned_utility_pruning_v2.py \
    --config "$CONFIG" \
    --aligned-cache-root "$RUN_ROOT/aligned_cache" \
    --checkpoint-dir "$RUN_ROOT/checkpoints" \
    --output "$output" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$shard" \
    --device cuda:0 \
    --attn-implementation flash_attention_2 &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do wait "$pid"; done

echo "phase=summary"
"$PYTHON_BIN" scripts/summarize_learned_utility_v2.py \
  --predictions "${prediction_paths[@]}" \
  --output "$RUN_ROOT/summary.json" \
  --expected-samples 50

echo "completed=$(date --iso-8601=seconds) summary=$RUN_ROOT/summary.json"
