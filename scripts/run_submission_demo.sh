#!/bin/bash
# Generate long-sequence demo for PRISM (mixed scenarios: wave hello → combat/sports/daily/gestures → wave goodbye)
# Output: outputs/submission_demo/submission_sequence/.../pred.npz (viewable in motion_vis_web)
#
# Usage (run from main repository root):
#   bash opensource/prism/scripts/run_submission_demo.sh
# Skip rewriter: REWRITER_SERVICE_URL= bash opensource/prism/scripts/run_submission_demo.sh
# Specify variants: VARIANTS=submission_showcase,v1_mixed,v2_mixed bash opensource/prism/scripts/run_submission_demo.sh
# Force 1 GPU (sequential): NUM_GPUS=1 bash opensource/prism/scripts/run_submission_demo.sh
# overlap_frames=1 is passed by default. Multi-GPU: variants split across GPUs, run in parallel.
#
# Or specify checkpoint (.pth or converted HF directory prism_1.4b):
#   bash opensource/prism/scripts/run_submission_demo.sh work_dirs/prism_1b_tp2m_hq_t5xxl_256text_aug_1frame/iter_11000.pth
#   bash opensource/prism/scripts/run_submission_demo.sh opensource/prism/pretrained_models/prism_1.4b

set -e
cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"

# Default: 1frame config + iter_11000; overlap_frames=1
CFG="${CFG:-configs/prism/prism_1b_tp2m_hq_t5xxl_256text_aug_1frame.py}"
CHECKPOINT="${1:-work_dirs/prism_1b_tp2m_hq_t5xxl_256text_aug_1frame/iter_11000.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submission_demo}"

# Default: generate all 15 demos (submission_showcase + v1~v14_mixed)
VARIANTS="${VARIANTS:-all}"
# rewriter: set REWRITER_SERVICE_URL to your HTTP endpoint (e.g. http://localhost:8080/v1); leave empty to skip
REWRITER_URL="${REWRITER_SERVICE_URL:-}"
echo "[run_submission_demo] cfg=$CFG checkpoint=$CHECKPOINT output_root=$OUTPUT_ROOT variants=$VARIANTS use_rewriter=${USE_REWRITER:-true} rewriter_url=${REWRITER_URL:+set}"
python3 scripts/evaluation/eval_babel_prism_submission_demo.py \
  --cfg "$CFG" \
  --checkpoint "$CHECKPOINT" \
  --output_root "$OUTPUT_ROOT" \
  --variants "$VARIANTS" \
  --overlap_frames 1 \
  --use_rewriter ${USE_REWRITER:-true} \
  --rewriter_service_url "${REWRITER_URL}"

echo "[Done] Output saved under $OUTPUT_ROOT/submission_sequence/; pred.npz can be previewed with motion_vis_web."
S"
else
  VARLIST="$VARIANTS"
fi

# Split variants into NGPUS chunks and launch parallel processes
IFS=',' read -ra ARR <<< "$VARLIST"
TOTAL=${#ARR[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "[Error] No variants to generate."
  exit 1
fi

echo "[run_submission_demo] cfg=$CFG checkpoint=$CHECKPOINT output_root=$OUTPUT_ROOT variants=$VARIANTS use_rewriter=${USE_REWRITER:-true} rewriter_url=${REWRITER_URL:+set} num_gpus=$NGPUS total_variants=$TOTAL"

if [ "$NGPUS" -eq 1 ]; then
  # Sequential
  python3 scripts/evaluation/eval_babel_prism_submission_demo.py \
    --cfg "$CFG" \
    --checkpoint "$CHECKPOINT" \
    --output_root "$OUTPUT_ROOT" \
    --variants "$VARLIST" \
    --overlap_frames 1 \
    --use_rewriter ${USE_REWRITER:-true} \
    --rewriter_service_url "${REWRITER_URL}"
else
  # Parallel: assign variants to GPUs (contiguous chunks)
  CHUNK_SIZE=$(( (TOTAL + NGPUS - 1) / NGPUS ))
  PIDS=()
  for ((gpu=0; gpu<NGPUS; gpu++)); do
    START=$((gpu * CHUNK_SIZE))
    [ "$START" -ge "$TOTAL" ] && break
    END=$(( (gpu + 1) * CHUNK_SIZE ))
    [ "$END" -gt "$TOTAL" ] && END=$TOTAL
    CHUNK=""
    for ((i=START; i<END; i++)); do
      CHUNK="${CHUNK}${ARR[$i]},"
    done
    CHUNK="${CHUNK%,}"
    [ -z "$CHUNK" ] && continue
    echo "[GPU $gpu] Variants: $CHUNK"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/evaluation/eval_babel_prism_submission_demo.py \
      --cfg "$CFG" \
      --checkpoint "$CHECKPOINT" \
      --output_root "$OUTPUT_ROOT" \
      --variants "$CHUNK" \
      --overlap_frames 1 \
      --use_rewriter ${USE_REWRITER:-true} \
      --rewriter_service_url "${REWRITER_URL}" &
    PIDS+=($!)
  done
  FAILED=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      FAILED=1
    fi
  done
  [ "$FAILED" -ne 0 ] && exit 1
fi

echo "[Done] Output saved under $OUTPUT_ROOT/submission_sequence/; pred.npz can be previewed with motion_vis_web."
