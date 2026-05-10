#!/usr/bin/env bash
set -euo pipefail

# Run direct answer-letter inference for the hidden test split.
# Example: bash scripts/run_inference.sh runs/final-trainval-budgetmax-r8-connector16/best_model
PY=${PY:-python}
CKPT=${1:-"runs/final-trainval-budgetmax-r8-connector16/best_model"}
SCORE_OUT=${2:-"runs/final_submission_scores.csv"}
TEST_CSV=${TEST_CSV:-data/test.csv}
IMAGE_DIR=${IMAGE_DIR:-data/images/test}
IMAGE_MODE=${IMAGE_MODE:-nosplit512}
MAX_LENGTH=${MAX_LENGTH:-1024}
CHOICE_TTA=${CHOICE_TTA:-deterministic}
CHOICE_TTA_MAX=${CHOICE_TTA_MAX:-8}
TTA_BATCH_SIZE=${TTA_BATCH_SIZE:-8}

EXTRA_ARGS=()
if [[ "${INCLUDE_CAPTION:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--include_caption --caption_max_chars "${CAPTION_MAX_CHARS:-256}")
fi
if [[ -n "${TTA_IMAGE_MODES:-}" ]]; then
    # shellcheck disable=SC2206
    MODES=($TTA_IMAGE_MODES)
    EXTRA_ARGS+=(--tta_image_modes "${MODES[@]}")
fi

$PY src/inference.py \
    --test_csv "$TEST_CSV" \
    --image_dir "$IMAGE_DIR" \
    --ckpts $CKPT \
    --out submission.csv \
    --score_out "$SCORE_OUT" \
    --image_mode "$IMAGE_MODE" \
    --max_length "$MAX_LENGTH" \
    --prompt_variants default exam context_first no_metadata \
    --metadata_fields subject grade topic \
    --choice_tta "$CHOICE_TTA" \
    --choice_tta_max "$CHOICE_TTA_MAX" \
    --tta_batch_size "$TTA_BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"
