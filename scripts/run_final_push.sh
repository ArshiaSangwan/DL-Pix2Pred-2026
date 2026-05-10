#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
FINAL_RUN=${FINAL_RUN:-runs/final-trainval-budgetmax-r8-connector16}
FINAL_CONFIG=${FINAL_CONFIG:-configs/final_trainval_budgetmax_connector_r8.json}
RUN_DIR=${RUN_DIR:-runs/final_push_pccp}
OUT=${OUT:-submission.csv}
TOP_K=${TOP_K:-5}
IMAGE_MODE=${IMAGE_MODE:-nosplit512}
MAX_LENGTH=${MAX_LENGTH:-1024}
CHOICE_TTA=${CHOICE_TTA:-deterministic}
CHOICE_TTA_MAX=${CHOICE_TTA_MAX:-8}
TTA_BATCH_SIZE=${TTA_BATCH_SIZE:-8}
MAKE_SOUP=${MAKE_SOUP:-1}
USE_PROVIDED_CAPTIONS=${USE_PROVIDED_CAPTIONS:-1}
BUILD_PROVIDED_CAPTIONS=${BUILD_PROVIDED_CAPTIONS:-$USE_PROVIDED_CAPTIONS}
CAPTION_DIR=${CAPTION_DIR:-data/pccp_captioned}

mkdir -p "$RUN_DIR"

if [[ "$BUILD_PROVIDED_CAPTIONS" == "1" ]]; then
  mkdir -p "$CAPTION_DIR"
  for split in train val test; do
    "$PY" scripts/build_provided_captions.py \
      --input_csv "data/${split}.csv" \
      --out_csv "$CAPTION_DIR/${split}_captioned.csv" \
      --manifest "$CAPTION_DIR/${split}_captioned_manifest.json" \
      --caption_col caption
  done
fi

if [[ ! -s "$FINAL_RUN/best_model/adapter_model.safetensors" ]]; then
  "$PY" src/train.py --config "$FINAL_CONFIG"
fi

if [[ "$USE_PROVIDED_CAPTIONS" == "1" ]]; then
  TEST_CSV=${TEST_CSV:-$CAPTION_DIR/test_captioned.csv}
  if [[ ! -f "$TEST_CSV" ]]; then
    echo "Captioned test CSV is missing: $TEST_CSV" >&2
    echo "Set BUILD_PROVIDED_CAPTIONS=1 or disable USE_PROVIDED_CAPTIONS." >&2
    exit 2
  fi
  INCLUDE_CAPTION_ARGS=(--include_caption --caption_max_chars "${CAPTION_MAX_CHARS:-256}")
else
  TEST_CSV=${TEST_CSV:-data/test.csv}
  INCLUDE_CAPTION_ARGS=()
fi
IMAGE_DIR=${IMAGE_DIR:-data/images/test}

# shellcheck disable=SC2206
PROMPT_VARIANTS=(${PROMPT_VARIANTS:-default exam context_first no_metadata answer_phrase})
# shellcheck disable=SC2206
METADATA_FIELDS=(${METADATA_FIELDS:-subject grade topic})

IMAGE_TTA_ARGS=()
if [[ -n "${TTA_IMAGE_MODES:-}" ]]; then
  # shellcheck disable=SC2206
  IMAGE_MODES=($TTA_IMAGE_MODES)
  IMAGE_TTA_ARGS=(--tta_image_modes "${IMAGE_MODES[@]}")
fi

mapfile -t CKPTS < <(
  "$PY" scripts/select_checkpoints.py \
    --run_dir "$FINAL_RUN" \
    --top_k "$TOP_K" \
    --min_epoch "${MIN_EPOCH:-1}" \
    --include_best \
    --format lines
)
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "No checkpoints found under $FINAL_RUN" >&2
  exit 1
fi

printf '%s\n' "${CKPTS[@]}" > "$RUN_DIR/selected_checkpoints.txt"
echo "Selected ${#CKPTS[@]} final-family checkpoints:"
sed 's/^/  /' "$RUN_DIR/selected_checkpoints.txt"

CKPT_SCORE="$RUN_DIR/final_family_checkpoint_scores.csv"
"$PY" src/inference.py \
  --test_csv "$TEST_CSV" \
  --image_dir "$IMAGE_DIR" \
  --ckpts "${CKPTS[@]}" \
  --out "$RUN_DIR/final_family_checkpoint_submission.csv" \
  --score_out "$CKPT_SCORE" \
  --image_mode "$IMAGE_MODE" \
  --max_length "$MAX_LENGTH" \
  --prompt_variants "${PROMPT_VARIANTS[@]}" \
  --metadata_fields "${METADATA_FIELDS[@]}" \
  --choice_tta "$CHOICE_TTA" \
  --choice_tta_max "$CHOICE_TTA_MAX" \
  --tta_batch_size "$TTA_BATCH_SIZE" \
  "${IMAGE_TTA_ARGS[@]}" \
  "${INCLUDE_CAPTION_ARGS[@]}"

SCORE_CSVS=("$CKPT_SCORE")

if [[ "$MAKE_SOUP" == "1" ]]; then
  SOUP_DIR="$RUN_DIR/final_family_adapter_soup"
  "$PY" scripts/soup_adapters.py \
    --adapters "${CKPTS[@]}" \
    --out_dir "$SOUP_DIR" \
    --force
  SOUP_SCORE="$RUN_DIR/final_family_soup_scores.csv"
  "$PY" src/inference.py \
    --test_csv "$TEST_CSV" \
    --image_dir "$IMAGE_DIR" \
    --ckpts "$SOUP_DIR" \
    --out "$RUN_DIR/final_family_soup_submission.csv" \
    --score_out "$SOUP_SCORE" \
    --image_mode "$IMAGE_MODE" \
    --max_length "$MAX_LENGTH" \
    --prompt_variants "${PROMPT_VARIANTS[@]}" \
    --metadata_fields "${METADATA_FIELDS[@]}" \
    --choice_tta "$CHOICE_TTA" \
    --choice_tta_max "$CHOICE_TTA_MAX" \
    --tta_batch_size "$TTA_BATCH_SIZE" \
    "${IMAGE_TTA_ARGS[@]}" \
    "${INCLUDE_CAPTION_ARGS[@]}"
  SCORE_CSVS+=("$SOUP_SCORE")
fi

"$PY" scripts/score_ensemble.py \
  --test_csv "$TEST_CSV" \
  --score_csvs "${SCORE_CSVS[@]}" \
  --out "$OUT" \
  --score_out "$RUN_DIR/final_pccp_scores.csv"

"$PY" - <<PY
import pandas as pd
sub = pd.read_csv("$OUT")
test = pd.read_csv("$TEST_CSV")
assert list(sub.columns) == ["id", "answer"], sub.columns.tolist()
assert len(sub) == len(test), (len(sub), len(test))
assert set(sub.id) == set(test.id)
merged = test[["id", "num_choices"]].merge(sub, on="id", how="left")
assert not merged["answer"].isna().any()
merged["answer"] = merged["answer"].astype(int)
bad = (merged["answer"] < 0) | (merged["answer"] >= merged["num_choices"])
assert not bad.any(), merged.loc[bad].head().to_dict("records")
print("$OUT is valid:", len(sub), "rows")
PY

echo "Final PCCP score dump: $RUN_DIR/final_pccp_scores.csv"
