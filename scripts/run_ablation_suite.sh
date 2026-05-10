#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
OUT_DIR=${OUT_DIR:-runs/eval}
BASELINE_ACC=${BASELINE_ACC:-0.8177}
CAPTION_DIR=${CAPTION_DIR:-data/pccp_captioned}

mkdir -p "$OUT_DIR" "$CAPTION_DIR"

echo "[0/4] Static checks and smoke test"
$PY -m py_compile src/*.py scripts/*.py
PY="$PY" bash scripts/run_smoke.sh

train_if_missing() {
  local config="$1"
  local marker="$2"
  if [[ -s "$marker" ]]; then
    echo "Found $marker; skipping training for $config"
  else
    echo "Training $config"
    $PY src/train.py --config "$config"
  fi
}

score_val_if_missing() {
  local name="$1"
  local ckpt="$2"
  shift 2
  local score_out="$OUT_DIR/${name}_scores.csv"
  local pred_out="$OUT_DIR/${name}_submission.csv"
  if [[ -s "$score_out" ]]; then
    echo "Found $score_out; skipping inference for $name"
    return
  fi
  if [[ ! -e "$ckpt" && "$ckpt" != HuggingFaceTB/* ]]; then
    echo "Skipping $name; checkpoint not found: $ckpt"
    return
  fi
  echo "Scoring $name"
  $PY src/inference.py \
    --test_csv data/val.csv \
    --image_dir data/images/val \
    --ckpts "$ckpt" \
    --out "$pred_out" \
    --score_out "$score_out" \
    "$@"
  $PY scripts/evaluate_scores.py --scores "$score_out" --labels data/val.csv > "$OUT_DIR/${name}_report.txt"
}

echo "[1/4] Train or reuse selector adapters"
train_if_missing configs/exp_lr3e-4.json runs/r8-letter-nosplit512-lr3e-4/best_model/adapter_model.safetensors
train_if_missing configs/budgetmax_connector_r8.json runs/budgetmax-r8-connector16-nosplit512-lr3e-4/best_model/adapter_model.safetensors
if [[ "${RUN_DORA_R6:-0}" == "1" ]]; then
  train_if_missing configs/r6_dora_attn_mlp_nosplit768_aug.json runs/r6-dora-attn-mlp-nosplit768-aug/best_model/adapter_model.safetensors
fi

echo "[2/4] Build provided-context caption CSVs"
for split in train val test; do
  $PY scripts/build_provided_captions.py \
    --input_csv "data/${split}.csv" \
    --out_csv "$CAPTION_DIR/${split}_captioned.csv" \
    --manifest "$CAPTION_DIR/${split}_captioned_manifest.json" \
    --caption_col caption
done

echo "[3/4] Validation scoring sweeps"
score_val_if_missing r8_ep5_prompt_choice_tta runs/r8-letter-nosplit512-lr3e-4/checkpoints/ckpt_ep5_acc0.8177 \
  --image_mode nosplit512 --max_length 1024 \
  --prompt_variants default exam context_first no_metadata answer_phrase \
  --choice_tta deterministic --choice_tta_max 8 \
  --metadata_fields subject grade topic

score_val_if_missing connector_best_prompt_choice_tta runs/budgetmax-r8-connector16-nosplit512-lr3e-4/best_model \
  --image_mode nosplit512 --max_length 1024 \
  --prompt_variants default exam context_first no_metadata answer_phrase \
  --choice_tta deterministic --choice_tta_max 8 \
  --metadata_fields subject grade topic

score_val_if_missing connector_best_image_tta runs/budgetmax-r8-connector16-nosplit512-lr3e-4/best_model \
  --image_mode nosplit512 --tta_image_modes nosplit512 nosplit768 split --max_length 1024 \
  --prompt_variants default exam context_first no_metadata answer_phrase \
  --choice_tta deterministic --choice_tta_max 8 \
  --metadata_fields subject grade topic

score_val_if_missing connector_best_provided_caption runs/budgetmax-r8-connector16-nosplit512-lr3e-4/best_model \
  --test_csv "$CAPTION_DIR/val_captioned.csv" \
  --image_mode nosplit512 --max_length 1024 \
  --prompt_variants default context_first answer_phrase \
  --include_caption --caption_max_chars 160 \
  --choice_tta deterministic --choice_tta_max 8 \
  --metadata_fields subject grade topic

if [[ -s runs/r6-dora-attn-mlp-nosplit768-aug/best_model/adapter_model.safetensors ]]; then
  score_val_if_missing dora_r6_aug runs/r6-dora-attn-mlp-nosplit768-aug/best_model \
    --image_mode nosplit768 --max_length 896 \
    --prompt_variants default question_first context_first answer_phrase \
    --choice_tta deterministic --choice_tta_max 8 \
    --metadata_fields subject grade topic
fi

echo "[4/4] Search score ensembles"
$PY scripts/search_score_ensemble.py \
  --labels data/val.csv \
  --score_dir "$OUT_DIR" \
  --out "$OUT_DIR/best_ensemble.json" \
  --pred_out "$OUT_DIR/best_ensemble_val_predictions.csv"

$PY - <<PY
import json
from pathlib import Path

baseline = float("$BASELINE_ACC")
ensemble = json.loads(Path("$OUT_DIR/best_ensemble.json").read_text())
acc = float(ensemble["best"]["accuracy"])
print(f"Best score ensemble validation accuracy: {acc:.4f}")
print("PASS" if acc >= baseline else f"Below baseline {baseline:.4f}; keep the r8 baseline/final-fit path.")
PY
