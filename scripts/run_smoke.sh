#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}
$PY src/train.py --config configs/smoke_r8_letter.json --dry_run
