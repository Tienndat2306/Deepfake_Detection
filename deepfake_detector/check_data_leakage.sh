#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/root/deepfake_detector/processed_faces}"

ls "${DATA_ROOT}/train/Fake/" | sed 's/_[0-9]*$//' | sort > /tmp/train_ids.txt
ls "${DATA_ROOT}/val/Fake/" | sed 's/_[0-9]*$//' | sort > /tmp/val_ids.txt

LEAK_COUNT="$(comm -12 /tmp/train_ids.txt /tmp/val_ids.txt | wc -l)"
echo "${LEAK_COUNT}"
