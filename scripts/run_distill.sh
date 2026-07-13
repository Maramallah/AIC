#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/contest-tracking-data}"
TEACHER_DIR="${TEACHER_DIR:-${REPO_ROOT}/teacher_predictions}"

# Base/pre-distillation LightFC checkpoint.
# Do NOT default to checkpoints/stage1_score0729.pth.tar.
INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/checkpoints/lightfc_base.pth.tar}"

WORK_DIR="${WORK_DIR:-${REPO_ROOT}/outputs/train_distill}"

python -u -m mtcaic4.distill \
  --data-root "${DATA_ROOT}" \
  --teacher-dir "${TEACHER_DIR}" \
  --init-ckpt "${INIT_CKPT}" \
  --work-dir "${WORK_DIR}" \
  --generate \
  --train \
  --clean-pairs \
  --pairs-per-seq 80 \
  --max-gap 100 \
  --batch-size 16 \
  --epochs 15 \
  --lr 5e-5 \
  --min-lr 1e-6 \
  --warmup 3 \
  --alpha 0.5 \
  --l1-weight 2.0 \
  --search-center-jitter 0.5 \
  --search-scale-jitter 0.1