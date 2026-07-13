#!/bin/bash
set -e

DATA_ROOT=${1}
CHECKPOINT=${2}
OUTPUT=${3:-outputs/submissions/submission.csv}

if [ -z "$DATA_ROOT" ]; then
  echo "Usage: bash scripts/run_inference.sh DATA_ROOT CHECKPOINT [OUTPUT]"
  exit 1
fi

if [ -z "$CHECKPOINT" ]; then
  echo "Usage: bash scripts/run_inference.sh DATA_ROOT CHECKPOINT [OUTPUT]"
  exit 1
fi

export PYTHONPATH=$(pwd)/src:$(pwd)/third_party/LightFC:$PYTHONPATH

python -m mtcaic4.infer \
  --repo-dir third_party/LightFC \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-csv "$OUTPUT"