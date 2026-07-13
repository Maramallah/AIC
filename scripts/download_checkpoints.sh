#!/bin/bash
set -e

REPO="https://github.com/youssef-othman12/MTC-AIC4-EagleAI/releases/download/v1.0-checkpoints"

mkdir -p checkpoints

# ── Required: final inference checkpoint ─────────────────────────────────────
if [ ! -f "checkpoints/best_checkpoint.pth.tar" ]; then
    echo "[1/3] Downloading best_checkpoint.pth.tar ..."
    wget -q --show-progress \
        -O checkpoints/best_checkpoint.pth.tar \
        "${REPO}/best_checkpoint.pth.tar"
    echo "      Verifying sha256 ..."
    echo "024521e99dc2915ff956b9be5c2c97cd4547b0062219aa40648f49e4be0b65cd  checkpoints/best_checkpoint.pth.tar" \
        | sha256sum -c -
    echo "      ✓ best_checkpoint.pth.tar"
else
    echo "[1/3] best_checkpoint.pth.tar already exists, skipping."
fi

# ── Optional: base LightFC checkpoint (needed only to re-run distillation) ───
if [ "${DOWNLOAD_OPTIONAL:-0}" = "1" ]; then
    if [ ! -f "checkpoints/lightfc_base.pth.tar" ]; then
        echo "[2/3] Downloading lightfc_base.pth.tar ..."
        wget -q --show-progress \
            -O checkpoints/lightfc_base.pth.tar \
            "${REPO}/lightfc_base.pth.tar"
        echo "      ✓ lightfc_base.pth.tar"
    else
        echo "[2/3] lightfc_base.pth.tar already exists, skipping."
    fi

    # ── Optional: OD teacher predictions (needed only to re-run distillation) ─
    if [ ! -d "od-predictions" ]; then
        echo "[3/3] Downloading od-predictions.zip ..."
        wget -q --show-progress \
            -O od-predictions.zip \
            "${REPO}/od-predictions.zip"
        echo "      Extracting ..."
        unzip -q od-predictions.zip
        rm od-predictions.zip
        echo "      ✓ od-predictions/"
    else
        echo "[3/3] od-predictions/ already exists, skipping."
    fi
else
    echo "[2/3] Skipping lightfc_base.pth.tar  (set DOWNLOAD_OPTIONAL=1 to download)"
    echo "[3/3] Skipping od-predictions.zip    (set DOWNLOAD_OPTIONAL=1 to download)"
fi

echo ""
echo "Done. Required assets are ready in checkpoints/"