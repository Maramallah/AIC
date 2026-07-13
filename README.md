# EagleAI — MTC-AIC4 UAV Single-Object Tracking

[![Competition](https://img.shields.io/badge/Competition-MTC--AIC4-blue)]()
[![Tracker](https://img.shields.io/badge/Tracker-LightFC%20%2F%20LightFC--ViT-teal)]()
[![Score](https://img.shields.io/badge/Public%20LB-0.7393-brightgreen)]()
[![Params](https://img.shields.io/badge/Params-5.5M-orange)]()
[![Checkpoint](https://img.shields.io/badge/Checkpoint-27.95%20MB-lightgray)]()

---

## ⚠️  Important Notes for Evaluators — Please Read First

1. **Use our provided checkpoint for evaluation.**
   We strongly recommend using `best_checkpoint.pth.tar` (available under GitHub Releases)
   to reproduce our submitted predictions. This checkpoint was generated in our Kaggle
   training notebook and has been validated against the original competition submission.

2. **Checkpoint reproducibility status.**
   Our `distill.py` script closely reproduces the Kaggle training notebook and achieves
   the same best epoch (epoch 14) and near-identical best val IoU (~0.8287 vs 0.8286).
   However, a newly trained checkpoint produces slightly different predictions
   (mean IoU ≈ 0.885 vs 1.00 against original). This is expected: tracking is autoregressive,
   so sub-pixel model weight differences compound over 74,293 frames.

3. **Kaggle notebook provided as reference.**
   The original Kaggle training + inference notebook is included at:
     notebooks/kaggle_train_infer.ipynb
   If any unexpected errors occur with the scripts, it can be run directly on a
   Kaggle environment (GPU T4 × 2) as an authoritative fallback.

4. **Official evaluation checkpoint:** `checkpoints/best_checkpoint.pth.tar`
   sha256 : 024521e99dc2915ff956b9be5c2c97cd4547b0062219aa40648f49e4be0b65cd
   Size   : 27.95 MB  |  Epoch: 14  |  Val IoU: 0.8286858484858558

5. **Expected accuracy:** ~ 0.7393

6. **Contact:** For any clarifications please reach out to:
   youssef Othman — yussufothman537@gmail.com

---

## Repository Structure

```
mtc-AIC4-EagleAI/
├── checkpoints/
│   └── .gitkeep                    # placeholder — checkpoint excluded from git
│                                   # download via scripts/download_checkpoints.sh
├── configs/
│   ├── inference_exact.yaml        # inference configuration (locked v6-conservative)
│   └── distill_stage1.yaml         # distillation configuration
├── docs/
│   ├── inference_notes.md          # inference pipeline details
│   ├── distillation_notes.md       # distillation notes and hyperparameters
│   └── docker_notes.md             # Docker build notes and known issues
├── notebooks/
│   └── lightfc-vit-distillation&inference.ipynb    # original Kaggle training + inference notebook
├── outputs/
│   ├── submissions/final_submission.csv     # final submission on kaggle acc = .7393 
│   ├── logs/.gitkeep
├── scripts/
│   ├── download_checkpoints.sh     # downloads best_checkpoint.pth.tar from Releases
│   ├── run_inference.sh            # end-to-end inference convenience script
│   └── run_distill.sh              # full distillation convenience script
├── src/
│   └── mtcaic4/
│       ├── __init__.py
│       ├── infer.py                # inference pipeline (locked v6-conservative)
│       └── distill.py             # distillation pipeline
├── third_party/
│   └── LightFC/                   # LightFC tracker (submodule)
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Method Overview

Our solution is built on [**LightFC-ViT**](https://arxiv.org/pdf/2310.05392), a lightweight one-stream tracker,
fine-tuned via knowledge distillation from stronger teacher trackers ([ODTrack](https://arxiv.org/pdf/2401.01686) / LoRAT).

**Key techniques applied:**
- **Knowledge distillation** — head-only fine-tuning using soft labels from ODTrack teachers combined with GT annotations ($\alpha$-weighted GIoU + L1 + teacher SmoothL1 loss). 
  > *Builds upon the framework introduced in [ODTrack (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28591/29149), utilizing its online temporal token tracking as a high-accuracy teacher to supervise lightweight models.*
- **EMA template update** — Exponential Moving Average template refresh conditioned on score, IoU, area, and interval thresholds to suppress drift in long UAV sequences. 
  > *Adapted from the conditional update mechanisms established in [MixFormer (CVPR 2022)](https://ieeexplore.ieee.org/document/10380715) and [STARK (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Yan_Learning_Spatio-Temporal_Transformer_for_Visual_Tracking_ICCV_2021_paper.pdf) to prevent the tracker from ingesting background noise during target occlusions.*
- **Anchor-based re-detection** — lost target recovery using spatial anchor priors to robustly handle target disappearance and reappearance in UAV scenarios. 
  > *Sourced from [SiamAPN (IEEE TGRS 2021)](https://vision4robotics.github.io/publication/2021_tgrs_siamapn_ext/), which formalized the use of predefined spatial anchors to exhaustively scan and instantly recover fast-moving aerial targets.*

**Efficiency profile (evaluated on Tesla T4):**

| Metric                        | Value            |
|-------------------------------|------------------|
| Parameters                    | 5.5 M            |
| Checkpoint size               | 27.95 MB         |
| GFLOPs — forward_backbone     | ~0.4 – 0.7       |
| GFLOPs — forward_tracking     | ~2.0 – 2.6       |
| GFLOPs — per-frame (steady)   | ~2.1 – 2.7       |
| Latency per frame (T4)        | ~5 ms            |
| **Public LB accuracy**        | **0.7393**       |

**Scoring formula:**
```
FinalScore = S_acc − 0.2 × S_eff
S_acc      = 0.6 × AUC + 0.4 × NormPrecision
S_eff      = 0.25×(FLOPs/30G) + 0.15×(Params/50M)
           + 0.35×(Latency/30ms) + 0.25×(Size/500MB)
```
LightFC was selected to optimize the combined accuracy-efficiency trade-off.

---

## Environment

| Component          | Specification                  |
|--------------------|-------------------------------|
| OS                 | Ubuntu (native or WSL)        |
| Environment manager| Conda                         |
| Python             | 3.10                          |
| GPU (training)     | Kaggle Tesla T4 × 2           |
| GPU (inference)    | Any CUDA-capable GPU          |
| PyTorch (Docker)   | 2.1.2                         |
| PyTorch (Kaggle)   | 2.10.0               |

---

## Step 0 — Download the Checkpoint

Run this once before any inference or evaluation:

```bash
bash scripts/download_checkpoints.sh
```

Verify integrity:
```bash
sha256sum checkpoints/best_checkpoint.pth.tar
# Expected: 024521e99dc2915ff956b9be5c2c97cd4547b0062219aa40648f49e4be0b65cd
```

---

## Option A — Docker (Recommended for Reproducibility)

> **Note:** Docker support is functional but still being stabilized on some host
> configurations (particularly WSL with limited disk space on C:).
> Ensure at least 50 GB of free disk before building a CUDA image.
> If Docker fails, proceed to Option B (Conda).

### A1 — Inference Only (Docker)

**Build:**
```bash
docker build --target inference -t mtc-aic4-eagleai:infer .
```

**Download checkpoint:**
```bash
bash scripts/download_checkpoints.sh
```

**Run inference:**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/checkpoints:/workspace/checkpoints:ro" \
  -v "$(pwd)/contest-tracking-data:/workspace/contest-tracking-data:ro" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  mtc-aic4-eagleai:infer \
  python -m mtcaic4.infer \
    --checkpoint checkpoints/best_checkpoint.pth.tar \
    --data_root contest-tracking-data \
    --output outputs/submissions/submission.csv
```

---

### A2 — Full Pipeline: Distillation + Inference (Docker)

> Requires the optional release assets: `lightfc_base.pth.tar` and `od-predictions.zip`
> (see GitHub Releases). Extract `od-predictions.zip` so the layout is:
> `od-predictions/train/*.txt` and `od-predictions/test/*.txt`

**Build:**
```bash
docker build --target full -t mtc-aic4-eagleai:full .
```

**Step 1 — Distillation:**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/checkpoints:/workspace/checkpoints" \
  -v "$(pwd)/contest-tracking-data:/workspace/contest-tracking-data:ro" \
  -v "$(pwd)/od-predictions:/workspace/od-predictions:ro" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  mtc-aic4-eagleai:full \
  bash scripts/run_distill.sh
```

**Step 2 — Inference with new checkpoint:**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/checkpoints:/workspace/checkpoints:ro" \
  -v "$(pwd)/contest-tracking-data:/workspace/contest-tracking-data:ro" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  mtc-aic4-eagleai:full \
  bash scripts/run_inference.sh
```

> **Reminder:** For official evaluation always prefer `best_checkpoint.pth.tar`.
> A newly trained checkpoint produces near-identical but not byte-identical results
> due to GPU non-determinism across different hardware and CUDA versions.

---

## Option B — Conda Environment (Fallback if Docker Fails)

### Setup

```bash
# Clone the repository
git clone https://github.com//mtc-AIC4-EagleAI.git
cd mtc-AIC4-EagleAI

# Initialize LightFC submodule
git submodule update --init --recursive

# Create and activate the Conda environment
conda create -n mtcaic4 python=3.10 -y
conda activate mtcaic4

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install Python dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Download the checkpoint
bash scripts/download_checkpoints.sh
```

---

### B1 — Inference Only (Conda)

```bash
conda activate mtcaic4

python -m mtcaic4.infer \
  --checkpoint checkpoints/best_checkpoint.pth.tar \
  --data_root contest-tracking-data \
  --output outputs/submissions/submission.csv
```

Or via the convenience script:
```bash
conda activate mtcaic4
bash scripts/run_inference.sh
```

---

### B2 — Full Pipeline: Distillation + Inference (Conda)

> Requires the optional release assets: `lightfc_base.pth.tar` and `od-predictions.zip`.
> Extract `od-predictions.zip` to the repo root before running.

**Step 1 — Distillation:**
```bash
conda activate mtcaic4

python -m mtcaic4.distill \
  --data_root contest-tracking-data \
  --teacher_preds od-predictions \
  --output_dir outputs/runs/distill_stage1
```

Or via the convenience script:
```bash
conda activate mtcaic4
bash scripts/run_distill.sh
```

**Step 2 — Inference with new checkpoint:**
```bash
python -m mtcaic4.infer \
  --checkpoint outputs/runs/distill_stage1/best_checkpoint.pth.tar \
  --data_root contest-tracking-data \
  --output outputs/submissions/submission_new.csv
```

---

## Inference Health Check (Expected Output)

Running on the full public_lb split (89 sequences, 74,293 frames)
with `best_checkpoint.pth.tar`:

```
EMA updates:          21517
EMA skip (score):     5734
EMA skip (interval):  42848
EMA skip (area):      3490
EMA skip (IoU):       615
Lost recoveries:      133
Matched frames:       74293 / 74293
```

Diff against original submitted CSV:
```
mean IoU            = 1.00000
frames IoU < 0.99   = 0.000%
frames IoU < 0.95   = 0.000%
```

---

## GitHub Release Assets

| Asset                     | Purpose                              | Required     |
|---------------------------|--------------------------------------|--------------|
| `best_checkpoint.pth.tar` | Final inference checkpoint (27.95 MB)| ✅ Yes        |
| `lightfc_base.pth.tar`    | Base checkpoint for distillation     | Optional     |
| `od-predictions.zip`      | Teacher soft labels (2.73 MB)        | Optional     |

All binary assets are distributed via GitHub Releases and are excluded from the
git repository to keep `git clone` clean and fast.

---

## Kaggle Notebook (Authoritative Fallback)

```
notebooks/kaggle_train_infer.ipynb
```

This is the original notebook used to produce `best_checkpoint.pth.tar` and the
competition submission. It can be run directly on Kaggle (GPU T4 × 2) if any
unexpected errors occur with the scripts above.

---

## Contact

For any technical questions or clarifications regarding reproduction:

**Yussuf Othman** — yussufothman537@gmail.com