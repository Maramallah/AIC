#!/usr/bin/env python3
"""
Distillation / fine-tuning pipeline for MTC-AIC4 UAV Tracking.

This script trains the LightFC student tracker using:
  - GT boxes from contest-tracking-data train split
  - teacher boxes, e.g. ODTrack/LoRAT predictions
  - a LightFC-compatible initialization checkpoint


"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW_API = False
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def from_nested(obj):
        if isinstance(obj, dict):
            return DotDict({k: DotDict.from_nested(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [DotDict.from_nested(x) for x in obj]
        return obj


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_lightfc_to_path(lightfc_dir: Path) -> None:
    if not (lightfc_dir / "lib").exists():
        raise FileNotFoundError(f"LightFC lib directory not found: {lightfc_dir / 'lib'}")
    sys.path.insert(0, str(lightfc_dir))


def patch_lightfc_runtime() -> None:
    """
    Runtime compatibility patches only.
    Avoid editing vendored third_party files during normal execution.
    """

    # torch._six shim for older code.
    if "torch._six" not in sys.modules:
        six_mod = types.ModuleType("torch._six")
        six_mod.string_classes = (str, bytes)
        sys.modules["torch._six"] = six_mod

    # Patch load_pretrain so model construction does not try to fetch external weights.
    try:
        load_mod = importlib.import_module("lib.utils.load")
        load_mod.load_pretrain = lambda model, path=None, **kwargs: model
    except Exception:
        pass

    try:
        pretrain_mod = importlib.import_module("lib.models.component.pretrain")
        pretrain_mod.load_pretrain = lambda model, path=None, **kwargs: model
    except Exception:
        pass


def safe_torch_load(path: str | Path, map_location="cpu"):
    """
    Torch 2.6+ defaults can reject old checkpoints unless weights_only=False.
    This wrapper is backward-compatible with older torch versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_grad_scaler():
    if _AMP_NEW_API:
        return make_grad_scaler()
    return GradScaler()


def amp_autocast():
    if _AMP_NEW_API:
        return autocast("cuda")
    return autocast()


def force_cuda(module: torch.nn.Module) -> None:
    """
    Match the notebook/inference behavior: force plain Tensor attributes/buffers to CUDA.
    """
    for name, buf in module.named_buffers(recurse=False):
        if buf is not None and not buf.is_cuda:
            module._buffers[name] = buf.cuda()

    for attr_name in list(module.__dict__.keys()):
        attr = getattr(module, attr_name, None)
        if isinstance(attr, torch.Tensor) and not attr.is_cuda:
            setattr(module, attr_name, attr.cuda())

    for child in module.children():
        force_cuda(child)


def load_cfg(lightfc_dir: Path, config_name: str):
    candidates = sorted((lightfc_dir / "experiments" / "lightfc").glob(f"{config_name}*.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No config YAML found for {config_name} under "
            f"{lightfc_dir / 'experiments' / 'lightfc'}"
        )

    with open(candidates[0], "r") as f:
        raw = yaml.safe_load(f)

    return DotDict.from_nested(raw), candidates[0]


def load_bboxes(txt_path: str | Path) -> np.ndarray:
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 4:
                boxes.append([float(parts[i]) for i in range(4)])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)

    return np.asarray(boxes, dtype=np.float32)


# ---------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def img_to_tensor_fp16(crop_uint8: np.ndarray) -> torch.Tensor:
    t = crop_uint8.astype(np.float32) / 255.0
    t = (t - MEAN) / STD
    return torch.from_numpy(t.transpose(2, 0, 1)).half()


def jitter_bbox(bbox_xywh, center_jitter: float, scale_jitter: float):
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = x + w / 2.0
    cy = y + h / 2.0

    max_dim = max(w, h)

    cx += np.random.uniform(-center_jitter, center_jitter) * max_dim
    cy += np.random.uniform(-center_jitter, center_jitter) * max_dim

    w *= np.exp(np.random.uniform(-scale_jitter, scale_jitter))
    h *= np.exp(np.random.uniform(-scale_jitter, scale_jitter))

    w = max(w, 1.0)
    h = max(h, 1.0)

    return [cx - w / 2.0, cy - h / 2.0, w, h]


def build_sequence_list(data_root: Path, teacher_dir: Path):
    manifest_path = data_root / "metadata" / "contestant_manifest.json"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_manifest = manifest["train"]
    sequences = []

    for key, info in train_manifest.items():
        gt_path = data_root / info["annotation_path"]
        video_path = data_root / info["video_path"]
        teacher_path = teacher_dir / "train" / f"{key.replace('/', '_')}.txt"

        if not gt_path.exists() or not video_path.exists() or not teacher_path.exists():
            continue

        gt = load_bboxes(gt_path)
        teacher = load_bboxes(teacher_path)

        if len(gt) == 0 or len(teacher) == 0:
            continue

        n = min(len(gt), len(teacher), int(info["n_frames"]))

        if n < 10:
            continue

        sequences.append(
            {
                "key": key,
                "video_path": str(video_path),
                "gt": gt[:n],
                "teacher": teacher[:n],
                "n_frames": n,
            }
        )

    return sequences


def split_sequences(sequences, val_ratio: float, seed: int):
    """
    Match the original notebook exactly:
      np.random.seed(SEED)
      indices = np.random.permutation(...)
      val = sorted(indices[:n_val])
      train = sorted(indices[n_val:])
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(sequences))
    n_val = max(1, int(len(sequences) * val_ratio))

    val_seqs = [sequences[i] for i in sorted(indices[:n_val])]
    train_seqs = [sequences[i] for i in sorted(indices[n_val:])]

    return train_seqs, val_seqs


def presample_pairs(n_frames: int, gt: np.ndarray, n_pairs: int, max_gap: int):
    valid = np.where((gt[:, 2] > 0) & (gt[:, 3] > 0))[0]

    if len(valid) < 2:
        return [], set()

    pairs = []
    attempts = 0

    while len(pairs) < n_pairs and attempts < n_pairs * 5:
        attempts += 1

        t = int(np.random.choice(valid))
        lo = max(int(valid[0]), t - max_gap)
        hi = min(int(valid[-1]), t + max_gap)

        candidates = valid[(valid >= lo) & (valid <= hi) & (valid != t)]

        if len(candidates) == 0:
            continue

        s = int(np.random.choice(candidates))
        pairs.append((t, s))

    needed = set()
    for t, s in pairs:
        needed.add(t)
        needed.add(s)

    return pairs, needed


def read_needed_frames(video_path: str, needed: set[int]):
    if not needed:
        return {}

    frames = {}
    max_idx = max(needed)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    idx = 0
    while idx <= max_idx:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if idx in needed:
            frames[idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        idx += 1

    cap.release()
    return frames


def burn_notebook_sanity_rng(
    sequences,
    sample_target_fn,
    transform_image_to_crop_fn,
    *,
    search_sz: int,
    search_factor: float,
    search_center_jitter: float,
    search_scale_jitter: float,
):
    """
    The original notebook ran a sanity check before generate_pairs().
    That sanity check consumed NumPy random numbers through jitter_bbox().
    To reproduce the notebook pair sampling/jitter sequence, keep this RNG burn.
    """
    if not sequences:
        return

    test_seq = sequences[0]
    test_frames = read_needed_frames(test_seq["video_path"], {0, 1, 2, 3, 4})

    for fidx in [0, 1, 2, 3, 4]:
        if fidx not in test_frames:
            continue

        gt_bbox = test_seq["gt"][fidx].tolist()
        if gt_bbox[2] <= 0:
            continue

        for _ in range(5):
            jit = jitter_bbox(
                gt_bbox,
                center_jitter=search_center_jitter,
                scale_jitter=search_scale_jitter,
            )

            try:
                _, rf, _ = sample_target_fn(
                    test_frames[fidx],
                    jit,
                    search_factor,
                    output_sz=search_sz,
                )
            except Exception:
                continue

            crop_sz_t = torch.tensor([search_sz, search_sz], dtype=torch.float32)
            gt_t = torch.tensor(gt_bbox, dtype=torch.float32)
            jit_t = torch.tensor(jit, dtype=torch.float32)

            _ = transform_image_to_crop_fn(
                gt_t,
                jit_t,
                rf,
                crop_sz_t,
                normalize=True,
            )

    del test_frames


def generate_pairs(
    sequences,
    split_name: str,
    pairs_dir: Path,
    sample_target_fn,
    transform_image_to_crop_fn,
    *,
    template_sz: int,
    search_sz: int,
    template_factor: float,
    search_factor: float,
    pairs_per_seq: int,
    max_gap: int,
    chunk_size: int,
    search_center_jitter: float,
    search_scale_jitter: float,
):
    pairs_dir.mkdir(parents=True, exist_ok=True)

    all_t, all_s, all_g, all_o = [], [], [], []
    chunk_idx = 0
    total_pairs = 0
    t0 = time.time()

    for seq_i, seq in enumerate(sequences):
        gt = seq["gt"]
        teacher = seq["teacher"]

        pairs, needed = presample_pairs(seq["n_frames"], gt, pairs_per_seq, max_gap)

        if not pairs:
            continue

        frames = read_needed_frames(seq["video_path"], needed)

        if not frames:
            continue

        for t_idx, s_idx in pairs:
            if t_idx not in frames or s_idx not in frames:
                continue

            t_frame = frames[t_idx]
            s_frame = frames[s_idx]

            t_gt = gt[t_idx].tolist()
            s_gt = gt[s_idx].tolist()
            s_teacher = teacher[s_idx].tolist()

            if t_gt[2] <= 0 or t_gt[3] <= 0:
                continue
            if s_gt[2] <= 0 or s_gt[3] <= 0:
                continue
            if s_teacher[2] <= 0 or s_teacher[3] <= 0:
                continue

            try:
                t_crop, _, _ = sample_target_fn(
                    t_frame,
                    t_gt,
                    template_factor,
                    output_sz=template_sz,
                )
            except Exception:
                continue

            s_jittered = jitter_bbox(
                s_gt,
                center_jitter=search_center_jitter,
                scale_jitter=search_scale_jitter,
            )

            try:
                s_crop, s_rf, _ = sample_target_fn(
                    s_frame,
                    s_jittered,
                    search_factor,
                    output_sz=search_sz,
                )
            except Exception:
                continue

            crop_sz_tensor = torch.tensor([search_sz, search_sz], dtype=torch.float32)
            jitter_tensor = torch.tensor(s_jittered, dtype=torch.float32)

            gt_tensor = torch.tensor(s_gt, dtype=torch.float32)
            gt_crop = transform_image_to_crop_fn(
                gt_tensor,
                jitter_tensor,
                s_rf,
                crop_sz_tensor,
                normalize=True,
            )

            gt_x, gt_y, gt_w, gt_h = gt_crop.tolist()
            gt_label = [gt_x + gt_w / 2.0, gt_y + gt_h / 2.0, gt_w, gt_h]

            teacher_tensor = torch.tensor(s_teacher, dtype=torch.float32)
            teacher_crop = transform_image_to_crop_fn(
                teacher_tensor,
                jitter_tensor,
                s_rf,
                crop_sz_tensor,
                normalize=True,
            )

            od_x, od_y, od_w, od_h = teacher_crop.tolist()
            teacher_label = [od_x + od_w / 2.0, od_y + od_h / 2.0, od_w, od_h]

            # Keep moderately centered labels.
            if gt_label[2] <= 0 or gt_label[3] <= 0:
                continue
            if gt_label[0] < 0.05 or gt_label[0] > 0.95:
                continue
            if gt_label[1] < 0.05 or gt_label[1] > 0.95:
                continue

            all_t.append(img_to_tensor_fp16(t_crop))
            all_s.append(img_to_tensor_fp16(s_crop))
            all_g.append(torch.tensor(gt_label, dtype=torch.float16))
            all_o.append(torch.tensor(teacher_label, dtype=torch.float16))

            total_pairs += 1

            if len(all_t) >= chunk_size:
                out_path = pairs_dir / f"{split_name}_chunk{chunk_idx:04d}.pt"
                torch.save(
                    {
                        "templates": torch.stack(all_t),
                        "searches": torch.stack(all_s),
                        "gt_labels": torch.stack(all_g),
                        "teacher_labels": torch.stack(all_o),
                    },
                    out_path,
                )

                chunk_idx += 1
                all_t.clear()
                all_s.clear()
                all_g.clear()
                all_o.clear()

        del frames

        if (seq_i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (seq_i + 1) / max(elapsed, 1e-6)
            eta = (len(sequences) - seq_i - 1) / max(rate, 1e-6)

            print(
                f"[{split_name}] {seq_i + 1}/{len(sequences)} | "
                f"{total_pairs} pairs | {elapsed:.0f}s | ETA {eta:.0f}s"
            )

    if all_t:
        out_path = pairs_dir / f"{split_name}_chunk{chunk_idx:04d}.pt"
        torch.save(
            {
                "templates": torch.stack(all_t),
                "searches": torch.stack(all_s),
                "gt_labels": torch.stack(all_g),
                "teacher_labels": torch.stack(all_o),
            },
            out_path,
        )
        chunk_idx += 1

    elapsed = time.time() - t0
    print(f"[{split_name}] done: {total_pairs} pairs, {chunk_idx} chunks, {elapsed:.0f}s")

    return total_pairs


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class ChunkDataset(Dataset):
    def __init__(self, pairs_dir: Path, split_name: str):
        chunks = sorted(pairs_dir.glob(f"{split_name}_chunk*.pt"))

        if not chunks:
            raise FileNotFoundError(f"No chunks found for split={split_name} in {pairs_dir}")

        t_list, s_list, g_list, o_list = [], [], [], []

        for cf in chunks:
            d = safe_torch_load(cf, map_location="cpu")
            t_list.append(d["templates"])
            s_list.append(d["searches"])
            g_list.append(d["gt_labels"])

            # backward-compatible key support
            if "teacher_labels" in d:
                o_list.append(d["teacher_labels"])
            else:
                o_list.append(d["od_labels"])

        self.templates = torch.cat(t_list)
        self.searches = torch.cat(s_list)
        self.gt_labels = torch.cat(g_list)
        self.teacher_labels = torch.cat(o_list)

        print(f"[{split_name}] loaded {len(self)} pairs")

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, idx):
        return (
            self.templates[idx].float(),
            self.searches[idx].float(),
            self.gt_labels[idx].float(),
            self.teacher_labels[idx].float(),
        )


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

def giou_loss_cxcywh(pred, target):
    p_x1 = pred[:, 0] - pred[:, 2] / 2.0
    p_y1 = pred[:, 1] - pred[:, 3] / 2.0
    p_x2 = pred[:, 0] + pred[:, 2] / 2.0
    p_y2 = pred[:, 1] + pred[:, 3] / 2.0

    t_x1 = target[:, 0] - target[:, 2] / 2.0
    t_y1 = target[:, 1] - target[:, 3] / 2.0
    t_x2 = target[:, 0] + target[:, 2] / 2.0
    t_y2 = target[:, 1] + target[:, 3] / 2.0

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    p_area = (p_x2 - p_x1).clamp(0) * (p_y2 - p_y1).clamp(0)
    t_area = (t_x2 - t_x1).clamp(0) * (t_y2 - t_y1).clamp(0)

    union = p_area + t_area - inter + 1e-7
    iou = inter / union

    enc_x1 = torch.min(p_x1, t_x1)
    enc_y1 = torch.min(p_y1, t_y1)
    enc_x2 = torch.max(p_x2, t_x2)
    enc_y2 = torch.max(p_y2, t_y2)

    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0) + 1e-7

    giou = iou - (enc_area - union) / enc_area

    return (1.0 - giou).mean(), iou.detach().mean()


def compute_distill_loss(pred, gt, teacher, alpha: float, l1_weight: float):
    giou_l, mean_iou = giou_loss_cxcywh(pred, gt)
    l1_l = F.l1_loss(pred, gt)
    gt_loss = giou_l + l1_weight * l1_l

    teacher_loss = F.smooth_l1_loss(pred, teacher)

    total = alpha * gt_loss + (1.0 - alpha) * teacher_loss

    return total, gt_loss.detach(), teacher_loss.detach(), mean_iou.detach()


def get_pred(output):
    pred = output["pred_boxes"]

    if pred.dim() == 3:
        pred = pred[:, 0, :]

    return pred


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def build_model(cfg, init_ckpt: Path):
    from lib.models.tracker_model import LightFC

    model = LightFC(cfg, env_num=0, training=False)

    ckpt = safe_torch_load(init_ckpt, map_location="cpu")
    state = ckpt.get("net", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded init checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    has_dense = any("rbr_dense" in k for k in model.state_dict().keys())
    has_reparam = any("rbr_reparam" in k for k in model.state_dict().keys())

    if not has_dense or has_reparam:
        raise RuntimeError(
            f"Checkpoint/model appears to be deploy-mode. "
            f"Training needs pre-deploy RepVGG branches. "
            f"rbr_dense={has_dense}, rbr_reparam={has_reparam}"
        )

    return model


def freeze_for_stage1(model):
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"Trainable params: {trainable / 1e6:.2f}M")
    print(f"Frozen params:    {frozen / 1e6:.2f}M")


def train_one_epoch(model, loader, optimizer, scaler, alpha, l1_weight, grad_clip):
    model.train()

    # Frozen modules should keep BN/stat behavior stable.
    model.backbone.eval()
    model.fusion.eval()

    loss_sum = 0.0
    iou_sum = 0.0
    n = 0

    for templates, searches, gt_labels, teacher_labels in loader:
        templates = templates.cuda(non_blocking=True)
        searches = searches.cuda(non_blocking=True)
        gt_labels = gt_labels.cuda(non_blocking=True)
        teacher_labels = teacher_labels.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_autocast():
            output = model(templates, searches)
            pred = get_pred(output)
            loss, _, _, iou = compute_distill_loss(
                pred,
                gt_labels,
                teacher_labels,
                alpha=alpha,
                l1_weight=l1_weight,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip,
        )

        scaler.step(optimizer)
        scaler.update()

        loss_sum += float(loss.detach().cpu())
        iou_sum += float(iou.cpu())
        n += 1

    return loss_sum / max(n, 1), iou_sum / max(n, 1)


@torch.no_grad()
def validate(model, loader, alpha, l1_weight):
    # Keep training forward path, but no updates.
    model.train()
    model.backbone.eval()
    model.fusion.eval()
    model.head.eval()

    loss_sum = 0.0
    iou_sum = 0.0
    n = 0

    for templates, searches, gt_labels, teacher_labels in loader:
        templates = templates.cuda(non_blocking=True)
        searches = searches.cuda(non_blocking=True)
        gt_labels = gt_labels.cuda(non_blocking=True)
        teacher_labels = teacher_labels.cuda(non_blocking=True)

        with amp_autocast():
            output = model(templates, searches)
            pred = get_pred(output)
            loss, _, _, iou = compute_distill_loss(
                pred,
                gt_labels,
                teacher_labels,
                alpha=alpha,
                l1_weight=l1_weight,
            )

        loss_sum += float(loss.cpu())
        iou_sum += float(iou.cpu())
        n += 1

    return loss_sum / max(n, 1), iou_sum / max(n, 1)


def make_scheduler(optimizer, epochs, lr, min_lr, warmup):
    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / float(max(warmup, 1))

        progress = (epoch - warmup) / max(1, epochs - warmup)
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_distillation(args, cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("Distillation training requires CUDA.")

    train_dataset = ChunkDataset(args.pairs_dir, "train")
    val_dataset = ChunkDataset(args.pairs_dir, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(cfg, args.init_ckpt)
    freeze_for_stage1(model)

    model.cuda()
    force_cuda(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = make_scheduler(
        optimizer,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup=args.warmup,
    )

    scaler = make_grad_scaler()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0
    best_epoch = -1

    print("=" * 72)
    print("STAGE 1 DISTILLATION")
    print(f"epochs={args.epochs}, lr={args.lr}, alpha={args.alpha}, l1_weight={args.l1_weight}")
    print(f"batches: train={len(train_loader)}, val={len(val_loader)}")
    print("=" * 72)

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_iou = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            alpha=args.alpha,
            l1_weight=args.l1_weight,
            grad_clip=args.grad_clip,
        )

        val_loss, val_iou = validate(
            model,
            val_loader,
            alpha=args.alpha,
            l1_weight=args.l1_weight,
        )

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        improved = val_iou > best_val_iou

        print(
            f"Ep {epoch + 1:02d}/{args.epochs} | "
            f"Train loss={train_loss:.4f} IoU={train_iou:.4f} | "
            f"Val loss={val_loss:.4f} IoU={val_iou:.4f} | "
            f"LR={lr_now:.2e} | {elapsed:.1f}s"
            f"{' *' if improved else ''}"
        )

        if improved:
            best_val_iou = val_iou
            best_epoch = epoch + 1

            ckpt_path = args.save_dir / "stage1_best.pth.tar"

            torch.save(
                {
                    "epoch": best_epoch,
                    "net": model.state_dict(),
                    "val_iou": best_val_iou,
                    "training_args": vars(args),
                },
                ckpt_path,
            )

    print(f"Best epoch: {best_epoch}")
    print(f"Best val IoU: {best_val_iou:.10f}")
    print(f"Saved to: {args.save_dir / 'stage1_best.pth.tar'}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    root = repo_root()

    p = argparse.ArgumentParser("MTC-AIC4 LightFC distillation")

    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--lightfc-dir", type=Path, default=root / "third_party" / "LightFC")
    p.add_argument("--config-name", type=str, default="baseline_v1_release_backbone_tinyvit")
    p.add_argument("--init-ckpt", type=Path, required=True)

    p.add_argument("--work-dir", type=Path, default=root / "outputs" / "train_distill")
    p.add_argument("--pairs-dir", type=Path, default=None)
    p.add_argument("--save-dir", type=Path, default=None)

    p.add_argument("--generate", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--clean-pairs", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.10)

    p.add_argument("--template-size", type=int, default=128)
    p.add_argument("--search-size", type=int, default=256)
    p.add_argument("--template-factor", type=float, default=2.0)
    p.add_argument("--search-factor", type=float, default=4.0)

    p.add_argument("--pairs-per-seq", type=int, default=80)
    p.add_argument("--max-gap", type=int, default=100)
    p.add_argument("--chunk-size", type=int, default=500)

    p.add_argument("--search-center-jitter", type=float, default=0.5)
    p.add_argument("--search-scale-jitter", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--l1-weight", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    args = p.parse_args()

    if args.pairs_dir is None:
        args.pairs_dir = args.work_dir / "train_pairs"

    if args.save_dir is None:
        args.save_dir = args.work_dir / "checkpoints"

    if not args.generate and not args.train:
        args.generate = True
        args.train = True

    return args


def main():
    args = parse_args()

    seed_everything(args.seed)

    add_lightfc_to_path(args.lightfc_dir)
    patch_lightfc_runtime()

    cfg, cfg_path = load_cfg(args.lightfc_dir, args.config_name)

    print(f"LightFC dir: {args.lightfc_dir}")
    print(f"Config:      {cfg_path}")
    print(f"Data root:   {args.data_root}")
    print(f"Teacher dir: {args.teacher_dir}")
    print(f"Init ckpt:   {args.init_ckpt}")
    print(f"Work dir:    {args.work_dir}")

    from lib.train.data.processing_utils import sample_target
    from lib.train.data.processing_utils import transform_image_to_crop

    if args.generate:
        if args.clean_pairs and args.pairs_dir.exists():
            shutil.rmtree(args.pairs_dir)

        args.pairs_dir.mkdir(parents=True, exist_ok=True)

        sequences = build_sequence_list(args.data_root, args.teacher_dir)
        print(f"Sequences with GT + teacher predictions: {len(sequences)}")

        train_seqs, val_seqs = split_sequences(sequences, args.val_ratio, args.seed)
        print(f"Split: train={len(train_seqs)}, val={len(val_seqs)}")

        # Match original notebook RNG state before generate_pairs().
        burn_notebook_sanity_rng(
            sequences,
            sample_target,
            transform_image_to_crop,
            search_sz=args.search_size,
            search_factor=args.search_factor,
            search_center_jitter=args.search_center_jitter,
            search_scale_jitter=args.search_scale_jitter,
        )

        n_train = generate_pairs(
            train_seqs,
            "train",
            args.pairs_dir,
            sample_target,
            transform_image_to_crop,
            template_sz=args.template_size,
            search_sz=args.search_size,
            template_factor=args.template_factor,
            search_factor=args.search_factor,
            pairs_per_seq=args.pairs_per_seq,
            max_gap=args.max_gap,
            chunk_size=args.chunk_size,
            search_center_jitter=args.search_center_jitter,
            search_scale_jitter=args.search_scale_jitter,
        )

        n_val = generate_pairs(
            val_seqs,
            "val",
            args.pairs_dir,
            sample_target,
            transform_image_to_crop,
            template_sz=args.template_size,
            search_sz=args.search_size,
            template_factor=args.template_factor,
            search_factor=args.search_factor,
            pairs_per_seq=args.pairs_per_seq,
            max_gap=args.max_gap,
            chunk_size=args.chunk_size,
            search_center_jitter=args.search_center_jitter,
            search_scale_jitter=args.search_scale_jitter,
        )

        disk_mb = sum(p.stat().st_size for p in args.pairs_dir.glob("*.pt")) / 1e6

        print(f"Pair generation complete: train={n_train}, val={n_val}, disk={disk_mb:.1f} MB")

    gc.collect()

    if args.train:
        train_distillation(args, cfg)


if __name__ == "__main__":
    main()