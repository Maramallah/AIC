import os
import sys
import csv
import json
import time
import glob
import types
import argparse

import cv2
import yaml
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="MTC-AIC4 LightFC v6-conservative exact inference"
    )

    parser.add_argument(
        "--repo-dir",
        type=str,
        default="third_party/LightFC",
        help="Path to LightFC repo/vendor directory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to contest-tracking-data",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_checkpoint.pth.tar",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/submissions/submission.csv",
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="public_lb",
        help="Manifest split to run, usually public_lb",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=0,
        help="Debug only: stop after N sequences. 0 means all sequences.",
    )
    parser.add_argument(
        "--copy-to-kaggle-submission",
        action="store_true",
        help="Also copy output CSV to /kaggle/working/submission.csv",
    )

    return parser.parse_args()


def setup_lightfc_repo(repo_dir):
    repo_dir = os.path.abspath(repo_dir)

    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"LightFC repo not found: {repo_dir}")

    # Match original notebook behavior.
    os.chdir(repo_dir)

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # torch._six shim for older LightFC code.
    if "torch._six" not in sys.modules:
        _six = types.ModuleType("torch._six")
        _six.string_classes = (str, bytes)
        sys.modules["torch._six"] = _six

    return repo_dir


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def from_nested(d):
        if isinstance(d, dict):
            return DotDict({k: DotDict.from_nested(v) for k, v in d.items()})
        if isinstance(d, list):
            return [DotDict.from_nested(x) for x in d]
        return d


def load_config(repo_dir):
    config_name = "baseline_v1_release_backbone_tinyvit"

    yaml_paths = glob.glob(
        os.path.join(
            repo_dir,
            "experiments",
            "lightfc",
            f"{config_name}*.yaml",
        )
    )

    if not yaml_paths:
        raise FileNotFoundError(
            f"No config YAML found for {config_name} under "
            f"{repo_dir}/experiments/lightfc"
        )

    yaml_path = yaml_paths[0]

    with open(yaml_path) as f:
        raw_cfg = yaml.safe_load(f)

    cfg = DotDict.from_nested(raw_cfg)

    print(f"✅ Config loaded: {yaml_path}")
    return cfg


def force_cuda(module):
    """
    Original notebook helper.

    Some LightFC/TinyViT modules keep tensors as plain attributes rather than
    registered parameters/buffers. model.cuda() does not always move those.
    This recursively moves such tensor attributes to CUDA.
    """
    for name, buf in module.named_buffers(recurse=False):
        if buf is not None and torch.is_tensor(buf) and not buf.is_cuda:
            module._buffers[name] = buf.cuda()

    for attr_name in list(module.__dict__.keys()):
        attr = getattr(module, attr_name, None)
        if isinstance(attr, torch.Tensor) and not attr.is_cuda:
            setattr(module, attr_name, attr.cuda())

    for child in module.children():
        force_cuda(child)


def load_model_v6_exact(cfg, checkpoint):
    """
    Match original submitted v6-conservative load/deploy path.

    Important:
    - Do NOT delete `ab`.
    - Do NOT call model.train() before switch_to_deploy().
    - Do NOT use model.to(DEVICE).
    - Do use model_inf.eval().cuda().
    - Do use force_cuda(model_inf).
    """
    from lib.models.tracker_model import LightFC

    model_inf = LightFC(cfg, env_num=0, training=False)

    print(f"✅ Loading checkpoint: {checkpoint}")
    s1_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if "net" not in s1_ckpt:
        raise KeyError(f"Checkpoint has no 'net' key. Keys: {list(s1_ckpt.keys())}")

    model_inf.load_state_dict(s1_ckpt["net"], strict=True)

    epoch = s1_ckpt.get("epoch", "?")
    val_iou = s1_ckpt.get("val_iou", s1_ckpt.get("val_IoU", None))

    if val_iou is not None:
        print(f"✅ Loaded (epoch {epoch}, val_IoU={val_iou:.4f})")
    else:
        print(f"✅ Loaded (epoch {epoch})")

    for m in model_inf.modules():
        if hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()

    model_inf.eval().cuda()
    force_cuda(model_inf)

    print("✅ Deployed")

    return model_inf


def cal_bbox(score_map_ctr, size_map, offset_map, return_score=False):
    max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)

    idx_y = idx // score_map_ctr.shape[-1]
    idx_x = idx % score_map_ctr.shape[-1]

    idx_expanded = idx.unsqueeze(1).expand(-1, size_map.size(1), -1)

    s = size_map.flatten(2).gather(2, idx_expanded).squeeze(-1)
    o = offset_map.flatten(2).gather(2, idx_expanded).squeeze(-1)

    cx = (idx_x.float() + o[:, 0:1]) / score_map_ctr.shape[-1]
    cy = (idx_y.float() + o[:, 1:2]) / score_map_ctr.shape[-1]

    w = s[:, 0:1]
    h = s[:, 1:2]

    bbox = torch.cat([cx, cy, w, h], dim=1)

    if return_score:
        return bbox, max_score.squeeze()

    return bbox


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter

    return inter / union if union > 0 else 0.0


def ema_update(z_old, z_new, alpha):
    if isinstance(z_old, dict):
        return {k: (1 - alpha) * z_old[k] + alpha * z_new[k] for k in z_old}
    return (1 - alpha) * z_old + alpha * z_new


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for reproducing the original submitted "
            "v6-conservative run."
        )

    repo_dir = setup_lightfc_repo(args.repo_dir)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    print(f"Using device: {device}")

    # Important:
    # Do NOT set v8 runtime flags here while reproducing original v6.
    # Original v6 used local torch.no_grad() contexts.

    cfg = load_config(repo_dir)
    model_inf = load_model_v6_exact(cfg, args.checkpoint)

    from lib.test.tracker.data_utils import Preprocessor
    from lib.train.data.processing_utils import sample_target as repo_sample_target
    from lib.test.utils.hann import hann2d
    from lib.utils.box_ops import clip_box

    preprocessor = Preprocessor()
    hann_window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()

    TEMPLATE_SZ = 128
    SEARCH_SZ = 256
    TEMPLATE_F = 2.0
    SEARCH_F = 4.0

    def preprocess(crop, mask):
        out = preprocessor.process(crop, mask)
        if hasattr(out, "tensors"):
            return out.tensors.cuda()
        return out.cuda()

    def decode_bbox(pred_bbox_tensor, s_rf, prev_bbox, H, W):
        pred = pred_bbox_tensor.squeeze().cpu().numpy()
        pred_img = pred * SEARCH_SZ / s_rf

        prev_x, prev_y, prev_w, prev_h = prev_bbox
        prev_cx = prev_x + prev_w / 2
        prev_cy = prev_y + prev_h / 2

        half_side = SEARCH_SZ / s_rf / 2

        img_cx = pred_img[0] - half_side + prev_cx
        img_cy = pred_img[1] - half_side + prev_cy
        img_w = pred_img[2]
        img_h = pred_img[3]

        img_x = img_cx - img_w / 2
        img_y = img_cy - img_h / 2

        bbox = clip_box(
            [float(img_x), float(img_y), float(img_w), float(img_h)],
            H,
            W,
            margin=2,
        )

        if isinstance(bbox, (torch.Tensor, np.ndarray)):
            bbox = [float(bbox[i]) for i in range(4)]
        elif not isinstance(bbox, list):
            bbox = list(bbox)

        return bbox

    # v6-conservative config
    USE_EMA = True
    EMA_ALPHA = 0.002
    EMA_SCORE_THRESH = 0.30
    EMA_AREA_RATIO_MIN = 0.25
    EMA_AREA_RATIO_MAX = 4.0
    EMA_IOU_THRESH = 0.4
    EMA_UPDATE_EVERY = 3
    ANCHOR_WEIGHT = 0.15
    USE_MULTISCALE = False
    SCALES = [1.0]
    LOST_THRESH = 0.15
    LOST_COUNT_EXPAND = 5
    LOST_SEARCH_F = 6.0

    print(f"✅ Config: EMA={USE_EMA} α={EMA_ALPHA} anchor={ANCHOR_WEIGHT}")
    print(f"   MultiScale={USE_MULTISCALE} {SCALES}")
    print(f"   LostRecovery: thresh={LOST_THRESH} expand_after={LOST_COUNT_EXPAND}")

    manifest_path = os.path.join(
        args.data_root,
        "metadata",
        "contestant_manifest.json",
    )

    with open(manifest_path) as f:
        manifest = json.load(f)

    if args.split not in manifest:
        raise KeyError(
            f"Split '{args.split}' not found in manifest. "
            f"Keys: {list(manifest.keys())}"
        )

    test_manifest = manifest[args.split]

    results = {}
    total_frames = 0

    ema_update_count = 0
    ema_skip_score = 0
    ema_skip_area = 0
    ema_skip_iou = 0
    ema_skip_interval = 0
    lost_recovery_count = 0

    t0 = time.time()

    for seq_idx, (key, info) in enumerate(test_manifest.items()):
        if args.max_seqs and seq_idx >= args.max_seqs:
            print(f"⚠️ Stopping early because --max-seqs={args.max_seqs}")
            break

        video_path = os.path.join(args.data_root, info["video_path"])
        ann_path = os.path.join(args.data_root, info["annotation_path"])
        n_frames = info["n_frames"]

        with open(ann_path) as f:
            first_line = f.readline().strip()

        parts = first_line.replace(",", " ").split()
        init_bbox = [float(parts[i]) for i in range(4)]

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"⚠️ Could not open video: {video_path}")
            continue

        seq_ema_alpha = 0.005 if n_frames < 200 else EMA_ALPHA

        seq_preds = []
        state = {"bbox": list(init_bbox), "lost_count": 0}

        z = None
        z0 = None

        ema_area = init_bbox[2] * init_bbox[3]
        prev_bbox_for_iou = list(init_bbox)
        frames_since_ema = 0

        for frame_idx in range(n_frames):
            ret, frame = cap.read()

            if not ret:
                seq_preds.append([0, 0, 0, 0])
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = frame_rgb.shape[:2]

            if frame_idx == 0:
                t_crop, t_rf, t_mask = repo_sample_target(
                    frame_rgb,
                    init_bbox,
                    TEMPLATE_F,
                    output_sz=TEMPLATE_SZ,
                )

                t_tensor = preprocess(t_crop, t_mask)

                with torch.no_grad():
                    z = model_inf.forward_backbone(t_tensor)

                z0 = z.clone()

                seq_preds.append(init_bbox)

            else:
                current_search_f = SEARCH_F

                if state["lost_count"] > LOST_COUNT_EXPAND:
                    current_search_f = LOST_SEARCH_F
                    lost_recovery_count += 1

                z_track = (1 - ANCHOR_WEIGHT) * z + ANCHOR_WEIGHT * z0

                best_score = float("-inf")
                best_bbox = None
                best_peak = 0.0

                scales = SCALES if USE_MULTISCALE else [1.0]

                for scale in scales:
                    try:
                        s_crop, s_rf, s_mask = repo_sample_target(
                            frame_rgb,
                            state["bbox"],
                            current_search_f * scale,
                            output_sz=SEARCH_SZ,
                        )
                    except Exception:
                        continue

                    s_tensor = preprocess(s_crop, s_mask)

                    with torch.no_grad():
                        output = model_inf.forward_tracking(z_track, s_tensor)

                    response = hann_window * output["score_map"]

                    pred_bbox, peak_score = cal_bbox(
                        response,
                        output["size_map"],
                        output["offset_map"],
                        return_score=True,
                    )

                    pk = peak_score.item()

                    if pk > best_score:
                        best_score = pk
                        best_peak = pk
                        best_bbox = decode_bbox(
                            pred_bbox,
                            s_rf,
                            state["bbox"],
                            H,
                            W,
                        )

                if best_bbox is None:
                    best_bbox = list(state["bbox"])
                    best_peak = 0.0

                bbox = best_bbox

                prev_bbox_for_iou = list(state["bbox"])
                state["bbox"] = bbox

                seq_preds.append(bbox)
                frames_since_ema += 1

                if best_peak < LOST_THRESH:
                    state["lost_count"] = state.get("lost_count", 0) + 1
                else:
                    state["lost_count"] = 0

                if USE_EMA:
                    if best_peak <= EMA_SCORE_THRESH:
                        ema_skip_score += 1

                    elif frames_since_ema < EMA_UPDATE_EVERY:
                        ema_skip_interval += 1

                    else:
                        cur_area = bbox[2] * bbox[3]
                        area_ratio = cur_area / max(ema_area, 1e-6)

                        if not (EMA_AREA_RATIO_MIN < area_ratio < EMA_AREA_RATIO_MAX):
                            ema_skip_area += 1

                        elif compute_iou(prev_bbox_for_iou, bbox) <= EMA_IOU_THRESH:
                            ema_skip_iou += 1

                        else:
                            try:
                                new_t_crop, new_t_rf, new_t_mask = repo_sample_target(
                                    frame_rgb,
                                    bbox,
                                    TEMPLATE_F,
                                    output_sz=TEMPLATE_SZ,
                                )

                                new_t_tensor = preprocess(new_t_crop, new_t_mask)

                                with torch.no_grad():
                                    z_new = model_inf.forward_backbone(new_t_tensor)

                                z = ema_update(z, z_new, seq_ema_alpha)

                                ema_update_count += 1
                                frames_since_ema = 0
                                ema_area = 0.9 * ema_area + 0.1 * cur_area

                            except Exception:
                                pass

        cap.release()

        results[key] = seq_preds
        total_frames += len(seq_preds)

        if (seq_idx + 1) % 15 == 0:
            elapsed = time.time() - t0
            fps = total_frames / elapsed if elapsed > 0 else 0.0

            print(
                f"  {seq_idx + 1}/{len(test_manifest)} | "
                f"{total_frames} frames | {fps:.0f} FPS | "
                f"EMA: {ema_update_count} | Lost: {lost_recovery_count}"
            )

    elapsed = time.time() - t0
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"✅ Done: {total_frames} frames in {elapsed:.0f}s ({fps:.0f} FPS)")
    print(f"   EMA updates:       {ema_update_count}")
    print(f"   EMA skip (score):  {ema_skip_score}")
    print(f"   EMA skip (interval): {ema_skip_interval}")
    print(f"   EMA skip (area):   {ema_skip_area}")
    print(f"   EMA skip (IoU):    {ema_skip_iou}")
    print(f"   Lost recoveries:   {lost_recovery_count}")
    print(f"{'=' * 60}")

    print("\n🔎 Health check against original submitted v6-conservative:")
    print("   Expected: EMA=21517   Lost=133")
    print(f"   This run: EMA={ema_update_count}   Lost={lost_recovery_count}")

    expected_match = (
        ema_update_count == 21517
        and lost_recovery_count == 133
    )

    if expected_match:
        print("   ✅ Exact original submitted run matched")
    else:
        print("   ❌ Does NOT match original submitted run")

    sample_sub_path = os.path.join(
        args.data_root,
        "metadata",
        "sample_submission.csv",
    )

    with open(sample_sub_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        all_rows = [row for row in reader]

    sample_ids = [r[0] for r in all_rows]

    output_dir = os.path.dirname(args.output_csv)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    matched = 0

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row_id in sample_ids:
            seq, frame_str = row_id.rsplit("_", 1)
            frame_idx = int(frame_str)

            preds = results.get(seq, [])

            if 0 <= frame_idx < len(preds):
                x, y, w, h = preds[frame_idx]
                writer.writerow(
                    [
                        row_id,
                        f"{x:.2f}",
                        f"{y:.2f}",
                        f"{w:.2f}",
                        f"{h:.2f}",
                    ]
                )
                matched += 1

            elif preds:
                x, y, w, h = preds[-1]
                writer.writerow(
                    [
                        row_id,
                        f"{x:.2f}",
                        f"{y:.2f}",
                        f"{w:.2f}",
                        f"{h:.2f}",
                    ]
                )
                matched += 1

            else:
                writer.writerow(
                    [
                        row_id,
                        "0.00",
                        "0.00",
                        "0.00",
                        "0.00",
                    ]
                )

    print(f"\n✅ Submission: {args.output_csv}")
    print(f"   Matched: {matched}/{len(sample_ids)}")

    if args.copy_to_kaggle_submission:
        import shutil

        kaggle_sub = "/kaggle/working/submission.csv"
        os.makedirs(os.path.dirname(kaggle_sub), exist_ok=True)
        shutil.copy2(args.output_csv, kaggle_sub)
        print(f"✅ Copied to {kaggle_sub}")


if __name__ == "__main__":
    main()