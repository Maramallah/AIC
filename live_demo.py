'''
import os, sys, time, argparse, cv2, torch
import numpy as np
import onnxruntime as ort
from src.mtcaic4.infer import setup_lightfc_repo, load_config
from third_party.LightFC.lib.test.tracker.data_utils import Preprocessor
from third_party.LightFC.lib.train.data.processing_utils import sample_target as repo_sample_target
from third_party.LightFC.lib.test.utils.hann import hann2d
from third_party.LightFC.lib.utils.box_ops import clip_box
'''


import sys
import os
import types 

# ── 1d: Patch compatibility issues ───────────────────────────
# torch._six shim
if "torch._six" not in sys.modules:
    _six = types.ModuleType("torch._six")
    _six.string_classes = (str, bytes)
    sys.modules["torch._six"] = _six


# ============ FIX PATHS ============
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Add LightFC to path
lightfc_path = os.path.join(project_root, 'third_party', 'LightFC')
sys.path.insert(0, lightfc_path)

# ============ IMPORTS ============
import time
import argparse
import cv2
import torch
import numpy as np
import onnxruntime as ort

# Import from src (your code)
from src.mtcaic4.infer import setup_lightfc_repo, load_config

# Import from LightFC (third party)
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target as repo_sample_target
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import clip_box


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Orin Nano - ONNX v6-Exact Tracker")
    parser.add_argument("--repo-dir", type=str, default="third_party/LightFC")
    parser.add_argument("--onnx-dir", type=str, default="checkpoints/onnx_models")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()

def compute_iou(box1, box2):
    """Pure Python IoU calculation matching the v6 script."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0: return 0.0
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

def decode_bbox(pred_bbox_tensor, s_rf, prev_bbox, H, W, SEARCH_SZ):
    pred = pred_bbox_tensor.squeeze().cpu().numpy() if torch.is_tensor(pred_bbox_tensor) else pred_bbox_tensor.squeeze()
    pred_img = pred * SEARCH_SZ / s_rf
    prev_x, prev_y, prev_w, prev_h = prev_bbox
    prev_cx = prev_x + prev_w / 2
    prev_cy = prev_y + prev_h / 2
    half_side = SEARCH_SZ / s_rf / 2
    img_x = (pred_img[0] - half_side + prev_cx) - pred_img[2] / 2
    img_y = (pred_img[1] - half_side + prev_cy) - pred_img[3] / 2
    bbox = clip_box([float(img_x), float(img_y), float(pred_img[2]), float(pred_img[3])], H, W, margin=2)
    return [float(bbox[i]) for i in range(4)] if isinstance(bbox, (torch.Tensor, np.ndarray)) else list(bbox)

def main():
    args = parse_args()
    repo_dir = setup_lightfc_repo(args.repo_dir)

    #ADDING PATHS MANUALLY 

    backbone_path = 'D:/aic final trial1/MTC-AIC4-EagleAI/checkpoints/onnx_models/lightfc_backbone.onnx'
    network_path = 'D:/aic final trial1/MTC-AIC4-EagleAI/checkpoints/onnx_models/lightfc_network.onnx'


    print("🚀 Loading ONNX Runtime Sessions (TensorRT -> CUDA -> CPU)...")
    #for laptop usage
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    #for jetson usage 
    #providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    #backbone_session = ort.InferenceSession(os.path.join(args.onnx_dir, "lightfc_backbone.onnx"), providers=providers)
    backbone_session = ort.InferenceSession(backbone_path, providers=providers)
    network_session = ort.InferenceSession(network_path, providers=providers)


    preprocessor = Preprocessor()
    hann_window = hann2d(torch.tensor([16, 16]).long(), centered=True).numpy() 

    def preprocess(crop, mask):
        out = preprocessor.process(crop, mask)
        return out.tensors.cpu().numpy() if hasattr(out, "tensors") else out.cpu().numpy()

    # Base Constants
    TEMPLATE_SZ, SEARCH_SZ, TEMPLATE_F, SEARCH_F = 128, 256, 2.0, 4.0
    
    # v6-Conservative Constants
    EMA_ALPHA = 0.002
    EMA_SCORE_THRESH = 0.30
    EMA_AREA_RATIO_MIN = 0.25
    EMA_AREA_RATIO_MAX = 4.0
    EMA_IOU_THRESH = 0.4
    EMA_UPDATE_EVERY = 3
    ANCHOR_WEIGHT = 0.15
    LOST_THRESH = 0.15
    LOST_COUNT_EXPAND = 5
    LOST_SEARCH_F = 6.0

    cap = cv2.VideoCapture(args.camera)
    ret, frame = cap.read()
    
    roi = cv2.selectROI('Jetson Target Selection', frame, False)
    cv2.destroyWindow('Jetson Target Selection')
    init_bbox = [float(v) for v in roi]
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]

    # Initialize First Frame
    t_crop, _, t_mask = repo_sample_target(frame_rgb, init_bbox, TEMPLATE_F, output_sz=TEMPLATE_SZ)
    np_template = preprocess(t_crop, t_mask)
    
    print("⚙️ Warming up Engine (Wait 1-2 mins if TensorRT is compiling)...")
    ort_z0 = backbone_session.run(None, {'template': np_template})[0]

    # v6 Tracking State Initialization
    z = ort_z0.copy()  # Current dynamic template
    z0 = ort_z0.copy() # Initial anchor template
    state = {"bbox": list(init_bbox), "lost_count": 0}
    ema_area = init_bbox[2] * init_bbox[3]
    prev_bbox_for_iou = list(init_bbox)
    frames_since_ema = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Lost Recovery Expansion Logic
        current_search_f = SEARCH_F
        if state["lost_count"] > LOST_COUNT_EXPAND:
            current_search_f = LOST_SEARCH_F

        # 2. Anchor Weighting (NumPy operation)
        z_track = (1 - ANCHOR_WEIGHT) * z + ANCHOR_WEIGHT * z0

        try:
            s_crop, s_rf, s_mask = repo_sample_target(frame_rgb, state["bbox"], current_search_f, output_sz=SEARCH_SZ)
            np_search = preprocess(s_crop, s_mask)

            # Pass the dynamically anchored template (z_track) instead of the static one
            ort_out_list = network_session.run(None, {'template_features': z_track, 'search_region': np_search})
            
            ort_score, ort_size, ort_offset = ort_out_list[0], ort_out_list[1], ort_out_list[2]

            response = hann_window * ort_score
            max_score_idx = np.argmax(response.flatten())
            pk = response.flatten()[max_score_idx]
            
            idx_y = max_score_idx // response.shape[-1]
            idx_x = max_score_idx % response.shape[-1]
            
            s = ort_size[0, :, idx_y, idx_x]
            o = ort_offset[0, :, idx_y, idx_x]
            
            cx = (idx_x + o[0]) / response.shape[-1]
            cy = (idx_y + o[1]) / response.shape[-1]
            pred_bbox = np.array([cx, cy, s[0], s[1]])

            best_bbox = decode_bbox(pred_bbox, s_rf, state["bbox"], H, W, SEARCH_SZ)
        except Exception as e:
            best_bbox, pk = list(state["bbox"]), 0.0

        state["bbox"] = best_bbox

        # 3. Update Lost Tracker Count
        if pk < LOST_THRESH:
            state["lost_count"] += 1
        else:
            state["lost_count"] = 0

        # 4. EMA Update Logic
        if pk > EMA_SCORE_THRESH and frames_since_ema >= EMA_UPDATE_EVERY:
            cur_area = best_bbox[2] * best_bbox[3]
            area_ratio = cur_area / max(ema_area, 1e-6)

            if (EMA_AREA_RATIO_MIN < area_ratio < EMA_AREA_RATIO_MAX) and (compute_iou(prev_bbox_for_iou, best_bbox) > EMA_IOU_THRESH):
                try:
                    new_t_crop, _, new_t_mask = repo_sample_target(frame_rgb, best_bbox, TEMPLATE_F, output_sz=TEMPLATE_SZ)
                    new_t_tensor = preprocess(new_t_crop, new_t_mask)

                    ort_z_new = backbone_session.run(None, {'template': new_t_tensor})[0]

                    # EMA Update on NumPy Arrays
                    z = (1 - EMA_ALPHA) * z + EMA_ALPHA * ort_z_new
                    frames_since_ema = 0
                    ema_area = 0.9 * ema_area + 0.1 * cur_area
                except Exception:
                    pass

        frames_since_ema += 1
        prev_bbox_for_iou = list(best_bbox)

        # Rendering
        status_text, color = ("TRACKING", (0, 255, 0)) if pk > 0.15 else ("LOST", (0, 0, 255))
        x, y, w, h = map(int, best_bbox)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"Jetson FPS: {fps:.1f} | Score: {pk:.2f} | {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('MTC-AIC4 Jetson Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()