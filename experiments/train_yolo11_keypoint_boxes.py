import os
import glob
import math
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET
import re

import cv2
import numpy as np
import ast
from datetime import datetime
from ultralytics import YOLO
import sys as _sys
# Ensure project root is on PYTHONPATH for 'src' imports
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from src.config_loader import load_config
from build_keypoint_boxes_dataset import build_detection_dataset


# Load config
CFG = load_config()

# Paths (from config)
# If DATASET_ROOT contains a CVAT for images export (annotations.xml + images),
# the script will parse CVAT XML; otherwise it will use prior label formats.
DATASET_ROOT = CFG["paths"].get("cvat_dataset_root", "data/Curling_keypoints_data")
OUTPUT_DATASET_ROOT = CFG["paths"].get("yolo_dataset_root", "data/curling_yolo11_boxes25")
YAML_PATH = CFG["paths"].get("yolo_dataset_yaml", f"{OUTPUT_DATASET_ROOT}/data.yaml")
OUTPUT_RUNS = CFG["paths"].get("runs_dir", "outputs/runs/yolo11_boxes25")
MODEL_WEIGHTS = CFG["paths"].get("base_model_weights", "yolo11s.pt")  # requires ultralytics>=8.3

# If images are very large, normalized widths/heights of 25 px boxes will be very small.
# You can optionally downscale images during dataset build. Set to None to disable.
RESIZE_MAX_WIDTH = 1280
BOX_SIZE_PX = 25
BUILD_DATASET = False  # set to False to skip (re)building the dataset


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(root: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, e)))
    files.sort()
    return files


def train_val_split(images: List[str], val_ratio: float = 0.1) -> Tuple[List[str], List[str]]:
    n = len(images)
    val_n = max(1, int(round(n * val_ratio)))
    return images[val_n:], images[:val_n]


def read_image_size(path: str) -> Tuple[int, int]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    return w, h


def parse_cvat_annotations(xml_path: str) -> Tuple[Dict[str, List[Tuple[int, float, float]]], List[str]]:
    """
    Parse CVAT for images XML. Returns:
    - mapping: image_name -> list of (cls_id, x_px, y_px)
    - names: list of class names in order (for YAML)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # labels
    names: List[str] = []
    label_to_id: Dict[str, int] = {}
    labels_el = root.find('meta').find('task').find('labels') if root.find('meta') is not None else None
    if labels_el is not None:
        for lab in labels_el.findall('label'):
            name_el = lab.find('name')
            if name_el is not None:
                nm = name_el.text.strip()
                if nm not in label_to_id:
                    label_to_id[nm] = len(names)
                    names.append(nm)
    mapping: Dict[str, List[Tuple[int, float, float]]] = {}
    for img_el in root.findall('image'):
        img_name = img_el.get('name')
        if not img_name:
            continue
        acc: List[Tuple[int, float, float]] = []
        # points elements directly under <image>
        for p in img_el.findall('points'):
            lbl = p.get('label') or ""
            if lbl not in label_to_id:
                # add unseen label
                label_to_id[lbl] = len(names)
                names.append(lbl)
            cls_id = label_to_id[lbl]
            pts_str = p.get('points') or ""
            # CVAT points are like "x,y" or "x1,y1;x2,y2;...". Treat each pair as a separate keypoint.
            for pair in pts_str.split(';'):
                pair = pair.strip()
                if not pair:
                    continue
                try:
                    x_str, y_str = pair.split(',')
                    x = float(x_str); y = float(y_str)
                    acc.append((cls_id, x, y))
                except Exception:
                    continue
        # points elements inside <skeleton> under <image>
        for sk in img_el.findall('skeleton'):
            for p in sk.findall('points'):
                lbl = p.get('label') or ""
                if lbl not in label_to_id:
                    label_to_id[lbl] = len(names)
                    names.append(lbl)
                cls_id = label_to_id[lbl]
                pts_str = p.get('points') or ""
                for pair in pts_str.split(';'):
                    pair = pair.strip()
                    if not pair:
                        continue
                    try:
                        x_str, y_str = pair.split(',')
                        x = float(x_str); y = float(y_str)
                        acc.append((cls_id, x, y))
                    except Exception:
                        continue
        if acc:
            mapping[os.path.basename(img_name)] = acc
    return mapping, names if names else [f"kpt_{i}" for i in range(22)]


def parse_keypoint_label_file(label_path: str) -> List[Tuple[float, float]]:
    """
    Attempt to parse a keypoint label file. Supports simple formats:
    - Lines: "x y" (normalized [0,1] or pixel coords)
    - Lines: "cls x y" (cls ignored)
    If values <=1.0, treated as normalized; otherwise pixels.
    Returns list of (x, y) in the SAME numeric space; caller will convert.
    """
    points: List[Tuple[float, float]] = []
    if not os.path.exists(label_path):
        return points
    with open(label_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize separators: replace commas with spaces
    content_norm = content.replace(",", " ")
    lines = [ln.strip() for ln in content_norm.splitlines() if ln.strip()]

    # First, try Ultralytics YOLO Pose format per line:
    # cls kpt1x kpt1y kpt1v kpt2x kpt2y kpt2v ... (for N keypoints)
    # Extract ALL x,y regardless of visibility (some datasets set v=0 for occluded but valid points)
    lines_pose_xy: List[Tuple[float, float]] = []
    pose_ok = True
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 3:
            try:
                # try parse class id
                _ = float(parts[0])
                coords = list(map(float, parts[1:]))
                # group into triplets (x,y,v) or pairs (x,y) if visibility is omitted
                if len(coords) >= 2:
                    if len(coords) % 3 == 0:
                        for i in range(0, len(coords), 3):
                            x = coords[i]
                            y = coords[i + 1]
                            # v = coords[i + 2]  # ignored intentionally
                            lines_pose_xy.append((x, y))
                        continue
                    elif len(coords) % 2 == 0:
                        for i in range(0, len(coords), 2):
                            x = coords[i]
                            y = coords[i + 1]
                            lines_pose_xy.append((x, y))
                        continue
                    else:
                        pose_ok = False
                        break
            except Exception:
                pose_ok = False
                break
        else:
            pose_ok = False
            break
    if pose_ok and lines_pose_xy:
        return lines_pose_xy

    # Next, try line-wise parsing supporting formats:
    #  - "x y"
    #  - "cls x y"
    #  - "... multiple numbers ..." -> fallback to regex
    tmp: List[Tuple[float, float]] = []
    line_ok = True
    for ln in lines:
        parts = ln.split()
        if len(parts) == 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                tmp.append((x, y))
                continue
            except Exception:
                line_ok = False
                break
        elif len(parts) >= 3:
            # assume cls x y
            try:
                x, y = float(parts[1]), float(parts[2])
                tmp.append((x, y))
                continue
            except Exception:
                line_ok = False
                break
        else:
            line_ok = False
            break

    if line_ok and tmp:
        return tmp

    # Fallback: collect all float numbers and group into pairs
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", content_norm)
    vals = []
    for n in nums:
        try:
            vals.append(float(n))
        except Exception:
            pass
    if len(vals) >= 2:
        for i in range(0, len(vals) - 1, 2):
            points.append((vals[i], vals[i + 1]))
    return points


def convert_points_to_boxes(
    image_path: str,
    label_path: str,
    out_label_path: str,
    box_size_px: int = 25,
    num_classes: int = 22,
    original_size: Tuple[int, int] | None = None
) -> int:
    """
    Convert keypoints to 25 px square boxes centered at each keypoint.
    Write YOLO detection labels: cls cx cy w h (normalized).
    Classes are 0..21 for 22 keypoints, assigned by order.
    Returns number of boxes written.
    """
    ensure_dir(os.path.dirname(out_label_path))
    w, h = read_image_size(image_path)
    pts = parse_keypoint_label_file(label_path)
    if not pts:
        # No points parsed; leave label empty but warn in console
        print(f"[WARN] No keypoints parsed for: {os.path.basename(image_path)} from {label_path}")
        with open(out_label_path, "w", encoding="utf-8") as f:
            pass
        return 0

    # Determine if normalized
    normalized = False
    if pts:
        maxv = max(max(abs(px), abs(py)) for px, py in pts)
        if maxv <= 1.001:
            normalized = True

    half = box_size_px / 2.0
    entries: List[str] = []
    for idx, (px, py) in enumerate(pts):
        if normalized:
            cx = px * w
            cy = py * h
        else:
            cx, cy = px, py
            # If labels are in pixel units and the saved image size differs from original,
            # scale coordinates accordingly.
            if original_size is not None:
                orig_w, orig_h = original_size
                if orig_w > 0 and orig_h > 0:
                    scale_x = w / float(orig_w)
                    scale_y = h / float(orig_h)
                    cx *= scale_x
                    cy *= scale_y
        # Clamp center
        cx = min(max(cx, 0.0), w - 1.0)
        cy = min(max(cy, 0.0), h - 1.0)

        x1 = max(0.0, cx - half)
        y1 = max(0.0, cy - half)
        x2 = min(w - 1.0, cx + half)
        y2 = min(h - 1.0, cy + half)

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        yolo_cx = (x1 + x2) / 2.0 / w
        yolo_cy = (y1 + y2) / 2.0 / h
        yolo_w = bw / w
        yolo_h = bh / h

        cls_id = idx  # per-order mapping 0..21
        entries.append(f"{cls_id} {yolo_cx:.6f} {yolo_cy:.6f} {yolo_w:.6f} {yolo_h:.6f}")


def write_boxes_from_pixel_points(
    image_path: str,
    out_label_path: str,
    points_with_cls: List[Tuple[int, float, float]],
    box_size_px: int,
    original_size: Tuple[int, int] | None = None
) -> int:
    ensure_dir(os.path.dirname(out_label_path))
    w, h = read_image_size(image_path)
    half = box_size_px / 2.0
    entries: List[str] = []
    for cls_id, px, py in points_with_cls:
        cx, cy = px, py
        if original_size is not None:
            orig_w, orig_h = original_size
            if orig_w > 0 and orig_h > 0:
                scale_x = w / float(orig_w)
                scale_y = h / float(orig_h)
                cx *= scale_x
                cy *= scale_y
        cx = min(max(cx, 0.0), w - 1.0)
        cy = min(max(cy, 0.0), h - 1.0)
        x1 = max(0.0, cx - half); y1 = max(0.0, cy - half)
        x2 = min(w - 1.0, cx + half); y2 = min(h - 1.0, cy + half)
        bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
        ycx = (x1 + x2) / 2.0 / w
        ycy = (y1 + y2) / 2.0 / h
        yw = bw / w; yh = bh / h
        entries.append(f"{cls_id} {ycx:.6f} {ycy:.6f} {yw:.6f} {yh:.6f}")
    with open(out_label_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e + "\n")
    return len(entries)


    with open(out_label_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e + "\n")
    return len(entries)


def build_detection_dataset_wrapper():
    # If CVAT annotations.xml exists, use CVAT conversion path
    # Use dedicated builder with configurable box size
    build_detection_dataset(
        dataset_root=DATASET_ROOT,
        output_dataset_root=OUTPUT_DATASET_ROOT,
        yaml_path=YAML_PATH,
        box_size_px=BOX_SIZE_PX,
        resize_max_width=RESIZE_MAX_WIDTH,
    )

    # Fallback to previous non-CVAT formats
    images_dir = os.path.join(DATASET_ROOT, "images")
    labels_dir = os.path.join(DATASET_ROOT, "labels")
    mode = "yolo_split"  # or "darknet_single"
    if not os.path.isdir(images_dir):
        images_dir = os.path.join(DATASET_ROOT, "images", "train")
        labels_dir = os.path.join(DATASET_ROOT, "labels", "train")
    if not os.path.isdir(images_dir):
        images_dir = os.path.join(DATASET_ROOT, "obj_train_data")
        labels_dir = images_dir
        mode = "darknet_single"

    all_images = list_images(images_dir)
    if not all_images and os.path.isdir(os.path.join(images_dir, "train")):
        images_dir = os.path.join(images_dir, "train")
        labels_dir = os.path.join(labels_dir, "train")
        all_images = list_images(images_dir)
    if not all_images:
        raise FileNotFoundError(f"No images found under {images_dir}")

    def paired_label_path(img_path: str) -> str:
        name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        return os.path.join(labels_dir, name)

    def has_label(img_path: str) -> bool:
        p = paired_label_path(img_path)
        return os.path.exists(p) and os.path.getsize(p) > 0

    images_with_labels = [p for p in all_images if has_label(p)]
    if not images_with_labels:
        raise FileNotFoundError("No paired label files found for any image. Check source labels.")

    train_imgs, val_imgs = train_val_split(images_with_labels, val_ratio=0.1)

    # Prepare output dirs
    out_train_img = os.path.join(OUTPUT_DATASET_ROOT, "images", "train")
    out_val_img = os.path.join(OUTPUT_DATASET_ROOT, "images", "val")
    out_train_lbl = os.path.join(OUTPUT_DATASET_ROOT, "labels", "train")
    out_val_lbl = os.path.join(OUTPUT_DATASET_ROOT, "labels", "val")
    ensure_dir(out_train_img); ensure_dir(out_val_img)
    ensure_dir(out_train_lbl); ensure_dir(out_val_lbl)

    # (paired_label_path already defined above)

    # Classes 0..21
    for subset, imgs, out_img_dir, out_lbl_dir in [
        ("train", train_imgs, out_train_img, out_train_lbl),
        ("val", val_imgs, out_val_img, out_val_lbl),
    ]:
        for img_path in imgs:
            dst_img = os.path.join(out_img_dir, os.path.basename(img_path))
            # Load and optionally resize before saving to dataset
            im = cv2.imread(img_path)
            if im is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue
            orig_h, orig_w = im.shape[:2]
            new_w, new_h = orig_w, orig_h
            if RESIZE_MAX_WIDTH is not None and orig_w > RESIZE_MAX_WIDTH:
                scale = RESIZE_MAX_WIDTH / float(orig_w)
                new_w = RESIZE_MAX_WIDTH
                new_h = int(round(orig_h * scale))
                im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst_img, im)
            src_lbl = paired_label_path(img_path)
            dst_lbl = os.path.join(out_lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            # src_lbl presence was validated in has_label; still guard
            if not os.path.exists(src_lbl) or os.path.getsize(src_lbl) == 0:
                print(f"[WARN] Missing/empty source label for {os.path.basename(img_path)} -> skipped")
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    pass
                continue
            _ = convert_points_to_boxes(
                dst_img, src_lbl, dst_lbl,
                box_size_px=25, num_classes=22,
                original_size=(orig_w, orig_h)
            )

    # Sanity log
    tr_lbls = list_images(out_train_lbl)
    vl_lbls = list_images(out_val_lbl)
    print(f"Generated labels: train={len(os.listdir(out_train_lbl))}, val={len(os.listdir(out_val_lbl))}")

    # Write YAML
    ensure_dir(os.path.dirname(YAML_PATH))
    names = [f"kpt_{i}" for i in range(22)]
    yaml_content = (
        f"path: \"{OUTPUT_DATASET_ROOT}\"\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 22\n"
        f"names: {names}\n"
    )
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def select_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(epochs: int = 100, imgsz: int = 640, batch: int = 8, run_name: str = "train"):
    device = select_device()
    model = YOLO(MODEL_WEIGHTS)
    results = model.train(
        data=YAML_PATH,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=OUTPUT_RUNS,
        name=run_name,
        exist_ok=True,
        verbose=True,
    )
    return results


# -----------------------------
# Jaccard (IoU) metric on val set
# -----------------------------
def _load_yolo_label_file(label_path: str):
    boxes = []  # (cls, cx, cy, w, h) normalized
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx = float(parts[1]); cy = float(parts[2])
                w = float(parts[3]); h = float(parts[4])
                boxes.append((cls, cx, cy, w, h))
            except Exception:
                continue
    return boxes


def _yolo_to_xyxy_px(norm_boxes, img_w: int, img_h: int):
    xyxy = []  # (cls, x1,y1,x2,y2)
    for cls, cx, cy, w, h in norm_boxes:
        bw = w * img_w; bh = h * img_h
        x1 = (cx * img_w) - bw / 2.0
        y1 = (cy * img_h) - bh / 2.0
        x2 = x1 + bw
        y2 = y1 + bh
        xyxy.append((cls, max(0.0, x1), max(0.0, y1), min(img_w - 1.0, x2), min(img_h - 1.0, y2)))
    return xyxy


def _iou(a, b):
    # a,b: (x1,y1,x2,y2)
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_jaccard_on_val(iou_thresh: float = 0.01, conf_thresh: float = 0.25, weights_path: str | None = None, run_name: str = "train") -> float:
    """Compute mean Jaccard (IoU) across val set by greedy matching per image.

    Additionally computes per-class mean IoU for all classes encountered in GT matches.
    Results are written to jaccard_val.txt with overall and per-class breakdown.
    """
    device = select_device()
    # Prefer provided weights (e.g., freshly trained best.pt); fallback to default MODEL_WEIGHTS
    model = YOLO(weights_path if (weights_path and os.path.exists(weights_path)) else MODEL_WEIGHTS)

    val_img_dir = os.path.join(OUTPUT_DATASET_ROOT, "images", "val")
    val_lbl_dir = os.path.join(OUTPUT_DATASET_ROOT, "labels", "val")
    img_paths = list_images(val_img_dir)
    if not img_paths:
        print("[Jaccard] No validation images found.")
        return 0.0

    ious_all = []
    per_class_ious: Dict[int, List[float]] = {}
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        gt_norm = _load_yolo_label_file(os.path.join(val_lbl_dir, base))
        gt = [(cls, x1, y1, x2, y2) for cls, x1, y1, x2, y2 in _yolo_to_xyxy_px(gt_norm, w, h)]

        # Predict
        res = model(img, device=device, verbose=False)
        preds = []  # (cls, x1,y1,x2,y2)
        for r in res:
            if getattr(r, 'boxes', None) is None:
                continue
            orig_h, orig_w = r.orig_img.shape[:2]
            sx = w / float(orig_w); sy = h / float(orig_h)
            for b in r.boxes:
                conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                if conf < conf_thresh:
                    continue
                cls = int(b.cls[0]) if hasattr(b, 'cls') else -1
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                preds.append((cls, x1 * sx, y1 * sy, x2 * sx, y2 * sy))

        # Greedy match by IoU per class
        used_pred = set()
        for g in gt:
            g_cls, gx1, gy1, gx2, gy2 = g
            best = 0.0
            best_j = -1
            for j, p in enumerate(preds):
                if j in used_pred:
                    continue
                p_cls, px1, py1, px2, py2 = p
                if p_cls != g_cls:
                    continue
                iou = _iou((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
                if iou > best:
                    best = iou
                    best_j = j
            if best_j >= 0 and best >= iou_thresh:
                used_pred.add(best_j)
                ious_all.append(best)
                per_class_ious.setdefault(g_cls, []).append(best)

    mean_iou = float(np.mean(ious_all)) if ious_all else 0.0
    print(f"[Jaccard] Mean IoU on val (threshold {iou_thresh}): {mean_iou:.4f} over {len(ious_all)} matches")
    # Try read class names from YAML for nicer reporting and to enumerate all classes
    class_names: Dict[int, str] = {}
    num_classes: int | None = None
    try:
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            yaml_txt = f.read()
        for line in yaml_txt.splitlines():
            if line.strip().startswith("names:"):
                # expect a single-line Python-like list; parse safely
                list_str = line.split("names:", 1)[1].strip()
                names_list = ast.literal_eval(list_str)
                for i, nm in enumerate(names_list):
                    class_names[i] = str(nm)
                num_classes = len(names_list)
            if num_classes is None and line.strip().startswith("nc:"):
                try:
                    num_classes = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
    except Exception:
        pass

    # Save to file
    metrics_path = os.path.join(OUTPUT_RUNS, run_name, "jaccard_val.txt")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"mean_iou@{iou_thresh}: {mean_iou:.6f}\n")
        # Per-class breakdown for ALL classes (write zeros if no matches)
        if num_classes is None:
            # Fallback: infer a reasonable upper bound
            if class_names:
                num_classes = max(class_names.keys()) + 1
            else:
                num_classes = 23  # default fallback for this dataset
        f.write("per_class:\n")
        for cls_id in range(num_classes):
            vals = per_class_ious.get(cls_id, [])
            cls_mean = float(np.mean(vals)) if vals else 0.0
            name_suffix = f" ({class_names.get(cls_id, str(cls_id))})"
            f.write(f"  class_{cls_id}{name_suffix}: {cls_mean:.6f} over {len(vals)} matches\n")
    return mean_iou


def infer_and_export_centers_csv(
    images_dir: str,
    weights_path: str,
    out_csv_path: str
):
    device = select_device()
    model = YOLO(weights_path)
    image_paths = list_images(images_dir)
    rows: List[str] = ["image,class_id,cx_px,cy_px,conf"]
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model(img, device=device, verbose=False)
        for r in res:
            if getattr(r, "boxes", None) is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0]) if hasattr(b, "cls") else -1
                conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                rows.append(f"{os.path.basename(img_path)},{cls_id},{cx:.2f},{cy:.2f},{conf:.4f}")
    ensure_dir(os.path.dirname(out_csv_path))
    with open(out_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


def main():
    # 1) Build detection dataset from keypoints (15px squares)
    if BUILD_DATASET:
        build_detection_dataset_wrapper()
    else:
        print("[INFO] Skipping dataset build (BUILD_DATASET=False)")

    # 2) Train with unique run name to avoid overlay in results.png
    run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_model(epochs=50, imgsz=640, batch=8, run_name=run_name)

    # 2.5) Compute and save Jaccard (IoU) on validation set using best weights
    best_weights = str(Path(OUTPUT_RUNS) / run_name / "weights" / "best.pt")
    compute_jaccard_on_val(iou_thresh=0.5, conf_thresh=0.25, weights_path=best_weights, run_name=run_name)

    # 3) Inference on validation images and export centers to CSV
    best_weights = str(Path(OUTPUT_RUNS) / run_name / "weights" / "best.pt")
    infer_images = os.path.join(OUTPUT_DATASET_ROOT, "images", "val")
    out_csv = CFG["paths"].get(
        "predicted_csv",
        os.path.join(OUTPUT_DATASET_ROOT, "predicted_keypoints.csv"),
    )
    infer_and_export_centers_csv(infer_images, best_weights, out_csv)
    print("Saved CSV:", out_csv)


if __name__ == "__main__":
    main()


