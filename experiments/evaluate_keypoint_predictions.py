import os
import csv
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import sys as _sys
from pathlib import Path as _Path
# Ensure project root is on PYTHONPATH for 'src' imports
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from src.config_loader import load_config


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_images(dir_path: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def parse_cvat_points(xml_path: str) -> Tuple[
    Dict[str, Dict[int, Tuple[float, float]]],  # gt points per image
    List[str],                                   # class names
    Dict[str, Tuple[int, int]]                   # original image sizes per image
]:
    """
    Parse CVAT annotations.xml containing 22 keypoints per image (pixel coords).
    Returns:
      - gt: mapping image_name -> {class_id -> (x_px, y_px)}
      - class_names: index-aligned list of class names (order from <meta><task><labels>)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1) Build class name -> id mapping from <meta>/<task>/<labels>
    class_names: List[str] = []
    name_to_id: Dict[str, int] = {}
    labels_el = root.find("meta")
    if labels_el is not None:
        task_el = labels_el.find("task")
        if task_el is not None:
            labs = task_el.find("labels")
            if labs is not None:
                for lab in labs.findall("label"):
                    nm_el = lab.find("name")
                    if nm_el is not None and nm_el.text:
                        nm = nm_el.text.strip()
                        if nm not in name_to_id:
                            name_to_id[nm] = len(class_names)
                            class_names.append(nm)

    # 2) Collect points per image, resolving labels to class ids
    gt: Dict[str, Dict[int, Tuple[float, float]]] = {}
    orig_sizes: Dict[str, Tuple[int, int]] = {}
    for img_el in root.findall("image"):
        img_name = img_el.get("name")
        if not img_name:
            continue
        try:
            ow = int(float(img_el.get("width") or 0))
            oh = int(float(img_el.get("height") or 0))
            if ow > 0 and oh > 0:
                orig_sizes[os.path.basename(img_name)] = (ow, oh)
        except Exception:
            pass
        by_cls: Dict[int, Tuple[float, float]] = {}

        def add_points(label: Optional[str], points_str: Optional[str]):
            if not label or not points_str:
                return
            if label not in name_to_id:
                # unseen label -> append to mapping to keep deterministic indexing
                name_to_id[label] = len(class_names)
                class_names.append(label)
            cls_id = name_to_id[label]
            # points like "x,y" or "x1,y1;x2,y2;..."; use the first pair per label as canonical
            for pair in points_str.split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                try:
                    xs, ys = pair.split(",")
                    x = float(xs); y = float(ys)
                    # Keep the first occurrence per class (assuming one GT point per class)
                    if cls_id not in by_cls:
                        by_cls[cls_id] = (x, y)
                    # stop after the first
                    break
                except Exception:
                    continue

        # points directly under <image>
        for p in img_el.findall("points"):
            add_points(p.get("label"), p.get("points"))

        # points under <skeleton>
        for sk in img_el.findall("skeleton"):
            for p in sk.findall("points"):
                add_points(p.get("label"), p.get("points"))

        if by_cls:
            gt[os.path.basename(img_name)] = by_cls

    return gt, class_names, orig_sizes


def load_predictions_csv(csv_path: str) -> Dict[str, Dict[int, Tuple[float, float, float]]]:
    """
    Load predictions from CSV with flexible headers.
    Supports either columns [image, class_id, cx_px, cy_px, conf] or [image, class_id, x, y, conf].
    Returns mapping: image_name -> {class_id -> (x_px, y_px, conf)} keeping the HIGHEST conf per class.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = {k.strip().lower() for k in (reader.fieldnames or [])}

        # Determine coordinate keys
        x_key = "cx_px" if "cx_px" in fieldnames else ("x" if "x" in fieldnames else None)
        y_key = "cy_px" if "cy_px" in fieldnames else ("y" if "y" in fieldnames else None)
        if x_key is None or y_key is None or "image" not in fieldnames or "class_id" not in fieldnames:
            raise ValueError("CSV must contain: image, class_id and coordinates (cx_px/cy_px or x/y)")

        preds: Dict[str, Dict[int, Tuple[float, float, float]]] = {}
        for row in reader:
            try:
                img = row.get("image", "").strip()
                cls = int(float(row.get("class_id", "-1")))
                x = float(row.get(x_key, "nan"))
                y = float(row.get(y_key, "nan"))
                conf = float(row.get("conf", "0") or 0)
            except Exception:
                continue
            if not img:
                continue
            by_cls = preds.setdefault(img, {})
            # keep highest confidence per class
            if cls not in by_cls or conf >= by_cls[cls][2]:
                by_cls[cls] = (x, y, conf)

    return preds


def _candidate_image_paths(images_dir: str, image_name: str) -> List[str]:
    paths: List[str] = []
    # 1) Provided directory
    paths.append(os.path.join(images_dir, image_name))
    # 2) If provided directory looks like .../images/{split}, try sibling splits and parent images
    parent = os.path.dirname(images_dir)
    base = os.path.basename(parent)
    split = os.path.basename(images_dir)
    if base == "images":
        # sibling train/val
        for sib in ["train", "val"]:
            if sib != split:
                paths.append(os.path.join(parent, sib, image_name))
        # parent images without split
        paths.append(os.path.join(parent, image_name))
    else:
        # Also try images/train and images/val under parent
        paths.append(os.path.join(parent, "images", image_name))
        paths.append(os.path.join(parent, "images", "train", image_name))
        paths.append(os.path.join(parent, "images", "val", image_name))
    return paths


def image_size(images_dir: str, image_name: str) -> Tuple[int, int]:
    tried: List[str] = []
    for p in _candidate_image_paths(images_dir, image_name):
        tried.append(p)
        if not os.path.isfile(p):
            continue
        im = cv2.imread(p)
        if im is not None:
            h, w = im.shape[:2]
            return w, h
    hint = "\n".join(tried)
    raise FileNotFoundError(
        f"Cannot locate/read image '{image_name}'. Tried paths:\n{hint}\n"
        f"Pass --images pointing to the correct directory with this file."
    )


def compute_per_point_errors(
    gt: Dict[str, Dict[int, Tuple[float, float]]],
    preds: Dict[str, Dict[int, Tuple[float, float, float]]],
    images_dir: str,
    num_classes: Optional[int] = None,
    orig_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
) -> List[Dict[str, float]]:
    """
    Compute per-image, per-class errors.
    Returns a list of rows with keys:
      image, class_id, image_w, image_h, gt_x, gt_y, pred_x, pred_y, conf, err_px, err_norm, matched
    """
    # Determine set of classes (ensure stable full range if provided)
    all_cls_ids = set()
    for _, m in gt.items():
        all_cls_ids.update(m.keys())
    if num_classes is not None:
        all_cls_ids.update(range(num_classes))
    cls_ids_sorted = sorted(all_cls_ids)

    # Restrict evaluation strictly to validation images present in images_dir
    val_image_set = set(list_images(images_dir))

    rows: List[Dict[str, float]] = []
    for img_name, gt_points in gt.items():
        if img_name not in val_image_set:
            continue
        w, h = image_size(images_dir, img_name)
        diag = float(np.hypot(w, h)) if (w > 0 and h > 0) else 1.0
        pred_points = preds.get(img_name, {})

        # rescale GT if needed
        sx = sy = 1.0
        if orig_sizes and img_name in orig_sizes:
            ow, oh = orig_sizes[img_name]
            if ow > 0 and oh > 0 and (ow != w or oh != h):
                sx = w / float(ow)
                sy = h / float(oh)

        for cid in cls_ids_sorted:
            has_gt = cid in gt_points
            has_pred = cid in pred_points
            if has_gt:
                gx, gy = gt_points[cid]
                gx, gy = gx * sx, gy * sy
            else:
                gx = gy = float("nan")

            if has_pred:
                px, py, conf = pred_points[cid]
            else:
                px = py = float("nan")
                conf = 0.0

            if has_gt and has_pred:
                err_px = float(np.hypot(px - gx, py - gy))
                err_norm = err_px / diag if diag > 0 else 0.0
                matched = 1
            else:
                # Penalty if missing either GT or pred
                err_px = diag
                err_norm = 1.0
                matched = 0

            rows.append({
                "image": img_name,
                "class_id": cid,
                "image_w": w,
                "image_h": h,
                "gt_x": gx,
                "gt_y": gy,
                "pred_x": px,
                "pred_y": py,
                "conf": conf,
                "err_px": err_px,
                "err_norm": err_norm,
                "matched": matched,
            })

    return rows


def save_detailed(out_dir: str, rows: List[Dict[str, float]], class_names: Optional[List[str]]):
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "keypoint_pixel_errors_detailed.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "class_id", "class_name", "image_w", "image_h",
            "gt_x", "gt_y", "pred_x", "pred_y", "conf", "err_px", "err_norm", "matched"
        ])
        for r in rows:
            cid = int(r["class_id"]) if r.get("class_id") is not None else -1
            cname = class_names[cid] if (class_names and 0 <= cid < len(class_names)) else str(cid)
            w.writerow([
                r.get("image"), cid, cname, int(r.get("image_w", 0)), int(r.get("image_h", 0)),
                r.get("gt_x"), r.get("gt_y"), r.get("pred_x"), r.get("pred_y"),
                r.get("conf"), r.get("err_px"), r.get("err_norm"), int(r.get("matched", 0))
            ])


def summarize_by_image(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    by_image: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        by_image.setdefault(str(r.get("image")), []).append(r)
    summaries: List[Dict[str, float]] = []
    for img, items in by_image.items():
        err_norms = [float(x.get("err_norm", 1.0)) for x in items]
        err_pxs = [float(x.get("err_px", 0.0)) for x in items]
        matched = [int(x.get("matched", 0)) for x in items]
        confs = [float(x.get("conf", 0.0)) for x in items if int(x.get("matched", 0)) == 1]
        image_w = int(items[0].get("image_w", 0)) if items else 0
        image_h = int(items[0].get("image_h", 0)) if items else 0
        n = len(items)
        summaries.append({
            "image": img,
            "image_w": image_w,
            "image_h": image_h,
            "classes": n,
            "matched": int(sum(matched)),
            "match_rate": (float(sum(matched)) / float(n)) if n else 0.0,
            "mean_err_norm": float(np.mean(err_norms)) if err_norms else 1.0,
            "median_err_norm": float(np.median(err_norms)) if err_norms else 1.0,
            "mean_err_px": float(np.mean(err_pxs)) if err_pxs else 0.0,
            "median_err_px": float(np.median(err_pxs)) if err_pxs else 0.0,
            "mean_conf_matched": float(np.mean(confs)) if confs else 0.0,
        })
    return summaries


def save_image_summary(out_dir: str, image_summaries: List[Dict[str, float]]):
    path = os.path.join(out_dir, "keypoint_pixel_errors_by_image.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "image_w", "image_h", "classes", "matched", "match_rate",
            "mean_err_norm", "median_err_norm", "mean_err_px", "median_err_px", "mean_conf_matched"
        ])
        for s in image_summaries:
            w.writerow([
                s.get("image"), int(s.get("image_w", 0)), int(s.get("image_h", 0)),
                int(s.get("classes", 0)), int(s.get("matched", 0)), float(s.get("match_rate", 0.0)),
                float(s.get("mean_err_norm", 0.0)), float(s.get("median_err_norm", 0.0)),
                float(s.get("mean_err_px", 0.0)), float(s.get("median_err_px", 0.0)),
                float(s.get("mean_conf_matched", 0.0)),
            ])


def print_overall_summary(image_summaries: List[Dict[str, float]]):
    if not image_summaries:
        print("No validation images matched between annotations and predictions.")
        return
    match_rates = [float(s.get("match_rate", 0.0)) for s in image_summaries]
    mean_err_norms = [float(s.get("mean_err_norm", 1.0)) for s in image_summaries]
    median_err_norms = [float(s.get("median_err_norm", 1.0)) for s in image_summaries]
    print("Overall summary (validation set only):")
    print(f"  images: {len(image_summaries)}")
    print(f"  mean(match_rate): {np.mean(match_rates):.4f}")
    print(f"  mean(mean_err_norm): {np.mean(mean_err_norms):.4f}")
    print(f"  mean(median_err_norm): {np.mean(median_err_norms):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate keypoint predictions against CVAT annotations")
    parser.add_argument("--annotations", default=None, help="Path to CVAT annotations.xml")
    parser.add_argument("--images", default=None, help="Directory with images (for sizes)")
    parser.add_argument("--predictions", default=None, help="CSV with predictions: image,class_id,x,y[,conf]")
    parser.add_argument("--out", default=None, help="Output directory for metrics report")
    parser.add_argument("--num-classes", type=int, default=None, help="Optional number of classes (22)")
    args = parser.parse_args()

    CFG = load_config()
    annotations = args.annotations or CFG["paths"].get("annotations_xml")
    images_dir = args.images or CFG["paths"].get("val_images_dir")
    predictions = args.predictions or CFG["paths"].get("predicted_csv")
    out_dir = args.out or os.path.join(CFG["paths"].get("output_dir", "outputs"), "eval_keypoints")

    gt, class_names, orig_sizes = parse_cvat_points(annotations)
    preds = load_predictions_csv(predictions)
    rows = compute_per_point_errors(gt, preds, images_dir, args.num_classes, orig_sizes)
    save_detailed(out_dir, rows, class_names)
    image_summaries = summarize_by_image(rows)
    save_image_summary(out_dir, image_summaries)
    print_overall_summary(image_summaries)
    print(f"Saved per-point metrics to: {os.path.join(out_dir, 'keypoint_pixel_errors_detailed.csv')}")
    print(f"Saved per-image summary to: {os.path.join(out_dir, 'keypoint_pixel_errors_by_image.csv')}")


if __name__ == "__main__":
    main()


