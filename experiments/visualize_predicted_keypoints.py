import os
import csv
from typing import Dict, List, Tuple

import cv2
import numpy as np
import sys as _sys
from pathlib import Path as _Path
# Ensure project root is on PYTHONPATH for 'src' imports
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from src.config_loader import load_config


# Inputs/outputs
CFG = load_config()
CSV_PATH = CFG["paths"].get("predicted_csv", "data/curling_yolo11_boxes25/predicted_keypoints.csv")
IMAGES_DIR = CFG["paths"].get("val_images_dir", "data/curling_yolo11_boxes25/images/val")
OUT_DIR = os.path.join(CFG["paths"].get("output_dir", "outputs"), "predicted_keypoints_vis")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_predictions(csv_path: str) -> Dict[str, List[Tuple[int, float, float, float]]]:
    """Return mapping: image_name -> list of (class_id, cx, cy, conf)."""
    by_image: Dict[str, List[Tuple[int, float, float, float]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            img_name = row[0].strip()
            try:
                cls_id = int(float(row[1]))
                cx = float(row[2])
                cy = float(row[3])
                conf = float(row[4])
            except Exception:
                continue
            by_image.setdefault(img_name, []).append((cls_id, cx, cy, conf))
    return by_image


def color_for_class(cls_id: int) -> Tuple[int, int, int]:
    """Generate a distinct BGR color for a class id using HSV wheel."""
    # Cycle hues deterministically over 0..179 (OpenCV HSV hue range)
    hue = int((cls_id * 179 / 23) % 179)  # 23 classes in dataset config
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_points(
    image_path: str,
    points: List[Tuple[int, float, float, float]],
    out_path: str,
    draw_labels: bool = True,
):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Cannot read image: {image_path}")
        return

    for cls_id, cx, cy, conf in points:
        color = color_for_class(cls_id)
        center = (int(round(cx)), int(round(cy)))
        cv2.circle(img, center, 6, color, thickness=-1, lineType=cv2.LINE_AA)
        if draw_labels:
            label = f"{cls_id}:{conf:.2f}"
            txt_pos = (center[0] + 6, center[1] - 6)
            cv2.putText(img, label, txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, img)


def main():
    preds = load_predictions(CSV_PATH)
    ensure_dir(OUT_DIR)

    # Process each image listed in the CSV
    for img_name, pts in preds.items():
        src = os.path.join(IMAGES_DIR, img_name)
        dst = os.path.join(OUT_DIR, img_name)
        draw_points(src, pts, dst, draw_labels=True)
        print(f"Saved: {dst} ({len(pts)} points)")


if __name__ == "__main__":
    main()


