import os
import glob
from typing import List, Tuple

import cv2


DATASET_DIR = \
    "/Users/glytov_ivan/Documents/curling/curling_analytics/data/curling_yolo11_boxes25"
OUT_DIR = \
    "/Users/glytov_ivan/Documents/curling/curling_analytics/outputs/preview_bboxes"
NUM_SAMPLES = 12  # how many images to visualize


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)




def list_images(dir_path: str) -> List[str]:
    files: List[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(glob.glob(os.path.join(dir_path, ext)))
    files.sort()
    return files


def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Return list of (cls, cx, cy, w, h) normalized [0,1]."""
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                # skip malformed
                continue
            try:
                cls = int(float(parts[0]))
                cx = float(parts[1]); cy = float(parts[2])
                w = float(parts[3]); h = float(parts[4])
                boxes.append((cls, cx, cy, w, h))
            except Exception:
                continue
    return boxes


def draw_boxes(img, boxes: List[Tuple[int, float, float, float, float]]):
    h, w = img.shape[:2]
    for (cls, cxn, cyn, wn, hn) in boxes:
        # Clamp normalized values into [0,1]
        cxn = max(0.0, min(1.0, cxn))
        cyn = max(0.0, min(1.0, cyn))
        wn = max(0.0, min(1.0, wn))
        hn = max(0.0, min(1.0, hn))
        cx = cxn * w; cy = cyn * h
        bw = wn * w; bh = hn * h
        # Enforce minimal visible size
        bw = max(bw, 2.0); bh = max(bh, 2.0)
        x1 = int(max(0, cx - bw / 2)); y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w - 1, cx + bw / 2)); y2 = int(min(h - 1, cy + bh / 2))
        color = (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # draw center for debugging
        cv2.circle(img, (int(cx), int(cy)), 2, (0, 0, 255), -1)
    return img


def main():
    ensure_dir(OUT_DIR)
    # Build a list of images that have corresponding non-empty label files
    candidates: List[str] = []
    for sub in ("train", "val"):
        img_dir = os.path.join(DATASET_DIR, "images", sub)
        lbl_dir = os.path.join(DATASET_DIR, "labels", sub)
        imgs = list_images(img_dir)
        for img_path in imgs:
            base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            lbl_path = os.path.join(lbl_dir, base)
            if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                candidates.append(img_path)
    samples = candidates[:NUM_SAMPLES] if len(candidates) > NUM_SAMPLES else candidates

    if not samples:
        raise FileNotFoundError(f"No images found under {DATASET_DIR}/images")

    for img_path in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Resolve label path based on image split
        split = "train" if "/images/train/" in img_path or "\\images\\train\\" in img_path else "val"
        label_dir = os.path.join(DATASET_DIR, "labels", split)
        label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        boxes = load_yolo_labels(label_path)
        vis = draw_boxes(img.copy(), boxes)
        out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, vis)
        print(f"Saved: {out_path} with {len(boxes)} boxes")


if __name__ == "__main__":
    main()


