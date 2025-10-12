import sys as _sys
from pathlib import Path as _Path
import cv2
import numpy as np
from typing import List, Dict
from tqdm import tqdm

# Ensure project root on PYTHONPATH to import from src
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics import YOLO
def classify_stone_team(image: np.ndarray, bbox: list[float]) -> str:
    """
    Простая классификация по цвету рукоятки в HSV.
    return: "red" / "yellow"
    """
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "red"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 100, 100]); red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100]); red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([20, 100, 100]); yellow_upper = np.array([30, 255, 255])

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    red_pixels = int(cv2.countNonZero(mask_red))
    yellow_pixels = int(cv2.countNonZero(mask_yellow))
    return "red" if red_pixels >= yellow_pixels else "yellow"


def detect_stones_yolo(video_path: str, weights_path: str, H: np.ndarray, conf: float = 0.25) -> List[Dict]:
    """
    Детектирует камни на каждом кадре видео с помощью YOLO-модели и
    возвращает список "треков" (по сути, по детекции на кадр) со списком позиций в px и в см.

    Формат результата совместим с create_birdseye_view:
    [
      {
        "stone_id": int,
        "team": "red"|"yellow",
        "positions_px": [{"frame": f, "x": cx, "y": cy}],
        "positions_real": [{"frame": f, "x": rx, "y": ry}],
        "final_position_real": { ... },
        "throw_start_frame": f,
        "throw_end_frame": f,
      },
      ...
    ]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    # Get total number of frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Видео: {total_frames} кадров, {fps:.1f} FPS")

    # Load YOLO weights robustly: if checkpoint is incompatible/corrupted, fallback to a generic detector
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"[WARN] Не удалось загрузить веса YOLO '{weights_path}': {e}")
        fallback = _PROJECT_ROOT / "experiments" / "yolo11s.pt"
        if not fallback.exists():
            # last resort: model id string
            fallback = "yolo11s.pt"
        print(f"[INFO] Использую запасные веса детектора: {fallback}")
        model = YOLO(str(fallback))
    
    frame_idx = 0
    stone_id_counter = 1
    stone_tracks: List[Dict] = []

    # Progress bar for video processing
    with tqdm(total=total_frames, desc="Обработка видео", unit="кадр") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model(frame, conf=conf, verbose=False)
            for r in results:
                if getattr(r, 'boxes', None) is None:
                    continue
                for b in r.boxes:
                    # center of bbox
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    team = classify_stone_team(frame, [x1, y1, x2, y2])

                    track = {
                        "stone_id": stone_id_counter,
                        "team": team,
                        "positions_px": [{"frame": frame_idx, "x": cx, "y": cy}],
                        "positions_real": [],
                    }

                    if H is not None:
                        pt = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), H)
                        rx, ry = map(float, pt[0][0].tolist())
                        track["positions_real"].append({"frame": frame_idx, "x": rx, "y": ry})
                        track["final_position_real"] = track["positions_real"][-1]

                    track["throw_start_frame"] = frame_idx
                    track["throw_end_frame"] = frame_idx

                    stone_tracks.append(track)
                    stone_id_counter += 1

            frame_idx += 1
            pbar.update(1)  # Update progress bar

    cap.release()
    return stone_tracks


