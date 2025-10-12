"""Stones detection, classification and tracking"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torchvision.transforms as T
from src.config_loader import load_config
import os

from groundingdino.util.inference import load_model, predict
from segment_anything import sam_model_registry, SamPredictor
import torch

_DINO_MODEL = None
_SAM_PREDICTOR = None

_TRANSFORM = T.Compose([T.ToTensor()])

def load_grounding_sam_models():
    global _DINO_MODEL, _SAM_PREDICTOR

    if _DINO_MODEL is None or _SAM_PREDICTOR is None:
        config = load_config()

        dino_config_path = config['model_paths']['groundingdino_config']
        dino_weights_path = config['model_paths']['groundingdino_weights']
        sam_weights_path = config['model_paths']['sam_weights']

        print("Загружаю GroundingDINO...")
        _DINO_MODEL = load_model(dino_config_path, dino_weights_path)

        print("Загружаю SAM...")
        sam = sam_model_registry["vit_l"](checkpoint=sam_weights_path)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        _SAM_PREDICTOR = SamPredictor(sam)

    return _DINO_MODEL, _SAM_PREDICTOR


def detect_stones_in_frame(image: np.ndarray, text_prompt: str = "curling stone.") -> list[dict]:
    """
    Детектирует камни в одном кадре с помощью GroundingSAM.

    image: numpy array, BGR
    text_prompt: promt with "." at the end
    список детекций: [{"bbox": [x1,y1,x2,y2], "mask": np.array, "score": float}, ...]
    """
    model, predictor = load_grounding_sam_models()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_tensor = _TRANSFORM(image_rgb).to("cuda" if torch.cuda.is_available() else "cpu")
    
    boxes, logits, _ = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25
    )
    
    if len(boxes) == 0:
        return []

    detections = []

    # SAM
    predictor.set_image(image_rgb)
    for box, logit in zip(boxes, logits):
        H, W, _ = image.shape
        box_xyxy = box * torch.Tensor([W, H, W, H])
        box_xyxy = box_xyxy.numpy()

        # SAM mask prediction
        masks, _, _ = predictor.predict(box=box_xyxy)

        # first mask (most convenient)
        mask = masks[0]

        detections.append({
            "bbox": box_xyxy.tolist(), 
            "mask": mask, # bool array
            "score": float(logit)
        })

    return detections


def classify_stone_team(image: np.ndarray, bbox: list[float]) -> str:
    """
    return: "red" / "yellow"
    """
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)

    if red_pixels > yellow_pixels:
        return "red"
    else:
        return "yellow"


class SimpleTracker:
    """
    Трекер на основе соответствия IoU
    """

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.active_tracks = []  # [{"id": 1, "bbox": [...], "last_seen": frame}, ...]

    def update(self, frame_num: int, detections: List[Dict]) -> List[Dict]:
        """
        return: список треков с ID
        """
        if len(self.active_tracks) == 0:
            for det in detections:
                det["track_id"] = self.next_id
                self.active_tracks.append({
                    "id": self.next_id,
                    "bbox": det["bbox"],
                    "last_seen": frame_num
                })
                self.next_id += 1
            return detections

        # выичсление IoU
        matched = set()
        used_tracks = set()

        for i, track in enumerate(self.active_tracks):
            best_iou = 0
            best_det_idx = -1

            for j, det in enumerate(detections):
                if j in matched:
                    continue
                iou = self._calculate_iou(track["bbox"], det["bbox"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = j

            if best_det_idx != -1:
                detections[best_det_idx]["track_id"] = track["id"]
                self.active_tracks[i]["bbox"] = detections[best_det_idx]["bbox"]
                self.active_tracks[i]["last_seen"] = frame_num
                matched.add(best_det_idx)
                used_tracks.add(i)

        # Новые треки для несопоставленных детекций
        for j, det in enumerate(detections):
            if j not in matched:
                det["track_id"] = self.next_id
                self.active_tracks.append({
                    "id": self.next_id,
                    "bbox": det["bbox"],
                    "last_seen": frame_num
                })
                self.next_id += 1

        # если на протяжении N кадрров трек потерян - удаление
        N = 40
        self.active_tracks = [t for t in self.active_tracks if frame_num - t["last_seen"] <= N]

        return detections

    @staticmethod
    def _calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


#########################


def detect_and_track_stones(video_path: str, homography_matrix: np.ndarray) -> list[dict]:
    """
    :return: список треков камней
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    frame_num = 0
    tracker = SimpleTracker()
    all_detections = [] 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_stones_in_frame(frame)

        for det in detections:
            det["team"] = classify_stone_team(frame, det["bbox"])

        tracked_detections = tracker.update(frame_num, detections)

        for det in tracked_detections:
            det["frame"] = frame_num

        all_detections.extend(tracked_detections)
        frame_num += 1

        if frame_num % 30 == 0:
            print(f"Обработано {frame_num} кадров...")

    cap.release()

    tracks_dict = {}
    for det in all_detections:
        track_id = det["track_id"]
        if track_id not in tracks_dict:
            tracks_dict[track_id] = {
                "stone_id": track_id,
                "team": det["team"],
                "positions_px": [],
                "positions_real": []
            }
        x1, y1, x2, y2 = det["bbox"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        tracks_dict[track_id]["positions_px"].append({
            "frame": det["frame"],
            "x": center_x,
            "y": center_y
        })

        # получение координат в см 
        if homography_matrix is not None:
            pixel_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            real_point = cv2.perspectiveTransform(pixel_point, homography_matrix)
            rx, ry = real_point[0][0]
            tracks_dict[track_id]["positions_real"].append({
                "frame": det["frame"],
                "x": float(rx),
                "y": float(ry)
            })

    stone_tracks = list(tracks_dict.values())

    for track in stone_tracks:
        if len(track["positions_real"]) > 0:
            track["throw_start_frame"] = track["positions_px"][0]["frame"]
            track["throw_end_frame"] = track["positions_px"][-1]["frame"]
            track["final_position_real"] = track["positions_real"][-1]

    print(f"Обнаружено {len(stone_tracks)} камней.")

    return stone_tracks