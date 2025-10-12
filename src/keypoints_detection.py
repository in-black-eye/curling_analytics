"""
Модуль для детекции ключевых точек кёрлингового поля с помощью дообученной млодели (сейчас - YOLOv11x-pose)
На вход: изображение (путь или numpy array)

Далее  - преобразование координат в px в реальные координаты (в см)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from src.config_loader import load_config 

_model = None
_det_model = None

def load_keypoints_model(model_path=None):
    global _model
    if _model is None:
        if model_path is None:
            config = load_config()
            # Prefer detection model for 22 keypoint boxes if configured
            model_path = config['model_paths'].get('keypoints_detector') or config['model_paths']['keypoints_model']
        
        print(f"Загружаю модель: {model_path}")
        _model = YOLO(model_path)
    return _model

def detect_keypoints(image_input, conf_threshold=0.2):
    """
    image_input: путь (str) или numpy array
    conf_threshold: порог уверенности  детекции
    
    return: [(x1, y1), (x2, y2), ...] — координаты ключевых точек
    """
    model = load_keypoints_model()

    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_input}")
    else:
        image = image_input

    results = model(image, conf=conf_threshold, verbose=False)

    keypoints_list = []

    # If this is a detection model (boxes), compute centers per class (we expect classes 1..22)
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        # Use highest-confidence box per class
        best_by_cls = {}
        for r in results:
            for b in r.boxes:
                conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                if conf < conf_threshold:
                    continue
                cls_id = int(b.cls[0]) if hasattr(b, 'cls') else -1
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if cls_id not in best_by_cls or conf > best_by_cls[cls_id][2]:
                    best_by_cls[cls_id] = (cx, cy, conf)
        # Build ordered by expected mapping: class 1 -> index 0, ..., class 22 -> index 21.
        for cls_id in range(1, 23):
            if cls_id in best_by_cls:
                cx, cy, _ = best_by_cls[cls_id]
                keypoints_list.append((float(cx), float(cy), cls_id - 1))
    else:
        # Pose model fallback
        for result in results:
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                kpts = result.keypoints.data[0].cpu().numpy()  # (N, 3) — x, y, confidence
                for i, kp in enumerate(kpts):
                    x, y, conf = kp
                    if conf >= conf_threshold:
                        keypoints_list.append((float(x), float(y), i))

    print(f"Найдено {len(keypoints_list)} keypoints.")
    return keypoints_list

def visualize_keypoints(image_input, keypoints, output_path=None):
    """
    Делает инференс ключевых точек

    image_input: путь / numpy array
    keypoints: список [(x1, y1), ...]
    output_path
    
    return: инференс изображения с нарисованными точками (numpy array)
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input.copy()

    for i, (x, y) in enumerate(keypoints):
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(image, str(i), (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Визуализация сохранена: {output_path}")

    return image
