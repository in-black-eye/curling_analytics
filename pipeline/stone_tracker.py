"""
Автономный модуль для обнаружения и трекинга камней в кёрлинге
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from collections import deque

# Опциональный импорт ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    print("Предупреждение: ultralytics не установлен. Будут использованы простые методы детекции камней.")


class StoneDetector:
    """Детектор камней в кёрлинге"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        """
        Инициализация детектора камней
        
        Args:
            model_path: Путь к модели YOLO для обнаружения камней
            conf_threshold: Порог уверенности для детекции
        """
        self.conf_threshold = conf_threshold
        self.model = None
        
        # Попытка загрузить модель
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
            self._load_model()
        else:
            # Попытка найти модель в стандартных местах
            self.model_path = self._find_model()
            if self.model_path:
                self._load_model()
            else:
                print("Предупреждение: Модель для обнаружения камней не найдена")
                print("Будут использованы простые методы детекции")
    
    def _find_model(self) -> Optional[str]:
        """Ищет модель в стандартных местах"""
        possible_paths = [
            "../models/stones_detection_and_classification_11s.pt",
            "models/stones_detection_and_classification_11s.pt",
            "yolo11s.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_model(self):
        """Загружает модель YOLO"""
        if not ULTRALYTICS_AVAILABLE:
            print("ultralytics не доступен, модель не загружена")
            self.model = None
            return
            
        try:
            print(f"Загружаю модель обнаружения камней: {self.model_path}")
            # Используем weights_only=False для совместимости с PyTorch 2.6+
            import torch
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            self.model = YOLO(self.model_path)
            print("Модель загружена успешно")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model = None
    
    def detect_stones(self, frame: np.ndarray) -> List[Dict]:
        """
        Обнаруживает камни на кадре
        
        Args:
            frame: Кадр изображения
            
        Returns:
            Список детекций камней
        """
        if self.model is None:
            return self._detect_stones_simple(frame)
        
        try:
            # Выполняем детекцию
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                        if conf < self.conf_threshold:
                            continue
                        
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                        
                        # Определяем команду по классу или цвету
                        team = self._classify_stone_team(frame, [x1, y1, x2, y2])
                        
                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "score": conf,
                            "team": team,
                            "class_id": cls_id
                        })
            
            return detections
            
        except Exception as e:
            print(f"Ошибка при обнаружении камней: {e}")
            return self._detect_stones_simple(frame)
    
    def _detect_stones_simple(self, frame: np.ndarray) -> List[Dict]:
        """Простая детекция камней по цвету (fallback)"""
        # Конвертируем в HSV для лучшей работы с цветом
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Определяем диапазоны цветов для красных и желтых камней
        # Красные камни
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Желтые камни
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        
        # Создаем маски
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        detections = []
        
        # Обрабатываем красные камни
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_red:
            area = cv2.contourArea(contour)
            if area > 100:  # Минимальная площадь
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "score": 0.8,
                    "team": "red",
                    "class_id": 0
                })
        
        # Обрабатываем желтые камни
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_yellow:
            area = cv2.contourArea(contour)
            if area > 100:  # Минимальная площадь
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "score": 0.8,
                    "team": "yellow",
                    "class_id": 1
                })
        
        return detections
    
    def _classify_stone_team(self, image: np.ndarray, bbox: List[float]) -> str:
        """
        Классифицирует камень по команде (цвету)
        
        Args:
            image: Изображение
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            "red" или "yellow"
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Обрезаем bbox до границ изображения
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return "red"
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Диапазоны цветов
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        # Создаем маски
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Подсчитываем пиксели
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        
        return "red" if red_pixels > yellow_pixels else "yellow"


class SimpleTracker:
    """Простой трекер камней на основе IoU"""
    
    def __init__(self, iou_threshold: float = 0.5, max_disappeared: int = 10):
        """
        Инициализация трекера
        
        Args:
            iou_threshold: Порог IoU для ассоциации
            max_disappeared: Максимальное количество кадров без детекции
        """
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.tracks = []
        self.next_id = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Обновляет треки с новыми детекциями
        
        Args:
            detections: Список новых детекций
            
        Returns:
            Список активных треков
        """
        # Если нет детекций, помечаем все треки как исчезнувшие
        if len(detections) == 0:
            for track in self.tracks:
                track['disappeared'] += 1
            return [t for t in self.tracks if t['disappeared'] < self.max_disappeared]
        
        # Если нет треков, создаем новые
        if len(self.tracks) == 0:
            for detection in detections:
                self._create_track(detection)
            return self.tracks
        
        # Вычисляем IoU между треками и детекциями
        iou_matrix = self._compute_iou_matrix(self.tracks, detections)
        
        # Ассоциируем треки с детекциями
        matched_tracks, matched_detections = self._associate_tracks(iou_matrix)
        
        # Обновляем совпавшие треки
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            self.tracks[track_idx].update({
                'bbox': detections[det_idx]['bbox'],
                'score': detections[det_idx]['score'],
                'team': detections[det_idx]['team'],
                'disappeared': 0
            })
        
        # Создаем новые треки для несовпавших детекций
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        for det_idx in unmatched_detections:
            self._create_track(detections[det_idx])
        
        # Помечаем несовпавшие треки как исчезнувшие
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['disappeared'] += 1
        
        # Удаляем старые треки
        self.tracks = [t for t in self.tracks if t['disappeared'] < self.max_disappeared]
        
        return self.tracks
    
    def _create_track(self, detection: Dict):
        """Создает новый трек"""
        track = {
            'track_id': self.next_id,
            'bbox': detection['bbox'],
            'score': detection['score'],
            'team': detection['team'],
            'disappeared': 0,
            'positions': [],
            'positions_real': []
        }
        self.tracks.append(track)
        self.next_id += 1
    
    def _compute_iou_matrix(self, tracks: List[Dict], detections: List[Dict]) -> np.ndarray:
        """Вычисляет матрицу IoU между треками и детекциями"""
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track['bbox'], detection['bbox'])
        
        return iou_matrix
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Вычисляет IoU между двумя bbox"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Вычисляем пересечение
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _associate_tracks(self, iou_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """Ассоциирует треки с детекциями на основе IoU"""
        matched_tracks = []
        matched_detections = []
        
        # Простой жадный алгоритм ассоциации
        while True:
            # Находим максимальный IoU
            max_iou = 0
            max_track_idx = -1
            max_det_idx = -1
            
            for i in range(iou_matrix.shape[0]):
                for j in range(iou_matrix.shape[1]):
                    if i not in matched_tracks and j not in matched_detections:
                        if iou_matrix[i, j] > max_iou:
                            max_iou = iou_matrix[i, j]
                            max_track_idx = i
                            max_det_idx = j
            
            # Если максимальный IoU меньше порога, прекращаем
            if max_iou < self.iou_threshold:
                break
            
            matched_tracks.append(max_track_idx)
            matched_detections.append(max_det_idx)
        
        return matched_tracks, matched_detections


class StoneTracker:
    """Основной класс для трекинга камней"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        """
        Инициализация трекера камней
        
        Args:
            model_path: Путь к модели YOLO
            conf_threshold: Порог уверенности
        """
        self.detector = StoneDetector(model_path, conf_threshold)
        self.tracker = SimpleTracker()
        self.frame_count = 0
    
    def track_stones_in_video(self, video_path: str, homography_matrix: np.ndarray = None) -> List[Dict]:
        """
        Трекинг камней в видео
        
        Args:
            video_path: Путь к видео
            homography_matrix: Матрица гомографии для преобразования координат
            
        Returns:
            Список треков камней
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")
        
        all_tracks = []
        self.frame_count = 0
        
        print("Начинаю трекинг камней...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Обнаруживаем камни
            detections = self.detector.detect_stones(frame)
            
            # Обновляем треки
            tracks = self.tracker.update(detections)
            
            # Сохраняем информацию о треках
            for track in tracks:
                bbox = track['bbox']
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                track_data = {
                    "track_id": track['track_id'],
                    "frame": self.frame_count,
                    "bbox": bbox,
                    "center": [center_x, center_y],
                    "team": track['team'],
                    "score": track['score']
                }
                
                # Преобразуем в реальные координаты если есть гомография
                if homography_matrix is not None:
                    pixel_point = np.array([[[center_x, center_y]]], dtype=np.float32)
                    real_point = cv2.perspectiveTransform(pixel_point, homography_matrix)
                    rx, ry = real_point[0][0]
                    track_data["center_real"] = [float(rx), float(ry)]
                
                all_tracks.append(track_data)
            
            self.frame_count += 1
            
            if self.frame_count % 30 == 0:
                print(f"Обработано {self.frame_count} кадров...")
        
        cap.release()
        
        # Группируем треки по track_id
        tracks_dict = {}
        for track_data in all_tracks:
            track_id = track_data["track_id"]
            if track_id not in tracks_dict:
                tracks_dict[track_id] = {
                    "track_id": track_id,
                    "team": track_data["team"],
                    "positions": [],
                    "positions_real": []
                }
            
            tracks_dict[track_id]["positions"].append({
                "frame": track_data["frame"],
                "bbox": track_data["bbox"],
                "center": track_data["center"],
                "score": track_data["score"]
            })
            
            if "center_real" in track_data:
                tracks_dict[track_id]["positions_real"].append({
                    "frame": track_data["frame"],
                    "x": track_data["center_real"][0],
                    "y": track_data["center_real"][1]
                })
        
        return list(tracks_dict.values())


def create_stone_tracker(model_path: str = None) -> StoneTracker:
    """
    Создает трекер камней
    
    Args:
        model_path: Путь к модели (опционально)
        
    Returns:
        Экземпляр StoneTracker
    """
    return StoneTracker(model_path)
