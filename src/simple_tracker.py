"""
Упрощенный трекер для демонстрации ByteTrack концепций
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2


class SimpleTrack:
    """Простое представление трека"""
    
    def __init__(self, bbox, score, track_id=None, team="unknown"):
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.score = score
        self.track_id = track_id
        self.team = team
        self.state = 'New'
        self.frame_id = 0
        self.history = [bbox.copy()]
        
    @property
    def center(self):
        """Центр bbox"""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def update(self, new_bbox, score, frame_id):
        """Обновить трек новым bbox"""
        self.bbox = np.array(new_bbox, dtype=np.float32)
        self.score = score
        self.frame_id = frame_id
        self.state = 'Tracked'
        self.history.append(new_bbox.copy())
    
    def predict(self):
        """Простое предсказание на основе истории"""
        if len(self.history) >= 2:
            # Простое линейное предсказание
            last_bbox = self.history[-1]
            prev_bbox = self.history[-2]
            
            # Вычисляем смещение
            dx = last_bbox[0] - prev_bbox[0]
            dy = last_bbox[1] - prev_bbox[1]
            
            # Предсказываем следующую позицию
            predicted_bbox = last_bbox + np.array([dx, dy, dx, dy])
            self.bbox = predicted_bbox


class SimpleByteTracker:
    """Упрощенная версия ByteTracker"""
    
    def __init__(self, 
                 track_thresh=0.5,
                 match_thresh=0.3,
                 max_disappeared=10):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_disappeared = max_disappeared
        
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0
        
    def update(self, detections: List[Dict]) -> List[SimpleTrack]:
        """Обновить трекер с новыми детекциями"""
        self.frame_id += 1
        
        # Фильтруем детекции по порогу
        valid_detections = [d for d in detections if d['score'] >= self.track_thresh]
        
        if len(valid_detections) == 0:
            # Нет детекций - помечаем все треки как потерянные
            for track in self.tracks:
                track.state = 'Lost'
            return self.tracks
        
        if len(self.tracks) == 0:
            # Первый кадр - создаем новые треки
            for det in valid_detections:
                track = SimpleTrack(
                    bbox=det['bbox'],
                    score=det['score'],
                    track_id=self.next_id,
                    team=det.get('team', 'unknown')
                )
                track.frame_id = self.frame_id
                self.tracks.append(track)
                self.next_id += 1
            return self.tracks
        
        # Предсказываем позиции существующих треков
        for track in self.tracks:
            track.predict()
        
        # Сопоставляем детекции с треками
        matched_tracks, matched_detections, unmatched_tracks, unmatched_detections = \
            self._associate_detections_to_tracks(valid_detections, self.tracks)
        
        # Обновляем сопоставленные треки
        for track, det in zip(matched_tracks, matched_detections):
            track.update(det['bbox'], det['score'], self.frame_id)
        
        # Создаем новые треки для несопоставленных детекций
        for det in unmatched_detections:
            track = SimpleTrack(
                bbox=det['bbox'],
                score=det['score'],
                track_id=self.next_id,
                team=det.get('team', 'unknown')
            )
            track.frame_id = self.frame_id
            self.tracks.append(track)
            self.next_id += 1
        
        # Удаляем старые треки
        self.tracks = [t for t in self.tracks if self.frame_id - t.frame_id <= self.max_disappeared]
        
        return self.tracks
    
    def _associate_detections_to_tracks(self, detections, tracks):
        """Сопоставить детекции с треками на основе IoU"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], [], tracks, detections
        
        # Вычисляем матрицу IoU
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                iou_matrix[i, j] = self._compute_iou(det['bbox'], track.bbox)
        
        # Простое жадное сопоставление
        matched_tracks = []
        matched_detections = []
        unmatched_tracks = list(tracks)
        unmatched_detections = list(detections)
        
        used_detections = set()
        used_tracks = set()
        
        # Находим лучшие совпадения
        for i in range(len(detections)):
            best_iou = 0
            best_track_idx = -1
            
            for j in range(len(tracks)):
                if j in used_tracks:
                    continue
                    
                iou = iou_matrix[i, j]
                if iou > best_iou and iou >= self.match_thresh:
                    best_iou = iou
                    best_track_idx = j
            
            if best_track_idx != -1:
                matched_detections.append(detections[i])
                matched_tracks.append(tracks[best_track_idx])
                used_detections.add(i)
                used_tracks.add(best_track_idx)
        
        # Обновляем списки несопоставленных
        unmatched_detections = [det for i, det in enumerate(detections) if i not in used_detections]
        unmatched_tracks = [track for i, track in enumerate(tracks) if i not in used_tracks]
        
        return matched_tracks, matched_detections, unmatched_tracks, unmatched_detections
    
    def _compute_iou(self, box1, box2):
        """Вычислить IoU между двумя bbox"""
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


def track_stones_with_simple_tracker(video_path: str, homography_matrix: np.ndarray) -> List[Dict]:
    """
    Трекинг камней с упрощенным трекером
    
    Args:
        video_path: Путь к видео
        homography_matrix: Матрица гомографии
        
    Returns:
        Список треков камней
    """
    # Создаем трекер
    tracker = SimpleByteTracker(
        track_thresh=0.5,
        match_thresh=0.3,
        max_disappeared=10
    )
    
    # Симулируем детекции для демонстрации
    print("Создание демонстрационных данных...")
    
    # Создаем синтетические треки камней
    stone_tracks = []
    
    # Камень 1: красный, движется по прямой
    track1 = {
        "stone_id": 1,
        "team": "red",
        "positions_px": [],
        "positions_real": []
    }
    
    for frame in range(30):
        x = 100 + frame * 5
        y = 200 + frame * 2
        track1["positions_px"].append({
            "frame": frame,
            "x": x,
            "y": y
        })
        
        # Преобразуем в реальные координаты
        if homography_matrix is not None:
            pixel_point = np.array([[[x, y]]], dtype=np.float32)
            real_point = cv2.perspectiveTransform(pixel_point, homography_matrix)
            rx, ry = real_point[0][0]
            track1["positions_real"].append({
                "frame": frame,
                "x": float(rx),
                "y": float(ry)
            })
    
    # Камень 2: желтый, движется по дуге
    track2 = {
        "stone_id": 2,
        "team": "yellow",
        "positions_px": [],
        "positions_real": []
    }
    
    for frame in range(10, 40):
        x = 150 + (frame - 10) * 4
        y = 150 + int(20 * np.sin((frame - 10) * 0.2))
        track2["positions_px"].append({
            "frame": frame,
            "x": x,
            "y": y
        })
        
        if homography_matrix is not None:
            pixel_point = np.array([[[x, y]]], dtype=np.float32)
            real_point = cv2.perspectiveTransform(pixel_point, homography_matrix)
            rx, ry = real_point[0][0]
            track2["positions_real"].append({
                "frame": frame,
                "x": float(rx),
                "y": float(ry)
            })
    
    stone_tracks = [track1, track2]
    
    # Добавляем метаданные
    for track in stone_tracks:
        if len(track["positions_px"]) > 0:
            track["throw_start_frame"] = track["positions_px"][0]["frame"]
            track["throw_end_frame"] = track["positions_px"][-1]["frame"]
            if len(track["positions_real"]) > 0:
                track["final_position_real"] = track["positions_real"][-1]
    
    print(f"✅ Создано {len(stone_tracks)} треков камней")
    
    return stone_tracks
