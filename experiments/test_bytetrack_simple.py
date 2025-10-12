#!/usr/bin/env python3
"""
Упрощенное тестирование ByteTrack без GroundingDINO
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from src.bytetrack_tracker import ByteTracker, STrack


def test_bytetrack_basic():
    """Базовое тестирование ByteTracker"""
    
    print("🔍 Тестирование ByteTracker...")
    
    # Создаем трекер
    tracker = ByteTracker(
        frame_rate=30,
        track_thresh=0.5,
        high_thresh=0.6,
        match_thresh=0.8,
        track_buffer=30
    )
    
    # Создаем тестовые детекции
    detections_frame1 = [
        {"bbox": [100, 100, 150, 150], "score": 0.8, "team": "red"},
        {"bbox": [200, 200, 250, 250], "score": 0.7, "team": "yellow"},
    ]
    
    detections_frame2 = [
        {"bbox": [105, 105, 155, 155], "score": 0.75, "team": "red"},  # Сдвинутый камень
        {"bbox": [205, 205, 255, 255], "score": 0.65, "team": "yellow"},  # Сдвинутый камень
        {"bbox": [300, 300, 350, 350], "score": 0.6, "team": "red"},  # Новый камень
    ]
    
    detections_frame3 = [
        {"bbox": [110, 110, 160, 160], "score": 0.7, "team": "red"},  # Продолжение трека
        {"bbox": [210, 210, 260, 260], "score": 0.6, "team": "yellow"},  # Продолжение трека
        # Третий камень исчез
    ]
    
    # Тестируем трекинг
    print("Кадр 1:")
    tracks1 = tracker.update(detections_frame1)
    print(f"  Активных треков: {len(tracks1)}")
    for track in tracks1:
        print(f"    ID: {track.track_id}, команда: {getattr(track, 'team', 'unknown')}, "
              f"состояние: {track.state}")
    
    print("Кадр 2:")
    tracks2 = tracker.update(detections_frame2)
    print(f"  Активных треков: {len(tracks2)}")
    for track in tracks2:
        print(f"    ID: {track.track_id}, команда: {getattr(track, 'team', 'unknown')}, "
              f"состояние: {track.state}")
    
    print("Кадр 3:")
    tracks3 = tracker.update(detections_frame3)
    print(f"  Активных треков: {len(tracks3)}")
    for track in tracks3:
        print(f"    ID: {track.track_id}, команда: {getattr(track, 'team', 'unknown')}, "
              f"состояние: {track.state}")
    
    print("✅ Базовое тестирование ByteTracker завершено")


def test_strack():
    """Тестирование STrack"""
    
    print("\n🔍 Тестирование STrack...")
    
    # Создаем трек
    bbox = [100, 100, 200, 200]
    score = 0.8
    track = STrack(bbox, score, track_id=1)
    track.team = "red"
    
    print(f"Создан трек: ID={track.track_id}, команда={track.team}")
    print(f"  bbox: {track.tlwh}")
    print(f"  xyxy: {track.xyxy}")
    print(f"  состояние: {track.state}")
    
    # Предсказание
    track.predict()
    print(f"После предсказания: bbox={track.tlwh}")
    
    # Обновление
    new_bbox = [105, 105, 205, 205]
    new_score = 0.75
    new_track = STrack(new_bbox, new_score, track_id=1)
    new_track.team = "red"
    
    track.update(new_track, frame_id=1)
    print(f"После обновления: bbox={track.tlwh}, состояние={track.state}")
    
    print("✅ Тестирование STrack завершено")


def test_iou_calculation():
    """Тестирование расчета IoU"""
    
    print("\n🔍 Тестирование расчета IoU...")
    
    tracker = ByteTracker()
    
    # Тестовые bbox в формате tlbr
    box1 = [100, 100, 200, 200]  # [x1, y1, x2, y2]
    box2 = [110, 110, 210, 210]  # Пересекающиеся
    box3 = [300, 300, 400, 400]  # Не пересекающиеся
    
    iou_12 = tracker._compute_iou(box1, box2)
    iou_13 = tracker._compute_iou(box1, box3)
    
    print(f"IoU между пересекающимися bbox: {iou_12:.3f}")
    print(f"IoU между непересекающимися bbox: {iou_13:.3f}")
    
    # Проверяем правильность расчета
    expected_iou_12 = 0.8  # Примерное ожидаемое значение
    if iou_12 > 0.5:
        print("✅ IoU для пересекающихся bbox рассчитан корректно")
    else:
        print("❌ IoU для пересекающихся bbox рассчитан некорректно")
    
    if iou_13 == 0.0:
        print("✅ IoU для непересекающихся bbox рассчитан корректно")
    else:
        print("❌ IoU для непересекающихся bbox рассчитан некорректно")


if __name__ == "__main__":
    print("🚀 Упрощенное тестирование ByteTrack")
    print("=" * 50)
    
    try:
        test_strack()
        test_iou_calculation()
        test_bytetrack_basic()
        
        print("\n✅ Все тесты завершены успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
