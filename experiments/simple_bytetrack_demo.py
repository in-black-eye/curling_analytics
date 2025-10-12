#!/usr/bin/env python3
"""
Простая демонстрация ByteTrack с ручными детекциями
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from src.bytetrack_tracker import ByteTracker


def create_manual_detections():
    """Создает ручные детекции для демонстрации"""
    
    # Симулируем движение камня по траектории
    detections = []
    
    # Камень 1: движется по прямой
    for frame in range(30):
        x = 100 + frame * 5  # Движение вправо
        y = 200 + frame * 2  # Небольшое движение вниз
        detections.append({
            "frame": frame,
            "detections": [{
                "bbox": [x, y, x + 40, y + 40],
                "score": 0.8,
                "team": "red"
            }]
        })
    
    # Камень 2: появляется на 10 кадре, движется по дуге
    for frame in range(10, 40):
        x = 150 + (frame - 10) * 4
        y = 150 + int(20 * np.sin((frame - 10) * 0.2))  # Дугообразное движение
        detections.append({
            "frame": frame,
            "detections": [{
                "bbox": [x, y, x + 40, y + 40],
                "score": 0.7,
                "team": "yellow"
            }]
        })
    
    # Камень 3: появляется на 20 кадре, движется быстро
    for frame in range(20, 35):
        x = 200 + (frame - 20) * 8
        y = 250
        detections.append({
            "frame": frame,
            "detections": [{
                "bbox": [x, y, x + 40, y + 40],
                "score": 0.9,
                "team": "red"
            }]
        })
    
    return detections


def test_bytetrack_with_manual_data():
    """Тестирует ByteTrack с ручными данными"""
    
    print("🔍 Тестирование ByteTrack с ручными детекциями...")
    
    # Создаем трекер
    tracker = ByteTracker(
        frame_rate=30,
        track_thresh=0.5,
        high_thresh=0.6,
        match_thresh=0.8,
        track_buffer=30
    )
    
    # Получаем ручные детекции
    manual_detections = create_manual_detections()
    
    # Обрабатываем каждый кадр
    results = []
    for frame_data in manual_detections:
        frame_num = frame_data["frame"]
        detections = frame_data["detections"]
        
        # Обновляем трекер
        tracks = tracker.update(detections)
        
        # Сохраняем результаты
        result = {
            "frame": frame_num,
            "num_tracks": len(tracks),
            "track_ids": [t.track_id for t in tracks],
            "track_states": [t.state for t in tracks]
        }
        results.append(result)
        
        print(f"Кадр {frame_num:2d}: {len(tracks)} треков, ID: {[t.track_id for t in tracks]}")
    
    return results


def create_visualization(results):
    """Создает визуализацию результатов"""
    
    print("\n🎬 Создание визуализации...")
    
    # Параметры видео
    width, height = 800, 600
    fps = 30
    
    # Создаем видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "outputs/bytetrack_manual_demo.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Создаем кадры
    for result in results:
        frame_num = result["frame"]
        
        # Создаем белый фон
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Рисуем дом для кёрлинга
        house_center = (width // 2, height // 2)
        cv2.circle(frame, house_center, 100, (0, 0, 255), 2)  # Красный круг
        cv2.circle(frame, house_center, 60, (255, 255, 255), 2)  # Белый круг
        cv2.circle(frame, house_center, 30, (255, 0, 0), 2)  # Синий круг
        cv2.circle(frame, house_center, 10, (0, 0, 0), -1)  # Центр
        
        # Добавляем информацию о треках
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Active Tracks: {result['num_tracks']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        if result['track_ids']:
            track_ids_str = ", ".join(map(str, result['track_ids']))
            cv2.putText(frame, f"Track IDs: {track_ids_str}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Добавляем информацию о состояниях треков
        if result['track_states']:
            states_str = ", ".join(result['track_states'])
            cv2.putText(frame, f"States: {states_str}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Видео создано: {output_path}")


def analyze_results(results):
    """Анализирует результаты трекинга"""
    
    print("\n📊 Анализ результатов ByteTrack:")
    
    # Статистика по трекам
    total_tracks = len(set([track_id for result in results for track_id in result['track_ids']]))
    avg_tracks_per_frame = np.mean([result['num_tracks'] for result in results])
    
    print(f"  Всего уникальных треков: {total_tracks}")
    print(f"  Среднее количество треков на кадр: {avg_tracks_per_frame:.1f}")
    
    # Анализ стабильности
    track_consistency = []
    for i in range(1, len(results)):
        prev_ids = set(results[i-1]['track_ids'])
        curr_ids = set(results[i]['track_ids'])
        if prev_ids or curr_ids:
            consistency = len(prev_ids & curr_ids) / max(len(prev_ids | curr_ids), 1)
            track_consistency.append(consistency)
    
    if track_consistency:
        avg_consistency = np.mean(track_consistency)
        print(f"  Средняя стабильность трекинга: {avg_consistency:.3f}")
    
    # Анализ состояний треков
    all_states = [state for result in results for state in result['track_states']]
    state_counts = {}
    for state in all_states:
        state_counts[state] = state_counts.get(state, 0) + 1
    
    print(f"  Распределение состояний треков:")
    for state, count in state_counts.items():
        print(f"    {state}: {count}")


def main():
    """Основная функция"""
    
    print("🚀 Простая демонстрация ByteTrack")
    print("=" * 40)
    
    try:
        # Тестируем ByteTrack
        results = test_bytetrack_with_manual_data()
        
        # Анализируем результаты
        analyze_results(results)
        
        # Создаем визуализацию
        create_visualization(results)
        
        print("\n✅ Демонстрация завершена!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
