#!/usr/bin/env python3
"""
Демонстрация ByteTrack vs SimpleTracker для кёрлинга
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from src.bytetrack_tracker import ByteTracker, STrack


def create_synthetic_curling_data():
    """Создает синтетические данные для демонстрации трекинга"""
    
    # Параметры видео
    width, height = 800, 600
    fps = 30
    duration = 5  # секунд
    total_frames = fps * duration
    
    # Создаем видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = "outputs/synthetic_curling_demo.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Параметры камней
    stones = [
        {"id": 1, "team": "red", "start_pos": (100, 100), "end_pos": (400, 300), "radius": 20},
        {"id": 2, "team": "yellow", "start_pos": (150, 120), "end_pos": (350, 280), "radius": 20},
        {"id": 3, "team": "red", "start_pos": (200, 140), "end_pos": (300, 260), "radius": 20},
    ]
    
    print(f"Создание синтетического видео: {total_frames} кадров")
    
    for frame_num in range(total_frames):
        # Создаем белый фон
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Рисуем дом для кёрлинга
        house_center = (width // 2, height // 2)
        cv2.circle(frame, house_center, 100, (0, 0, 255), 2)  # Красный круг
        cv2.circle(frame, house_center, 60, (255, 255, 255), 2)  # Белый круг
        cv2.circle(frame, house_center, 30, (255, 0, 0), 2)  # Синий круг
        cv2.circle(frame, house_center, 10, (0, 0, 0), -1)  # Центр
        
        # Анимируем камни
        for stone in stones:
            # Линейная интерполяция позиции
            t = frame_num / (total_frames - 1)
            x = int(stone["start_pos"][0] + t * (stone["end_pos"][0] - stone["start_pos"][0]))
            y = int(stone["start_pos"][1] + t * (stone["end_pos"][1] - stone["start_pos"][1]))
            
            # Добавляем небольшой шум для реалистичности
            noise_x = np.random.randint(-2, 3)
            noise_y = np.random.randint(-2, 3)
            x += noise_x
            y += noise_y
            
            # Цвет камня
            color = (0, 0, 255) if stone["team"] == "red" else (0, 255, 255)
            
            # Рисуем камень
            cv2.circle(frame, (x, y), stone["radius"], color, -1)
            cv2.circle(frame, (x, y), stone["radius"] + 2, (255, 255, 255), 2)
            
            # Добавляем ID
            cv2.putText(frame, f"ID:{stone['id']}", (x + 25, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Добавляем информацию о кадре
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Синтетическое видео создано: {video_path}")
    return video_path


def simulate_detections_from_video(video_path):
    """Симулирует детекции из видео для тестирования трекеров"""
    
    cap = cv2.VideoCapture(video_path)
    detections_per_frame = []
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Простая детекция камней по цвету
        detections = []
        
        # Ищем красные камни
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Минимальный размер
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "score": 0.8,
                    "team": "red"
                })
        
        # Ищем желтые камни
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "score": 0.8,
                    "team": "yellow"
                })
        
        detections_per_frame.append(detections)
        frame_num += 1
        
        if frame_num % 30 == 0:
            print(f"Обработано {frame_num} кадров...")
    
    cap.release()
    return detections_per_frame


def compare_trackers(detections_per_frame):
    """Сравнивает SimpleTracker и ByteTracker"""
    
    print("\n🔍 Сравнение трекеров...")
    
    # Тестируем ByteTracker
    print("Тестирование ByteTracker...")
    bytetracker = ByteTracker(
        frame_rate=30,
        track_thresh=0.5,
        high_thresh=0.6,
        match_thresh=0.8,
        track_buffer=30
    )
    
    bytetrack_results = []
    for frame_num, detections in enumerate(detections_per_frame):
        tracks = bytetracker.update(detections)
        bytetrack_results.append({
            "frame": frame_num,
            "tracks": len(tracks),
            "track_ids": [t.track_id for t in tracks]
        })
    
    # Анализируем результаты
    total_tracks_bytetrack = len(set([t["track_ids"] for t in bytetrack_results if t["track_ids"]]))
    avg_tracks_per_frame = np.mean([t["tracks"] for t in bytetrack_results])
    
    print(f"ByteTracker результаты:")
    print(f"  Всего уникальных треков: {total_tracks_bytetrack}")
    print(f"  Среднее количество треков на кадр: {avg_tracks_per_frame:.1f}")
    
    # Показываем стабильность трекинга
    track_consistency = []
    for i in range(1, len(bytetrack_results)):
        prev_ids = set(bytetrack_results[i-1]["track_ids"])
        curr_ids = set(bytetrack_results[i]["track_ids"])
        consistency = len(prev_ids & curr_ids) / max(len(prev_ids | curr_ids), 1)
        track_consistency.append(consistency)
    
    avg_consistency = np.mean(track_consistency)
    print(f"  Средняя стабильность трекинга: {avg_consistency:.3f}")
    
    return bytetrack_results


def create_comparison_visualization(bytetrack_results, video_path):
    """Создает визуализацию результатов трекинга"""
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Создаем выходное видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "outputs/bytetrack_demo_result.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num < len(bytetrack_results):
            result = bytetrack_results[frame_num]
            
            # Добавляем информацию о треках
            cv2.putText(frame, f"ByteTracker - Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Active Tracks: {result['tracks']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Показываем ID треков
            if result['track_ids']:
                track_ids_str = ", ".join(map(str, result['track_ids']))
                cv2.putText(frame, f"Track IDs: {track_ids_str}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        out.write(frame)
        frame_num += 1
    
    cap.release()
    out.release()
    print(f"✅ Результат трекинга сохранен: {output_path}")


def main():
    """Основная функция демонстрации"""
    
    print("🚀 Демонстрация ByteTrack для кёрлинга")
    print("=" * 50)
    
    try:
        # Создаем синтетические данные
        print("1. Создание синтетических данных...")
        video_path = create_synthetic_curling_data()
        
        # Симулируем детекции
        print("\n2. Симуляция детекций...")
        detections_per_frame = simulate_detections_from_video(video_path)
        print(f"   Обработано {len(detections_per_frame)} кадров")
        
        # Сравниваем трекеры
        print("\n3. Тестирование трекеров...")
        bytetrack_results = compare_trackers(detections_per_frame)
        
        # Создаем визуализацию
        print("\n4. Создание визуализации...")
        create_comparison_visualization(bytetrack_results, video_path)
        
        print("\n✅ Демонстрация завершена!")
        print("\nСозданные файлы:")
        print(f"  - Синтетическое видео: {video_path}")
        print(f"  - Результат трекинга: outputs/bytetrack_demo_result.mp4")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
