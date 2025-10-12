#!/usr/bin/env python3
"""
Демонстрационный скрипт для прогонки тестового видео через ByteTrack
с созданием визуализации birds eye view
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from src.simple_tracker import track_stones_with_simple_tracker
from src.animated_visualizations import create_birdseye_video_with_legend
from src.config_loader import load_config


def create_demo_homography_matrix():
    """Создает демонстрационную матрицу гомографии"""
    # Простая матрица гомографии для демонстрации
    # В реальном проекте она должна вычисляться из ключевых точек
    homography_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return homography_matrix


def run_bytetrack_demo():
    """Запускает демонстрацию ByteTrack с тестовым видео"""
    
    print("🚀 Демонстрация ByteTrack для кёрлинга")
    print("=" * 50)
    
    # Загружаем конфигурацию
    try:
        config = load_config()
        print("✅ Конфигурация загружена")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки конфигурации: {e}")
        print("Используем значения по умолчанию")
        config = {
            'paths': {
                'video_data': 'data/videos',
                'output_dir': 'outputs'
            }
        }
    
    # Путь к тестовому видео
    video_path = os.path.join(config['paths']['video_data'], "example_4.mp4")
    
    if not os.path.exists(video_path):
        print(f"❌ Видео не найдено: {video_path}")
        print("Создаем демонстрационные данные...")
        
        # Создаем синтетическое видео для демонстрации
        video_path = create_synthetic_video()
    
    print(f"📹 Используем видео: {video_path}")
    
    # Создаем матрицу гомографии
    print("\n🔧 Создание матрицы гомографии...")
    homography_matrix = create_demo_homography_matrix()
    print("✅ Матрица гомографии создана")
    
    # Запускаем трекинг с ByteTrack
    print("\n🔍 Запуск трекинга камней с ByteTrack...")
    try:
        stone_tracks = track_stones_with_simple_tracker(video_path, homography_matrix)
        print(f"✅ Трекинг завершен. Обнаружено {len(stone_tracks)} камней")
        
        # Выводим информацию о треках
        for i, track in enumerate(stone_tracks):
            print(f"  Камень {i+1}:")
            print(f"    ID: {track.get('stone_id', '?')}")
            print(f"    Команда: {track.get('team', 'unknown')}")
            print(f"    Позиций: {len(track.get('positions_real', []))}")
            
            if track.get('positions_real'):
                start_pos = track['positions_real'][0]
                end_pos = track['positions_real'][-1]
                print(f"    Начальная позиция: ({start_pos['x']:.1f}, {start_pos['y']:.1f})")
                print(f"    Конечная позиция: ({end_pos['x']:.1f}, {end_pos['y']:.1f})")
    
    except Exception as e:
        print(f"❌ Ошибка при трекинге: {e}")
        return
    
    # Создаем визуализацию birds eye view
    print("\n🎬 Создание визуализации birds eye view...")
    try:
        output_dir = config['paths']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        video_path_animated = create_birdseye_video_with_legend(
            stone_tracks=stone_tracks,
            output_dir=output_dir,
            fps=30,
            court_size_cm=(500, 1300),
            img_size=(800, 2080),
            use_bytetrack=True
        )
        
        print(f"✅ Анимированное видео создано: {video_path_animated}")
        
        # Создаем статичную визуализацию
        create_static_visualization(stone_tracks, output_dir)
        
    except Exception as e:
        print(f"❌ Ошибка при создании визуализации: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 Демонстрация завершена успешно!")
    print(f"\nСозданные файлы:")
    print(f"  - Анимированное видео: {video_path_animated}")
    print(f"  - Статичная визуализация: {output_dir}/static_visualization.png")


def create_synthetic_video():
    """Создает синтетическое видео для демонстрации"""
    
    print("🎬 Создание синтетического видео...")
    
    # Параметры видео
    width, height = 800, 600
    fps = 30
    duration = 5  # секунд
    total_frames = fps * duration
    
    # Создаем видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = "data/videos/synthetic_curling_demo.mp4"
    os.makedirs("data/videos", exist_ok=True)
    
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Параметры камней
    stones = [
        {"id": 1, "team": "red", "start_pos": (100, 100), "end_pos": (400, 300), "radius": 20},
        {"id": 2, "team": "yellow", "start_pos": (150, 120), "end_pos": (350, 280), "radius": 20},
        {"id": 3, "team": "red", "start_pos": (200, 140), "end_pos": (300, 260), "radius": 20},
    ]
    
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


def create_static_visualization(stone_tracks, output_dir):
    """Создает статичную визуализацию треков"""
    
    print("📊 Создание статичной визуализации...")
    
    # Параметры изображения
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Рисуем дом
    house_center = (width // 2, height // 2)
    cv2.circle(img, house_center, 100, (0, 0, 255), 2)  # Красный круг
    cv2.circle(img, house_center, 60, (255, 255, 255), 2)  # Белый круг
    cv2.circle(img, house_center, 30, (255, 0, 0), 2)  # Синий круг
    cv2.circle(img, house_center, 10, (0, 0, 0), -1)  # Центр
    
    # Цвета команд
    colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255)
    }
    
    # Рисуем траектории камней
    for track in stone_tracks:
        team = track.get("team", "red")
        color = colors.get(team, (0, 0, 0))
        stone_id = track.get("stone_id", "?")
        
        positions = track.get("positions_real", [])
        if len(positions) > 1:
            # Рисуем траекторию
            points = []
            for pos in positions:
                x = int(pos["x"] * 2)  # Масштабирование для визуализации
                y = int(pos["y"] * 2)
                points.append((x, y))
            
            # Рисуем линию траектории
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i+1], color, 2, cv2.LINE_AA)
            
            # Рисуем точки
            for point in points:
                cv2.circle(img, point, 3, color, -1, cv2.LINE_AA)
            
            # Рисуем начальную и конечную точки
            if points:
                cv2.circle(img, points[0], 8, color, -1, cv2.LINE_AA)
                cv2.circle(img, points[-1], 8, color, -1, cv2.LINE_AA)
                
                # Добавляем ID
                cv2.putText(img, f"ID:{stone_id}", (points[-1][0] + 10, points[-1][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Добавляем заголовок
    cv2.putText(img, "ByteTrack Curling Analysis", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Добавляем информацию о треках
    y_offset = 60
    for i, track in enumerate(stone_tracks):
        team = track.get("team", "red")
        stone_id = track.get("stone_id", "?")
        positions_count = len(track.get("positions_real", []))
        
        text = f"Stone {i+1}: ID={stone_id}, Team={team}, Positions={positions_count}"
        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25
    
    # Сохраняем изображение
    output_path = os.path.join(output_dir, "static_visualization.png")
    cv2.imwrite(output_path, img)
    print(f"✅ Статичная визуализация сохранена: {output_path}")


if __name__ == "__main__":
    run_bytetrack_demo()
