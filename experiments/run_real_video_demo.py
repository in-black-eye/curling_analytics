#!/usr/bin/env python3
"""
Демонстрационный скрипт для прогонки реального видео через ByteTrack
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


def run_real_video_demo():
    """Запускает демонстрацию с реальным видео"""
    
    print("🚀 Демонстрация ByteTrack с реальным видео")
    print("=" * 50)
    
    # Загружаем конфигурацию
    try:
        config = load_config()
        print("✅ Конфигурация загружена")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки конфигурации: {e}")
        return
    
    # Путь к видео
    video_path = os.path.join(config['paths']['video_data'], "example_4.mp4")
    
    if not os.path.exists(video_path):
        print(f"❌ Видео не найдено: {video_path}")
        print("Попробуйте запустить: python3 experiments/run_bytetrack_demo.py")
        return
    
    print(f"📹 Используем видео: {video_path}")
    
    # Проверяем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Не удалось открыть видео")
        return
    
    # Получаем информацию о видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"📊 Информация о видео:")
    print(f"  Разрешение: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Кадров: {frame_count}")
    print(f"  Длительность: {duration:.1f} секунд")
    
    cap.release()
    
    # Создаем матрицу гомографии
    print("\n🔧 Создание матрицы гомографии...")
    homography_matrix = create_demo_homography_matrix()
    print("✅ Матрица гомографии создана")
    
    # Запускаем трекинг
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
        import traceback
        traceback.print_exc()
        return
    
    # Создаем визуализацию
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
        
        # Создаем сводку результатов
        create_results_summary(stone_tracks, output_dir, video_path)
        
    except Exception as e:
        print(f"❌ Ошибка при создании визуализации: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 Демонстрация завершена успешно!")
    print(f"\nСозданные файлы:")
    print(f"  - Анимированное видео: {video_path_animated}")
    print(f"  - Статичная визуализация: {output_dir}/static_visualization.png")
    print(f"  - Сводка результатов: {output_dir}/results_summary.txt")


def create_demo_homography_matrix():
    """Создает демонстрационную матрицу гомографии"""
    # Простая матрица гомографии для демонстрации
    homography_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return homography_matrix


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


def create_results_summary(stone_tracks, output_dir, video_path):
    """Создает сводку результатов анализа"""
    
    print("📋 Создание сводки результатов...")
    
    summary_path = os.path.join(output_dir, "results_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CURLING ANALYSIS RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Video: {video_path}\n")
        f.write(f"Analysis Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total Stones Detected: {len(stone_tracks)}\n\n")
        
        # Статистика по командам
        red_stones = [t for t in stone_tracks if t.get("team") == "red"]
        yellow_stones = [t for t in stone_tracks if t.get("team") == "yellow"]
        
        f.write("Team Statistics:\n")
        f.write(f"  Red Team: {len(red_stones)} stones\n")
        f.write(f"  Yellow Team: {len(yellow_stones)} stones\n\n")
        
        # Детальная информация о каждом камне
        f.write("Stone Details:\n")
        f.write("-" * 30 + "\n")
        
        for i, track in enumerate(stone_tracks):
            f.write(f"\nStone {i+1}:\n")
            f.write(f"  ID: {track.get('stone_id', '?')}\n")
            f.write(f"  Team: {track.get('team', 'unknown')}\n")
            f.write(f"  Positions: {len(track.get('positions_real', []))}\n")
            
            if track.get('positions_real'):
                start_pos = track['positions_real'][0]
                end_pos = track['positions_real'][-1]
                f.write(f"  Start Position: ({start_pos['x']:.2f}, {start_pos['y']:.2f})\n")
                f.write(f"  End Position: ({end_pos['x']:.2f}, {end_pos['y']:.2f})\n")
                
                # Вычисляем расстояние
                distance = np.sqrt((end_pos['x'] - start_pos['x'])**2 + 
                                 (end_pos['y'] - start_pos['y'])**2)
                f.write(f"  Distance Traveled: {distance:.2f} cm\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Analysis completed successfully!\n")
    
    print(f"✅ Сводка результатов сохранена: {summary_path}")


if __name__ == "__main__":
    run_real_video_demo()
