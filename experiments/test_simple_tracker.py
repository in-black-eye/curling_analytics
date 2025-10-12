#!/usr/bin/env python3
"""
Тестирование упрощенного трекера для кёрлинга
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


def test_simple_tracker():
    """Тестирование упрощенного трекера"""
    
    print("🔍 Тестирование упрощенного трекера...")
    
    # Создаем тестовую матрицу гомографии
    homography_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Тестовый путь к видео (не используется в упрощенной версии)
    video_path = "data/videos/example_4.mp4"
    
    # Получаем треки камней
    stone_tracks = track_stones_with_simple_tracker(video_path, homography_matrix)
    
    print(f"✅ Получено {len(stone_tracks)} треков камней")
    
    # Анализируем результаты
    for i, track in enumerate(stone_tracks):
        print(f"  Трек {i+1}:")
        print(f"    ID: {track.get('stone_id', '?')}")
        print(f"    Команда: {track.get('team', 'unknown')}")
        print(f"    Позиций в пикселях: {len(track.get('positions_px', []))}")
        print(f"    Позиций в реальных координатах: {len(track.get('positions_real', []))}")
        
        if track.get('positions_real'):
            start_pos = track['positions_real'][0]
            end_pos = track['positions_real'][-1]
            print(f"    Начальная позиция: ({start_pos['x']:.1f}, {start_pos['y']:.1f})")
            print(f"    Конечная позиция: ({end_pos['x']:.1f}, {end_pos['y']:.1f})")
    
    return stone_tracks


def create_visualization(stone_tracks):
    """Создает визуализацию треков"""
    
    print("\n🎬 Создание визуализации...")
    
    # Создаем выходную директорию
    output_dir = "outputs/simple_tracker_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем видео с треками
    video_path = create_birdseye_video_with_legend(
        stone_tracks=stone_tracks,
        output_dir=output_dir,
        fps=30,
        court_size_cm=(500, 1300),
        img_size=(800, 2080),
        use_bytetrack=True
    )
    
    print(f"✅ Видео создано: {video_path}")
    return video_path


def compare_tracking_methods():
    """Сравнивает различные методы трекинга"""
    
    print("\n📊 Сравнение методов трекинга...")
    
    # Создаем тестовую матрицу гомографии
    homography_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    video_path = "data/videos/example_4.mp4"
    
    # Тестируем упрощенный трекер
    print("Тестирование упрощенного трекера...")
    simple_tracks = track_stones_with_simple_tracker(video_path, homography_matrix)
    
    print(f"Упрощенный трекер: {len(simple_tracks)} треков")
    
    # Анализируем качество трекинга
    if simple_tracks:
        avg_positions = np.mean([len(track.get('positions_real', [])) for track in simple_tracks])
        print(f"Среднее количество позиций на трек: {avg_positions:.1f}")
        
        # Проверяем стабильность треков
        stable_tracks = 0
        for track in simple_tracks:
            positions = track.get('positions_real', [])
            if len(positions) > 5:  # Трек считается стабильным, если имеет много позиций
                stable_tracks += 1
        
        print(f"Стабильных треков: {stable_tracks}/{len(simple_tracks)}")
    
    return simple_tracks


def main():
    """Основная функция тестирования"""
    
    print("🚀 Тестирование упрощенного трекера для кёрлинга")
    print("=" * 60)
    
    try:
        # Тестируем трекер
        stone_tracks = test_simple_tracker()
        
        # Создаем визуализацию
        video_path = create_visualization(stone_tracks)
        
        # Сравниваем методы
        compare_tracking_methods()
        
        print("\n✅ Тестирование завершено!")
        print(f"\nСозданные файлы:")
        print(f"  - Видео с треками: {video_path}")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
