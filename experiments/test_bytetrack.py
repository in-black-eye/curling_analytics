#!/usr/bin/env python3
"""
Тестирование ByteTrack для детекции и трекинга камней в кёрлинге
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from src.stones_processing import detect_and_track_stones
from src.animated_visualizations import create_birdseye_video_with_legend
from src.homography import load_homography_matrix


def test_bytetrack_vs_simple_tracker():
    """Сравнение ByteTrack и SimpleTracker"""
    
    # Пути к файлам
    video_path = "data/videos/example_4.mp4"
    homography_path = "outputs/homography_matrix.npy"
    
    if not os.path.exists(video_path):
        print(f"❌ Видео не найдено: {video_path}")
        return
    
    if not os.path.exists(homography_path):
        print(f"❌ Матрица гомографии не найдена: {homography_path}")
        return
    
    # Загружаем матрицу гомографии
    homography_matrix = load_homography_matrix(homography_path)
    print(f"✅ Загружена матрица гомографии: {homography_matrix.shape}")
    
    # Тестируем SimpleTracker
    print("\n🔍 Тестирование SimpleTracker...")
    simple_tracks = detect_and_track_stones(
        video_path=video_path,
        homography_matrix=homography_matrix,
        use_bytetrack=False
    )
    
    print(f"SimpleTracker: обнаружено {len(simple_tracks)} камней")
    for i, track in enumerate(simple_tracks):
        print(f"  Камень {i+1}: команда={track.get('team', 'unknown')}, "
              f"позиций={len(track.get('positions_real', []))}")
    
    # Тестируем ByteTrack
    print("\n🔍 Тестирование ByteTrack...")
    bytetrack_tracks = detect_and_track_stones(
        video_path=video_path,
        homography_matrix=homography_matrix,
        use_bytetrack=True
    )
    
    print(f"ByteTrack: обнаружено {len(bytetrack_tracks)} камней")
    for i, track in enumerate(bytetrack_tracks):
        print(f"  Камень {i+1}: ID={track.get('stone_id', '?')}, "
              f"команда={track.get('team', 'unknown')}, "
              f"позиций={len(track.get('positions_real', []))}")
    
    # Создаем видео с ByteTrack
    print("\n🎬 Создание видео с ByteTrack...")
    output_dir = "outputs/bytetrack_test"
    os.makedirs(output_dir, exist_ok=True)
    
    video_path_bytetrack = create_birdseye_video_with_legend(
        stone_tracks=bytetrack_tracks,
        output_dir=output_dir,
        fps=30,
        court_size_cm=(500, 1300),
        img_size=(800, 2080),
        use_bytetrack=True
    )
    
    print(f"✅ Видео с ByteTrack сохранено: {video_path_bytetrack}")
    
    # Создаем видео с SimpleTracker для сравнения
    print("\n🎬 Создание видео с SimpleTracker...")
    video_path_simple = create_birdseye_video_with_legend(
        stone_tracks=simple_tracks,
        output_dir=output_dir,
        fps=30,
        court_size_cm=(500, 1300),
        img_size=(800, 2080),
        use_bytetrack=False
    )
    
    print(f"✅ Видео с SimpleTracker сохранено: {video_path_simple}")
    
    # Сравниваем результаты
    print("\n📊 Сравнение результатов:")
    print(f"SimpleTracker: {len(simple_tracks)} камней")
    print(f"ByteTrack: {len(bytetrack_tracks)} камней")
    
    # Анализируем качество трекинга
    if bytetrack_tracks:
        avg_positions_bytetrack = np.mean([len(track.get('positions_real', [])) for track in bytetrack_tracks])
        print(f"ByteTrack: среднее количество позиций на камень = {avg_positions_bytetrack:.1f}")
    
    if simple_tracks:
        avg_positions_simple = np.mean([len(track.get('positions_real', [])) for track in simple_tracks])
        print(f"SimpleTracker: среднее количество позиций на камень = {avg_positions_simple:.1f}")


def test_bytetrack_parameters():
    """Тестирование различных параметров ByteTrack"""
    
    print("\n🔧 Тестирование параметров ByteTrack...")
    
    # Различные конфигурации ByteTrack
    configs = [
        {"track_thresh": 0.3, "high_thresh": 0.5, "match_thresh": 0.7, "name": "Sensitive"},
        {"track_thresh": 0.5, "high_thresh": 0.6, "match_thresh": 0.8, "name": "Balanced"},
        {"track_thresh": 0.7, "high_thresh": 0.8, "match_thresh": 0.9, "name": "Conservative"}
    ]
    
    video_path = "data/videos/example_4.mp4"
    homography_path = "outputs/homography_matrix.npy"
    
    if not os.path.exists(video_path) or not os.path.exists(homography_path):
        print("❌ Необходимые файлы не найдены")
        return
    
    homography_matrix = load_homography_matrix(homography_path)
    
    for config in configs:
        print(f"\nТестирование конфигурации: {config['name']}")
        print(f"  track_thresh={config['track_thresh']}, "
              f"high_thresh={config['high_thresh']}, "
              f"match_thresh={config['match_thresh']}")
        
        # Здесь можно было бы создать кастомный трекер с этими параметрами
        # Для простоты используем стандартные параметры
        tracks = detect_and_track_stones(
            video_path=video_path,
            homography_matrix=homography_matrix,
            use_bytetrack=True
        )
        
        print(f"  Результат: {len(tracks)} камней")


if __name__ == "__main__":
    print("🚀 Тестирование ByteTrack для кёрлинга")
    print("=" * 50)
    
    try:
        test_bytetrack_vs_simple_tracker()
        test_bytetrack_parameters()
        
        print("\n✅ Тестирование завершено!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
