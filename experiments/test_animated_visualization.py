#!/usr/bin/env python3
"""
Тестовый скрипт для проверки анимированной визуализации bird's-eye view.
Создает демонстрационные данные камней и генерирует видео.
"""

import sys as _sys
from pathlib import Path as _Path
import numpy as np

# Ensure project root on PYTHONPATH
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from src.animated_visualizations import create_birdseye_video_with_legend


def create_demo_stone_tracks():
    """Создает демонстрационные данные треков камней для тестирования."""
    
    # Демонстрационные траектории камней
    stone_tracks = []
    
    # Камень 1 (красная команда) - движение слева направо
    stone1_track = {
        "stone_id": 1,
        "team": "red",
        "positions_real": [],
        "positions_px": []
    }
    
    # Создаем траекторию движения камня
    for frame in range(0, 60):
        x = 100 + frame * 3  # Движение слева направо
        y = 800 - frame * 2  # Небольшое движение вверх
        stone1_track["positions_real"].append({"frame": frame, "x": x, "y": y})
    
    stone1_track["final_position_real"] = stone1_track["positions_real"][-1]
    stone1_track["throw_start_frame"] = 0
    stone1_track["throw_end_frame"] = 59
    stone_tracks.append(stone1_track)
    
    # Камень 2 (желтая команда) - движение справа налево
    stone2_track = {
        "stone_id": 2,
        "team": "yellow",
        "positions_real": [],
        "positions_px": []
    }
    
    for frame in range(10, 70):
        x = 400 - (frame - 10) * 2.5  # Движение справа налево
        y = 850 - (frame - 10) * 1.5  # Небольшое движение вверх
        stone2_track["positions_real"].append({"frame": frame, "x": x, "y": y})
    
    stone2_track["final_position_real"] = stone2_track["positions_real"][-1]
    stone2_track["throw_start_frame"] = 10
    stone2_track["throw_end_frame"] = 69
    stone_tracks.append(stone2_track)
    
    # Камень 3 (красная команда) - движение к центру дома
    stone3_track = {
        "stone_id": 3,
        "team": "red",
        "positions_real": [],
        "positions_px": []
    }
    
    for frame in range(20, 80):
        # Движение по дуге к центру дома
        angle = (frame - 20) * 0.1
        x = 250 + 50 * np.cos(angle)  # Около центра поля по x
        y = 900 - (frame - 20) * 3   # Движение к центру дома
        stone3_track["positions_real"].append({"frame": frame, "x": x, "y": y})
    
    stone3_track["final_position_real"] = stone3_track["positions_real"][-1]
    stone3_track["throw_start_frame"] = 20
    stone3_track["throw_end_frame"] = 79
    stone_tracks.append(stone3_track)
    
    # Камень 4 (желтая команда) - движение к кнопке
    stone4_track = {
        "stone_id": 4,
        "team": "yellow",
        "positions_real": [],
        "positions_px": []
    }
    
    for frame in range(30, 90):
        # Точное движение к кнопке (центр дома)
        x = 250 + (frame - 30) * 0.5  # Очень точное движение по x
        y = 950 - (frame - 30) * 2.5  # Движение к кнопке
        stone4_track["positions_real"].append({"frame": frame, "x": x, "y": y})
    
    stone4_track["final_position_real"] = stone4_track["positions_real"][-1]
    stone4_track["throw_start_frame"] = 30
    stone4_track["throw_end_frame"] = 89
    stone_tracks.append(stone4_track)
    
    return stone_tracks


def main():
    """Основная функция для тестирования анимированной визуализации."""
    
    print("Создание демонстрационного видео bird's-eye view...")
    
    # Создаем демонстрационные данные
    stone_tracks = create_demo_stone_tracks()
    print(f"Создано {len(stone_tracks)} демонстрационных треков камней")
    
    # Создаем выходную директорию
    output_dir = "outputs/test_animation"
    
    # Генерируем видео
    video_path = create_birdseye_video_with_legend(
        stone_tracks=stone_tracks,
        output_dir=output_dir,
        fps=20,  # Более плавная анимация для демо
        court_size_cm=(500, 1300),
        img_size=(800, 2080)
    )
    
    print(f"Демонстрационное видео создано: {video_path}")
    print("Видео показывает:")
    print("   - Шаблон дома для кёрлинга с четырьмя кругами")
    print("   - Анимированные траектории камней")
    print("   - Легенду с информацией о командах")
    print("   - Счетчик кадров и количества камней")


if __name__ == "__main__":
    main()
