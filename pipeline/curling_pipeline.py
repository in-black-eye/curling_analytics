#!/usr/bin/env python3
"""
Автономный пайплайн обработки видео кёрлинга:
1. Загрузка видео и извлечение первого кадра
2. Обнаружение ключевых точек поля
3. Расчет матрицы гомографии
4. Обработка видео с обнаружением и трекингом камней
5. Создание визуализаций

Все необходимые модули включены в папку pipeline.
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import argparse

# Импортируем автономные модули из той же папки
from keypoints_detector import KeypointsDetector
from homography_utils import HomographyCalculator
from stone_tracker import StoneTracker
from visualization import CurlingVisualizer
from config import PipelineConfig


class CurlingPipeline:
    """Основной класс для пайплайна обработки видео кёрлинга"""
    
    def __init__(self, config_path: str = None):
        """Инициализация пайплайна с конфигурацией"""
        # Загружаем конфигурацию
        self.config = PipelineConfig(config_path)
        
        # Создаем выходную директорию
        self.output_dir = Path(self.config.get_output_dir())
        self.pipeline_output_dir = self.output_dir / "pipeline"
        self.pipeline_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем компоненты пайплайна
        self.keypoints_detector = KeypointsDetector(
            model_path=self.config.get_model_path("keypoints_detector"),
            conf_threshold=self.config.get_keypoints_confidence_threshold()
        )
        
        self.homography_calculator = HomographyCalculator()
        
        self.stone_tracker = StoneTracker(
            model_path=self.config.get_model_path("stones_detector"),
            conf_threshold=self.config.get_confidence_threshold()
        )
        
        self.visualizer = CurlingVisualizer()
        
        # Результаты обработки
        self.first_frame = None
        self.keypoints = []
        self.homography_matrix = None
        self.stone_tracks = []
        
        print(f"Пайплайн инициализирован. Выходная директория: {self.pipeline_output_dir}")
    
    def extract_first_frame(self, video_path: str) -> np.ndarray:
        """Извлекает первый кадр из видео"""
        print(f"Извлечение первого кадра из {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать первый кадр")
        
        cap.release()
        
        # Сохраняем первый кадр
        first_frame_path = self.pipeline_output_dir / "first_frame.jpg"
        cv2.imwrite(str(first_frame_path), frame)
        print(f"Первый кадр сохранен: {first_frame_path}")
        
        self.first_frame = frame
        return frame
    
    def detect_keypoints_on_frame(self, frame: np.ndarray) -> List[Tuple[float, float, int]]:
        """Обнаруживает ключевые точки на кадре"""
        print("Обнаружение ключевых точек...")
        
        # Используем детектор ключевых точек
        keypoints = self.keypoints_detector.detect_keypoints(frame)
        
        print(f"Найдено {len(keypoints)} ключевых точек")
        
        self.keypoints = keypoints
        
        # Визуализируем найденные точки
        if self.keypoints:
            vis_path = self.pipeline_output_dir / "keypoints_visualization.jpg"
            self.keypoints_detector.visualize_keypoints(frame, self.keypoints, str(vis_path))
        
        return self.keypoints
    
    def compute_homography(self) -> np.ndarray:
        """Вычисляет матрицу гомографии на основе найденных ключевых точек"""
        print("Вычисление матрицы гомографии...")
        
        if not self.keypoints:
            raise ValueError("Нет ключевых точек для вычисления гомографии")
        
        # Вычисляем матрицу гомографии
        homography_matrix = self.homography_calculator.compute_homography_matrix(self.keypoints)
        
        if homography_matrix is None:
            raise ValueError("Не удалось вычислить матрицу гомографии")
        
        self.homography_matrix = homography_matrix
        
        # Сохраняем матрицу гомографии
        homography_path = self.pipeline_output_dir / "homography_matrix.npy"
        np.save(str(homography_path), homography_matrix)
        print(f"Матрица гомографии сохранена: {homography_path}")
        
        # Валидируем гомографию
        first_frame_path = self.pipeline_output_dir / "first_frame.jpg"
        validation_path = self.pipeline_output_dir / "homography_validation.jpg"
        self.homography_calculator.validate_homography(homography_matrix, str(first_frame_path), str(validation_path))
        
        return homography_matrix
    
    def process_video_with_stones_detection(self, video_path: str) -> List[Dict]:
        """Обрабатывает видео с обнаружением и трекингом камней"""
        print("Обработка видео с обнаружением камней...")
        
        if self.homography_matrix is None:
            raise ValueError("Матрица гомографии не вычислена")
        
        # Используем трекер камней
        stone_tracks = self.stone_tracker.track_stones_in_video(
            video_path=video_path,
            homography_matrix=self.homography_matrix
        )
        
        self.stone_tracks = stone_tracks
        
        # Сохраняем результаты трекинга
        tracks_path = self.pipeline_output_dir / "stone_tracks.json"
        with open(tracks_path, 'w', encoding='utf-8') as f:
            json.dump(stone_tracks, f, indent=2, ensure_ascii=False)
        print(f"Результаты трекинга сохранены: {tracks_path}")
        
        return stone_tracks
    
    def create_visualizations(self, video_path: str) -> Tuple[str, str]:
        """Создает два типа визуализации: с bbox и bird's eye view"""
        print("Создание визуализаций...")
        
        if not self.stone_tracks:
            raise ValueError("Нет данных о треках камней")
        
        # 1. Создаем видео с bbox поверх оригинального видео
        bbox_video_path = self.pipeline_output_dir / "video_with_bbox.mp4"
        self.visualizer.create_bbox_video(video_path, self.stone_tracks, str(bbox_video_path))
        
        # 2. Создаем bird's eye view анимацию
        birdseye_video_path = self.pipeline_output_dir / "birdseye_animation_with_legend.mp4"
        vis_params = self.config.get_visualization_params()
        self.visualizer.create_birdseye_video(
            self.stone_tracks, 
            str(birdseye_video_path),
            fps=vis_params.get("fps", 30),
            court_size_cm=tuple(vis_params.get("court_size_cm", [500, 1300])),
            img_size=tuple(vis_params.get("img_size", [800, 2080]))
        )
        
        return str(bbox_video_path), str(birdseye_video_path)
    
    
    def run_full_pipeline(self, video_path: str) -> Dict[str, str]:
        """Запускает полный пайплайн обработки"""
        print("=" * 60)
        print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ОБРАБОТКИ ВИДЕО КЁРЛИНГА")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 1. Извлекаем первый кадр
            self.extract_first_frame(video_path)
            
            # 2. Обнаруживаем ключевые точки
            self.detect_keypoints_on_frame(self.first_frame)
            
            # 3. Вычисляем матрицу гомографии
            self.compute_homography()
            
            # 4. Обрабатываем видео с обнаружением камней
            self.process_video_with_stones_detection(video_path)
            
            # 5. Создаем визуализации
            bbox_video, birdseye_video = self.create_visualizations(video_path)
            
            # Результаты
            results = {
                "first_frame": str(self.pipeline_output_dir / "first_frame.jpg"),
                "keypoints_visualization": str(self.pipeline_output_dir / "keypoints_visualization.jpg"),
                "homography_matrix": str(self.pipeline_output_dir / "homography_matrix.npy"),
                "homography_validation": str(self.pipeline_output_dir / "homography_validation.jpg"),
                "stone_tracks": str(self.pipeline_output_dir / "stone_tracks.json"),
                "bbox_video": bbox_video,
                "birdseye_video": birdseye_video
            }
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("=" * 60)
            print("ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
            print(f"Время выполнения: {duration}")
            print("=" * 60)
            print("РЕЗУЛЬТАТЫ:")
            for key, path in results.items():
                print(f"  {key}: {path}")
            print("=" * 60)
            
            return results
            
        except Exception as e:
            print(f"ОШИБКА В ПАЙПЛАЙНЕ: {e}")
            raise


def main():
    """Основная функция для запуска пайплайна из командной строки"""
    parser = argparse.ArgumentParser(description="Пайплайн обработки видео кёрлинга")
    parser.add_argument("video_path", help="Путь к видео файлу")
    parser.add_argument("--config", help="Путь к файлу конфигурации")
    parser.add_argument("--output-dir", help="Директория для сохранения результатов")
    
    args = parser.parse_args()
    
    # Проверяем существование видео файла
    if not os.path.exists(args.video_path):
        print(f"Ошибка: Видео файл не найден: {args.video_path}")
        sys.exit(1)
    
    try:
        # Создаем и запускаем пайплайн
        pipeline = CurlingPipeline(args.config)
        
        if args.output_dir:
            pipeline.pipeline_output_dir = Path(args.output_dir)
            pipeline.pipeline_output_dir.mkdir(exist_ok=True)
        
        results = pipeline.run_full_pipeline(args.video_path)
        
        print("\nПайплайн завершен успешно!")
        print("Все результаты сохранены в директории:", pipeline.pipeline_output_dir)
        
    except Exception as e:
        print(f"Ошибка выполнения пайплайна: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
