#!/usr/bin/env python3
"""
Скрипт для тестирования отдельных компонентов пайплайна
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Импортируем автономные модули
from curling_pipeline import CurlingPipeline
from keypoints_detector import KeypointsDetector
from homography_utils import HomographyCalculator


def test_keypoints_detection():
    """Тестирует обнаружение ключевых точек"""
    print("=" * 50)
    print("ТЕСТ: Обнаружение ключевых точек")
    print("=" * 50)
    
    # Путь к первому кадру
    first_frame_path = Path("../outputs/pipeline/first_frame.jpg")
    
    if not first_frame_path.exists():
        print("Ошибка: Файл first_frame.jpg не найден")
        print("Сначала запустите полный пайплайн или создайте первый кадр вручную")
        return False
    
    try:
        # Загружаем изображение
        frame = cv2.imread(str(first_frame_path))
        if frame is None:
            print("Ошибка: Не удалось загрузить изображение")
            return False
        
        print(f"Загружено изображение: {frame.shape}")
        
        # Создаем детектор ключевых точек
        detector = KeypointsDetector()
        
        # Обнаруживаем ключевые точки
        keypoints = detector.detect_keypoints(frame)
        print(f"Найдено ключевых точек: {len(keypoints)}")
        
        # Выводим информацию о найденных точках
        for i, kp in enumerate(keypoints):
            if len(kp) == 3:
                x, y, idx = kp
                print(f"  Точка {i}: ({x:.1f}, {y:.1f}), индекс: {idx}")
            else:
                x, y = kp
                print(f"  Точка {i}: ({x:.1f}, {y:.1f})")
        
        # Создаем визуализацию
        output_path = Path("../outputs/pipeline/test_keypoints.jpg")
        detector.visualize_keypoints(frame, keypoints, str(output_path))
        print(f"Визуализация сохранена: {output_path}")
        
        return len(keypoints) > 0
        
    except Exception as e:
        print(f"Ошибка при тестировании обнаружения ключевых точек: {e}")
        return False


def test_homography_computation():
    """Тестирует вычисление матрицы гомографии"""
    print("=" * 50)
    print("ТЕСТ: Вычисление матрицы гомографии")
    print("=" * 50)
    
    try:
        # Создаем тестовые ключевые точки (симулируем обнаруженные)
        # Используем несколько ключевых точек для тестирования
        test_keypoints = [
            (100, 200, 0),   # l_hog
            (400, 200, 1),   # r_hog
            (100, 400, 2),   # l_back
            (400, 400, 3),   # r_back
            (250, 300, 8),   # center
        ]
        
        print(f"Тестовые ключевые точки: {len(test_keypoints)}")
        for kp in test_keypoints:
            print(f"  ({kp[0]}, {kp[1]}), индекс: {kp[2]}")
        
        # Создаем калькулятор гомографии
        calculator = HomographyCalculator()
        
        # Вычисляем матрицу гомографии
        homography_matrix = calculator.compute_homography_matrix(test_keypoints)
        
        if homography_matrix is not None:
            print("Матрица гомографии вычислена успешно!")
            print(f"Размер матрицы: {homography_matrix.shape}")
            print("Матрица:")
            print(homography_matrix)
            
            # Сохраняем матрицу
            output_path = Path("../outputs/pipeline/test_homography.npy")
            np.save(str(output_path), homography_matrix)
            print(f"Матрица сохранена: {output_path}")
            
            return True
        else:
            print("Ошибка: Не удалось вычислить матрицу гомографии")
            return False
            
    except Exception as e:
        print(f"Ошибка при тестировании вычисления гомографии: {e}")
        return False


def test_video_processing():
    """Тестирует обработку видео"""
    print("=" * 50)
    print("ТЕСТ: Обработка видео")
    print("=" * 50)
    
    video_path = Path("../data/videos/example_4.mp4")
    
    if not video_path.exists():
        print(f"Ошибка: Видео файл не найден: {video_path}")
        return False
    
    try:
        # Создаем пайплайн
        pipeline = CurlingPipeline()
        
        # Извлекаем первый кадр
        print("Извлечение первого кадра...")
        first_frame = pipeline.extract_first_frame(str(video_path))
        print(f"Первый кадр извлечен: {first_frame.shape}")
        
        # Обнаруживаем ключевые точки
        print("Обнаружение ключевых точек...")
        keypoints = pipeline.detect_keypoints_on_frame(first_frame)
        print(f"Найдено ключевых точек: {len(keypoints)}")
        
        if len(keypoints) < 4:
            print("Предупреждение: Найдено менее 4 ключевых точек, гомография может быть неточной")
        
        # Вычисляем гомографию
        print("Вычисление матрицы гомографии...")
        homography = pipeline.compute_homography()
        print("Матрица гомографии вычислена успешно")
        
        return True
        
    except Exception as e:
        print(f"Ошибка при тестировании обработки видео: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Тестирует полный пайплайн"""
    print("=" * 50)
    print("ТЕСТ: Полный пайплайн")
    print("=" * 50)
    
    video_path = Path("../data/videos/example_4.mp4")
    
    if not video_path.exists():
        print(f"Ошибка: Видео файл не найден: {video_path}")
        return False
    
    try:
        # Создаем пайплайн
        pipeline = CurlingPipeline()
        
        # Запускаем полный пайплайн
        results = pipeline.run_full_pipeline(str(video_path))
        
        print("Полный пайплайн выполнен успешно!")
        print("Созданные файлы:")
        for key, path in results.items():
            if os.path.exists(path):
                print(f"  ✓ {key}: {path}")
            else:
                print(f"  ✗ {key}: {path} (файл не найден)")
        
        return True
        
    except Exception as e:
        print(f"Ошибка при тестировании полного пайплайна: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запускает все тесты"""
    print("ЗАПУСК ТЕСТОВ ПАЙПЛАЙНА ОБРАБОТКИ ВИДЕО КЁРЛИНГА")
    print("=" * 60)
    
    # Создаем выходную директорию
    output_dir = Path("../outputs/pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tests = [
        ("Обнаружение ключевых точек", test_keypoints_detection),
        ("Вычисление матрицы гомографии", test_homography_computation),
        ("Обработка видео", test_video_processing),
        ("Полный пайплайн", test_full_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nЗапуск теста: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            status = "ПРОЙДЕН" if result else "ПРОВАЛЕН"
            print(f"Тест '{test_name}': {status}")
        except Exception as e:
            print(f"Тест '{test_name}': ОШИБКА - {e}")
            results[test_name] = False
    
    # Итоговый отчет
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ ПРОЙДЕН" if result else "✗ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
    
    print(f"\nВсего тестов: {total}")
    print(f"Пройдено: {passed}")
    print(f"Провалено: {total - passed}")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print(f"\n⚠️  {total - passed} ТЕСТОВ ПРОВАЛЕНО")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
