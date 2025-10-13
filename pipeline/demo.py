#!/usr/bin/env python3
"""
Демонстрационный скрипт для пайплайна обработки видео кёрлинга
"""

import os
import sys
import time
from pathlib import Path

# Импортируем автономный пайплайн
from curling_pipeline import CurlingPipeline


def print_banner():
    """Выводит баннер приложения"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🥌 ПАЙПЛАЙН ОБРАБОТКИ ВИДЕО КЁРЛИНГА 🥌           ║
    ║                                                              ║
    ║  Автоматический анализ видео кёрлинга с обнаружением        ║
    ║  ключевых точек поля, трекингом камней и созданием          ║
    ║  визуализаций                                                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_progress(step, total_steps, description):
    """Выводит прогресс выполнения"""
    progress = (step / total_steps) * 100
    bar_length = 40
    filled_length = int(bar_length * step // total_steps)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f"\r[{bar}] {progress:.1f}% - {description}", end='', flush=True)
    
    if step == total_steps:
        print()  # Новая строка в конце


def check_requirements():
    """Проверяет наличие необходимых файлов и зависимостей"""
    print("🔍 Проверка требований...")
    
    # Проверяем наличие видео файла
    video_path = Path("../data/videos/example_4.mp4")
    if not video_path.exists():
        print(f"❌ Видео файл не найден: {video_path}")
        print("   Убедитесь, что файл example_4.mp4 находится в директории data/videos/")
        return False
    else:
        print(f"✅ Видео файл найден: {video_path}")
    
    # Проверяем наличие конфигурации (опционально)
    config_path = Path("../configs/config.yaml")
    if not config_path.exists():
        print(f"⚠️  Файл конфигурации не найден: {config_path}")
        print("   Будет использована конфигурация по умолчанию")
    else:
        print(f"✅ Файл конфигурации найден: {config_path}")
    
    # Проверяем наличие моделей (опционально)
    models_dir = Path("../models")
    if not models_dir.exists():
        print(f"⚠️  Директория моделей не найдена: {models_dir}")
        print("   Будут использованы простые методы детекции")
    else:
        print(f"✅ Директория моделей найдена: {models_dir}")
    
    return True


def run_demo():
    """Запускает демонстрацию пайплайна"""
    print_banner()
    
    # Проверяем требования
    if not check_requirements():
        print("\n❌ Не все требования выполнены. Завершение работы.")
        return False
    
    print("\n🚀 Запуск демонстрации пайплайна...")
    
    # Путь к видео
    video_path = Path("../data/videos/example_4.mp4")
    
    try:
        # Создаем пайплайн
        print("\n📋 Инициализация пайплайна...")
        pipeline = CurlingPipeline()
        
        # Запускаем пайплайн с отслеживанием прогресса
        print("\n⚙️  Выполнение пайплайна:")
        
        start_time = time.time()
        
        # Этапы пайплайна
        steps = [
            "Извлечение первого кадра",
            "Обнаружение ключевых точек",
            "Вычисление матрицы гомографии",
            "Обработка видео с трекингом камней",
            "Создание визуализаций"
        ]
        
        # Выполняем пайплайн
        results = pipeline.run_full_pipeline(str(video_path))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Выводим результаты
        print("\n" + "="*60)
        print("🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("="*60)
        print(f"⏱️  Время выполнения: {duration:.1f} секунд")
        print(f"📁 Результаты сохранены в: {pipeline.pipeline_output_dir}")
        print("\n📄 Созданные файлы:")
        
        for key, path in results.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024 * 1024)  # MB
                print(f"   ✅ {key}: {Path(path).name} ({file_size:.1f} MB)")
            else:
                print(f"   ❌ {key}: {Path(path).name} (файл не найден)")
        
        print("\n🎬 Рекомендуется просмотреть:")
        print(f"   📹 Видео с bbox: {results.get('bbox_video', 'не создано')}")
        print(f"   🗺️  Bird's eye view: {results.get('birdseye_video', 'не создано')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении демонстрации: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_help():
    """Показывает справку по использованию"""
    help_text = """
📖 СПРАВКА ПО ПАЙПЛАЙНУ ОБРАБОТКИ ВИДЕО КЁРЛИНГА

🎯 НАЗНАЧЕНИЕ:
   Автоматический анализ видео кёрлинга с обнаружением ключевых точек поля,
   трекингом камней и созданием визуализаций.

📋 ЭТАПЫ ПАЙПЛАЙНА:
   1. Извлечение первого кадра из видео
   2. Обнаружение 22 ключевых точек поля с помощью YOLO
   3. Вычисление матрицы гомографии для преобразования координат
   4. Обработка видео с обнаружением и трекингом камней (ByteTrack)
   5. Создание двух типов визуализаций:
      - Видео с bounding box поверх оригинального
      - Bird's eye view анимация с траекториями

📁 СТРУКТУРА ПРОЕКТА:
   curling_analytics/
   ├── pipeline/                    # Пайплайн обработки
   │   ├── curling_pipeline.py     # Основной класс пайплайна
   │   ├── run_pipeline.py         # Простой запуск
   │   ├── demo.py                 # Демонстрация (этот файл)
   │   └── test_pipeline_components.py  # Тестирование
   ├── data/videos/                # Видео файлы
   │   └── example_4.mp4           # Основное видео для анализа
   ├── models/                     # Обученные модели
   └── outputs/pipeline/           # Результаты обработки

🚀 ЗАПУСК:
   python demo.py                  # Демонстрация
   python run_pipeline.py          # Быстрый запуск
   python curling_pipeline.py video.mp4  # Запуск с параметрами

📊 ВЫХОДНЫЕ ФАЙЛЫ:
   - first_frame.jpg              # Первый кадр видео
   - keypoints_visualization.jpg  # Визуализация ключевых точек
   - homography_matrix.npy        # Матрица гомографии
   - homography_validation.jpg    # Валидация гомографии
   - stone_tracks.json            # Данные о треках камней
   - video_with_bbox.mp4          # Видео с bbox
   - birdseye_animation_with_legend.mp4  # Bird's eye view

⚙️  КОНФИГУРАЦИЯ:
   Настройки в configs/config.yaml:
   - model_paths.keypoints_detector  # Модель для ключевых точек
   - model_params.confidence_threshold  # Порог уверенности
   - paths.output_dir              # Директория результатов

🔧 ТРЕБОВАНИЯ:
   - Python 3.8+
   - OpenCV, NumPy, Ultralytics YOLO
   - Обученные модели YOLO
   - Видео файл example_4.mp4

❓ ПОДДЕРЖКА:
   При возникновении проблем проверьте:
   1. Наличие всех необходимых файлов
   2. Корректность путей в конфигурации
   3. Доступность моделей
   4. Качество входного видео
    """
    print(help_text)


def main():
    """Основная функция"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        return
    
    success = run_demo()
    
    if success:
        print("\n🎊 Демонстрация завершена успешно!")
        print("   Спасибо за использование пайплайна обработки видео кёрлинга!")
    else:
        print("\n💥 Демонстрация завершена с ошибками.")
        print("   Проверьте логи выше для диагностики проблем.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
