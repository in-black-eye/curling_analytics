#!/usr/bin/env python3
"""
Простой скрипт для запуска пайплайна обработки видео кёрлинга
"""

import os
import sys
from pathlib import Path

# Импортируем автономный пайплайн
from curling_pipeline import CurlingPipeline


def main():
    """Запуск пайплайна для видео example_4.mp4"""
    
    # Путь к видео файлу (относительно папки pipeline)
    video_path = Path("../data/videos/example_4.mp4")
    
    if not video_path.exists():
        print(f"Ошибка: Видео файл не найден: {video_path}")
        print("Убедитесь, что файл example_4.mp4 находится в директории data/videos/")
        sys.exit(1)
    
    print("Запуск пайплайна обработки видео кёрлинга...")
    print(f"Видео: {video_path}")
    
    try:
        # Создаем и запускаем пайплайн
        pipeline = CurlingPipeline()
        results = pipeline.run_full_pipeline(str(video_path))
        
        print("\n" + "="*60)
        print("ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        print("="*60)
        print("Созданные файлы:")
        for key, path in results.items():
            print(f"  {key}: {path}")
        print("="*60)
        
    except Exception as e:
        print(f"Ошибка выполнения пайплайна: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
