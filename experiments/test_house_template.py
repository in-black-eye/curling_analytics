#!/usr/bin/env python3
"""
Тестовый скрипт для проверки нового шаблона дома для кёрлинга.`
Создает статичное изображение с правильными размерами согласно официальным правилам.
"""

import sys as _sys
from pathlib import Path as _Path
import cv2
import numpy as np
import os

# Ensure project root on PYTHONPATH
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from src.animated_visualizations import create_curling_house_template


def main():
    """Создает тестовое изображение шаблона дома для кёрлинга."""
    
    print("Создание тестового шаблона дома для кёрл[инга...")
    
    # Создаем выходную директорию
    output_dir = "outputs/test_house"
    os.makedirs(output_dir, exist_ok=True)
    
    # Тестируем разные масштабы
    scales = [
        (1.0, "1cm_per_px"),      # 1 см на пиксель
        (0.5, "0.5cm_per_px"),    # 0.5 см на пиксель (более детально)
        (2.0, "2cm_per_px"),      # 2 см на пиксель (менее детально)
    ]
    
    for scale_cm_per_px, scale_name in scales:
        print(f"Создание шаблона с масштабом {scale_name}...")
        
        # Создаем шаблон дома
        house_center = (400, 400)  # Центр изображения
        house_template = create_curling_house_template(house_center, scale_cm_per_px)
        
        # Сохраняем изображение
        output_path = os.path.join(output_dir, f"house_template_{scale_name}.png")
        cv2.imwrite(output_path, house_template)
        
        # Выводим информацию о размерах
        height, width = house_template.shape[:2]
        print(f"   Размер изображения: {width}x{height} пикселей")
        
        # Вычисляем размеры кругов в пикселях для этого масштаба
        radius_6ft_px = int(182.9 / scale_cm_per_px)  # 6 футов
        radius_4ft_px = int(121.9 / scale_cm_per_px)  # 4 фута
        radius_2ft_px = int(61.0 / scale_cm_per_px)   # 2 фута
        radius_6in_px = int(15.24 / scale_cm_per_px)  # 6 дюймов
        
        print(f"   Радиусы кругов в пикселях:")
        print(f"     - 6 футов (1,829 м): {radius_6ft_px} px")
        print(f"     - 4 фута (1,219 м): {radius_4ft_px} px")
        print(f"     - 2 фута (0,610 м): {radius_2ft_px} px")
        print(f"     - 6 дюймов (15,24 см): {radius_6in_px} px")
        print(f"   Сохранено: {output_path}")
        print()
    
    # Создаем также полное поле с домом
    print("Создание полного поля с домом...")
    
    # Размеры поля (как в основном пайплайне)
    court_width_cm, court_height_cm = 500, 1300
    img_width_px, img_height_px = 800, 2080
    
    # Масштаб для полного поля
    scale_cm_per_px = court_width_cm / img_width_px
    house_center = (img_width_px // 2, int(img_height_px * 0.6))
    
    # Создаем полное поле
    full_field = create_full_field_with_house(
        img_width_px, img_height_px, 
        court_width_cm, court_height_cm,
        house_center, scale_cm_per_px
    )
    
    full_field_path = os.path.join(output_dir, "full_field_with_house.png")
    cv2.imwrite(full_field_path, full_field)
    
    print(f"   Размер поля: {court_width_cm}x{court_height_cm} см")
    print(f"   Размер изображения: {img_width_px}x{img_height_px} пикселей")
    print(f"   Масштаб: {scale_cm_per_px:.3f} см/пиксель")
    print(f"   Полное поле сохранено: {full_field_path}")
    
    print("\nТестирование завершено!")
    print("Все файлы сохранены в директории: outputs/test_house/")


def create_full_field_with_house(img_width_px: int, img_height_px: int,
                                court_width_cm: int, court_height_cm: int,
                                house_center: tuple, scale_cm_per_px: float):
    """Создает полное поле с домом для тестирования."""
    
    # Создаем белый фон
    img = np.ones((img_height_px, img_width_px, 3), dtype=np.uint8) * 255
    
    # Создаем шаблон дома
    house_template = create_curling_house_template(house_center, scale_cm_per_px)
    
    # Накладываем дом на поле
    house_h, house_w = house_template.shape[:2]
    house_x1 = house_center[0] - house_w // 2
    house_y1 = house_center[1] - house_h // 2
    house_x2 = house_x1 + house_w
    house_y2 = house_y1 + house_h
    
    # Проверяем границы
    if (house_x1 >= 0 and house_y1 >= 0 and 
        house_x2 <= img_width_px and house_y2 <= img_height_px):
        img[house_y1:house_y2, house_x1:house_x2] = house_template
    
    # Добавляем границы поля
    cv2.rectangle(img, (0, 0), (img_width_px-1, img_height_px-1), (0, 0, 0), 2)
    
    # Добавляем центральную линию
    cv2.line(img, (img_width_px//2, 0), (img_width_px//2, img_height_px), (0, 0, 0), 1)
    
    return img


if __name__ == "__main__":
    main()
