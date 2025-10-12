#!/usr/bin/env python3
"""
Тестирование нового шаблона поля для кёрлинга
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from src.animated_visualizations import create_curling_field_template, create_curling_house_template


def test_field_template():
    """Тестирование нового шаблона поля"""
    
    print("🔍 Тестирование шаблона поля кёрлинга...")
    
    # Создаем выходную директорию
    output_dir = "outputs/field_template_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Тестируем различные масштабы
    scales = [0.5, 1.0, 2.0]  # см на пиксель
    
    for scale in scales:
        print(f"\nТестирование масштаба: {scale} см/пиксель")
        
        # Создаем шаблон поля
        field_center = (400, 300)  # Центр изображения
        field_template = create_curling_field_template(field_center, scale)
        
        # Сохраняем шаблон
        filename = f"field_template_{scale}cm_per_px.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, field_template)
        print(f"  ✅ Сохранен: {filepath}")
        print(f"  Размер: {field_template.shape[1]}x{field_template.shape[0]} пикселей")
        
        # Создаем изображение с шаблоном поля
        img_width = 800
        img_height = 600
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Размещаем шаблон в центре
        x1 = field_center[0] - field_template.shape[1] // 2
        y1 = field_center[1] - field_template.shape[0] // 2
        x2 = x1 + field_template.shape[1]
        y2 = y1 + field_template.shape[0]
        
        if (x1 >= 0 and y1 >= 0 and x2 <= img_width and y2 <= img_height):
            img[y1:y2, x1:x2] = field_template
            
            # Добавляем информацию о масштабе
            cv2.putText(img, f"Scale: {scale} cm/px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img, f"Field size: {field_template.shape[1]}x{field_template.shape[0]}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Сохраняем полное изображение
            full_filename = f"field_with_template_{scale}cm_per_px.png"
            full_filepath = os.path.join(output_dir, full_filename)
            cv2.imwrite(full_filepath, img)
            print(f"  ✅ Полное изображение: {full_filepath}")


def test_house_template():
    """Тестирование старого шаблона дома для сравнения"""
    
    print("\n🔍 Тестирование шаблона дома (для сравнения)...")
    
    output_dir = "outputs/field_template_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем шаблон дома
    house_center = (400, 300)
    house_template = create_curling_house_template(house_center, 1.0)
    
    # Сохраняем шаблон
    filepath = os.path.join(output_dir, "house_template_1cm_per_px.png")
    cv2.imwrite(filepath, house_template)
    print(f"  ✅ Сохранен: {filepath}")
    print(f"  Размер: {house_template.shape[1]}x{house_template.shape[0]} пикселей")


def compare_templates():
    """Сравнение шаблонов поля и дома"""
    
    print("\n📊 Сравнение шаблонов...")
    
    output_dir = "outputs/field_template_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем оба шаблона
    center = (400, 300)
    scale = 1.0
    
    field_template = create_curling_field_template(center, scale)
    house_template = create_curling_house_template(center, scale)
    
    # Создаем сравнительное изображение
    max_width = max(field_template.shape[1], house_template.shape[1])
    max_height = max(field_template.shape[0], house_template.shape[0])
    
    comparison_img = np.ones((max_height * 2 + 50, max_width, 3), dtype=np.uint8) * 255
    
    # Размещаем шаблон поля сверху
    field_x1 = (max_width - field_template.shape[1]) // 2
    field_y1 = 0
    field_x2 = field_x1 + field_template.shape[1]
    field_y2 = field_y1 + field_template.shape[0]
    comparison_img[field_y1:field_y2, field_x1:field_x2] = field_template
    
    # Размещаем шаблон дома снизу
    house_x1 = (max_width - house_template.shape[1]) // 2
    house_y1 = max_height + 50
    house_x2 = house_x1 + house_template.shape[1]
    house_y2 = house_y1 + house_template.shape[0]
    comparison_img[house_y1:house_y2, house_x1:house_x2] = house_template
    
    # Добавляем подписи
    cv2.putText(comparison_img, "Field Template (New)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(comparison_img, "House Template (Old)", (10, max_height + 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Сохраняем сравнение
    comparison_path = os.path.join(output_dir, "template_comparison.png")
    cv2.imwrite(comparison_path, comparison_img)
    print(f"  ✅ Сравнение сохранено: {comparison_path}")
    
    # Выводим статистику
    print(f"\nСтатистика шаблонов:")
    print(f"  Поле: {field_template.shape[1]}x{field_template.shape[0]} пикселей")
    print(f"  Дом:  {house_template.shape[1]}x{house_template.shape[0]} пикселей")


def test_field_measurements():
    """Тестирование размеров поля"""
    
    print("\n📏 Тестирование размеров поля...")
    
    # Параметры из кода
    tee_to_back = 1.829          # м
    tee_to_hog = 6.401           # м
    line_width = 0.0127          # м
    hog_line_width = 0.1016      # м
    field_width = 4.0            # м
    
    # Радиусы домов
    radii_m = [1.829, 1.219, 0.610, 0.1524]  # м
    
    print(f"Параметры поля:")
    print(f"  Ширина поля: {field_width} м")
    print(f"  Длина поля: {tee_to_back + tee_to_hog + hog_line_width:.3f} м")
    print(f"  Расстояние от центра до back line: {tee_to_back} м")
    print(f"  Расстояние от центра до hog line: {tee_to_hog} м")
    print(f"  Ширина тонких линий: {line_width} м")
    print(f"  Ширина hog line: {hog_line_width} м")
    print(f"  Радиусы домов: {radii_m} м")
    
    # Тестируем с масштабом 1 см/пиксель
    scale_cm_per_px = 1.0
    scale_m_per_px = scale_cm_per_px / 100.0
    
    field_width_px = int(field_width / scale_m_per_px)
    field_height_px = int((tee_to_back + tee_to_hog + hog_line_width) / scale_m_per_px)
    
    print(f"\nВ пикселях (масштаб {scale_cm_per_px} см/пиксель):")
    print(f"  Ширина поля: {field_width_px} пикселей")
    print(f"  Высота поля: {field_height_px} пикселей")
    print(f"  Радиусы домов: {[int(r / scale_m_per_px) for r in radii_m]} пикселей")


def main():
    """Основная функция тестирования"""
    
    print("🚀 Тестирование нового шаблона поля кёрлинга")
    print("=" * 60)
    
    try:
        # Тестируем размеры
        test_field_measurements()
        
        # Тестируем шаблоны
        test_field_template()
        test_house_template()
        
        # Сравниваем шаблоны
        compare_templates()
        
        print("\n✅ Тестирование завершено!")
        print(f"\nСозданные файлы в outputs/field_template_test/:")
        print("  - field_template_*.png - шаблоны поля в разных масштабах")
        print("  - field_with_template_*.png - полные изображения с шаблонами")
        print("  - house_template_*.png - старые шаблоны дома")
        print("  - template_comparison.png - сравнение шаблонов")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
