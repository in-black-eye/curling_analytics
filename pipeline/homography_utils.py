"""
Автономный модуль для работы с гомографией в кёрлинге
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class HomographyCalculator:
    """Калькулятор гомографии для кёрлингового поля"""
    
    def __init__(self):
        """Инициализация калькулятора гомографии"""
        self.real_coords = self._get_real_world_coordinates()
    
    def _get_real_world_coordinates(self) -> np.ndarray:
        """
        Возвращает реальные физические координаты ключевых точек в сантиметрах.
        Позиция точки в списке совпадает с номером точки в скелете CVAT (0 -> 21)
        """
        real_coords = np.array([
            (0, 731.54),      # 0: "l_hog"
            (500, 731.54),    # 1: "r_hog"
            (0, 365.76),      # 2: "l_back"
            (500, 365.76),    # 3: "r_back"
            (0, 548.64),      # 4: "l_tee"
            (500, 548.64),    # 5: "r_tee"
            (250, 1188.72),   # 6: "hog_central"
            (250, 365.76),    # 7: "p_central"
            (250, 548.64),    # 8: "center"
            (250, 1188.72),   # 9: "a_central"
            (250, 563.84),    # 10: "button_ant"
            (265.2, 548.64),  # 11: "button_r"
            (250, 533.44),    # 12: "button_post"
            (234.8, 548.64),  # 13: "button_l"
            (250, 609.64),    # 14: "inner_ant"
            (311, 548.64),    # 15: "inner_r"
            (250, 487.64),    # 16: "inner_post"
            (189, 548.64),    # 17: "inner_l"
            (250, 670.54),    # 18: "outer_ant"
            (371.9, 548.64),  # 19: "outer_r"
            (250, 426.74),    # 20: "outer_post"
            (128.1, 548.64)   # 21: "outer_l"
        ], dtype=np.float32)
        
        return real_coords
    
    def compute_homography_matrix(self, keypoints_pxls: List[Tuple[float, float, int]]) -> Optional[np.ndarray]:
        """
        Вычисляет матрицу гомографии на основе ключевых точек
        
        Args:
            keypoints_pxls: Список кортежей (x, y, idx) - координаты и индексы ключевых точек
            
        Returns:
            Матрица гомографии 3x3 или None при ошибке
        """
        if len(keypoints_pxls) == 0:
            print("Ошибка: Нет ключевых точек для вычисления гомографии")
            return None
        
        # Извлекаем координаты и индексы
        src_points = []
        dst_points = []
        
        for kp in keypoints_pxls:
            if len(kp) == 3:
                x, y, idx = kp
                if 0 <= idx < len(self.real_coords):
                    src_points.append((x, y))
                    dst_points.append(self.real_coords[idx])
            elif len(kp) == 2:
                x, y = kp
                src_points.append((x, y))
                # Используем первые доступные реальные координаты
                if len(dst_points) < len(self.real_coords):
                    dst_points.append(self.real_coords[len(dst_points)])
        
        if len(src_points) < 4:
            print(f"Недостаточно точек для гомографии: найдено {len(src_points)}, минимум 4")
            return None
        
        # Ограничиваем количество точек
        if len(src_points) > len(self.real_coords):
            src_points = src_points[:len(self.real_coords)]
            dst_points = dst_points[:len(self.real_coords)]
        
        print(f"Используем {len(src_points)} точек для гомографии")
        
        # Преобразуем в numpy arrays
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Вычисляем гомографию с RANSAC
        H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        
        if H is None:
            print("Не удалось вычислить гомографию")
            return None
        
        # Статистика inliers
        inliers = np.sum(status)
        total = len(status)
        print(f"Inliers: {inliers}/{total} ({inliers/total*100:.1f}%)")
        
        return H
    
    def validate_homography(self, H: np.ndarray, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Валидирует гомографию, накладывая реальные координаты обратно на изображение
        
        Args:
            H: Матрица гомографии
            image_path: Путь к изображению
            output_path: Путь для сохранения результата
            
        Returns:
            Изображение с наложенными реальными координатами
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        
        # Преобразуем реальные координаты в формат для perspectiveTransform
        real_coords = self.real_coords.reshape(-1, 1, 2)
        
        # Обратная гомография
        H_inv = np.linalg.inv(H)
        pixel_coords = cv2.perspectiveTransform(real_coords, H_inv)
        
        # Рисуем точки на изображении
        for i, pt in enumerate(pixel_coords):
            x, y = int(pt[0][0]), int(pt[0][1])
            # Проверяем, что точка в пределах изображения
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(image, str(i), (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Валидация гомографии сохранена: {output_path}")
        
        return image
    
    def transform_point(self, point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
        """
        Преобразует точку из пиксельных координат в реальные координаты
        
        Args:
            point: Точка (x, y) в пикселях
            H: Матрица гомографии
            
        Returns:
            Точка (x, y) в реальных координатах (см)
        """
        pixel_point = np.array([[[point[0], point[1]]]], dtype=np.float32)
        real_point = cv2.perspectiveTransform(pixel_point, H)
        return (float(real_point[0][0][0]), float(real_point[0][0][1]))
    
    def transform_points_batch(self, points: List[Tuple[float, float]], H: np.ndarray) -> List[Tuple[float, float]]:
        """
        Преобразует список точек из пиксельных координат в реальные координаты
        
        Args:
            points: Список точек (x, y) в пикселях
            H: Матрица гомографии
            
        Returns:
            Список точек (x, y) в реальных координатах (см)
        """
        if not points:
            return []
        
        pixel_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        real_points = cv2.perspectiveTransform(pixel_points, H)
        
        return [(float(pt[0][0]), float(pt[0][1])) for pt in real_points]


def create_homography_calculator() -> HomographyCalculator:
    """
    Создает калькулятор гомографии
    
    Returns:
        Экземпляр HomographyCalculator
    """
    return HomographyCalculator()
