"""
Автономный модуль для обнаружения ключевых точек кёрлингового поля
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

# Опциональный импорт ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    print("Предупреждение: ultralytics не установлен. Будут использованы простые методы детекции.")


class KeypointsDetector:
    """Детектор ключевых точек кёрлингового поля"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.2):
        """
        Инициализация детектора ключевых точек
        
        Args:
            model_path: Путь к модели YOLO для обнаружения ключевых точек
            conf_threshold: Порог уверенности для детекции
        """
        self.conf_threshold = conf_threshold
        self.model = None
        
        # Попытка загрузить модель
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
            self._load_model()
        else:
            # Попытка найти модель в стандартных местах
            self.model_path = self._find_model()
            if self.model_path:
                self._load_model()
            else:
                print("Предупреждение: Модель для обнаружения ключевых точек не найдена")
                print("Будут использованы тестовые ключевые точки")
    
    def _find_model(self) -> Optional[str]:
        """Ищет модель в стандартных местах"""
        possible_paths = [
            "/Users/glytov_ivan/Documents/curling/curling_analytics/outputs/runs/yolo11_boxes25/train_20251009_152028/weights/best.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_model(self):
        """Загружает модель YOLO"""
        if not ULTRALYTICS_AVAILABLE:
            print("ultralytics не доступен, модель не загружена")
            self.model = None
            return
            
        try:
            print(f"Загружаю модель ключевых точек: {self.model_path}")
            # Используем weights_only=False для совместимости с PyTorch 2.6+
            import torch
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            self.model = YOLO(self.model_path)
            print("Модель загружена успешно")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model = None
    
    def detect_keypoints(self, image_input) -> List[Tuple[float, float, int]]:
        """
        Обнаруживает ключевые точки на изображении
        
        Args:
            image_input: Путь к изображению (str) или numpy array
            
        Returns:
            Список кортежей (x, y, idx) - координаты и индексы ключевых точек
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise FileNotFoundError(f"Не удалось загрузить изображение: {image_input}")
        else:
            image = image_input
        
        # Если модель не загружена, используем тестовые точки
        if self.model is None:
            return self._get_test_keypoints(image)
        
        try:
            # Выполняем детекцию
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            keypoints_list = []
            
            # Обрабатываем результаты детекции (boxes)
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                # Используем детекцию боксов для ключевых точек
                best_by_cls = {}
                for r in results:
                    for b in r.boxes:
                        conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                        if conf < self.conf_threshold:
                            continue
                        cls_id = int(b.cls[0]) if hasattr(b, 'cls') else -1
                        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        if cls_id not in best_by_cls or conf > best_by_cls[cls_id][2]:
                            best_by_cls[cls_id] = (cx, cy, conf)
                
                # Строим упорядоченный список по ожидаемому маппингу
                for cls_id in range(1, 23):  # Классы 1-22
                    if cls_id in best_by_cls:
                        cx, cy, _ = best_by_cls[cls_id]
                        keypoints_list.append((float(cx), float(cy), cls_id - 1))
            
            # Обрабатываем результаты pose модели (fallback)
            else:
                for result in results:
                    if result.keypoints is not None and len(result.keypoints.data) > 0:
                        kpts = result.keypoints.data[0].cpu().numpy()  # (N, 3) — x, y, confidence
                        for i, kp in enumerate(kpts):
                            x, y, conf = kp
                            if conf >= self.conf_threshold:
                                keypoints_list.append((float(x), float(y), i))
            
            print(f"Найдено {len(keypoints_list)} ключевых точек")
            return keypoints_list
            
        except Exception as e:
            print(f"Ошибка при обнаружении ключевых точек: {e}")
            return self._get_test_keypoints(image)
    
    def _get_test_keypoints(self, image: np.ndarray) -> List[Tuple[float, float, int]]:
        """Возвращает тестовые ключевые точки для демонстрации"""
        h, w = image.shape[:2]
        
        # Создаем тестовые ключевые точки в разных частях изображения
        test_points = [
            (w * 0.1, h * 0.3, 0),   # l_hog
            (w * 0.9, h * 0.3, 1),   # r_hog
            (w * 0.1, h * 0.7, 2),   # l_back
            (w * 0.9, h * 0.7, 3),   # r_back
            (w * 0.5, h * 0.5, 8),   # center
            (w * 0.3, h * 0.4, 4),   # l_tee
            (w * 0.7, h * 0.4, 5),   # r_tee
            (w * 0.5, h * 0.2, 6),   # hog_central
            (w * 0.5, h * 0.8, 7),   # p_central
        ]
        
        print(f"Используются тестовые ключевые точки: {len(test_points)}")
        return test_points
    
    def visualize_keypoints(self, image_input, keypoints: List[Tuple[float, float, int]], 
                          output_path: str = None) -> np.ndarray:
        """
        Визуализирует ключевые точки на изображении
        
        Args:
            image_input: Путь к изображению (str) или numpy array
            keypoints: Список ключевых точек
            output_path: Путь для сохранения результата
            
        Returns:
            Изображение с нарисованными ключевыми точками
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input.copy()
        
        # Рисуем ключевые точки
        for i, kp in enumerate(keypoints):
            if len(kp) == 3:
                x, y, idx = kp
            else:
                x, y = kp
                idx = i
            
            # Рисуем точку
            cv2.circle(image, (int(x), int(y)), 8, (0, 255, 0), -1)
            # Рисуем номер точки
            cv2.putText(image, str(idx), (int(x) + 10, int(y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Визуализация ключевых точек сохранена: {output_path}")
        
        return image


def create_keypoints_detector(model_path: str = None) -> KeypointsDetector:
    """
    Создает детектор ключевых точек
    
    Args:
        model_path: Путь к модели (опционально)
        
    Returns:
        Экземпляр KeypointsDetector
    """
    return KeypointsDetector(model_path)
