"""
Автономный модуль конфигурации для пайплайна кёрлинга
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class PipelineConfig:
    """Конфигурация пайплайна обработки видео кёрлинга"""
    
    def __init__(self, config_path: str = None):
        """
        Инициализация конфигурации
        
        Args:
            config_path: Путь к файлу конфигурации (опционально)
        """
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        else:
            # Попытка найти конфигурацию в стандартных местах
            default_config_path = self._find_config_file()
            if default_config_path:
                self._load_from_file(default_config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию по умолчанию"""
        return {
            "model_paths": {
                "keypoints_detector": None,  # Будет найдена автоматически
                "stones_detector": None,     # Будет найдена автоматически
            },
            "model_params": {
                "confidence_threshold": 0.25,
                "keypoints_confidence_threshold": 0.2,
                "team_colors": ["red", "yellow"]
            },
            "paths": {
                "output_dir": "outputs",
                "video_data": "data/videos",
                "models_dir": "models"
            },
            "tracking": {
                "iou_threshold": 0.5,
                "max_disappeared": 10
            },
            "visualization": {
                "fps": 30,
                "court_size_cm": [500, 1300],
                "img_size": [800, 2080]
            }
        }
    
    def _find_config_file(self) -> str:
        """Ищет файл конфигурации в стандартных местах"""
        possible_paths = [
            "../configs/config.yaml",
            "configs/config.yaml",
            "config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_from_file(self, config_path: str):
        """Загружает конфигурацию из файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            
            # Объединяем с конфигурацией по умолчанию
            self._merge_config(file_config)
            print(f"Конфигурация загружена из: {config_path}")
            
        except Exception as e:
            print(f"Ошибка загрузки конфигурации из {config_path}: {e}")
            print("Используется конфигурация по умолчанию")
    
    def _merge_config(self, file_config: Dict[str, Any]):
        """Объединяет конфигурацию из файла с конфигурацией по умолчанию"""
        def merge_dicts(default: Dict, override: Dict):
            for key, value in override.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dicts(default[key], value)
                else:
                    default[key] = value
        
        merge_dicts(self.config, file_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получает значение конфигурации по ключу
        
        Args:
            key: Ключ конфигурации (может быть вложенным, например "model_paths.keypoints_detector")
            default: Значение по умолчанию
            
        Returns:
            Значение конфигурации
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Устанавливает значение конфигурации
        
        Args:
            key: Ключ конфигурации
            value: Значение
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_model_path(self, model_type: str) -> str:
        """
        Получает путь к модели
        
        Args:
            model_type: Тип модели ("keypoints_detector" или "stones_detector")
            
        Returns:
            Путь к модели или None
        """
        return self.get(f"model_paths.{model_type}")
    
    def get_output_dir(self) -> str:
        """Получает директорию для выходных файлов"""
        return self.get("paths.output_dir", "outputs")
    
    def get_video_data_dir(self) -> str:
        """Получает директорию с видео данными"""
        return self.get("paths.video_data", "data/videos")
    
    def get_confidence_threshold(self) -> float:
        """Получает порог уверенности для детекции"""
        return self.get("model_params.confidence_threshold", 0.25)
    
    def get_keypoints_confidence_threshold(self) -> float:
        """Получает порог уверенности для ключевых точек"""
        return self.get("model_params.keypoints_confidence_threshold", 0.2)
    
    def get_tracking_params(self) -> Dict[str, Any]:
        """Получает параметры трекинга"""
        return self.get("tracking", {})
    
    def get_visualization_params(self) -> Dict[str, Any]:
        """Получает параметры визуализации"""
        return self.get("visualization", {})
    
    def save(self, config_path: str):
        """
        Сохраняет конфигурацию в файл
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"Конфигурация сохранена в: {config_path}")
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
    
    def print_config(self):
        """Выводит текущую конфигурацию"""
        print("Текущая конфигурация:")
        print("-" * 40)
        for key, value in self.config.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        print("-" * 40)


def create_config(config_path: str = None) -> PipelineConfig:
    """
    Создает конфигурацию пайплайна
    
    Args:
        config_path: Путь к файлу конфигурации (опционально)
        
    Returns:
        Экземпляр PipelineConfig
    """
    return PipelineConfig(config_path)
