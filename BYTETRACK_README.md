# ByteTrack Integration for Curling Stone Tracking

## Обзор

Этот проект интегрирует алгоритм ByteTrack для улучшенного трекинга камней в кёрлинге. ByteTrack обеспечивает более стабильное и точное отслеживание объектов по сравнению с простыми методами трекинга.

## Основные компоненты

### 1. ByteTracker (`src/bytetrack_tracker.py`)
- Полная реализация алгоритма ByteTrack
- Kalman фильтр для предсказания движения
- Ассоциация детекций с треками на основе IoU
- Управление состояниями треков (New, Tracked, Lost, Removed)

### 2. SimpleTracker (`src/simple_tracker.py`)
- Упрощенная версия трекера для демонстрации
- Базовый трекинг без сложных алгоритмов
- Подходит для тестирования и отладки

### 3. Обновленная визуализация (`src/animated_visualizations.py`)
- Поддержка отображения ID треков
- Информация о типе трекера в легенде
- Улучшенная визуализация траекторий

## Использование

### Базовое использование

```python
from src.stones_processing import detect_and_track_stones
from src.animated_visualizations import create_birdseye_video_with_legend

# Использование ByteTrack (по умолчанию)
stone_tracks = detect_and_track_stones(
    video_path="path/to/video.mp4",
    homography_matrix=homography_matrix,
    use_bytetrack=True
)

# Создание анимированного видео
video_path = create_birdseye_video_with_legend(
    stone_tracks=stone_tracks,
    output_dir="outputs",
    use_bytetrack=True
)
```

### Переключение между трекерами

```python
# ByteTrack (рекомендуется)
stone_tracks = detect_and_track_stones(video_path, homography_matrix, use_bytetrack=True)

# SimpleTracker (для сравнения)
stone_tracks = detect_and_track_stones(video_path, homography_matrix, use_bytetrack=False)
```

## Тестирование

### Запуск тестов

```bash
# Тестирование упрощенного трекера
python3 experiments/test_simple_tracker.py

# Тестирование ByteTrack компонентов
python3 experiments/test_bytetrack_simple.py

# Демонстрация трекинга
python3 experiments/simple_bytetrack_demo.py
```

### Ожидаемые результаты

- **ByteTrack**: Более стабильное трекинг с меньшим количеством переключений ID
- **SimpleTracker**: Базовый трекинг для сравнения
- **Визуализация**: Отображение ID треков и информации о трекере

## Параметры конфигурации

### ByteTracker параметры

```python
tracker = ByteTracker(
    frame_rate=30,           # FPS видео
    track_thresh=0.5,        # Минимальный порог confidence для трекинга
    high_thresh=0.6,         # Высокий порог confidence
    match_thresh=0.8,        # Порог IoU для сопоставления
    track_buffer=30          # Буфер кадров для потерянных треков
)
```

### SimpleTracker параметры

```python
tracker = SimpleByteTracker(
    track_thresh=0.5,        # Минимальный порог confidence
    match_thresh=0.3,        # Порог IoU для сопоставления
    max_disappeared=10       # Максимальное количество кадров без детекции
)
```

## Преимущества ByteTrack

1. **Стабильность**: Меньше переключений ID между кадрами
2. **Точность**: Лучшее сопоставление детекций с треками
3. **Робастность**: Обработка временных потерь детекций
4. **Предсказание**: Kalman фильтр для предсказания движения

## Структура данных треков

```python
{
    "stone_id": 1,                    # Уникальный ID трека
    "team": "red",                    # Команда камня
    "positions_px": [                 # Позиции в пикселях
        {"frame": 0, "x": 100, "y": 200},
        {"frame": 1, "x": 105, "y": 202},
        # ...
    ],
    "positions_real": [               # Позиции в реальных координатах
        {"frame": 0, "x": 50.0, "y": 100.0},
        {"frame": 1, "x": 52.5, "y": 101.0},
        # ...
    ],
    "throw_start_frame": 0,           # Начальный кадр броска
    "throw_end_frame": 29,            # Конечный кадр броска
    "final_position_real": {          # Финальная позиция
        "frame": 29,
        "x": 147.5,
        "y": 129.0
    }
}
```

## Файлы проекта

- `src/bytetrack_tracker.py` - Основная реализация ByteTrack
- `src/simple_tracker.py` - Упрощенный трекер
- `src/stones_processing.py` - Интеграция трекеров
- `src/animated_visualizations.py` - Визуализация с поддержкой ByteTrack
- `experiments/test_*.py` - Тестовые скрипты
- `main.py` - Основной скрипт с ByteTrack

## Требования

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- tqdm

## Устранение неполадок

### Проблема: Треки не создаются
- Проверьте пороги confidence в параметрах трекера
- Убедитесь, что детекции имеют достаточный score

### Проблема: Частые переключения ID
- Увеличьте `match_thresh` для более строгого сопоставления
- Уменьшите `track_buffer` для быстрого удаления потерянных треков

### Проблема: Треки теряются
- Увеличьте `track_buffer` для более длительного удержания треков
- Уменьшите `match_thresh` для более мягкого сопоставления

## Лицензия

Этот проект использует алгоритм ByteTrack, который доступен под лицензией MIT.
