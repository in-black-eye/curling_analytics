"""
Автономный модуль для визуализации результатов анализа кёрлинга
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple

# Опциональный импорт tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Простая заглушка для tqdm
    def tqdm(iterable, **kwargs):
        return iterable
    TQDM_AVAILABLE = False
    print("Предупреждение: tqdm не установлен. Прогресс-бар не будет отображаться.")


class CurlingVisualizer:
    """Визуализатор для кёрлинга"""
    
    def __init__(self):
        """Инициализация визуализатора"""
        self.colors = {
            "red": (0, 0, 255),      # BGR: красный
            "yellow": (0, 255, 255)  # BGR: желтый
        }
    
    def create_curling_field_template(self, field_center: Tuple[int, int], 
                                    scale_cm_per_px: float = 1.0) -> np.ndarray:
        """
        Создает шаблон поля для кёрлинга
        
        Args:
            field_center: (x, y) координаты центра поля в пикселях
            scale_cm_per_px: масштаб - сколько сантиметров в одном пикселе
            
        Returns:
            numpy array с шаблоном поля
        """
        # Параметры разметки (в метрах)
        tee_to_back = 1.829          # расстояние от центра до наружного края back line
        tee_to_hog = 6.401           # расстояние от центра до внутреннего края hog line
        line_width = 0.0127          # ширина тонких линий (tee, back)
        hog_line_width = 0.1016      # ширина hog line
        
        # Радиусы домов (внешние края кругов)
        radii_m = [1.829, 1.219, 0.610, 0.1524]  # 6ft, 4ft, 2ft, 6in
        colors = ['red', 'white', 'blue', 'white']
        
        # Преобразуем в пиксели (метры -> сантиметры -> пиксели)
        scale_m_per_px = scale_cm_per_px / 100.0
        radii_px = [int(radius_m / scale_m_per_px) for radius_m in radii_m]
        
        # Размеры поля в пикселях
        field_width_px = int(4.0 / scale_m_per_px)  # 4 метра ширина
        field_height_px = int((tee_to_back + tee_to_hog + hog_line_width) / scale_m_per_px)
        
        # Создаем изображение поля
        img = np.ones((field_height_px, field_width_px, 3), dtype=np.uint8) * 255
        
        # Координаты центра поля (дом в верхней части)
        cx, cy = field_width_px // 2, int(tee_to_back / scale_m_per_px)
        
        # Рисуем концентрические круги дома
        for radius_px, color in zip(radii_px, colors):
            if color == 'red':
                cv2.circle(img, (cx, cy), radius_px, (0, 0, 255), -1, cv2.LINE_AA)  # Красный в BGR
            elif color == 'white':
                cv2.circle(img, (cx, cy), radius_px, (255, 255, 255), -1, cv2.LINE_AA)  # Белый в BGR
            elif color == 'blue':
                cv2.circle(img, (cx, cy), radius_px, (255, 0, 0), -1, cv2.LINE_AA)  # Синий в BGR
        
        # Рисуем контуры кругов
        for radius_px in radii_px:
            cv2.circle(img, (cx, cy), radius_px, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Tee line (через центр, y = cy)
        tee_line_width_px = max(1, int(line_width / scale_m_per_px))
        if tee_line_width_px > 0:
            tee_line = np.zeros((tee_line_width_px, field_width_px, 3), dtype=np.uint8)
            y_start = max(0, cy - tee_line_width_px//2)
            y_end = min(field_height_px, cy + tee_line_width_px//2)
            if y_end > y_start:
                img[y_start:y_end, :] = tee_line[:y_end-y_start, :]
        
        # Back line (наружный край на y = cy - tee_to_back) - теперь вверху
        back_line_y = cy - int(tee_to_back / scale_m_per_px)
        back_line_width_px = max(1, int(line_width / scale_m_per_px))
        if back_line_width_px > 0 and back_line_y >= 0:
            back_line = np.zeros((back_line_width_px, field_width_px, 3), dtype=np.uint8)
            y_start = max(0, back_line_y - back_line_width_px//2)
            y_end = min(field_height_px, back_line_y + back_line_width_px//2)
            if y_end > y_start:
                img[y_start:y_end, :] = back_line[:y_end-y_start, :]
        
        # Hog line (внутренний край на y = cy + tee_to_hog) - теперь внизу
        hog_line_y = cy + int(tee_to_hog / scale_m_per_px)
        hog_line_width_px = max(1, int(hog_line_width / scale_m_per_px))
        if hog_line_width_px > 0 and hog_line_y < field_height_px:
            hog_line = np.zeros((hog_line_width_px, field_width_px, 3), dtype=np.uint8)
            y_start = max(0, hog_line_y)
            y_end = min(field_height_px, hog_line_y + hog_line_width_px)
            if y_end > y_start:
                img[y_start:y_end, :] = hog_line[:y_end-y_start, :]
        
        # Центральная линия (вертикальная, от back line до hog line)
        center_line_width = 2
        cv2.line(img, (cx, back_line_y), (cx, hog_line_y + hog_line_width_px), (0, 0, 0), center_line_width, cv2.LINE_AA)
        
        # Переворачиваем шаблон поля по вертикали, чтобы дом был вверху
        img = cv2.flip(img, 0)
        
        return img
    
    def create_bbox_video(self, video_path: str, stone_tracks: List[Dict], 
                         output_path: str) -> str:
        """
        Создает видео с bbox для камней поверх оригинального видео
        
        Args:
            video_path: Путь к оригинальному видео
            stone_tracks: Данные о треках камней
            output_path: Путь для сохранения результата
            
        Returns:
            Путь к созданному видео
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Настраиваем выходное видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Группируем треки по кадрам
        tracks_by_frame = {}
        for track in stone_tracks:
            for pos in track.get("positions", []):
                frame_num = pos["frame"]
                if frame_num not in tracks_by_frame:
                    tracks_by_frame[frame_num] = []
                tracks_by_frame[frame_num].append({
                    "bbox": pos["bbox"],
                    "team": track.get("team", "unknown"),
                    "track_id": track.get("track_id", "?"),
                    "score": pos.get("score", 0.0)
                })
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Рисуем bbox для текущего кадра
            if frame_num in tracks_by_frame:
                for detection in tracks_by_frame[frame_num]:
                    bbox = detection["bbox"]
                    team = detection["team"]
                    track_id = detection["track_id"]
                    score = detection["score"]
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Цвет bbox в зависимости от команды
                    color = self.colors.get(team, (0, 0, 0))
                    
                    # Рисуем bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Добавляем текст с информацией
                    label = f"{team} ID:{track_id} {score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
            frame_num += 1
            
            if frame_num % 30 == 0:
                print(f"Обработано {frame_num} кадров для bbox видео...")
        
        cap.release()
        out.release()
        
        print(f"Видео с bbox сохранено: {output_path}")
        return output_path
    
    def create_birdseye_video(self, stone_tracks: List[Dict], output_path: str,
                            fps: int = 30, court_size_cm: Tuple[int, int] = (500, 1300),
                            img_size: Tuple[int, int] = (800, 2080)) -> str:
        """
        Создает bird's eye view анимацию с траекториями камней
        
        Args:
            stone_tracks: Данные о треках камней
            output_path: Путь для сохранения результата
            fps: FPS выходного видео
            court_size_cm: Размеры поля в см (width, height)
            img_size: Размеры изображения в пикселях (width, height)
            
        Returns:
            Путь к созданному видео
        """
        w_px, h_px = img_size
        cw, ch = court_size_cm
        
        # Коэффициенты масштабирования
        sx = w_px / float(cw) if cw > 0 else 1.0
        sy = h_px / float(ch) if ch > 0 else 1.0
        
        # Находим диапазон кадров
        all_frames = set()
        for track in stone_tracks:
            for pos in track.get("positions_real", []):
                all_frames.add(pos["frame"])
        
        if not all_frames:
            raise ValueError("Нет данных о позициях камней")
        
        min_frame = min(all_frames)
        max_frame = max(all_frames)
        total_frames = max_frame - min_frame + 1
        
        print(f"Создание bird's eye view видео: {total_frames} кадров, {fps} FPS")
        
        # Настройка видео writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w_px, h_px))
        
        # Создаем шаблон поля
        field_center = (w_px // 2, int(h_px * 0.6))
        
        # Вычисляем масштаб: сколько сантиметров в одном пикселе
        scale_cm_per_px = cw / w_px  # ширина поля в см / ширина изображения в пикселях
        field_template = self.create_curling_field_template(field_center, scale_cm_per_px)
        
        # Позиция шаблона поля
        field_x1 = field_center[0] - field_template.shape[1] // 2
        field_y1 = field_center[1] - field_template.shape[0] // 2
        field_x2 = field_x1 + field_template.shape[1]
        field_y2 = field_y1 + field_template.shape[0]
        
        # Подсчитываем камни по командам
        red_stones = [t for t in stone_tracks if t.get("team") == "red"]
        yellow_stones = [t for t in stone_tracks if t.get("team") == "yellow"]
        
        with tqdm(total=total_frames, desc="Создание bird's eye view", unit="кадр") as pbar:
            for frame_num in range(min_frame, max_frame + 1):
                # Создаем белый фон
                img = np.ones((h_px, w_px, 3), dtype=np.uint8) * 255
                
                # Рисуем траектории и позиции камней
                for track in stone_tracks:
                    team = track.get("team", "red")
                    color = self.colors.get(team, (0, 0, 0))
                    stone_id = track.get("track_id", "?")
                    
                    trajectory_points = []
                    for pos in track.get("positions_real", []):
                        if pos["frame"] <= frame_num:
                            x_cm, y_cm = pos["x"], pos["y"]
                            x_px = int(round(x_cm * sx))
                            y_px = int(round((ch - y_cm) * sy))
                            trajectory_points.append((x_px, y_px))
                    
                    # Рисуем полную траекторию как соединенные линии
                    if len(trajectory_points) > 1:
                        for i in range(1, len(trajectory_points)):
                            cv2.line(img, trajectory_points[i-1], trajectory_points[i], 
                                    color, 2, cv2.LINE_AA)
                    
                    # Рисуем все предыдущие позиции как точки
                    for point in trajectory_points:
                        cv2.circle(img, point, 3, color, -1, cv2.LINE_AA)
                    
                    # Рисуем текущую позицию камня
                    if trajectory_points:
                        current_pos = trajectory_points[-1]
                        cv2.circle(img, current_pos, 6, color, -1, cv2.LINE_AA)
                        cv2.circle(img, current_pos, 8, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Добавляем ID трека
                        cv2.putText(img, f"ID:{stone_id}", 
                                   (current_pos[0] + 10, current_pos[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Накладываем шаблон поля с прозрачностью поверх треков
                if (field_x1 >= 0 and field_y1 >= 0 and 
                    field_x2 <= w_px and field_y2 <= h_px):
                    # Накладываем шаблон поля с прозрачностью (alpha = 0.7)
                    alpha = 0.7
                    img[field_y1:field_y2, field_x1:field_x2] = (
                        alpha * field_template + (1 - alpha) * img[field_y1:field_y2, field_x1:field_x2]
                    ).astype(np.uint8)
                
                # Добавляем легенду в левом верхнем углу
                legend_y = 30
                cv2.putText(img, "CURLING ANALYSIS", (10, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                legend_y += 35
                
                # Информация о трекере
                cv2.putText(img, "Tracker: SimpleTracker", (10, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                legend_y += 25
                
                cv2.putText(img, f"Frame: {frame_num}", (10, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                legend_y += 25
                
                # Легенда команд
                cv2.circle(img, (20, legend_y), 8, self.colors["red"], -1)
                cv2.putText(img, f"Red Team ({len(red_stones)} stones)", (35, legend_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                legend_y += 25
                
                cv2.circle(img, (20, legend_y), 8, self.colors["yellow"], -1)
                cv2.putText(img, f"Yellow Team ({len(yellow_stones)} stones)", (35, legend_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                out.write(img)
                pbar.update(1)
        
        out.release()
        print(f"Bird's eye view видео сохранено: {output_path}")
        return output_path


def create_visualizer() -> CurlingVisualizer:
    """
    Создает визуализатор
    
    Returns:
        Экземпляр CurlingVisualizer
    """
    return CurlingVisualizer()
