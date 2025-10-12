import os
import cv2
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm


def create_curling_field_template(field_center: Tuple[int, int], scale_cm_per_px: float = 1.0) -> np.ndarray:
    """
    Создает полный шаблон поля для кёрлинга согласно официальным правилам.
    
    Параметры разметки (в метрах):
    - tee_to_back = 1.829          # расстояние от центра до наружного края back line
    - tee_to_hog = 6.401           # расстояние от центра до внутреннего края hog line
    - line_width = 0.0127          # ширина тонких линий (tee, back)
    - hog_line_width = 0.1016      # ширина hog line
    
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
    
    # Координаты центра поля
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
    
    # Back line (наружный край на y = cy - tee_to_back)
    back_line_y = cy - int(tee_to_back / scale_m_per_px)
    back_line_width_px = max(1, int(line_width / scale_m_per_px))
    if back_line_width_px > 0 and back_line_y >= 0:
        back_line = np.zeros((back_line_width_px, field_width_px, 3), dtype=np.uint8)
        y_start = max(0, back_line_y - back_line_width_px//2)
        y_end = min(field_height_px, back_line_y + back_line_width_px//2)
        if y_end > y_start:
            img[y_start:y_end, :] = back_line[:y_end-y_start, :]
    
    # Hog line (внутренний край на y = cy + tee_to_hog)
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
    
    return img


def create_curling_house_template(house_center: Tuple[int, int], scale_cm_per_px: float = 1.0) -> np.ndarray:
    """
    Создает шаблон дома для кёрлинга (упрощенная версия для обратной совместимости).
    
    Args:
        house_center: (x, y) координаты центра дома в пикселях
        scale_cm_per_px: масштаб - сколько сантиметров в одном пикселе
        
    Returns:
        numpy array с шаблоном дома
    """
    # Радиусы в метрах согласно коду пользователя
    radii_m = [1.829, 1.219, 0.610, 0.1524]  # 6ft, 4ft, 2ft, 6in
    colors = ['red', 'white', 'blue', 'white']
    
    # Преобразуем в пиксели (метры -> сантиметры -> пиксели)
    radii_px = [int(radius_m * 100 / scale_cm_per_px) for radius_m in radii_m]
    
    # Размер изображения (с запасом для линий)
    max_radius_px = max(radii_px)
    img_size = max_radius_px * 2 + 100  # запас для линий
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Координаты центра относительно изображения
    cx, cy = img_size // 2, img_size // 2
    
    # Рисуем четыре концентрических круга согласно правилам
    for radius_px, color in zip(radii_px, colors):
        if color == 'red':
            cv2.circle(img, (cx, cy), radius_px, (0, 0, 255), 3, cv2.LINE_AA)  # Красный в BGR
        elif color == 'white':
            cv2.circle(img, (cx, cy), radius_px, (255, 255, 255), 3, cv2.LINE_AA)  # Белый в BGR
        elif color == 'blue':
            cv2.circle(img, (cx, cy), radius_px, (255, 0, 0), 3, cv2.LINE_AA)  # Синий в BGR
    
    # Центральная точка (кнопка) - маленькая черная точка
    cv2.circle(img, (cx, cy), 2, (0, 0, 0), -1, cv2.LINE_AA)
    
    # Линии через центр (крест)
    cross_length = 20
    cv2.line(img, (cx - cross_length, cy), (cx + cross_length, cy), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - cross_length), (cx, cy + cross_length), (0, 0, 0), 2, cv2.LINE_AA)
    
    return img


def create_birdseye_video(stone_tracks: List[Dict], output_dir: str, 
                         fps: int = 30, court_size_cm: Tuple[int, int] = (500, 1300),
                         img_size: Tuple[int, int] = (800, 2080)) -> str:
    """
    Создает анимированное видео bird's-eye view с траекториями камней.
    
    Args:
        stone_tracks: список треков камней
        output_dir: директория для сохранения видео
        fps: FPS выходного видео
        court_size_cm: размеры поля в см (width, height)
        img_size: размеры изображения в пикселях (width, height)
        
    Returns:
        путь к созданному видео файлу
    """
    os.makedirs(output_dir, exist_ok=True)
    
    w_px, h_px = img_size
    cw, ch = court_size_cm
    
    # Коэффициенты масштабирования
    sx = w_px / float(cw) if cw > 0 else 1.0
    sy = h_px / float(ch) if ch > 0 else 1.0
    
    # Цвета команд
    colors = {
        "red": (0, 0, 255),      # BGR: красный
        "yellow": (0, 255, 255)  # BGR: желтый
    }
    
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
    
    print(f"Создание видео: {total_frames} кадров, {fps} FPS")
    
    # Настройка видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, "birdseye_animation.mp4")
    out = cv2.VideoWriter(video_path, fourcc, fps, (w_px, h_px))
    
    # Создаем шаблон поля
    field_center = (w_px // 2, int(h_px * 0.6))  # Поле в верхней части изображения
    
    # Вычисляем масштаб: сколько сантиметров в одном пикселе
    scale_cm_per_px = cw / w_px  # ширина поля в см / ширина изображения в пикселях
    field_template = create_curling_field_template(field_center, scale_cm_per_px)
    
    # Накладываем шаблон поля на каждое изображение
    field_x1 = field_center[0] - field_template.shape[1] // 2
    field_y1 = field_center[1] - field_template.shape[0] // 2
    field_x2 = field_x1 + field_template.shape[1]
    field_y2 = field_y1 + field_template.shape[0]
    
    # Обрабатываем каждый кадр
    with tqdm(total=total_frames, desc="Создание видео", unit="кадр") as pbar:
        for frame_num in range(min_frame, max_frame + 1):
            # Создаем белый фон
            img = np.ones((h_px, w_px, 3), dtype=np.uint8) * 255
            
            # Накладываем шаблон поля
            if (field_x1 >= 0 and field_y1 >= 0 and 
                field_x2 <= w_px and field_y2 <= h_px):
                img[field_y1:field_y2, field_x1:field_x2] = field_template
            
            # Рисуем траектории и текущие позиции камней
            for track in stone_tracks:
                team = track.get("team", "red")
                color = colors.get(team, (0, 0, 0))
                
                # Рисуем траекторию до текущего кадра
                trajectory_points = []
                for pos in track.get("positions_real", []):
                    if pos["frame"] <= frame_num:
                        x_cm, y_cm = pos["x"], pos["y"]
                        x_px = int(round(x_cm * sx))
                        y_px = int(round((ch - y_cm) * sy))  # origin at bottom-left
                        trajectory_points.append((x_px, y_px))
                
                # Рисуем траекторию как линию
                if len(trajectory_points) > 1:
                    cv2.line(img, trajectory_points[0], trajectory_points[-1], 
                            color, 2, cv2.LINE_AA)
                
                # Рисуем все предыдущие позиции как точки
                for point in trajectory_points:
                    cv2.circle(img, point, 3, color, -1, cv2.LINE_AA)
                
                # Рисуем текущую позицию камня (если есть) как более крупный круг
                if trajectory_points:
                    current_pos = trajectory_points[-1]
                    # Рисуем камень как круг
                    cv2.circle(img, current_pos, 6, color, -1, cv2.LINE_AA)
                    # Белая обводка для контраста
                    cv2.circle(img, current_pos, 8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Добавляем информацию о кадре
            cv2.putText(img, f"Frame: {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Добавляем информацию о количестве камней
            visible_stones = sum(1 for track in stone_tracks 
                               for pos in track.get("positions_real", [])
                               if pos["frame"] <= frame_num)
            cv2.putText(img, f"Stones: {visible_stones}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            out.write(img)
            pbar.update(1)
    
    out.release()
    print(f"✅ Видео сохранено: {video_path}")
    return video_path


def create_birdseye_video_with_legend(stone_tracks: List[Dict], output_dir: str, 
                                    fps: int = 30, court_size_cm: Tuple[int, int] = (500, 1300),
                                    img_size: Tuple[int, int] = (800, 2080),
                                    use_bytetrack: bool = True) -> str:
    """
    Расширенная версия с легендой команд и поддержкой ByteTrack.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    w_px, h_px = img_size
    cw, ch = court_size_cm
    
    # Коэффициенты масштабирования
    sx = w_px / float(cw) if cw > 0 else 1.0
    sy = h_px / float(ch) if ch > 0 else 1.0
    
    # Цвета команд
    colors = {
        "red": (0, 0, 255),      # BGR: красный
        "yellow": (0, 255, 255)  # BGR: желтый
    }
    
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
    
    print(f"Создание видео с легендой: {total_frames} кадров, {fps} FPS")
    
    # Настройка видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, "birdseye_animation_with_legend.mp4")
    out = cv2.VideoWriter(video_path, fourcc, fps, (w_px, h_px))
    
    # Создаем шаблон поля
    field_center = (w_px // 2, int(h_px * 0.6))
    
    # Вычисляем масштаб: сколько сантиметров в одном пикселе
    scale_cm_per_px = cw / w_px  # ширина поля в см / ширина изображения в пикселях
    field_template = create_curling_field_template(field_center, scale_cm_per_px)
    
    # Позиция шаблона поля
    field_x1 = field_center[0] - field_template.shape[1] // 2
    field_y1 = field_center[1] - field_template.shape[0] // 2
    field_x2 = field_x1 + field_template.shape[1]
    field_y2 = field_y1 + field_template.shape[0]
    
    # Подсчитываем камни по командам
    red_stones = [t for t in stone_tracks if t.get("team") == "red"]
    yellow_stones = [t for t in stone_tracks if t.get("team") == "yellow"]
    
    with tqdm(total=total_frames, desc="Создание видео с легендой", unit="кадр") as pbar:
        for frame_num in range(min_frame, max_frame + 1):
            # Создаем белый фон
            img = np.ones((h_px, w_px, 3), dtype=np.uint8) * 255
            
            # Накладываем шаблон поля
            if (field_x1 >= 0 and field_y1 >= 0 and 
                field_x2 <= w_px and field_y2 <= h_px):
                img[field_y1:field_y2, field_x1:field_x2] = field_template
            
            # Рисуем траектории и позиции камней
            for track in stone_tracks:
                team = track.get("team", "red")
                color = colors.get(team, (0, 0, 0))
                stone_id = track.get("stone_id", "?")
                
                trajectory_points = []
                for pos in track.get("positions_real", []):
                    if pos["frame"] <= frame_num:
                        x_cm, y_cm = pos["x"], pos["y"]
                        x_px = int(round(x_cm * sx))
                        y_px = int(round((ch - y_cm) * sy))
                        trajectory_points.append((x_px, y_px))
                
                # Рисуем траекторию как линию
                if len(trajectory_points) > 1:
                    cv2.line(img, trajectory_points[0], trajectory_points[-1], 
                            color, 2, cv2.LINE_AA)
                
                # Рисуем все предыдущие позиции как точки
                for point in trajectory_points:
                    cv2.circle(img, point, 3, color, -1, cv2.LINE_AA)
                
                # Рисуем текущую позицию камня
                if trajectory_points:
                    current_pos = trajectory_points[-1]
                    cv2.circle(img, current_pos, 6, color, -1, cv2.LINE_AA)
                    cv2.circle(img, current_pos, 8, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Добавляем ID трека (только для ByteTrack)
                    if use_bytetrack:
                        cv2.putText(img, f"ID:{stone_id}", 
                                   (current_pos[0] + 10, current_pos[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Добавляем легенду в левом верхнем углу
            legend_y = 30
            cv2.putText(img, "CURLING ANALYSIS", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            legend_y += 35
            
            # Информация о трекере
            tracker_name = "ByteTrack" if use_bytetrack else "SimpleTracker"
            cv2.putText(img, f"Tracker: {tracker_name}", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            legend_y += 25
            
            cv2.putText(img, f"Frame: {frame_num}", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            legend_y += 25
            
            # Легенда команд
            cv2.circle(img, (20, legend_y), 8, colors["red"], -1)
            cv2.putText(img, f"Red Team ({len(red_stones)} stones)", (35, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            legend_y += 25
            
            cv2.circle(img, (20, legend_y), 8, colors["yellow"], -1)
            cv2.putText(img, f"Yellow Team ({len(yellow_stones)} stones)", (35, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            out.write(img)
            pbar.update(1)
    
    out.release()
    print(f"✅ Видео с легендой сохранено: {video_path}")
    return video_path
