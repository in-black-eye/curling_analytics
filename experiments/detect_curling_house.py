# Эксперимент по сегментации эллипсов дома для кёрлинга с помощью SAM
# Используется vit_b

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- 1. Загрузка модели SAM ---
sam_checkpoint = "sam_weights/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# --- 2. Загрузка изображения ---
image_path = "/Users/glytov_ivan/Documents/curling/curling_analytics/data/first_frame.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# --- 3. Цветовая сегментация в HSV ---
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = mask_red1 | mask_red2

# --- 4. 🔴 ЗАМЕНА: вместо cv2.moments — фильтрация связных компонент ---
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_red, connectivity=8)

# Параметры фильтрации (настройте под ваше видео!)
min_area = 1000        # минимальная площадь (в пикселях)
max_y = image.shape[0] * 0.8  # дом почти никогда не выше 80% высоты кадра

valid_components = []
for i in range(1, num_labels):  # 0 — это фон
    area = stats[i, cv2.CC_STAT_AREA]
    cy = centroids[i][1]
    
    if area >= min_area and cy <= max_y:
        valid_components.append((area, centroids[i]))

if valid_components:
    # Выбираем самую большую компоненту
    valid_components.sort(key=lambda x: x[0], reverse=True)
    cx, cy = valid_components[0][1]
    cx, cy = int(cx), int(cy)
    prompt_point = np.array([[cx, cy]])
    print(f"✅ Центр дома (после фильтрации): ({cx}, {cy})")
    
    # --- 5. Визуализация: отрисовка центра на исходном изображении ---
    image_with_center = image.copy()
    cv2.circle(image_with_center, (cx, cy), radius=10, color=(0, 0, 255), thickness=-1)  # красная точка
    cv2.circle(image_with_center, (cx, cy), radius=12, color=(255, 255, 255), thickness=2)  # белая окантовка
    
    cv2.imwrite("first_frame_with_center.jpg", image_with_center)
    print("✅ Изображение с отмеченным центром сохранено как 'first_frame_with_center.jpg'")
    
    cv2.imshow("Исходное изображение", image)
    cv2.imshow("Цветовая маска (красный)", mask_red)
    cv2.imshow("Центр дома", image_with_center)
    print("Нажмите любую клавишу в любом окне OpenCV, чтобы продолжить...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("❌ Ни одна компонента не прошла фильтрацию")
    cv2.imshow("Маска (без подходящих компонент)", mask_red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()



point_x = cx
point_y = cy
# Это ваш ПРОМПТ для SAM:
input_point = np.array([[point_x, point_y]])
input_label = np.array([1])  # 1 = "это объект, который мы хотим выделить"

# --- 4. Запуск SAM ---
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,  # хотим одну маску
)

mask = masks[0]  # бинарная маска (True/False)

# --- 5. Найти контуры и эллипс ---
mask_uint8 = (mask * 255).astype(np.uint8)
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Берём самый большой контур (дом — самый крупный круг)
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) >= 5:  # нужно минимум 5 точек для fitEllipse
        ellipse = cv2.fitEllipse(largest_contour)
        center, axes, angle = ellipse
        cx, cy = int(center[0]), int(center[1])
        
        print(f"Центр дома: ({cx}, {cy})")
        
        # Визуализация
        output = image.copy()
        cv2.ellipse(output, ellipse, (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imwrite("/Users/glytov_ivan/Documents/curling/curling_analytics/data/result.jpg", output)
        print("Результат сохранён в result.jpg")

        # --- Автоматическое получение 4 точек из эллипса ---
        (cx_img, cy_img), (major_axis, minor_axis), angle_deg = ellipse
        a = major_axis / 2.0
        b = minor_axis / 2.0
        angle_rad = np.radians(angle_deg)

        # Точки на изображении (src)
        p0 = (cx_img, cy_img)
        p1 = (cx_img + a * np.cos(angle_rad), cy_img + a * np.sin(angle_rad))  # правый
        p2 = (cx_img - a * np.cos(angle_rad), cy_img - a * np.sin(angle_rad))  # левый
        p3 = (cx_img - b * np.sin(angle_rad), cy_img + b * np.cos(angle_rad))  # вперёд (перпендикуляр)

        src_points = np.array([p0, p1, p2, p3], dtype=np.float32)

        # Точки на 2D-карте (dst) — в условных единицах
        dst_points = np.array([
            [0.0, 0.0],    # центр
            [1.0, 0.0],    # правый
            [-1.0, 0.0],   # левый
            [0.0, -1.0]    # вперёд
        ], dtype=np.float32)

        # Вычисляем гомографию
        H, mask = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)

        print("✅ Гомография вычислена автоматически!")
        print("Матрица H:\n", H)

        # Сохраняем
        np.save("/Users/glytov_ivan/Documents/curling/curling_analytics/data/homography.npy", H)

        # Визуализация всех 4 точек
        output_final = image.copy()
        cv2.ellipse(output_final, ellipse, (0, 255, 0), 2)
        colors = [(0,0,255), (255,0,0), (255,0,0), (0,255,255)]
        labels = ["Центр", "Право", "Лево", "Вперёд"]
        for i, pt in enumerate([p0, p1, p2, p3]):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(output_final, (x, y), 6, colors[i], -1)
            cv2.putText(output_final, labels[i], (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

        cv2.imwrite("/Users/glytov_ivan/Documents/curling/curling_analytics/data/final_homography_points.jpg", output_final)
        print("✅ Финальное изображение с точками сохранено.")

    else:
        print("Не удалось найти эллипс: контур слишком мал")
else:
    print("Контуры не найдены")