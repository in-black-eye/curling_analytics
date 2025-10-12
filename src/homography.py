from src.keypoints_detection import detect_keypoints
import cv2
import numpy as np

def get_real_world_coordinates():
    """
    Возвращает реальные физические координаты ключевых точек в сантиметрах.
    Позиций точки в списке совпадает с номером точки в скелете CVAT (1 -> 22)  
    """
    real_coords = np.array([
    (0, 731.54), # "l_hog"
    (500, 731.54), # "r_hog"
    (0, 365.76), # "l_back"
    (500, 365.76), # "r_back"
    (0, 548.64), # "l_tee"
    (500, 548.64), # "r_tee"
    (250, 1188.72), # "hog_central"
    (250, 365.76), # "p_central"
    (250, 548.64), # "center"
    (250, 1188.72), # "a_central"
    (250, 563.84), # "button_ant"
    (265.2, 548.64), # "button_r"
    (250, 533.44), # "button_post"
    (234.8, 548.64), # "button_l"
    (250, 609.64), # "inner_ant"
    (311, 548.64), # "inner_r"
    (250, 487.64), # "inner_post"
    (189, 548.64), # "inner_l"
    (250, 670.54), # "outer_ant"
    (371.9, 548.64), # "outer_r"
    (250, 426.74), # "outer_post"
    (128.1, 548.64) # "outer_l"
    ], dtype=np.float32)
    
    return real_coords

def compute_homography_matrix(keypoints_pxls):
    """
    keypoints_pxls: list of tuples (x,y) in pxls or (x,y,idx)

    return: homography_matrix
    """
    if len(keypoints_pxls) == 0:
        return None

    real_coords = get_real_world_coordinates()

    # keypoints_pxls may contain tuples (x,y) or (x,y,idx). We'll map by idx when possible.
    src_points = [kp if len(kp) == 2 else (kp[0], kp[1]) for kp in keypoints_pxls]
    idxs = [None if len(kp) == 2 else int(kp[2]) for kp in keypoints_pxls]

    # If indices provided (0..21), sort and align with corresponding real coordinates
    if all(i is not None for i in idxs):
        pairs = sorted(zip(idxs, src_points), key=lambda t: t[0])
        ordered_src = []
        ordered_dst = []
        
        for idx, src_pt in pairs:
            if 0 <= idx < len(real_coords):
                ordered_src.append(src_pt)
                ordered_dst.append(real_coords[idx])
        
        if len(ordered_src) < 4:
            print(f"Недостаточно точек для гомографии: найдено {len(ordered_src)}, минимум 4")
            return None
            
        print(f"Используем {len(ordered_src)} из {len(real_coords)} точек для гомографии")
        src_points = ordered_src
        dst_points = ordered_dst
    else:
        # Fallback: use all detected points with corresponding real coordinates
        if len(src_points) < 4:
            print(f"Недостаточно точек для гомографии: найдено {len(src_points)}, минимум 4")
            return None
            
        if len(src_points) > len(real_coords):
            print(f"Слишком много точек: найдено {len(src_points)}, максимум {len(real_coords)}")
            src_points = src_points[:len(real_coords)]
        
        # Take corresponding real coordinates
        dst_points = real_coords[:len(src_points)].tolist()
        print(f"Используем {len(src_points)} точек для гомографии")

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    if H is None:
        print("Не удалось вычислить гомографию")
        return None

    inliers = np.sum(status)
    total = len(status)
    print(f"Inliers: {inliers}/{total} ({inliers/total*100:.1f}%)")

    return H

def validate_homography(H: np.ndarray, image_path: str, output_path: str = None):
    """ Накладывает реальные координаты обратно на изображение.
    """
    
    image = cv2.imread(image_path)

    real_coords = get_real_world_coordinates()
    real_coords = np.array(real_coords, dtype=np.float32).reshape(-1, 1, 2)

    # Обратная гомография 
    H_inv = np.linalg.inv(H)
    pixel_coords = cv2.perspectiveTransform(real_coords, H_inv)

    for i, pt in enumerate(pixel_coords):
        x, y = int(pt[0][0]), int(pt[0][1])
        cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(image, str(i), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Инференс гомографии: {output_path}")

    return image