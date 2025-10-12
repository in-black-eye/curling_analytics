import os
import cv2
import numpy as np

def create_birdseye_view(stone_tracks, output_dir: str, court_size_cm=(500, 1300), img_size=(800, 2080)):
    """
    Рисует треки камней в координатах реального поля (см) в виде birds-eye.
    - court_size_cm: (width_cm, height_cm)
    - img_size: (width_px, height_px) результирующего изображения
    """
    os.makedirs(output_dir, exist_ok=True)
    w_px, h_px = img_size
    img = np.ones((h_px, w_px, 3), dtype=np.uint8) * 255

    cw, ch = court_size_cm
    sx = w_px / float(cw) if cw > 0 else 1.0
    sy = h_px / float(ch) if ch > 0 else 1.0

    colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255)
    }

    for track in stone_tracks:
        team = track.get("team", "red")
        color = colors.get(team, (0, 0, 0))
        pts = []
        for pos in track.get("positions_real", []):
            x_cm, y_cm = pos["x"], pos["y"]
            x_px = int(round(x_cm * sx))
            y_px = int(round((ch - y_cm) * sy))  # origin at bottom-left
            pts.append((x_px, y_px))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], color, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[-1], 8, color, -1, cv2.LINE_AA)

    out_path = os.path.join(output_dir, "birdseye.png")
    cv2.imwrite(out_path, img)
    return out_path


def plot_throw_progression(stone_tracks, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # Placeholder: could be implemented with matplotlib; currently no-op
    return os.path.join(output_dir, "progression.txt")
