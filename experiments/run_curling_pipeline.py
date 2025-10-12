import os
import sys as _sys
from pathlib import Path as _Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# Ensure project root on PYTHONPATH
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from src.config_loader import load_config
from src.keypoints_detection import load_keypoints_model, detect_keypoints
from src.homography import compute_homography_matrix
from src.visualizations import create_birdseye_view
from src.animated_visualizations import create_birdseye_video_with_legend
from experiments.detect_stones_yolo import detect_stones_yolo


def read_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")
    ok, frame0 = cap.read()
    cap.release()
    if not ok or frame0 is None:
        raise RuntimeError("Не удалось прочитать первый кадр из видео")
    return frame0


def main():
    parser = argparse.ArgumentParser(description="Run full curling pipeline: keypoints -> homography -> stones(YOLO) -> birds-eye")
    parser.add_argument(
        "--video",
        default="data/videos/example_4.mp4",
        help="Path to input video (relative to project root or absolute)",
    )
    parser.add_argument(
        "--keypoints-weights",
        default="outputs/runs/yolo11_boxes25/train_20251009_152028/weights/best.pt",
        help="Path to YOLO keypoint-detector (22 boxes) weights",
    )
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold for keypoint detections")
    parser.add_argument(
        "--stones-weights",
        default="/Users/glytov_ivan/Documents/curling/curling_analytics/models/stones_detection_and_classification_11s.pt",
        help="YOLO weights for stone detection",
    )
    parser.add_argument("--stones-conf", type=float, default=0.25, help="Confidence threshold for stone detections")
    parser.add_argument("--video-fps", type=int, default=15, help="FPS for output video")
    args = parser.parse_args()

    cfg = load_config()
    output_dir = cfg["paths"].get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    video_path = args.video if os.path.isabs(args.video) else os.path.join(str(_PROJECT_ROOT), args.video)
    weights_path = args.keypoints_weights if os.path.isabs(args.keypoints_weights) else os.path.join(str(_PROJECT_ROOT), args.keypoints_weights)

    print("Запуск пайплайна анализа кёрлинга...")
    
    # 1) Load keypoint model (sets global cache)
    print("Загрузка модели детекции ключевых точек...")
    load_keypoints_model(weights_path)

    # 2) Read first frame and detect keypoints (pixel coords)
    print("Детекция ключевых точек на первом кадре...")
    frame0 = read_first_frame(video_path)
    keypoints = detect_keypoints(frame0, conf_threshold=args.conf)
    if len(keypoints) == 0:
        raise RuntimeError("Ключевые точки не найдены на первом кадре. Проверьте путь к весам и порог conf.")
    
    print(f"Найдено {len(keypoints)} ключевых точек")

    # 3) Compute homography from pixel -> real-world (см)
    print("Вычисление матрицы гомографии...")
    H = compute_homography_matrix(keypoints)
    if H is None or not isinstance(H, np.ndarray) or H.size == 0:
        raise RuntimeError("Не удалось вычислить матрицу гомографии. Попробуйте снизить порог conf или проверить качество детекции ключевых точек.")
    print("Матрица гомографии вычислена успешно")

    # 4) Detect stones with YOLO model (no tracker) and map via H
    print("Детекция камней в видео...")
    stones_weights = args.stones_weights if os.path.isabs(args.stones_weights) else os.path.join(str(_PROJECT_ROOT), args.stones_weights)
    stone_tracks = detect_stones_yolo(video_path, stones_weights, H, conf=args.stones_conf)
    print(f"Обработано {len(stone_tracks)} детекций камней")

    # 5) Create static bird's-eye view
    print("Создание статичного bird's-eye view...")
    birds_out = create_birdseye_view(stone_tracks, output_dir)

    # 6) Create animated bird's-eye view video
    print("Создание анимированного bird's-eye view видео...")
    video_out = create_birdseye_video_with_legend(stone_tracks, output_dir, fps=args.video_fps)

    print("Pipeline finished successfully!")
    print(f"Статичный bird's-eye: {birds_out}")
    print(f"Анимированное видео: {video_out}")


if __name__ == "__main__":
    main()


