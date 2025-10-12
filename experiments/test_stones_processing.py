import os
import cv2
import numpy as np
from src.config_loader import load_config
from src.keypoints_detection import detect_keypoints
from src.homography import compute_homography_matrix
from src.stones_processing import detect_and_track_stones

import json

from src.utils.extract_first_frame import extract_first_frame

def main():
    config = load_config()

    video_path = os.path.join(config['paths']['raw_data'], "example_4.mp4")

    first_frame_path = os.path.join(config['paths']['raw_data'], "house_image.jpg")
    extract_first_frame(video_path, first_frame_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"Изображение дома не найдено: {first_frame_path}")

    keypoints_pxls = detect_keypoints(first_frame_path)

    homography_matrix = compute_homography_matrix(keypoints_pxls)

    stone_tracks = detect_and_track_stones(video_path, homography_matrix)

    print(f"\nОбработано {len(stone_tracks)} треков:")
    for track in stone_tracks:
        print(f"  ID: {track['stone_id']} | Команда: {track['team']} | Кадров: {len(track['positions_px'])}")
        if len(track['positions_real']) > 0:
            final = track['final_position_real']
            print(f"     Финальная позиция: ({final['x']:.2f}, {final['y']:.2f}) м")

    output_path = os.path.join(config['paths']['output_dir'], "stone_tracks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stone_tracks, f, indent=2, ensure_ascii=False)
    print(f"\nРезультаты сохранены в: {output_path}")

if __name__ == "__main__":
    main()
