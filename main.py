# from src.s3_utils import download_file_from_s3, list_s3_objects
from src.keypoints_detection import detect_keypoints
from src.homography import compute_homography_matrix
from src.stones_processing import detect_and_track_stones
from src.visualizations import create_birdseye_view, plot_throw_progression
from src.config_loader import load_config
import os
import cv2

def main():
    config = load_config()

    # bucket = config['s3']['bucket_name']
    # video_files = list_s3_objects(bucket)
    # for video_file in video_files:
    #     if video_file.endswith(".mp4"):
    #         local_path = os.path.join(config['paths']['raw_data'], video_file.split('/')[-1])
    #         download_file_from_s3(bucket, video_file, local_path)

    # Extract first frame from video as reference image
    video_path = os.path.join(config['paths']['video_data'], "example_4.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Не удалось прочитать первый кадр из видео")

    # Detect keypoints on first frame (model path from config handled inside)
    keypoints = detect_keypoints(frame0)

    homography_matrix = compute_homography_matrix(keypoints)

    # Используем ByteTrack для более точного трекинга
    stone_tracks = detect_and_track_stones(video_path, homography_matrix, use_bytetrack=True)

    create_birdseye_view(stone_tracks, config['paths']['output_dir'])
    plot_throw_progression(stone_tracks, config['paths']['output_dir'])
    
    # Создаем анимированное видео с ByteTrack
    from src.animated_visualizations import create_birdseye_video_with_legend
    video_path_animated = create_birdseye_video_with_legend(
        stone_tracks=stone_tracks,
        output_dir=config['paths']['output_dir'],
        fps=30,
        court_size_cm=(500, 1300),
        img_size=(800, 2080),
        use_bytetrack=True
    )
    print(f"✅ Анимированное видео создано: {video_path_animated}")

    # Optional judging/statistics could be implemented later

    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()
