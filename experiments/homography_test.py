from src.keypoints_detection import detect_keypoints, visualize_keypoints
from src.homography import compute_homography_matrix, validate_homography


image_path = "/home/jovyan/curling_analytics/data/raw/inference_2.jpg"
keypoints = detect_keypoints(image_path)

# print("Ключевые точки:", keypoints)

visualize_keypoints(image_path, keypoints, "outputs/keypoints_test.jpg")

hom_matrix = compute_homography_matrix(keypoints)

validate_homography(hom_matrix, image_path, output_path="outputs/homography_test.jpg")