import cv2
from posture_analysis.segmentation import BodySegmentation
from posture_analysis.pose_estimation import PoseEstimator
from posture_analysis.fusion import AlignmentAnalyzer

def main(image_path):
    # Load image
    image = cv2.imread(image_path)

    # --- Step 1: Segmentation ---
    segmenter = BodySegmentation()
    mask = segmenter.get_mask(image)

    # --- Step 2: Pose Estimation ---
    pose_estimator = PoseEstimator()
    landmarks = pose_estimator.get_landmarks(image)

    # --- Step 3: Fusion / Analysis ---
    analyzer = AlignmentAnalyzer()
    result = analyzer.evaluate_leg_alignment(landmarks, mask)

    # --- Step 4: Display Result ---
    print(result["summary"])
    cv2.imshow("Result", result["visualized_image"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = "user_data/samples/test_image.jpg"
    main(path)
