import cv2
import os
from datetime import datetime
from posture_analysis.segmentation import BodySegmentation
from posture_analysis.pose_estimation import PoseEstimator
from posture_analysis.posture_evaluator import PostureEvaluator

def main(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- Step 1: Segmentation ---
    segmenter = BodySegmentation(model_path="assets/models/yolov8n-seg.pt")
    mask = segmenter.get_mask(image)

    # --- Step 2: Pose Estimation ---
    pose_estimator = PoseEstimator()
    landmarks = pose_estimator.get_landmarks(image)

    # --- Step 3: Fusion & Analysis ---
    analyzer = PostureEvaluator()
    print("Landmarks:", landmarks)
    result = analyzer.analyze(landmarks, mask, image)

    # --- Step 4: Display Result in resizable window ---
    print(result["summary"])
    cv2.namedWindow("Knock-Knee / Bow-Leg Detection", cv2.WINDOW_NORMAL)  # <--- make it resizable
    cv2.imshow("Knock-Knee / Bow-Leg Detection", result["visualized_image"])
    
    # --- Step 5: Save image if user wants ---
    print("Press 'y' to save the image, or any other key to exit.")
    key = cv2.waitKey(0)
    if key == ord('y'):
        # Create exports folder if it doesn't exist
        exports_dir = "exports"
        os.makedirs(exports_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base_name}_result_{timestamp}.png"
        output_path = os.path.join(exports_dir, output_filename)
        
        # Save the image
        cv2.imwrite(output_path, result["visualized_image"])
        print(f"Image saved to {output_path}.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = "user_data/legsamples/8.jpg"
    main(path)
