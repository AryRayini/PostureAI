import cv2
import numpy as np
from posture_analysis.utils import calculate_angle

class AlignmentAnalyzer:
    def evaluate_leg_alignment(self, landmarks, mask):
        """
        Analyze leg alignment (normal, knock-knee, or bow-leg)
        using MediaPipe landmarks and YOLO segmentation mask.
        """

        # If no landmarks were detected → can't analyze anything
        if landmarks is None:
            return {"summary": "No person detected", "visualized_image": mask}

        # === Key landmark indices from MediaPipe Pose ===
        # Left leg: hip=23, knee=25, ankle=27
        # Right leg: hip=24, knee=26, ankle=28
        left_hip, left_knee, left_ankle = 23, 25, 27
        right_hip, right_knee, right_ankle = 24, 26, 28

        # Calculate the knee joint angles (hip–knee–ankle)
        left_angle = calculate_angle(landmarks[left_hip], landmarks[left_knee], landmarks[left_ankle])
        right_angle = calculate_angle(landmarks[right_hip], landmarks[right_knee], landmarks[right_ankle])

        # === Thresholds for classification (tunable by physiotherapist) ===
        threshold_knock = 165   # below this → likely knock-knee
        threshold_bow = 175     # above this → likely bow-leg

        # Default assumption
        condition = "Normal"

        # Classify based on both leg angles
        if left_angle < threshold_knock and right_angle < threshold_knock:
            condition = "Knock-Knee"
        elif left_angle > threshold_bow and right_angle > threshold_bow:
            condition = "Bow-Leg"

        # Build readable summary string
        summary = f"Left angle: {left_angle:.2f}, Right angle: {right_angle:.2f} → {condition}"

        # === Optional visualization ===
        # Convert mask to color so we can draw text on it
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, condition, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        # Return both textual result and visual overlay
        return {"summary": summary, "visualized_image": vis}
