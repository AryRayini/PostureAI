import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self):
        # Initialize MediaPipe Pose model
        # static_image_mode=True → designed for single images, not continuous video
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def get_landmarks(self, image):
        """
        Detect body landmarks (keypoints) from a given image using MediaPipe Pose.
        Returns a list of (x, y, z) tuples for each detected landmark.
        If no pose is found, returns None.
        """
        
        # Convert image from BGR (OpenCV format) to RGB (MediaPipe format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run pose detection on the image
        results = self.pose.process(image_rgb)

        # If no landmarks are detected, return None
        if not results.pose_landmarks:
            return None

        # Extract all detected landmark coordinates
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))  # (normalized x, y, z)

        # Return list of (x, y, z) positions for all body keypoints
        return landmarks
