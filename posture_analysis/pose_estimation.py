import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self, static_mode=True, selfie_mode=False):
        """
        PoseEstimator initializes MediaPipe Pose for static image processing.
        Args:
            static_mode (bool): Whether to treat input as a static image.
            selfie_mode (bool): Whether the input image is a mirrored selfie (front camera).
        """
        self.pose = mp.solutions.pose.Pose(static_image_mode=static_mode)
        self.is_selfie = selfie_mode

    def get_landmarks(self, image):
        """
        Detect body landmarks (keypoints) from a given image using MediaPipe Pose.
        Converts normalized coordinates to pixel coordinates.
        Corrects mirrored images if selfie_mode=True.
        Returns:
            List of dictionaries with x, y, z, visibility OR None if no pose found.
        """

        # Convert image to RGB (MediaPipe format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # Run pose detection
        results = self.pose.process(image_rgb)

        # If no landmarks are detected, return None
        if not results.pose_landmarks:
            return None

        # Convert normalized coordinates → pixel coordinates
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": int(lm.x * width),
                "y": int(lm.y * height),
                "z": lm.z,  # z stays normalized (depth info)
                "visibility": lm.visibility
            })

        # If image is mirrored (selfie/front camera), flip horizontally
        if self.is_selfie:
            landmarks = self._correct_mirrored_landmarks(landmarks, width)

        return landmarks

    def _correct_mirrored_landmarks(self, landmarks, image_width):
        """
        Correct mirrored coordinates for selfie (front camera) images.
        Flips the x-coordinates horizontally to match real-world left/right.
        """
        corrected = []
        for lm in landmarks:
            corrected.append({
                "x": image_width - lm["x"],
                "y": lm["y"],
                "z": lm["z"],
                "visibility": lm["visibility"]
            })
        return corrected
