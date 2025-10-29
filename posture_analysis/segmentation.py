# posture_analysis/segmentation.py
from ultralytics import YOLO
import cv2
import numpy as np

class BodySegmentation:
    """
    YOLO-based body segmentation for leg masks.
    Produces a binary mask: white = body, black = background.
    """
    def __init__(self, model_path="assets/models/yolov8n-seg.pt"):
        self.model = YOLO(model_path)

    def get_mask(self, image):
        """
        Predicts body segmentation mask from an image.
        Returns: np.ndarray of shape (H, W), dtype=uint8
        """
        results = self.model.predict(source=image, verbose=False)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for r in results:
            if hasattr(r, "masks") and r.masks is not None:
                for m in r.masks.data:
                    mask_resized = cv2.resize(
                        (m.cpu().numpy() * 255).astype(np.uint8),
                        (image.shape[1], image.shape[0])
                    )
                    mask = cv2.bitwise_or(mask, mask_resized)

        # Ensure mask is strictly 0 or 255
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        return mask
