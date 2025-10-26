from ultralytics import YOLO
import cv2
import numpy as np

class BodySegmentation:
    def __init__(self, model_path="assets/models/yolov8n-seg.pt"):
        # Load YOLO segmentation model
        self.model = YOLO(model_path)

    def get_mask(self, image):
        # --- Step 1: Run segmentation ---
        results = self.model.predict(source=image, verbose=False)

        # --- Step 2: Initialize empty mask ---
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # --- Step 3: Extract body masks from YOLO output ---
        for r in results:
            if hasattr(r, "masks") and r.masks is not None:
                for m in r.masks.data:
                    # Resize mask to match original image
                    mask_resized = cv2.resize((m.cpu().numpy() * 255).astype(np.uint8),
                                            (image.shape[1], image.shape[0]))
                    mask = cv2.bitwise_or(mask, mask_resized)

        # --- Step 4: Return final mask (white = body, black = background) ---
        return mask
