from ultralytics import YOLO
import cv2
import numpy as np

class BodySegmentation:
    def __init__(self, model_path="assets/models/yolov8n-seg.pt"):
        self.model = YOLO(model_path)

    def get_mask(self, image):
        results = self.model.predict(source=image, verbose=False)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for r in results:
            if hasattr(r, "masks") and r.masks is not None:
                for m in r.masks.data:
                    mask = cv2.bitwise_or(mask, (m.cpu().numpy() * 255).astype(np.uint8))
        return mask
