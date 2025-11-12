# posture_analysis/segmentation.py
from ultralytics import YOLO
import cv2
import numpy as np

class BodySegmentation:
    """
    YOLO-based body segmentation for leg masks.
    Produces a binary mask: white = body, black = background.
    """
    def __init__(self, model_path="assets/models/yolov8n-seg.pt", conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def get_mask(self, image, verbose=False):
        """
        Predicts body segmentation mask from an image.
        Only includes detections of 'person' class (class 0 in COCO).
        Args:
            image: Input image
            verbose: Whether to print debug information
        Returns: np.ndarray of shape (H, W), dtype=uint8
        """
        # Run prediction with confidence threshold
        # Note: YOLO class 0 is 'person' in COCO dataset
        results = self.model.predict(
            source=image, 
            verbose=False,
            conf=self.conf_threshold,
            classes=[0]  # Only detect person class
        )
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        person_detected = False

        for r in results:
            # Process masks if available (already filtered to person class by classes=[0])
            if hasattr(r, "masks") and r.masks is not None and len(r.masks.data) > 0:
                # Get confidence scores from boxes if available for logging
                confidences = []
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                    try:
                        # Extract confidences from boxes tensor
                        confidences = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else r.boxes.conf.numpy()
                    except:
                        try:
                            confidences = [float(c) for c in r.boxes.conf] if hasattr(r.boxes.conf, '__iter__') else []
                        except:
                            pass
                
                # Process each mask (they're already filtered to person class)
                for i, m in enumerate(r.masks.data):
                    try:
                        mask_resized = cv2.resize(
                            (m.cpu().numpy() * 255).astype(np.uint8),
                            (image.shape[1], image.shape[0])
                        )
                        mask = cv2.bitwise_or(mask, mask_resized)
                        person_detected = True
                        if verbose and i < len(confidences):
                            print(f"✓ Person detected with confidence: {confidences[i]:.2f}")
                    except Exception as e:
                        if verbose:
                            print(f"⚠ Warning processing mask {i}: {e}")
                        continue
                        
            # If no masks but boxes exist, person was detected but segmentation failed
            elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                try:
                    # Check if any boxes have sufficient confidence
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else r.boxes.conf.numpy()
                    if len(confs) > 0 and np.any(confs >= self.conf_threshold):
                        person_detected = True
                        if verbose:
                            max_conf = float(np.max(confs))
                            print(f"✓ Person detected (box only, confidence: {max_conf:.2f}), but no segmentation mask available")
                except Exception as e:
                    if verbose:
                        print(f"⚠ Could not check box confidences: {e}")

        if verbose:
            if not person_detected:
                print("⚠ Warning: No person detected by YOLO segmentation!")
                print(f"  - Confidence threshold: {self.conf_threshold}")
                print("  - Try lowering conf_threshold if person is visible in image")
            else:
                mask_pixels = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                coverage = (mask_pixels / total_pixels) * 100
                print(f"✓ Mask created: {coverage:.1f}% of image covered")

        # Ensure mask is strictly 0 or 255
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        return mask
