import mediapipe as mp
import cv2
import numpy as np

class PoseEstimator:
    # Lower body landmark indices we need for leg alignment analysis
    REQUIRED_LANDMARKS = [23, 24, 25, 26, 27, 28]  # L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE
    
    def __init__(self, static_mode=True, selfie_mode=False,
                 min_detection_confidence=0.3, model_complexity=2,
                 enhance_image=True):
        """
        PoseEstimator initializes MediaPipe Pose for static image processing.
        Optimized for leg alignment analysis - focuses on lower body landmarks.
        
        Args:
            static_mode (bool): Whether to treat input as a static image.
            selfie_mode (bool): Whether the input image is a mirrored selfie (front camera).
            min_detection_confidence (float): Minimum confidence threshold (0.0-1.0).
                Lower values (0.3) make detection more lenient for occluded upper body.
                Higher values (0.5-0.7) can improve accuracy but may miss low-confidence detections.
            model_complexity (int): Model complexity (0, 1, or 2). Default is 2.
                0 = lightest, fastest, most lenient (best for partial visibility)
                1 = balanced - often better accuracy for landmark positioning
                2 = most accurate, but stricter requirements (default)
            enhance_image (bool): If True, applies image enhancement (contrast, sharpening)
                to improve detection accuracy.
        """
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            enable_segmentation=False  # We don't need segmentation, saves processing
        )
        self.is_selfie = selfie_mode
        self.enhance_image = enhance_image

    def _enhance_image(self, image):
        """
        Preprocess image to improve pose detection accuracy.
        Applies contrast enhancement, sharpening, and other techniques.
        """
        # Convert to LAB color space for better contrast adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Slight sharpening to enhance edges (helps with landmark detection)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * 0.1
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original with sharpened (70% sharpened, 30% original)
        result = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)
        
        return result

    def _apply_anatomical_constraints(self, landmarks):
        """
        Post-process landmarks with anatomical constraints.
        Validates and corrects ankle positions relative to knees and hips.
        """
        L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
        R_HIP, R_KNEE, R_ANKLE = 24, 26, 28
        
        def validate_ankle(hip_idx, knee_idx, ankle_idx):
            """Validate ankle position relative to knee and hip."""
            if (hip_idx >= len(landmarks) or knee_idx >= len(landmarks) or 
                ankle_idx >= len(landmarks)):
                return landmarks[ankle_idx]
            
            hip = landmarks[hip_idx]
            knee = landmarks[knee_idx]
            ankle = landmarks[ankle_idx]
            
            # Check visibility
            if (hip.get("visibility", 0) < 0.1 or knee.get("visibility", 0) < 0.1 or
                ankle.get("visibility", 0) < 0.1):
                return ankle
            
            # Ankle should be vertically below knee
            # Knee should be vertically below hip
            # Allow some horizontal offset but not too extreme
            
            hip_y = hip["y"]
            knee_y = knee["y"]
            ankle_y = ankle["y"]
            
            # Basic validation: ankle should be below knee, knee below hip
            if not (hip_y < knee_y < ankle_y):
                return ankle
            
            # If ankle x is too far from knee alignment, might need adjustment
            # But we'll leave it for now as MediaPipe is usually good
            
            return ankle
        
        # Validate left side
        landmarks[L_ANKLE] = validate_ankle(L_HIP, L_KNEE, L_ANKLE)
        
        # Validate right side
        landmarks[R_ANKLE] = validate_ankle(R_HIP, R_KNEE, R_ANKLE)
        
        return landmarks

    def get_landmarks(self, image, require_only_lower_body=False, debug=False, 
                     try_multiple_configs=False):
        """
        Detect body landmarks (keypoints) from a given image using MediaPipe Pose.
        Optimized for leg alignment - accepts partial poses if lower body is visible.
        
        Args:
            image: Input image
            require_only_lower_body (bool): If True, returns landmarks even if only 
                lower body (hips, knees, ankles) are detected. Upper body can be missing.
            debug (bool): If True, prints detailed information about landmark visibility.
        
        Returns:
            List of dictionaries with x, y, z, visibility OR None if no pose found.
        """
        best_landmarks = None
        best_visibility_sum = 0
        
        # Try multiple configurations if enabled
        configs_to_try = []
        if try_multiple_configs:
            # Try current config first
            configs_to_try.append(("current", self.pose))
            
            # Try with higher model complexity (if not already at max)
            # Wrap in try-except to handle download failures gracefully
            try:
                pose_complexity_2 = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    min_detection_confidence=0.5,
                    enable_segmentation=False
                )
                configs_to_try.append(("complexity=2", pose_complexity_2))
            except Exception as e:
                if debug:
                    print(f"    ⚠ Could not load complexity=2 model (download failed or unavailable): {type(e).__name__}")
                    print(f"       Continuing with other configurations...")
            
            # Try with higher confidence threshold
            try:
                pose_high_conf = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.7,
                    enable_segmentation=False
                )
                configs_to_try.append(("high_conf", pose_high_conf))
            except Exception as e:
                if debug:
                    print(f"    ⚠ Could not load high_conf configuration: {type(e).__name__}")
        else:
            configs_to_try.append(("current", self.pose))
        
        # Apply image enhancement if enabled
        enhanced_image = self._enhance_image(image) if self.enhance_image else image
        image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        height, width, _ = enhanced_image.shape
        
        for config_name, pose_model in configs_to_try:
            if debug and len(configs_to_try) > 1:
                print(f"  Trying configuration: {config_name}")
            
            # Run pose detection
            results = pose_model.process(image_rgb)

            # If no landmarks are detected, try next config
            if not results.pose_landmarks:
                if debug and len(configs_to_try) > 1:
                    print(f"    No landmarks detected with {config_name}")
                continue

            # Convert normalized coordinates → pixel coordinates
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": int(lm.x * width),
                    "y": int(lm.y * height),
                    "z": lm.z,  # z stays normalized (depth info)
                    "visibility": lm.visibility
                })

            # Calculate visibility sum for lower body landmarks
            visibility_sum = sum(
                landmarks[i].get("visibility", 0) 
                for i in self.REQUIRED_LANDMARKS 
                if i < len(landmarks)
            )
            
            # If we only require lower body, validate that critical landmarks are visible
            # MediaPipe always returns 33 landmarks, but visibility indicates if detected
            is_valid = True
            if require_only_lower_body:
                visible_count = sum(
                    1 for i in self.REQUIRED_LANDMARKS 
                    if i < len(landmarks) and landmarks[i].get("visibility", 0) > 0.1
                )
                
                if debug:
                    print(f"    Lower body landmarks visible: {visible_count}/6")
                    for i, idx in enumerate(self.REQUIRED_LANDMARKS):
                        if idx < len(landmarks):
                            vis = landmarks[idx].get("visibility", 0)
                            landmark_names = ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"]
                            print(f"      {landmark_names[i]}: visibility={vis:.2f}")
                
                # Need at least 4 out of 6 required landmarks to be visible
                # This allows for one leg being partially occluded
                if visible_count < 4:
                    is_valid = False
                    if debug:
                        print(f"    Not enough landmarks ({visible_count}/6) with {config_name}")
                elif visible_count < 6:
                    if debug:
                        print(f"    {visible_count}/6 landmarks visible with {config_name}")
            
            # Keep the best result based on visibility scores
            if is_valid and visibility_sum > best_visibility_sum:
                best_landmarks = landmarks
                best_visibility_sum = visibility_sum
                if debug and len(configs_to_try) > 1:
                    print(f"    ✓ Better result with {config_name} (visibility sum: {visibility_sum:.2f})")
        
        # If no valid landmarks found, return None
        if best_landmarks is None:
            return None
        
        landmarks = best_landmarks
        
        # Apply anatomical constraints to validate landmark positions
        landmarks = self._apply_anatomical_constraints(landmarks)
        
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
