import cv2
import numpy as np
from posture_analysis.utils import calculate_angle
from posture_analysis.mirror_utils import detect_and_correct_mirror

class LegAnalyzer:
    """
    LegAnalyzer:
      - robust handling of landmark formats
      - knee angle + knee/hip ratio + lateral offset analysis
      - smooth classification
      - visual debug overlay
      - mirror detection + confidence
    """
    def _validate_ankle(self, hip_idx, knee_idx, ankle_idx, landmarks, mask=None):
        """
        Correct ankle positions based on:
        1. Vertical order (hip < knee < ankle)
        2. Lateral drift relative to knee
        3. Symmetry with the opposite leg
        4. Constraining to segmentation mask if available
        """
        hip = landmarks[hip_idx]
        knee = landmarks[knee_idx]
        ankle = landmarks[ankle_idx]

        if hip is None or knee is None or ankle is None:
            return ankle

        x, y, z = ankle

        # 1️⃣ Vertical order correction
        if y <= knee[1]:
            y = knee[1] + 1

        # 2️⃣ Lateral drift relative to knee
        max_offset = abs(knee[1] - hip[1]) * 0.35  # 30–35% of thigh
        if abs(x - knee[0]) > max_offset:
            x = knee[0]

        # 3️⃣ Symmetry check: nudge ankle toward opposite leg
        # Use other leg indices: assume L/R based on hip index
        if hip_idx in [self.L_HIP, self.L_KNEE, self.L_ANKLE]:
            opp_ankle = landmarks.get(self.R_ANKLE, None)
            opp_knee = landmarks.get(self.R_KNEE, None)
        else:
            opp_ankle = landmarks.get(self.L_ANKLE, None)
            opp_knee = landmarks.get(self.L_KNEE, None)

        if opp_ankle is not None and opp_knee is not None:
            # Maintain roughly symmetric x-distance to knee
            leg_offset = x - knee[0]
            opp_offset = opp_ankle[0] - opp_knee[0]
            if abs(leg_offset - opp_offset) > max_offset:
                x = knee[0] + np.clip(leg_offset, -max_offset, max_offset)

        # 4️⃣ Constrain to segmentation mask if provided
        if mask is not None and len(mask.shape) == 2:
            h, w = mask.shape
            px, py = int(round(x)), int(round(y))
            # If outside mask, move to nearest non-zero pixel in vertical line
            if 0 <= px < w and 0 <= py < h:
                if mask[py, px] == 0:
                    # Search upward and downward in mask for first non-zero pixel
                    search_range = 10
                    found = False
                    for dy in range(1, search_range):
                        for ny in [py - dy, py + dy]:
                            if 0 <= ny < h and mask[ny, px] > 0:
                                y = ny
                                found = True
                                break
                        if found:
                            break

        return (x, y, z)


    # MediaPipe landmark indices
    L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
    R_HIP, R_KNEE, R_ANKLE = 24, 26, 28

    NORMAL_ANGLE_MIN = 176.0
    NORMAL_ANGLE_MAX = 183.0
    NORMAL_RATIO_MIN = 0.88
    NORMAL_RATIO_MAX = 1.06

    def _to_pixel(self, lm, image_shape):
        if lm is None:
            return None
        h, w = image_shape[:2]
        if isinstance(lm, dict):
            try:
                return (float(lm["x"]), float(lm["y"]), lm.get("z", 0.0))
            except Exception:
                return None
        try:
            x0 = float(lm[0])
            y0 = float(lm[1])
        except Exception:
            return None
        if x0 > 1.0 or y0 > 1.0:
            return (x0, y0, lm[2] if len(lm) > 2 else 0.0)
        else:
            return (x0 * w, y0 * h, lm[2] if len(lm) > 2 else 0.0)

    def _safe_get(self, landmarks, idx):
        try:
            return landmarks[idx]
        except Exception:
            return None

    def evaluate_leg_alignment(self, landmarks, mask, image):
        if landmarks is None or len(landmarks) == 0:
            return {"summary": "No person detected", "visualized_image": image}

        h, w = image.shape[:2]

        # Normalize landmarks
        norm_landmarks = []
        for lm in landmarks:
            if lm is None:
                norm_landmarks.append(None)
                continue
            if isinstance(lm, dict):
                try:
                    x_px = float(lm["x"])
                    y_px = float(lm["y"])
                except Exception:
                    norm_landmarks.append(None)
                    continue
                norm_landmarks.append((x_px / w, y_px / h, lm.get("z", 0.0)))
            else:
                try:
                    x0 = float(lm[0])
                    y0 = float(lm[1])
                except Exception:
                    norm_landmarks.append(None)
                    continue
                if x0 > 1.0 or y0 > 1.0:
                    norm_landmarks.append((x0 / w, y0 / h, lm[2] if len(lm) > 2 else 0.0))
                else:
                    norm_landmarks.append((x0, y0, lm[2] if len(lm) > 2 else 0.0))

        # Mirror detection - disabled for posture analysis
        # For posture analysis, we don't need to correct mirroring as we analyze relative positions
        mirror_info = {"mirrored": False, "confidence": 0.0}
        corrected_norm = norm_landmarks
        mirrored = False
        mirror_confidence = 0.0

        # Map normalized landmarks to pixels
        lm_px = {}
        for i in range(max(self.R_ANKLE + 1, len(landmarks))):
            raw = self._safe_get(corrected_norm, i)
            lm_px[i] = self._to_pixel(raw, image.shape) if raw is not None else None

        # Required joints
        Lh, Lk, La = lm_px.get(self.L_HIP), lm_px.get(self.L_KNEE), self._validate_ankle(self.L_HIP, self.L_KNEE, self.L_ANKLE, lm_px, mask)
        Rh, Rk, Ra = lm_px.get(self.R_HIP), lm_px.get(self.R_KNEE), self._validate_ankle(self.R_HIP, self.R_KNEE, self.R_ANKLE, lm_px, mask)


        if any(p is None for p in [Lh, Lk, La, Rh, Rk, Ra]):
            return {"summary": "Insufficient landmarks for analysis", "visualized_image": image}

        # Angles
        left_angle = calculate_angle(Lh, Lk, La)
        right_angle = calculate_angle(Rh, Rk, Ra)

        # Knee/Hip ratio and Ankle/Hip ratio
        lk_x, rk_x = Lk[0], Rk[0]
        lh_x, rh_x = Lh[0], Rh[0]
        la_x, ra_x = La[0], Ra[0]
        
        knee_dist = abs(lk_x - rk_x)
        hip_dist = abs(lh_x - rh_x) + 1e-6
        ankle_dist = abs(la_x - ra_x)
        
        knee_hip_ratio = knee_dist / hip_dist
        ankle_hip_ratio = ankle_dist / hip_dist

        # Lateral offset for bow/knock detection
        left_offset = Lk[0] - Lh[0]
        right_offset = Rk[0] - Rh[0]
        avg_offset = (left_offset + right_offset) / 2.0
        hip_half = hip_dist / 2.0 + 1e-6
        mid_ratio = avg_offset / hip_half  # مثبت ⇒ Bow-Leg، منفی ⇒ Knock-Knee

        # Thresholds
        ANGLE_EPS = 4.0
        NORMAL_ANGLE_LOW = 180.0 - ANGLE_EPS
        NORMAL_ANGLE_HIGH = 180.0 + ANGLE_EPS
        RATIO_EPS = 0.12
        NORMAL_RATIO_MIN = self.NORMAL_RATIO_MIN - RATIO_EPS
        NORMAL_RATIO_MAX = self.NORMAL_RATIO_MAX + RATIO_EPS

        # Classification with improved logic using ankle-hip ratio
        # Key insight: Bow-legs have ankles closer together than hips
        # Knock-knees have ankles further apart than hips
        
        # Check for knock-knee (knees pointing inward, ankles further apart)
        knock_knee_indicators = 0
        if left_angle < NORMAL_ANGLE_LOW:
            knock_knee_indicators += 1
        if right_angle < NORMAL_ANGLE_LOW:
            knock_knee_indicators += 1
        if ankle_hip_ratio > 1.1:  # Ankles further apart than hips
            knock_knee_indicators += 2  # Strong indicator
        if knee_hip_ratio < 0.85:  # Knees closer together than hips
            knock_knee_indicators += 1
        if mid_ratio < -0.12:  # Knees positioned inward relative to hips
            knock_knee_indicators += 1
            
        # Check for bow-leg (knees pointing outward, ankles closer together)
        bow_leg_indicators = 0
        if left_angle > NORMAL_ANGLE_HIGH:
            bow_leg_indicators += 1
        if right_angle > NORMAL_ANGLE_HIGH:
            bow_leg_indicators += 1
        if ankle_hip_ratio < 0.7:  # Ankles much closer together than hips
            bow_leg_indicators += 2  # Strong indicator
        if knee_hip_ratio > 1.15:  # Knees further apart than hips
            bow_leg_indicators += 1
        if mid_ratio > 0.08:  # Knees positioned outward relative to hips
            bow_leg_indicators += 1
        
        # Determine condition based on strongest indicators
        # Prioritize ankle-hip ratio as it's the most reliable indicator
        if ankle_hip_ratio < 0.7:  # Strong bow-leg indicator
            condition = "Bow-Leg"
        elif ankle_hip_ratio > 1.1:  # Strong knock-knee indicator
            condition = "Knock-Knee"
        elif bow_leg_indicators >= 2:
            condition = "Bow-Leg"
        elif knock_knee_indicators >= 2:
            condition = "Knock-Knee"
        elif (
            NORMAL_ANGLE_LOW <= left_angle <= NORMAL_ANGLE_HIGH
            and NORMAL_ANGLE_LOW <= right_angle <= NORMAL_ANGLE_HIGH
            and NORMAL_RATIO_MIN <= knee_hip_ratio <= NORMAL_RATIO_MAX
            and 0.7 <= ankle_hip_ratio <= 1.1  # Normal ankle-hip ratio
            and -0.05 <= mid_ratio <= 0.05
        ):
            condition = "Normal"
        else:
            condition = "Normal"  # Default to normal if unclear

        # Visualization
        vis = image.copy()
        def draw_point(pt, color=(0, 255, 0), r=4):
            if pt is not None:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), r, color, -1)
        for pt in [Lh, Lk, La, Rh, Rk, Ra]:
            draw_point(pt)
        cv2.line(vis, (int(Lh[0]), int(Lh[1])), (int(Lk[0]), int(Lk[1])), (255, 0, 0), 2)
        cv2.line(vis, (int(Lk[0]), int(Lk[1])), (int(La[0]), int(La[1])), (255, 0, 0), 2)
        cv2.line(vis, (int(Rh[0]), int(Rh[1])), (int(Rk[0]), int(Rk[1])), (0, 0, 255), 2)
        cv2.line(vis, (int(Rk[0]), int(Rk[1])), (int(Ra[0]), int(Ra[1])), (0, 0, 255), 2)

        if mask is not None and len(mask.shape) == 2:
            # Ensure mask has same dimensions as visualization image
            if mask.shape[:2] != vis.shape[:2]:
                mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = vis.copy()
            overlay[:, :, 2] = np.maximum(overlay[:, :, 2], mask)
            vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)

        # Text info
        cv2.putText(vis, f"L:{left_angle:.1f} R:{right_angle:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        cv2.putText(vis, f"knee/hip ratio:{knee_hip_ratio:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(vis, f"ankle/hip ratio:{ankle_hip_ratio:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(vis, f"mid_ratio:{mid_ratio:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(vis, condition, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0) if condition=="Normal" else (0,0,255), 3)
        cv2.putText(vis, f"Mirrored: {mirrored}, conf: {mirror_confidence:.2f}", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        summary = f"Left angle: {left_angle:.2f}, Right angle: {right_angle:.2f}, knee/hip: {knee_hip_ratio:.2f}, ankle/hip: {ankle_hip_ratio:.2f}, mid_ratio: {mid_ratio:.2f} → {condition} | Mirrored: {mirrored}, conf: {mirror_confidence:.2f}"

        return {"summary": summary, "visualized_image": vis}
