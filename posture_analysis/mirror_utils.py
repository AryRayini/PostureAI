# posture_analysis/mirror_utils.py

from typing import List, Tuple, Dict

LEFT_RIGHT_PAIRS = [
    (11, 12),  # shoulders
    (23, 24),  # hips
    (13, 14),  # elbows
    (15, 16),  # wrists
    (25, 26),  # knees
    (27, 28),  # ankles
]

def to_pixel(lm, image_shape):
    """Convert landmark to pixel coordinates."""
    h, w = image_shape[:2]
    if isinstance(lm, dict):
        # Handle dictionary format from PoseEstimator
        return (lm["x"], lm["y"], lm.get("z", 0.0))
    else:
        # Handle tuple format (normalized coordinates)
        return (lm[0] * w, lm[1] * h, lm[2])

def swap_left_right(landmarks):
    """Swap all left-right landmarks in MediaPipe's landmark list."""
    if landmarks is None:
        return None
    out = list(landmarks)
    for l_idx, r_idx in LEFT_RIGHT_PAIRS:
        if l_idx < len(out) and r_idx < len(out):
            out[l_idx], out[r_idx] = out[r_idx], out[l_idx]
    return out

def detect_and_correct_mirror(landmarks, image_shape, exif_mirrored=None, vote_threshold=0.8):
    """
    Detect if the image is mirrored and correct landmarks if needed.
    Uses a higher threshold to be more conservative about mirror detection
    to avoid incorrect landmark swapping for posture analysis.

    Returns:
        {
            "mirrored": bool,
            "confidence": float,
            "votes": dict,
            "corrected_landmarks": list
        }
    """
    if landmarks is None:
        return {"mirrored": False, "confidence": 0.0, "votes": {}, "corrected_landmarks": None}

    # EXIF override
    if exif_mirrored is True:
        return {"mirrored": True, "confidence": 1.0, "votes": {}, "corrected_landmarks": swap_left_right(landmarks)}

    votes = {}
    valid_votes = 0
    mirrored_votes = 0

    # Focus on upper body landmarks for more reliable mirror detection
    # Shoulders and hips are more reliable indicators than knees/ankles
    upper_body_pairs = [(11, 12), (23, 24)]  # shoulders, hips
    
    for l_idx, r_idx in upper_body_pairs:
        if l_idx >= len(landmarks) or r_idx >= len(landmarks):
            continue

        l_px = to_pixel(landmarks[l_idx], image_shape)
        r_px = to_pixel(landmarks[r_idx], image_shape)

        if l_px[0] is None or r_px[0] is None:
            continue

        vote = 1 if l_px[0] > r_px[0] else 0
        votes[(l_idx, r_idx)] = vote
        valid_votes += 1
        mirrored_votes += vote

    if valid_votes == 0:
        return {"mirrored": False, "confidence": 0.0, "votes": votes, "corrected_landmarks": landmarks}

    confidence = mirrored_votes / valid_votes
    # Use higher threshold and require strong evidence for mirroring
    mirrored = confidence >= vote_threshold and confidence >= 0.9

    corrected = swap_left_right(landmarks) if mirrored else landmarks

    return {
        "mirrored": mirrored,
        "confidence": confidence,
        "votes": votes,
        "corrected_landmarks": corrected
    }
