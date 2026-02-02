"""
Lordosis Analysis Configuration Settings

This module contains medical standards, thresholds, and configuration
parameters for lordosis (گودی کمر) posture analysis.
"""

# Normal lordosis angle ranges (in degrees)
NORMAL_LUMBAR_LORDOSIS = {
    'adult_male': (30, 80),
    'adult_female': (30, 80),
    'adolescent': (20, 60),
    'elderly': (20, 70)
}

# Pelvic tilt angle ranges (in degrees)
NORMAL_PELVIC_TILT = {
    'anterior': (10, 15),  # Normal anterior pelvic tilt
    'posterior': (-5, 5)   # Normal posterior pelvic tilt (negative values)
}

# Severity classification thresholds
SEVERITY_THRESHOLDS = {
    'mild_lordosis': 15,      # Degrees above normal range
    'moderate_lordosis': 25,  # Degrees above normal range
    'severe_lordosis': 35,    # Degrees above normal range
    
    'mild_kyphosis': -10,     # Degrees below normal range (posterior tilt)
    'moderate_kyphosis': -20, # Degrees below normal range
    'severe_kyphosis': -30    # Degrees below normal range
}

# Measurement tolerances
MEASUREMENT_TOLERANCE = {
    'angle_tolerance': 2.0,      # Degrees tolerance for angle measurements
    'distance_tolerance': 0.05,  # Percentage tolerance for distance measurements
    'confidence_threshold': 0.7  # Minimum confidence for landmark detection
}

# Spine landmark configuration
SPINE_LANDMARKS = {
    'c7': 'C7 Vertebra (Base of neck)',
    't1': 'T1 Vertebra (Top of thoracic)',
    't12': 'T12 Vertebra (Bottom of thoracic)',
    'l1': 'L1 Vertebra (Top of lumbar)',
    'l2': 'L2 Vertebra',
    'l3': 'L3 Vertebra',
    'l4': 'L4 Vertebra',
    'l5': 'L5 Vertebra (Bottom of lumbar)',
    's1': 'S1 Vertebra (Sacrum)'
}

# MediaPipe landmark indices for spine approximation
# Note: MediaPipe doesn't have direct spine landmarks, so we use
# shoulder, hip, and other landmarks to approximate spine position
MEDIPIPE_SPINE_INDICES = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'spine_base': 24  # Using hip as spine base reference
}

# Analysis configuration
ANALYSIS_CONFIG = {
    'enable_multi_view': True,      # Support for lateral view analysis
    'enable_curve_analysis': True,  # Analyze spinal curve shape
    'enable_progress_tracking': True, # Track changes over time
    'generate_visual_reports': True, # Create visual analysis reports
    'medical_validation': True      # Include medical-grade validation
}

# Report configuration
REPORT_CONFIG = {
    'include_angle_measurements': True,
    'include_curve_analysis': True,
    'include_recommendations': True,
    'include_visual_overlays': True,
    'include_confidence_scores': True
}

# User interface settings
UI_CONFIG = {
    'show_real_time_feedback': True,
    'show_measurement_guides': True,
    'show_severity_indicators': True,
    'enable_progress_charts': True,
    'language': 'en'  # 'en' for English, 'fa' for Persian
}

# Camera and image requirements
IMAGE_CONFIG = {
    'min_resolution': (1920, 1080),
    'recommended_distance': (2.0, 3.0),  # Meters
    'camera_height': 'hip_level',
    'background_contrast': 'high',
    'lighting_quality': 'diffused'
}

# Medical reference standards
MEDICAL_STANDARDS = {
    'reference': 'Scoliosis Research Society (SRS) standards',
    'measurement_method': 'Sagittal Vertical Axis (SVA)',
    'validation_required': True,
    'professional_review_required': True
}
