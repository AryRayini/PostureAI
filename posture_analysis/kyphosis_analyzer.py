import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
# Removed POSTURE_CONFIG import as it doesn't exist
from configs.lordosis_settings import (
    MEDIPIPE_SPINE_INDICES,
    NORMAL_PELVIC_TILT,
    SEVERITY_THRESHOLDS,
    MEASUREMENT_TOLERANCE,
    ANALYSIS_CONFIG
)
from posture_analysis.utils import calculate_angle, calculate_distance
from posture_analysis.pose_estimation import PoseEstimator
from posture_analysis.lordosis_visualization import LordosisVisualizer


@dataclass
class KyphosisMetrics:
    """Data class for kyphosis measurement results"""
    shoulder_angle: float = 0.0                    # Shoulder angle in degrees
    thoracic_curve_angle: float = 0.0              # Thoracic curve angle
    head_position_angle: float = 0.0               # Head position relative to shoulders
    forward_head_distance: float = 0.0             # Forward head posture distance
    severity: str = "normal"                       # Severity classification
    confidence: float = 0.0                        # Measurement confidence
    recommendations: List[str] = None              # List of recommendations


class KyphosisAnalyzer:
    """Advanced kyphosis analysis using computer vision and medical standards"""
    
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.min_confidence = MEASUREMENT_TOLERANCE['confidence_threshold']
        
    def analyze(self, landmarks, mask, image) -> Dict:
        """
        Complete kyphosis analysis workflow
        
        Args:
            landmarks: MediaPipe pose landmarks
            mask: Segmentation mask
            image: Original image
            
        Returns:
            Dict containing analysis results and visualizations
        """
        try:
            # Step 1: Calculate kyphosis metrics
            metrics = self.calculate_kyphosis_metrics(landmarks, image.shape)
            
            # Step 2: Generate analysis report
            report = self.generate_kyphosis_report(metrics)
            
            # Step 3: Create visualizations
            visualized_image = self.create_kyphosis_visualization(
                image, metrics, landmarks
            )
            
            return {
                "summary": report['summary'],
                "severity": metrics.severity,
                "confidence": metrics.confidence,
                "metrics": {
                    "shoulder_angle": metrics.shoulder_angle,
                    "thoracic_curve": metrics.thoracic_curve_angle,
                    "head_position": metrics.head_position_angle,
                    "forward_head_distance": metrics.forward_head_distance
                },
                "recommendations": metrics.recommendations,
                "visualized_image": visualized_image
            }
            
        except Exception as e:
            return {
                "summary": f"Error in kyphosis analysis: {str(e)}",
                "severity": "error",
                "confidence": 0.0,
                "visualized_image": image
            }
    
    def calculate_kyphosis_metrics(self, landmarks, image_shape) -> KyphosisMetrics:
        """
        Calculate comprehensive kyphosis metrics
        
        Args:
            landmarks: MediaPipe pose landmarks
            image_shape: Image dimensions
            
        Returns:
            KyphosisMetrics object with all measurements
        """
        try:
            height, width = image_shape[:2]
            
            # Calculate shoulder angle
            shoulder_angle = self.calculate_shoulder_angle(landmarks, width, height)
            
            # Calculate thoracic curve angle
            thoracic_angle = self.calculate_thoracic_curve_angle(landmarks, width, height)
            
            # Calculate head position
            head_angle = self.calculate_head_position(landmarks, width, height)
            
            # Calculate forward head distance
            forward_distance = self.calculate_forward_head_distance(landmarks, width, height)
            
            # Classify severity
            severity = self.classify_kyphosis_severity(shoulder_angle, thoracic_angle, head_angle)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(shoulder_angle, thoracic_angle, head_angle, severity)
            
            return KyphosisMetrics(
                shoulder_angle=shoulder_angle,
                thoracic_curve_angle=thoracic_angle,
                head_position_angle=head_angle,
                forward_head_distance=forward_distance,
                severity=severity,
                confidence=0.8,  # Default confidence
                recommendations=recommendations
            )
            
        except Exception as e:
            return KyphosisMetrics(
                severity="error",
                confidence=0.0,
                recommendations=["Error in measurement calculation"]
            )
    
    def calculate_shoulder_angle(self, landmarks, width, height) -> float:
        """
        Calculate shoulder angle relative to vertical axis
        
        Forward shoulder posture indicates kyphosis
        """
        try:
            # Get shoulder landmarks
            left_shoulder = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['left_shoulder'])
            right_shoulder = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['right_shoulder'])
            
            if left_shoulder and right_shoulder:
                # Convert to pixel coordinates
                ls_px = (int(left_shoulder[0] * width), int(left_shoulder[1] * height))
                rs_px = (int(right_shoulder[0] * width), int(right_shoulder[1] * height))
                
                # Calculate shoulder line angle relative to horizontal
                horizontal_ref = (ls_px[0] + 100, ls_px[1])
                shoulder_angle = calculate_angle(ls_px, rs_px, horizontal_ref)
                
                return abs(shoulder_angle)
            
            return 0.0
            
        except:
            return 0.0
    
    def calculate_thoracic_curve_angle(self, landmarks, width, height) -> float:
        """
        Calculate thoracic spine curve angle
        
        Uses shoulder and hip references to estimate thoracic curvature
        """
        try:
            # Get reference points
            left_shoulder = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['left_shoulder'])
            right_shoulder = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['right_shoulder'])
            left_hip = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['left_hip'])
            right_hip = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['right_hip'])
            
            if left_shoulder and right_shoulder and left_hip and right_hip:
                # Calculate midpoints
                shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                              (left_shoulder[1] + right_shoulder[1]) / 2)
                hip_mid = ((left_hip[0] + right_hip[0]) / 2, 
                          (left_hip[1] + right_hip[1]) / 2)
                
                # Convert to pixel coordinates
                sm_px = (int(shoulder_mid[0] * width), int(shoulder_mid[1] * height))
                hm_px = (int(hip_mid[0] * width), int(hip_mid[1] * height))
                
                # Calculate angle between shoulder-hip line and vertical
                vertical_ref = (sm_px[0], sm_px[1] - 100)
                curve_angle = calculate_angle(sm_px, hm_px, vertical_ref)
                
                return abs(curve_angle)
            
            return 0.0
            
        except:
            return 0.0
    
    def calculate_head_position(self, landmarks, width, height) -> float:
        """
        Calculate head position relative to shoulders
        
        Forward head posture is a key indicator of kyphosis
        """
        try:
            # Get head and shoulder landmarks
            # Note: MediaPipe doesn't have direct head landmark, using nose as approximation
            nose_idx = 0  # MediaPipe nose landmark
            left_shoulder = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['left_shoulder'])
            right_shoulder = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['right_shoulder'])
            
            if hasattr(landmarks, 'landmark') and len(landmarks.landmark) > nose_idx:
                nose = landmarks.landmark[nose_idx]
                nose_px = (int(nose.x * width), int(nose.y * height))
                
                if left_shoulder and right_shoulder:
                    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                                  (left_shoulder[1] + right_shoulder[1]) / 2)
                    sm_px = (int(shoulder_mid[0] * width), int(shoulder_mid[1] * height))
                    
                    # Calculate angle between nose-shoulder line and vertical
                    vertical_ref = (sm_px[0], sm_px[1] - 100)
                    head_angle = calculate_angle(nose_px, sm_px, vertical_ref)
                    
                    return abs(head_angle)
            
            return 0.0
            
        except:
            return 0.0
    
    def calculate_forward_head_distance(self, landmarks, width, height) -> float:
        """
        Calculate forward head posture distance
        
        Distance from ear to shoulder in horizontal plane
        """
        try:
            # Get reference points (using nose as head approximation)
            nose_idx = 0
            left_shoulder = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['left_shoulder'])
            right_shoulder = self.get_landmark_coords(landmarks, MEDIPIPE_SPINE_INDICES['right_shoulder'])
            
            if hasattr(landmarks, 'landmark') and len(landmarks.landmark) > nose_idx:
                nose = landmarks.landmark[nose_idx]
                nose_px = (int(nose.x * width), int(nose.y * height))
                
                if left_shoulder and right_shoulder:
                    # Calculate horizontal distance from nose to shoulder line
                    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2 * width
                    forward_distance = abs(nose_px[0] - shoulder_mid_x)
                    
                    return forward_distance
            
            return 0.0
            
        except:
            return 0.0
    
    def get_landmark_coords(self, landmarks, index) -> Optional[Tuple[float, float]]:
        """Get coordinates from MediaPipe landmark"""
        try:
            if hasattr(landmarks, 'landmark') and len(landmarks.landmark) > index:
                landmark = landmarks.landmark[index]
                return (landmark.x, landmark.y)
            return None
        except:
            return None
    
    def classify_kyphosis_severity(self, shoulder_angle: float, thoracic_angle: float, head_angle: float) -> str:
        """
        Classify kyphosis severity based on measurements
        
        Uses multiple indicators for comprehensive assessment
        """
        try:
            # Normal ranges (approximate)
            normal_shoulder_angle = 0  # Shoulders should be level
            normal_thoracic_angle = 20  # Normal thoracic curve
            normal_head_angle = 0  # Head should be aligned
            
            # Check for kyphosis indicators
            shoulder_forward = shoulder_angle > 10  # Shoulders rolled forward
            excessive_curve = thoracic_angle > 40   # Excessive thoracic curve
            forward_head = head_angle > 15          # Forward head posture
            
            # Severity classification
            if shoulder_forward and excessive_curve and forward_head:
                return "severe_kyphosis"
            elif (shoulder_forward and excessive_curve) or (excessive_curve and forward_head):
                return "moderate_kyphosis"
            elif shoulder_forward or excessive_curve or forward_head:
                return "mild_kyphosis"
            else:
                return "normal"
                
        except:
            return "error"
    
    def generate_recommendations(self, shoulder_angle: float, thoracic_angle: float, 
                               head_angle: float, severity: str) -> List[str]:
        """Generate personalized recommendations based on analysis results"""
        
        recommendations = []
        
        if severity == "normal":
            recommendations.append("Posture is within normal range. Maintain good posture habits.")
            recommendations.append("Continue regular stretching and strengthening exercises.")
            
        elif "kyphosis" in severity:
            recommendations.append("Kyphosis detected. Focus on posture correction exercises.")
            recommendations.append("Strengthen upper back and shoulder muscles.")
            recommendations.append("Stretch chest and shoulder muscles regularly.")
            recommendations.append("Practice chin tucks to correct forward head posture.")
            recommendations.append("Consider consulting a physiotherapist for personalized guidance.")
            
            if "severe" in severity:
                recommendations.append("Severe kyphosis detected. Professional evaluation recommended.")
                recommendations.append("Consider ergonomic assessment of work environment.")
            
            elif "moderate" in severity:
                recommendations.append("Moderate kyphosis. Consistent exercise program needed.")
                recommendations.append("Focus on daily posture awareness and correction.")
            
            elif "mild" in severity:
                recommendations.append("Mild kyphosis. Early intervention can prevent progression.")
                recommendations.append("Focus on preventive exercises and posture habits.")
        
        # Add general recommendations
        recommendations.append("Ensure proper ergonomics at work and home.")
        recommendations.append("Take regular breaks from sitting to stretch and move.")
        recommendations.append("Maintain a strong core to support good posture.")
        recommendations.append("Consider professional evaluation for persistent posture issues.")
        
        return recommendations
    
    def generate_kyphosis_report(self, metrics: KyphosisMetrics) -> Dict:
        """Generate comprehensive kyphosis analysis report"""
        
        report = {
            "summary": f"Kyphosis analysis complete - {metrics.severity.replace('_', ' ').title()}",
            "metrics": {
                "shoulder_angle": f"{metrics.shoulder_angle:.1f}°",
                "thoracic_curve": f"{metrics.thoracic_curve_angle:.1f}°", 
                "head_position": f"{metrics.head_position_angle:.1f}°",
                "forward_head_distance": f"{metrics.forward_head_distance:.1f}px"
            },
            "severity": metrics.severity,
            "confidence": f"{metrics.confidence:.2f}",
            "recommendations": metrics.recommendations
        }
        
        return report
    
    def create_kyphosis_visualization(self, image, metrics: KyphosisMetrics, landmarks) -> np.ndarray:
        """
        Create visual overlay showing kyphosis analysis results
        
        Args:
            image: Original image
            metrics: Kyphosis measurements
            landmarks: MediaPipe landmarks for reference
            
        Returns:
            Image with visualization overlay
        """
        try:
            # Create copy for visualization
            viz_image = image.copy()
            height, width = image.shape[:2]
            
            # Draw kyphosis-specific visualizations
            if hasattr(landmarks, 'landmark') and len(landmarks.landmark) > 12:
                # Get key landmarks
                left_shoulder = landmarks.landmark[11]
                right_shoulder = landmarks.landmark[12]
                left_hip = landmarks.landmark[23]
                right_hip = landmarks.landmark[24]
                nose = landmarks.landmark[0]
                
                # Convert to pixel coordinates
                ls_px = (int(left_shoulder.x * width), int(left_shoulder.y * height))
                rs_px = (int(right_shoulder.x * width), int(right_shoulder.y * height))
                lh_px = (int(left_hip.x * width), int(left_hip.y * height))
                rh_px = (int(right_hip.x * width), int(right_hip.y * height))
                nose_px = (int(nose.x * width), int(nose.y * height))
                
                # Draw shoulder line
                cv2.line(viz_image, ls_px, rs_px, (255, 0, 0), 2)  # Blue
                
                # Draw shoulder-hip line
                sm_px = ((ls_px[0] + rs_px[0]) // 2, (ls_px[1] + rs_px[1]) // 2)
                hm_px = ((lh_px[0] + rh_px[0]) // 2, (lh_px[1] + rh_px[1]) // 2)
                cv2.line(viz_image, sm_px, hm_px, (0, 255, 0), 2)  # Green
                
                # Draw head position line
                cv2.line(viz_image, nose_px, sm_px, (0, 0, 255), 2)  # Red
                
                # Draw vertical reference
                vertical_ref = (sm_px[0], sm_px[1] - 100)
                cv2.line(viz_image, sm_px, vertical_ref, (255, 255, 255), 1, cv2.LINE_DOTTED)
            
            # Add measurement text
            text_y = 30
            cv2.putText(viz_image, f"Shoulder Angle: {metrics.shoulder_angle:.1f}°", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_y += 30
            cv2.putText(viz_image, f"Thoracic Curve: {metrics.thoracic_curve_angle:.1f}°", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_y += 30
            cv2.putText(viz_image, f"Head Position: {metrics.head_position_angle:.1f}°", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_y += 30
            cv2.putText(viz_image, f"Forward Head: {metrics.forward_head_distance:.1f}px", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_y += 30
            cv2.putText(viz_image, f"Severity: {metrics.severity.replace('_', ' ').title()}", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add severity indicator
            severity_color = {
                'normal': (0, 255, 0),
                'mild_kyphosis': (255, 255, 0),
                'moderate_kyphosis': (255, 165, 0),
                'severe_kyphosis': (255, 0, 0)
            }.get(metrics.severity, (255, 255, 255))
            
            cv2.rectangle(viz_image, (width - 120, 10), (width - 10, 60), severity_color, -1)
            cv2.putText(viz_image, "KYPHOSIS", (width - 110, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(viz_image, metrics.severity.replace('_', ' ').upper(), (width - 110, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return viz_image
            
        except Exception as e:
            print(f"Error in kyphosis visualization: {e}")
            return image  # Return original image if visualization fails
