import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io


@dataclass
class VisualizationConfig:
    """Configuration for lordosis visualization"""
    spine_color: Tuple[int, int, int] = (0, 255, 0)      # Green
    landmark_colors: Dict[str, Tuple[int, int, int]] = None
    measurement_color: Tuple[int, int, int] = (255, 255, 255)  # White
    severity_colors: Dict[str, Tuple[int, int, int]] = None
    font_scale: float = 0.7
    line_thickness: int = 2
    landmark_radius: int = 5
    
    def __post_init__(self):
        if self.landmark_colors is None:
            self.landmark_colors = {
                'c7': (255, 0, 0),      # Blue
                't1': (0, 255, 255),    # Yellow
                't12': (255, 255, 0),   # Cyan
                'l1': (0, 0, 255),      # Red
                'l5': (255, 0, 255),    # Magenta
                's1': (0, 255, 0)       # Green
            }
        
        if self.severity_colors is None:
            self.severity_colors = {
                'normal': (0, 255, 0),           # Green
                'mild_lordosis': (0, 255, 255),  # Yellow
                'moderate_lordosis': (0, 165, 255),  # Orange
                'severe_lordosis': (0, 0, 255),  # Red
                'mild_kyphosis': (255, 255, 0),  # Cyan
                'moderate_kyphosis': (255, 165, 0),  # Light Blue
                'severe_kyphosis': (255, 0, 0)   # Blue
            }


class LordosisVisualizer:
    """Advanced visualization tools for lordosis analysis results"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def create_comprehensive_visualization(self, original_image: np.ndarray, 
                                         spine_landmarks: Dict, 
                                         metrics: Dict,
                                         landmarks: Optional = None) -> np.ndarray:
        """
        Create comprehensive visualization with all lordosis analysis elements
        
        Args:
            original_image: Original input image
            spine_landmarks: Detected spine landmarks
            metrics: Lordosis measurement metrics
            landmarks: MediaPipe landmarks for reference
            
        Returns:
            Enhanced image with comprehensive visualization
        """
        # Create base visualization
        viz_image = self.create_base_visualization(original_image, spine_landmarks, metrics)
        
        # Add measurement overlays
        viz_image = self.add_measurement_overlays(viz_image, spine_landmarks, metrics)
        
        # Add severity indicators
        viz_image = self.add_severity_indicators(viz_image, metrics)
        
        # Add recommendations summary
        viz_image = self.add_recommendations_overlay(viz_image, metrics.get('recommendations', []))
        
        return viz_image
    
    def create_base_visualization(self, image: np.ndarray, 
                                spine_landmarks: Dict,
                                metrics: Dict) -> np.ndarray:
        """Create base visualization with spine curve and landmarks"""
        viz_image = image.copy()
        height, width = image.shape[:2]
        
        # Draw spine curve
        if spine_landmarks:
            spine_points = self.extract_spine_points(spine_landmarks, width, height)
            if len(spine_points) >= 2:
                cv2.polylines(viz_image, [np.array(spine_points)], False, 
                             self.config.spine_color, self.config.line_thickness)
        
        # Draw individual landmarks
        for landmark_name, point in spine_landmarks.items():
            if point and landmark_name in self.config.landmark_colors:
                px_point = (int(point[0] * width), int(point[1] * height))
                color = self.config.landmark_colors[landmark_name]
                
                # Draw landmark circle
                cv2.circle(viz_image, px_point, self.config.landmark_radius, color, -1)
                
                # Draw landmark label
                cv2.putText(viz_image, landmark_name.upper(), 
                           (px_point[0] + 10, px_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, color, 2)
        
        return viz_image
    
    def extract_spine_points(self, spine_landmarks: Dict, width: int, height: int) -> List[Tuple[int, int]]:
        """Extract spine points in correct order and convert to pixel coordinates"""
        ordered_landmarks = ['c7', 't1', 't12', 'l1', 'l5', 's1']
        points = []
        
        for landmark_name in ordered_landmarks:
            if landmark_name in spine_landmarks and spine_landmarks[landmark_name]:
                point = spine_landmarks[landmark_name]
                points.append((int(point[0] * width), int(point[1] * height)))
        
        return points
    
    def add_measurement_overlays(self, image: np.ndarray, 
                               spine_landmarks: Dict,
                               metrics: Dict) -> np.ndarray:
        """Add measurement text and angle indicators"""
        viz_image = image.copy()
        height, width = image.shape[:2]
        
        # Add measurement text
        measurements = [
            f"Pelvic Tilt: {metrics.get('pelvic_tilt', 0):.1f}°",
            f"Lumbar Lordosis: {metrics.get('lumbar_lordosis', 0):.1f}°",
            f"SVA: {metrics.get('sagittal_vertical_axis', 0):.1f}px",
            f"Curve Depth: {metrics.get('curve_depth', 0):.1f}px"
        ]
        
        text_y = 30
        for measurement in measurements:
            cv2.putText(viz_image, measurement, (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                       self.config.measurement_color, 2)
            text_y += 30
        
        # Add angle visualization if we have spine points
        if spine_landmarks and len(spine_landmarks) >= 2:
            viz_image = self.draw_angle_visualization(viz_image, spine_landmarks, width, height)
        
        return viz_image
    
    def draw_angle_visualization(self, image: np.ndarray, 
                               spine_landmarks: Dict,
                               width: int, height: int) -> np.ndarray:
        """Draw angle measurement visualization on the image"""
        viz_image = image.copy()
        
        # Draw lumbar lordosis angle
        if 'l1' in spine_landmarks and 'l5' in spine_landmarks:
            l1 = (int(spine_landmarks['l1'][0] * width), int(spine_landmarks['l1'][1] * height))
            l5 = (int(spine_landmarks['l5'][0] * width), int(spine_landmarks['l5'][1] * height))
            
            # Draw L1-L5 line
            cv2.line(viz_image, l1, l5, (255, 255, 0), 2)
            
            # Draw angle arc (simplified)
            center = ((l1[0] + l5[0]) // 2, (l1[1] + l5[1]) // 2)
            cv2.circle(viz_image, center, 20, (255, 255, 0), 2)
        
        return viz_image
    
    def add_severity_indicators(self, image: np.ndarray, metrics: Dict) -> np.ndarray:
        """Add visual severity indicators"""
        viz_image = image.copy()
        height, width = image.shape[:2]
        severity = metrics.get('severity', 'normal')
        
        # Get severity color
        color = self.config.severity_colors.get(severity, (255, 255, 255))
        
        # Create severity badge
        badge_width, badge_height = 200, 80
        badge_x, badge_y = width - badge_width - 20, 20
        
        # Draw severity background
        cv2.rectangle(viz_image, (badge_x, badge_y), 
                     (badge_x + badge_width, badge_y + badge_height), color, -1)
        
        # Draw severity text
        cv2.putText(viz_image, "LORDOSIS", (badge_x + 10, badge_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw severity level
        severity_text = severity.replace('_', ' ').upper()
        cv2.putText(viz_image, severity_text, (badge_x + 10, badge_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add confidence indicator
        confidence = metrics.get('confidence', 0.0)
        confidence_text = f"Confidence: {confidence:.2f}"
        cv2.putText(viz_image, confidence_text, (badge_x + 10, badge_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return viz_image
    
    def add_recommendations_overlay(self, image: np.ndarray, 
                                  recommendations: List[str]) -> np.ndarray:
        """Add recommendations text overlay"""
        viz_image = image.copy()
        height, width = image.shape[:2]
        
        # Create recommendations panel
        panel_height = min(200, len(recommendations) * 25)
        panel_y = height - panel_height - 20
        
        # Draw panel background
        cv2.rectangle(viz_image, (20, panel_y), (width - 20, height - 20), 
                     (0, 0, 0), -1)
        cv2.rectangle(viz_image, (20, panel_y), (width - 20, height - 20), 
                     (255, 255, 255), 2)
        
        # Add title
        cv2.putText(viz_image, "RECOMMENDATIONS", (30, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add recommendations
        text_y = panel_y + 50
        for i, recommendation in enumerate(recommendations[:6]):  # Limit to 6 recommendations
            # Wrap long text
            wrapped_text = self.wrap_text(recommendation, 50)
            for line in wrapped_text:
                if text_y < height - 30:
                    cv2.putText(viz_image, f"• {line}", (40, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    text_y += 20
        
        return viz_image
    
    def wrap_text(self, text: str, max_length: int) -> List[str]:
        """Wrap text to fit within maximum length"""
        if len(text) <= max_length:
            return [text]
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) <= max_length:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines
    
    def create_spinal_curve_plot(self, spine_landmarks: Dict, 
                               metrics: Dict) -> np.ndarray:
        """
        Create a matplotlib plot showing the spinal curve
        
        Args:
            spine_landmarks: Detected spine landmarks
            metrics: Lordosis metrics
            
        Returns:
            Matplotlib plot as numpy array
        """
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Extract spine points
        if spine_landmarks:
            x_points = []
            y_points = []
            labels = []
            
            for landmark_name in ['c7', 't1', 't12', 'l1', 'l5', 's1']:
                if landmark_name in spine_landmarks and spine_landmarks[landmark_name]:
                    point = spine_landmarks[landmark_name]
                    x_points.append(point[0])
                    y_points.append(point[1])
                    labels.append(landmark_name.upper())
            
            if len(x_points) >= 2:
                # Plot spine curve
                ax.plot(x_points, y_points, 'g-', linewidth=3, label='Spinal Curve')
                
                # Plot landmarks
                for i, (x, y, label) in enumerate(zip(x_points, y_points, labels)):
                    color = self.config.landmark_colors.get(label.lower(), (0, 0, 0))
                    ax.plot(x, y, 'o', color=color, markersize=8)
                    ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Customize plot
        ax.set_title(f"Spinal Curve Analysis - {metrics.get('severity', 'Normal').replace('_', ' ').title()}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Horizontal Position")
        ax.set_ylabel("Vertical Position")
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        ax.legend()
        
        # Add measurement annotations
        pelvic_tilt = metrics.get('pelvic_tilt', 0)
        lumbar_angle = metrics.get('lumbar_lordosis', 0)
        
        ax.text(0.02, 0.98, f"Pelvic Tilt: {pelvic_tilt:.1f}°", transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.text(0.02, 0.90, f"Lumbar Angle: {lumbar_angle:.1f}°", transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Convert plot to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        plot_array = np.asarray(buf)
        
        plt.close(fig)
        return plot_array
    
    def create_severity_heatmap(self, metrics: Dict) -> np.ndarray:
        """
        Create a heatmap visualization showing severity levels
        
        Args:
            metrics: Lordosis metrics
            
        Returns:
            Heatmap visualization as numpy array
        """
        # Create heatmap image
        heatmap = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Define severity zones
        severity_zones = [
            ('normal', (0, 255, 0), "Normal"),
            ('mild_lordosis', (0, 255, 255), "Mild Lordosis"),
            ('moderate_lordosis', (0, 165, 255), "Moderate Lordosis"),
            ('severe_lordosis', (0, 0, 255), "Severe Lordosis"),
            ('mild_kyphosis', (255, 255, 0), "Mild Kyphosis"),
            ('moderate_kyphosis', (255, 165, 0), "Moderate Kyphosis"),
            ('severe_kyphosis', (255, 0, 0), "Severe Kyphosis")
        ]
        
        # Draw severity zones
        zone_height = 200 // len(severity_zones)
        current_y = 0
        
        for severity, color, label in severity_zones:
            # Draw zone
            cv2.rectangle(heatmap, (0, current_y), (300, current_y + zone_height), color, -1)
            
            # Add label
            cv2.putText(heatmap, label, (10, current_y + zone_height // 2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            current_y += zone_height
        
        # Highlight current severity
        current_severity = metrics.get('severity', 'normal')
        for i, (severity, color, label) in enumerate(severity_zones):
            if severity == current_severity:
                y_pos = i * zone_height
                cv2.rectangle(heatmap, (250, y_pos + 5), (290, y_pos + zone_height - 5), 
                             (255, 255, 255), 3)
                break
        
        return heatmap
    
    def create_progress_chart(self, historical_data: List[Dict]) -> np.ndarray:
        """
        Create a progress chart showing lordosis measurements over time
        
        Args:
            historical_data: List of historical lordosis measurements
            
        Returns:
            Progress chart as numpy array
        """
        if not historical_data:
            return np.zeros((200, 400, 3), dtype=np.uint8)
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Extract data
        dates = [data.get('date', f"Day {i}") for i, data in enumerate(historical_data)]
        pelvic_tilts = [data.get('pelvic_tilt', 0) for data in historical_data]
        lumbar_angles = [data.get('lumbar_lordosis', 0) for data in historical_data]
        
        # Plot pelvic tilt
        ax1.plot(dates, pelvic_tilts, 'b-', linewidth=2, marker='o', label='Pelvic Tilt')
        ax1.set_title('Pelvic Tilt Progress')
        ax1.set_ylabel('Angle (degrees)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot lumbar angle
        ax2.plot(dates, lumbar_angles, 'r-', linewidth=2, marker='s', label='Lumbar Angle')
        ax2.set_title('Lumbar Lordosis Progress')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Angle (degrees)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Convert to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        chart_array = np.asarray(buf)
        
        plt.close(fig)
        return chart_array
    
    def create_report_summary(self, metrics: Dict, recommendations: List[str]) -> np.ndarray:
        """
        Create a summary report visualization
        
        Args:
            metrics: Lordosis metrics
            recommendations: List of recommendations
            
        Returns:
            Report summary as numpy array
        """
        # Create report image
        report = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add title
        cv2.putText(report, "LORDOSIS ANALYSIS REPORT", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        
        # Add metrics
        metrics_text = [
            f"Severity: {metrics.get('severity', 'Normal').replace('_', ' ').title()}",
            f"Pelvic Tilt: {metrics.get('pelvic_tilt', 0):.1f}°",
            f"Lumbar Lordosis: {metrics.get('lumbar_lordosis', 0):.1f}°",
            f"SVA: {metrics.get('sagittal_vertical_axis', 0):.1f}px",
            f"Curve Depth: {metrics.get('curve_depth', 0):.1f}px",
            f"Confidence: {metrics.get('confidence', 0):.2f}"
        ]
        
        text_y = 80
        for metric in metrics_text:
            cv2.putText(report, metric, (20, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            text_y += 30
        
        # Add recommendations header
        cv2.putText(report, "RECOMMENDATIONS:", (20, text_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add recommendations
        text_y += 50
        for i, rec in enumerate(recommendations[:5]):  # Show first 5 recommendations
            wrapped = self.wrap_text(rec, 60)
            for line in wrapped:
                if text_y < 380:
                    cv2.putText(report, f"• {line}", (30, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
                    text_y += 20
        
        return report
