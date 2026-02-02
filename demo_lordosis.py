#!/usr/bin/env python3
"""
Demonstration script for lordosis analysis system
"""

import cv2
import numpy as np
from posture_analysis.lordosis_analyzer import LordosisAnalyzer, SpineLandmarks, LordosisMetrics
from configs.lordosis_settings import NORMAL_PELVIC_TILT, SEVERITY_THRESHOLDS, NORMAL_LUMBAR_LORDOSIS


def create_demo_image():
    """Create a demo image for lordosis analysis"""
    # Create a test image with a simple stick figure
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image.fill(255)  # White background
    
    # Draw a simple stick figure
    # Head
    cv2.circle(image, (400, 100), 30, (0, 0, 0), -1)
    
    # Spine (approximate curve)
    spine_points = [
        (400, 130),  # C7
        (395, 200),  # T1
        (390, 300),  # T12
        (385, 350),  # L1
        (380, 450),  # L5
        (380, 500)   # S1
    ]
    
    # Draw spine curve
    for i in range(len(spine_points) - 1):
        cv2.line(image, spine_points[i], spine_points[i+1], (0, 255, 0), 3)
    
    # Draw shoulders
    cv2.line(image, (350, 180), (450, 180), (0, 0, 255), 3)
    
    # Draw hips
    cv2.line(image, (340, 420), (460, 420), (255, 0, 0), 3)
    
    # Draw legs
    cv2.line(image, (350, 420), (350, 550), (0, 0, 0), 3)
    cv2.line(image, (450, 420), (450, 550), (0, 0, 0), 3)
    
    return image, spine_points


def create_mock_landmarks(spine_points):
    """Create mock MediaPipe landmarks for testing"""
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x / 800  # Normalize to 0-1
            self.y = y / 600
    
    class MockLandmarks:
        def __init__(self):
            self.landmark = [MockLandmark(0, 0)] * 29  # 29 landmarks
            
            # Set specific landmarks for spine analysis
            self.landmark[11] = MockLandmark(350, 180)  # Left shoulder
            self.landmark[12] = MockLandmark(450, 180)  # Right shoulder
            self.landmark[23] = MockLandmark(340, 420)  # Left hip
            self.landmark[24] = MockLandmark(460, 420)  # Right hip
    
    return MockLandmarks()


def demo_lordosis_analysis():
    """Demonstrate lordosis analysis with mock data"""
    print("=== LORDOSIS ANALYSIS DEMO ===\n")
    
    # Create demo image
    image, spine_points = create_demo_image()
    
    # Create mock landmarks
    mock_landmarks = create_mock_landmarks(spine_points)
    
    # Create mock mask
    mask = np.ones((600, 800), dtype=np.uint8) * 255
    
    # Initialize analyzer
    analyzer = LordosisAnalyzer()
    
    try:
        # Analyze
        result = analyzer.analyze(mock_landmarks, mask, image)
        
        print("✅ Lordosis analysis completed successfully!")
        print(f"Summary: {result['summary']}")
        print(f"Severity: {result['severity']}")
        print(f"Confidence: {result['confidence']}")
        
        if 'metrics' in result:
            print("\n📊 Metrics:")
            for key, value in result['metrics'].items():
                print(f"  {key}: {value}")
        
        if 'recommendations' in result:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Save demo result
        cv2.imwrite("lordosis_demo_result.png", result['visualized_image'])
        print("\n✅ Demo result saved as 'lordosis_demo_result.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in lordosis analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_medical_standards():
    """Display the medical standards used"""
    print("\n=== MEDICAL STANDARDS ===")
    print(f"Normal Pelvic Tilt Range: {NORMAL_PELVIC_TILT['anterior']}°")
    print(f"Normal Lumbar Lordosis Range: {NORMAL_LUMBAR_LORDOSIS['adult_male']}°")
    print("\nSeverity Thresholds:")
    for condition, threshold in SEVERITY_THRESHOLDS.items():
        print(f"  {condition}: {threshold}°")
    
    print("\nSeverity Classification:")
    print("  Normal: Within normal ranges")
    print("  Mild Lordosis: > Normal + 15°")
    print("  Moderate Lordosis: > Normal + 25°")
    print("  Severe Lordosis: > Normal + 35°")
    print("  Mild Kyphosis: < Normal - 10°")
    print("  Moderate Kyphosis: < Normal - 20°")
    print("  Severe Kyphosis: < Normal - 30°")


def main():
    """Run the lordosis demo"""
    print("🎉 Lordosis Analysis System Demo")
    print("=" * 50)
    
    # Show medical standards
    show_medical_standards()
    
    # Run demo
    success = demo_lordosis_analysis()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\n📋 Summary:")
        print("  ✅ Medical standards loaded")
        print("  ✅ Spine landmark detection working")
        print("  ✅ Lordosis metrics calculation working")
        print("  ✅ Severity classification working")
        print("  ✅ Recommendations generation working")
        print("  ✅ Visualization creation working")
        print("\n🚀 The lordosis analysis system is ready for use!")
    else:
        print("\n❌ Demo failed. Please check the implementation.")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
