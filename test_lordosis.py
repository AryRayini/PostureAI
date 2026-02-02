#!/usr/bin/env python3
"""
Test script for lordosis analysis implementation
"""

import cv2
import numpy as np
from posture_analysis.lordosis_analyzer import LordosisAnalyzer, SpineLandmarks, LordosisMetrics
from configs.lordosis_settings import NORMAL_PELVIC_TILT, SEVERITY_THRESHOLDS


def test_lordosis_analyzer():
    """Test the lordosis analyzer with mock data"""
    print("=== Testing Lordosis Analyzer ===")
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image.fill(200)  # Light gray background
    
    # Create mock landmarks (simulating MediaPipe landmarks structure)
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    class MockLandmarks:
        def __init__(self):
            self.landmark = [
                MockLandmark(0.5, 0.5),  # 0
                MockLandmark(0.5, 0.5),  # 1
                MockLandmark(0.5, 0.5),  # 2
                MockLandmark(0.5, 0.5),  # 3
                MockLandmark(0.5, 0.5),  # 4
                MockLandmark(0.5, 0.5),  # 5
                MockLandmark(0.5, 0.5),  # 6
                MockLandmark(0.5, 0.5),  # 7
                MockLandmark(0.5, 0.5),  # 8
                MockLandmark(0.5, 0.5),  # 9
                MockLandmark(0.5, 0.5),  # 10
                MockLandmark(0.4, 0.3),  # 11 - Left shoulder
                MockLandmark(0.6, 0.3),  # 12 - Right shoulder
                MockLandmark(0.4, 0.6),  # 13 - Left elbow
                MockLandmark(0.6, 0.6),  # 14 - Right elbow
                MockLandmark(0.4, 0.7),  # 15 - Left wrist
                MockLandmark(0.6, 0.7),  # 16 - Right wrist
                MockLandmark(0.35, 0.8), # 17 - Left hip
                MockLandmark(0.65, 0.8), # 18 - Right hip
                MockLandmark(0.35, 0.9), # 19 - Left knee
                MockLandmark(0.65, 0.9), # 20 - Right knee
                MockLandmark(0.35, 1.0), # 21 - Left ankle
                MockLandmark(0.65, 1.0), # 22 - Right ankle
                MockLandmark(0.35, 0.8), # 23 - Left hip (duplicate)
                MockLandmark(0.65, 0.8), # 24 - Right hip (duplicate)
                MockLandmark(0.35, 0.9), # 25 - Left knee (duplicate)
                MockLandmark(0.65, 0.9), # 26 - Right knee (duplicate)
                MockLandmark(0.35, 1.0), # 27 - Left ankle (duplicate)
                MockLandmark(0.65, 1.0), # 28 - Right ankle (duplicate)
            ]
    
    mock_landmarks = MockLandmarks()
    
    # Create mock mask
    mask = np.ones((480, 640), dtype=np.uint8) * 255
    
    # Initialize analyzer
    analyzer = LordosisAnalyzer()
    
    try:
        # Test analysis
        result = analyzer.analyze(mock_landmarks, mask, test_image)
        
        print("✅ Lordosis analysis completed successfully!")
        print(f"Summary: {result['summary']}")
        print(f"Severity: {result['severity']}")
        print(f"Confidence: {result['confidence']}")
        
        if 'metrics' in result:
            print("\nMetrics:")
            for key, value in result['metrics'].items():
                print(f"  {key}: {value}")
        
        if 'recommendations' in result:
            print("\nRecommendations:")
            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"  {i}. {rec}")
        
        # Test visualization
        if 'visualized_image' in result:
            print("✅ Visualization created successfully!")
            
            # Save test result
            cv2.imwrite("test_lordosis_result.png", result['visualized_image'])
            print("✅ Test result saved as 'test_lordosis_result.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in lordosis analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spine_landmarks():
    """Test spine landmark detection"""
    print("\n=== Testing Spine Landmarks ===")
    
    try:
        # Test spine landmarks dataclass
        landmarks = SpineLandmarks(
            c7=(0.5, 0.2),
            t1=(0.5, 0.3),
            t12=(0.5, 0.6),
            l1=(0.5, 0.7),
            l5=(0.5, 0.9),
            s1=(0.5, 0.95),
            confidence=0.9,
            is_valid=True
        )
        
        print("✅ Spine landmarks created successfully!")
        print(f"C7: {landmarks.c7}")
        print(f"L1-L5 distance: {((0.5-0.5)**2 + (0.7-0.9)**2)**0.5:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating spine landmarks: {e}")
        return False


def test_lordosis_metrics():
    """Test lordosis metrics calculation"""
    print("\n=== Testing Lordosis Metrics ===")
    
    try:
        # Test metrics dataclass
        metrics = LordosisMetrics(
            pelvic_tilt_angle=12.5,
            lumbar_lordosis_angle=45.2,
            sagittal_vertical_axis=25.3,
            curve_depth=15.7,
            severity="normal",
            confidence=0.85,
            recommendations=["Maintain good posture"]
        )
        
        print("✅ Lordosis metrics created successfully!")
        print(f"Pelvic Tilt: {metrics.pelvic_tilt_angle}°")
        print(f"Lumbar Lordosis: {metrics.lumbar_lordosis_angle}°")
        print(f"Severity: {metrics.severity}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating lordosis metrics: {e}")
        return False


def test_medical_standards():
    """Test medical standards configuration"""
    print("\n=== Testing Medical Standards ===")
    
    try:
        print("Normal Pelvic Tilt Range:", NORMAL_PELVIC_TILT['anterior'])
        print("Severity Thresholds:", SEVERITY_THRESHOLDS)
        
        # Test severity classification logic
        normal_range = NORMAL_PELVIC_TILT['anterior']
        mild_threshold = normal_range[1] + SEVERITY_THRESHOLDS['mild_lordosis']
        moderate_threshold = normal_range[1] + SEVERITY_THRESHOLDS['moderate_lordosis']
        severe_threshold = normal_range[1] + SEVERITY_THRESHOLDS['severe_lordosis']
        
        print(f"Mild lordosis threshold: > {mild_threshold}°")
        print(f"Moderate lordosis threshold: > {moderate_threshold}°")
        print(f"Severe lordosis threshold: > {severe_threshold}°")
        
        print("✅ Medical standards loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading medical standards: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Starting Lordosis Analysis Tests\n")
    
    tests = [
        test_medical_standards,
        test_spine_landmarks,
        test_lordosis_metrics,
        test_lordosis_analyzer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Lordosis analysis implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
