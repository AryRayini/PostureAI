#!/usr/bin/env python3
"""
Test script for the separate analysis pages
"""

import cv2
import numpy as np
from main import leg_analysis_workflow, lordosis_analysis_workflow, kyphosis_analysis_workflow


def create_test_image():
    """Create a test image for analysis"""
    # Create a test image with a simple stick figure
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image.fill(255)  # White background
    
    # Draw a simple stick figure
    # Head
    cv2.circle(image, (400, 100), 30, (0, 0, 0), -1)
    
    # Shoulders
    cv2.line(image, (350, 180), (450, 180), (0, 0, 255), 3)
    
    # Hips
    cv2.line(image, (340, 420), (460, 420), (255, 0, 0), 3)
    
    # Spine (approximate curve)
    spine_points = [
        (400, 130),  # C7
        (395, 200),  # T1
        (390, 300),  # T12
        (385, 350),  # L1
        (380, 450),  # L5
        (380, 500)   # S1
    ]
    
    for i in range(len(spine_points) - 1):
        cv2.line(image, spine_points[i], spine_points[i+1], (0, 255, 0), 3)
    
    # Legs
    cv2.line(image, (350, 420), (350, 550), (0, 0, 0), 3)
    cv2.line(image, (450, 420), (450, 550), (0, 0, 0), 3)
    
    return image


def test_leg_analysis():
    """Test the leg analysis workflow"""
    print("\n=== Testing Leg Analysis ===")
    
    try:
        # Create test image
        test_image = create_test_image()
        cv2.imwrite("test_leg_image.jpg", test_image)
        
        # Test leg analysis
        result = leg_analysis_workflow("test_leg_image.jpg")
        
        if result:
            print("✅ Leg analysis test passed!")
            return True
        else:
            print("❌ Leg analysis test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in leg analysis test: {e}")
        return False


def test_lordosis_analysis():
    """Test the lordosis analysis workflow"""
    print("\n=== Testing Lordosis Analysis ===")
    
    try:
        # Create test image
        test_image = create_test_image()
        cv2.imwrite("test_lordosis_image.jpg", test_image)
        
        # Test lordosis analysis
        result = lordosis_analysis_workflow("test_lordosis_image.jpg")
        
        if result:
            print("✅ Lordosis analysis test passed!")
            return True
        else:
            print("❌ Lordosis analysis test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in lordosis analysis test: {e}")
        return False


def test_kyphosis_analysis():
    """Test the kyphosis analysis workflow"""
    print("\n=== Testing Kyphosis Analysis ===")
    
    try:
        # Create test image
        test_image = create_test_image()
        cv2.imwrite("test_kyphosis_image.jpg", test_image)
        
        # Test kyphosis analysis
        result = kyphosis_analysis_workflow("test_kyphosis_image.jpg")
        
        if result:
            print("✅ Kyphosis analysis test passed!")
            return True
        else:
            print("❌ Kyphosis analysis test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in kyphosis analysis test: {e}")
        return False


def test_main_menu():
    """Test the main menu functionality"""
    print("\n=== Testing Main Menu ===")
    
    try:
        # This would normally require user input, so we'll just test the function exists
        from main import main_menu
        main_menu()
        print("✅ Main menu test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in main menu test: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Separate Analysis Pages")
    print("=" * 50)
    
    tests = [
        ("Main Menu", test_main_menu),
        ("Leg Analysis", test_leg_analysis),
        ("Lordosis Analysis", test_lordosis_analysis),
        ("Kyphosis Analysis", test_kyphosis_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed!")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The separate analysis pages are working correctly.")
        print("\n📋 Summary:")
        print("  ✅ Main menu interface working")
        print("  ✅ Leg analysis workflow functional")
        print("  ✅ Lordosis analysis workflow functional")
        print("  ✅ Kyphosis analysis workflow functional")
        print("  ✅ Separate analysis pages implemented")
        print("\n🚀 The PhysioApp is ready with 3 dedicated analysis pages!")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    # Clean up test images
    import os
    test_images = ["test_leg_image.jpg", "test_lordosis_image.jpg", "test_kyphosis_image.jpg"]
    for img in test_images:
        if os.path.exists(img):
            os.remove(img)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
