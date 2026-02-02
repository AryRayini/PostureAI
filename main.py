import cv2
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from posture_analysis.segmentation import BodySegmentation
from posture_analysis.pose_estimation import PoseEstimator
from posture_analysis.leg_analyzer import LegAnalyzer
from posture_analysis.lordosis_analyzer import LordosisAnalyzer
from posture_analysis.kyphosis_analyzer import KyphosisAnalyzer


def main_menu():
    """Main menu for selecting analysis type"""
    print("\n" + "="*60)
    print("           PHYSIOAPP - POSTURE ANALYSIS SYSTEM")
    print("="*60)
    print("\nPlease select the type of analysis you want to perform:")
    print("\n1. 🦵 Leg Analysis (Bowlegs/Knock Knees)")
    print("2. 📐 Lordosis Analysis (گودی کمر)")
    print("3. 🏔️ Kyphosis Analysis (قوز پشت)")
    print("4. 🔍 Complete Posture Analysis (All)")
    print("5. 🧪 Test System (Demo)")
    print("6. 🚪 Exit")
    print("\n" + "="*60)


def select_image_file():
    """Open file dialog to select image"""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        file_path = filedialog.askopenfilename(
            title="Select Posture Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        
        if not file_path:
            print("❌ No file selected.")
            return None
        
        return file_path
        
    except Exception as e:
        print(f"❌ Error opening file dialog: {e}")
        return None


def load_image(image_path):
    """Load image with error handling"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image


def create_mask(image):
    """Create segmentation mask"""
    segmenter = BodySegmentation(model_path="assets/models/yolov8n-seg.pt")
    return segmenter.get_mask(image)


def create_landmarks(image):
    """Create pose landmarks"""
    pose_estimator = PoseEstimator()
    return pose_estimator.get_landmarks(image)


def leg_analysis_workflow(image_path):
    """Complete leg analysis workflow"""
    print("\n🦵 LEG ANALYSIS WORKFLOW")
    print("-" * 40)
    
    try:
        # Load and process image
        image = load_image(image_path)
        mask = create_mask(image)
        landmarks = create_landmarks(image)
        
        # Run leg analysis
        leg_analyzer = LegAnalyzer()
        result = leg_analyzer.evaluate_leg_alignment(landmarks, mask, image)
        
        # Display results
        print("=== LEG ANALYSIS RESULTS ===")
        print(result["summary"])
        
        # Display visualization
        cv2.namedWindow("Leg Analysis", cv2.WINDOW_NORMAL)
        cv2.imshow("Leg Analysis", result["visualized_image"])
        
        # Save option
        print("\nPress 'y' to save the leg analysis image, or any other key to exit.")
        key = cv2.waitKey(0)
        if key == ord('y'):
            save_result(result["visualized_image"], image_path, "leg_analysis")
        
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"❌ Error in leg analysis: {e}")
        return False


def lordosis_analysis_workflow(image_path):
    """Complete lordosis analysis workflow"""
    print("\n📐 LORDOSIS ANALYSIS WORKFLOW (گودی کمر)")
    print("-" * 40)
    
    try:
        # Load and process image
        image = load_image(image_path)
        mask = create_mask(image)
        landmarks = create_landmarks(image)
        
        # Check if person detected
        if landmarks is None or len(landmarks) == 0:
            print("❌ No person detected in image.")
            return False
        
        # Run lordosis analysis
        lordosis_analyzer = LordosisAnalyzer()
        result = lordosis_analyzer.analyze(landmarks, mask, image)
        
        # Display results
        print("=== LORDOSIS ANALYSIS RESULTS ===")
        print(result["summary"])
        
        if "severity" in result:
            print(f"Severity: {result['severity']}")
        
        if "confidence" in result:
            print(f"Confidence: {result['confidence']}")
        
        if "metrics" in result:
            print("\n📊 Metrics:")
            for key, value in result["metrics"].items():
                print(f"  {key}: {value}")
        
        if "recommendations" in result:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        # Display visualization
        cv2.namedWindow("Lordosis Analysis", cv2.WINDOW_NORMAL)
        cv2.imshow("Lordosis Analysis", result["visualized_image"])
        
        # Save option
        print("\nPress 'y' to save the lordosis analysis image, or any other key to exit.")
        key = cv2.waitKey(0)
        if key == ord('y'):
            save_result(result["visualized_image"], image_path, "lordosis_analysis")
        
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"❌ Error in lordosis analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def kyphosis_analysis_workflow(image_path):
    """Complete kyphosis analysis workflow"""
    print("\n🏔️ KYPHOSIS ANALYSIS WORKFLOW (قوز پشت)")
    print("-" * 40)
    
    try:
        # Load and process image
        image = load_image(image_path)
        mask = create_mask(image)
        landmarks = create_landmarks(image)
        
        # Run kyphosis analysis
        kyphosis_analyzer = KyphosisAnalyzer()
        result = kyphosis_analyzer.analyze(landmarks, mask, image)
        
        # Display results
        print("=== KYPHOSIS ANALYSIS RESULTS ===")
        print(result["summary"])
        
        if "severity" in result:
            print(f"Severity: {result['severity']}")
        
        if "confidence" in result:
            print(f"Confidence: {result['confidence']}")
        
        # Display visualization
        cv2.namedWindow("Kyphosis Analysis", cv2.WINDOW_NORMAL)
        cv2.imshow("Kyphosis Analysis", result["visualized_image"])
        
        # Save option
        print("\nPress 'y' to save the kyphosis analysis image, or any other key to exit.")
        key = cv2.waitKey(0)
        if key == ord('y'):
            save_result(result["visualized_image"], image_path, "kyphosis_analysis")
        
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"❌ Error in kyphosis analysis: {e}")
        return False


def complete_analysis_workflow(image_path):
    """Complete posture analysis workflow (all analyses)"""
    print("\n🔍 COMPLETE POSTURE ANALYSIS WORKFLOW")
    print("-" * 40)
    
    try:
        # Load and process image once
        image = load_image(image_path)
        mask = create_mask(image)
        landmarks = create_landmarks(image)
        
        # Run all analyses
        print("Running Leg Analysis...")
        leg_analyzer = LegAnalyzer()
        leg_result = leg_analyzer.evaluate_leg_alignment(landmarks, mask, image)
        
        print("Running Lordosis Analysis...")
        lordosis_analyzer = LordosisAnalyzer()
        lordosis_result = lordosis_analyzer.analyze(landmarks, mask, image)
        
        print("Running Kyphosis Analysis...")
        kyphosis_analyzer = KyphosisAnalyzer()
        kyphosis_result = kyphosis_analyzer.analyze(landmarks, mask, image)
        
        # Display comprehensive results
        print("\n" + "="*60)
        print("           COMPREHENSIVE POSTURE ANALYSIS")
        print("="*60)
        
        print("\n🦵 LEG ANALYSIS:")
        print(leg_result["summary"])
        
        print("\n📐 LORDOSIS ANALYSIS:")
        print(lordosis_result["summary"])
        if "severity" in lordosis_result:
            print(f"Severity: {lordosis_result['severity']}")
        
        print("\n🏔️ KYPHOSIS ANALYSIS:")
        print(kyphosis_result["summary"])
        if "severity" in kyphosis_result:
            print(f"Severity: {kyphosis_result['severity']}")
        
        print("\n" + "="*60)
        
        # Display all visualizations
        cv2.namedWindow("Complete Posture Analysis", cv2.WINDOW_NORMAL)
        cv2.imshow("Complete Posture Analysis", leg_result["visualized_image"])
        
        # Save option
        print("\nPress 'y' to save the complete analysis image, or any other key to exit.")
        key = cv2.waitKey(0)
        if key == ord('y'):
            save_result(leg_result["visualized_image"], image_path, "complete_analysis")
        
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"❌ Error in complete analysis: {e}")
        return False


def test_system():
    """Test system with demo data"""
    print("\n🧪 TESTING SYSTEM WITH DEMO DATA")
    print("-" * 40)
    
    try:
        # Run the demo
        import subprocess
        result = subprocess.run(["python", "demo_lordosis.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System test completed successfully!")
            print("Demo results saved as 'lordosis_demo_result.png'")
        else:
            print("❌ System test failed!")
            print("Error:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running system test: {e}")
        return False


def save_result(image, original_path, analysis_type):
    """Save analysis result with timestamp"""
    try:
        exports_dir = "exports"
        os.makedirs(exports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_filename = f"{base_name}_{analysis_type}_{timestamp}.png"
        output_path = os.path.join(exports_dir, output_filename)
        
        cv2.imwrite(output_path, image)
        print(f"✅ Analysis result saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving result: {e}")


def main():
    """Main application loop"""
    print("🚀 Starting PhysioApp Posture Analysis System...")
    
    while True:
        try:
            # Show main menu
            main_menu()
            
            # Get user choice
            choice = input("\nPlease enter your choice (1-6): ").strip()
            
            if choice == '6':
                print("\n👋 Thank you for using PhysioApp! Goodbye!")
                break
            
            # Get image path using file browser
            if choice in ['1', '2', '3', '4']:
                print("\n📁 Opening file browser to select image...")
                image_path = select_image_file()
                
                if not image_path:
                    print("❌ No image selected. Please try again.")
                    continue
                
                print(f"✅ Selected image: {os.path.basename(image_path)}")
            
            # Execute selected analysis
            if choice == '1':
                leg_analysis_workflow(image_path)
                print("\nPress 'Q' to quit or any other key to return to main menu.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Thank you for using PhysioApp! Goodbye!")
                    break
                cv2.destroyAllWindows()
                
            elif choice == '2':
                lordosis_analysis_workflow(image_path)
                print("\nPress 'Q' to quit or any other key to return to main menu.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Thank you for using PhysioApp! Goodbye!")
                    break
                cv2.destroyAllWindows()
                
            elif choice == '3':
                kyphosis_analysis_workflow(image_path)
                print("\nPress 'Q' to quit or any other key to return to main menu.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Thank you for using PhysioApp! Goodbye!")
                    break
                cv2.destroyAllWindows()
                
            elif choice == '4':
                complete_analysis_workflow(image_path)
                print("\nPress 'Q' to quit or any other key to return to main menu.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Thank you for using PhysioApp! Goodbye!")
                    break
                cv2.destroyAllWindows()
                
            elif choice == '5':
                test_system()
                print("\nPress 'Q' to quit or any other key to return to main menu.")
                input("Press Enter to continue...")
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using PhysioApp! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or contact support.")


if __name__ == "__main__":
    main()
