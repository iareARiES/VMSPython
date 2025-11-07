#!/usr/bin/env python3
"""
Face Registration Script
========================
Register faces in the database for face recognition.

This script captures 30 images from each angle (front, left, right) for better recognition accuracy.
Total: 90 images per person.

Usage:
    python register_face.py
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from vms_gui.detection.engine import DetectionEngine
from vms_gui.detection.face_recognition import FaceRecognizer


def find_face_detection_model():
    """Find the face detection model."""
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    # Look for face detection models
    for model_file in models_dir.glob("*.onnx"):
        model_name = model_file.stem.lower()
        if "face" in model_name or model_name == "best":
            return str(model_file)
    
    return None


def find_recognition_model():
    """Find the face recognition/embedding model."""
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    # Look for recognition models (w600k, mbf, etc.)
    for model_file in models_dir.glob("*.onnx"):
        model_name = model_file.stem.lower()
        if "w600k" in model_name or "mbf" in model_name:
            return str(model_file)
    
    return None


def capture_images_for_angle(cap, engine, recognizer, name, angle_name, count=30, delay=0.2):
    """
    Capture multiple images for a specific angle.
    
    Args:
        cap: VideoCapture object
        engine: DetectionEngine
        recognizer: FaceRecognizer
        name: Person's name
        angle_name: "Front", "Left", or "Right"
        count: Number of images to capture
        delay: Delay between captures in seconds
    """
    print(f"\n{'='*60}")
    print(f"Capturing {count} images from {angle_name} angle")
    print(f"{'='*60}")
    print(f"Position the person's face to face {angle_name.lower()}")
    print("Press 'q' to cancel, or wait for automatic capture...")
    print()
    
    captured = 0
    last_capture_time = 0
    
    while captured < count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        
        # Run face detection
        detections = []
        try:
            if "face_detector" in engine.models:
                runner = engine.models["face_detector"]
                detections = runner.infer(frame, conf_threshold=0.5)
                detections = [d for d in detections if d.get("class") == "face"]
        except Exception as e:
            pass
        
        # Find the largest face
        largest_face = None
        largest_area = 0
        current_face_bbox = None
        
        for det in detections:
            if det.get("class") == "face":
                x1, y1, x2, y2 = det["bbox"]
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_face = det
                    current_face_bbox = (x1, y1, x2, y2)
        
        # Draw detection
        display_frame = frame.copy()
        
        if largest_face:
            x1, y1, x2, y2 = current_face_bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extract face ROI
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y2 = max(y1 + 1, min(y2, frame_h))
            
            face_roi = frame[y1:y2, x1:x2]
            
            # Auto-capture if face detected and enough time has passed
            current_time = time.time()
            if face_roi.size > 0 and face_roi.shape[0] > 20 and face_roi.shape[1] > 20:
                if current_time - last_capture_time >= delay:
                    try:
                        # Register face
                        success = recognizer.register_face(name, face_roi)
                        if success:
                            captured += 1
                            last_capture_time = current_time
                            print(f"  [{captured}/{count}] Captured from {angle_name}")
                        else:
                            print(f"  Warning: Failed to register image {captured + 1}")
                    except Exception as e:
                        print(f"  Error capturing image: {e}")
            
            # Draw label
            label = f"{angle_name}: {captured}/{count} captured"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, f"No face detected - {angle_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show progress
        progress_text = f"{angle_name} Angle: {captured}/{count} images"
        cv2.putText(display_frame, progress_text, (10, display_frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to cancel", (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Face Registration", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\nCancelled. Captured {captured}/{count} images from {angle_name}.")
            return captured
    
    print(f"\n✓ Completed {angle_name} angle: {captured} images captured")
    return captured


def main():
    """Main registration function."""
    print("=" * 60)
    print("Face Registration Tool")
    print("=" * 60)
    print("This script will capture 30 images from each angle:")
    print("  - Front (30 images)")
    print("  - Left (30 images)")
    print("  - Right (30 images)")
    print("Total: 90 images per person")
    print("=" * 60)
    print()
    
    # Find models
    print("Looking for models...")
    face_detection_model = find_face_detection_model()
    recognition_model = find_recognition_model()
    
    if not face_detection_model:
        print("ERROR: Face detection model not found!")
        print("Please ensure you have a face detection model (e.g., best.onnx) in the models/ directory")
        return
    
    if not recognition_model:
        print("ERROR: Face recognition model not found!")
        print("Please ensure you have a recognition model (e.g., w600k_mbf.onnx) in the models/ directory")
        return
    
    print(f"Face detection model: {face_detection_model}")
    print(f"Recognition model: {recognition_model}")
    print()
    
    # Initialize detection engine
    print("Loading models...")
    engine = DetectionEngine()
    engine.load_model("face_detector", face_detection_model, model_type="face")
    engine.set_model_config("face_detector", enabled=True, conf=0.5, enabled_classes={"face": True})
    
    # Initialize face recognizer
    try:
        recognizer = FaceRecognizer(recognition_model)
        print("Face recognizer loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load face recognizer: {e}")
        return
    
    # Get person's name
    print("\n" + "=" * 60)
    name = input("Enter the person's name to register: ").strip()
    if not name:
        print("Name cannot be empty. Exiting.")
        return
    
    print(f"\nRegistering: {name}")
    print("=" * 60)
    
    # Open camera
    print("\nOpening camera...")
    camera_source = 0
    try:
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {camera_source}")
            print("Try a different camera index (0, 1, 2, etc.)")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Camera opened successfully")
        print()
    except Exception as e:
        print(f"ERROR: Failed to open camera: {e}")
        return
    
    # Capture from each angle
    total_captured = 0
    
    # Front angle
    front_count = capture_images_for_angle(cap, engine, recognizer, name, "Front", count=30, delay=0.2)
    total_captured += front_count
    
    if front_count < 30:
        response = input(f"\nOnly captured {front_count}/30 from front. Continue? (y/n): ")
        if response.lower() != 'y':
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nRegistration incomplete. Total images: {total_captured}")
            return
    
    # Left angle
    print("\n" + "=" * 60)
    print("Now turn the person's face to the LEFT (their left)")
    input("Press Enter when ready to continue...")
    left_count = capture_images_for_angle(cap, engine, recognizer, name, "Left", count=30, delay=0.2)
    total_captured += left_count
    
    if left_count < 30:
        response = input(f"\nOnly captured {left_count}/30 from left. Continue? (y/n): ")
        if response.lower() != 'y':
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nRegistration incomplete. Total images: {total_captured}")
            return
    
    # Right angle
    print("\n" + "=" * 60)
    print("Now turn the person's face to the RIGHT (their right)")
    input("Press Enter when ready to continue...")
    right_count = capture_images_for_angle(cap, engine, recognizer, name, "Right", count=30, delay=0.2)
    total_captured += right_count
    
    # Summary
    print("\n" + "=" * 60)
    print("Registration Summary")
    print("=" * 60)
    print(f"Name: {name}")
    print(f"Front angle: {front_count}/30 images")
    print(f"Left angle: {left_count}/30 images")
    print(f"Right angle: {right_count}/30 images")
    print(f"Total: {total_captured}/90 images")
    print("=" * 60)
    
    if total_captured >= 30:
        print(f"\n✓ Successfully registered {name} with {total_captured} images")
    else:
        print(f"\n⚠ Warning: Only {total_captured} images captured (recommended: 90)")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nRegistration session ended.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nRegistration interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
