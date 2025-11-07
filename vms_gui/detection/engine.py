"""
Detection engine for video processing and object detection.
"""

import cv2
import numpy as np
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: ONNX Runtime not available. Install with: pip install onnxruntime")

from ..config import COCO_CLASSES

# Detect platform
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


def detect_model_classes(model_path):
    """Detect classes from ONNX model by checking metadata or output shape."""
    if not HAS_ONNX:
        return []
    
    model_name = Path(model_path).stem.lower()
    
    # Method 1: Check filename FIRST for specific model types (face, fire, etc.)
    # This takes priority to avoid false positives from test inference
    if "face" in model_name or model_name == "best":
        # "best.onnx" is a face detection model
        print(f"Detected face model from filename: {Path(model_path).name}")
        return ["face"]
    elif "fire" in model_name or "smoke" in model_name:
        print(f"Detected fire/smoke model from filename: {Path(model_path).name}")
        return ["fire", "smoke"]
    
    try:
        # Create a temporary session to inspect the model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=[('CPUExecutionProvider', {})]
        )
        
        # Method 2: Check metadata for class names
        try:
            metadata = session.get_modelmeta()
            if metadata:
                # Check for custom metadata
                for key, value in metadata.custom_metadata_map.items():
                    if 'class' in key.lower() or 'label' in key.lower():
                        if isinstance(value, str):
                            # Try to parse as JSON or comma-separated
                            try:
                                import json
                                classes = json.loads(value)
                                if isinstance(classes, list) and len(classes) > 0:
                                    print(f"Detected classes from metadata: {classes}")
                                    return classes
                            except:
                                # Try comma-separated
                                classes = [c.strip() for c in value.split(',') if c.strip()]
                                if classes:
                                    print(f"Detected classes from metadata: {classes}")
                                    return classes
        except:
            pass
        
        # Method 3: For general models, try test inference to find actual class range
        # Only do this for models that don't match specific patterns
        try:
            outputs = session.get_outputs()
            if len(outputs) > 0:
                input_shape = session.get_inputs()[0].shape
                if len(input_shape) >= 4:
                    h = input_shape[2] if input_shape[2] else 640
                    w = input_shape[3] if input_shape[3] else 640
                    # Create dummy input
                    dummy_input = np.random.randn(1, 3, int(h), int(w)).astype(np.float32)
                    dummy_outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
                    
                    if len(dummy_outputs) > 0:
                        output = dummy_outputs[0]
                        # Find max class_id in output
                        if len(output.shape) >= 2 and output.shape[1] >= 6:
                            # Get all unique class_ids from output
                            class_ids = set()
                            for det in output[0]:
                                if len(det) >= 6:
                                    cls_id = int(det[5])
                                    if cls_id >= 0:
                                        class_ids.add(cls_id)
                            
                            if class_ids:
                                max_class_id = max(class_ids)
                                # Only return COCO classes if we have a reasonable range
                                # If max_class_id is 0 or very small, it might be a single-class model
                                if max_class_id == 0:
                                    # Single class model - return first COCO class
                                    print(f"Detected single-class model (class_id=0), using: {COCO_CLASSES[0]}")
                                    return [COCO_CLASSES[0]]
                                elif max_class_id < 10:
                                    # Small number of classes - return up to max
                                    classes = COCO_CLASSES[:max_class_id + 1]
                                    print(f"Detected {len(classes)} classes from model output: {classes}")
                                    return classes
                                else:
                                    # Large range - likely full COCO
                                    classes = COCO_CLASSES[:max_class_id + 1] if max_class_id < len(COCO_CLASSES) else COCO_CLASSES
                                    print(f"Detected {len(classes)} COCO classes from model output")
                                    return classes
        except Exception as e:
            print(f"Could not detect classes from model output: {e}")
        
        # Default: Return full COCO classes for general detection models
        print(f"Using default COCO classes for model: {Path(model_path).name}")
        return COCO_CLASSES
            
    except Exception as e:
        print(f"Error detecting classes from model {model_path}: {e}")
        # Final fallback: check filename again
        if "face" in model_name or model_name == "best":
            return ["face"]
        elif "fire" in model_name or "smoke" in model_name:
            return ["fire", "smoke"]
        return COCO_CLASSES


class VideoCapture:
    """Direct video capture."""
    def __init__(self, source, width=None, height=None):
        self.source = source
        self.cap = None
        self.target_width = width
        self.target_height = height
    
    def open(self):
        """Open camera with platform-specific handling."""
        # Convert source to appropriate type
        if isinstance(self.source, str):
            if self.source.startswith("/dev/video"):
                # Linux device path
                if IS_LINUX and hasattr(cv2, 'CAP_V4L2'):
                    try:
                        self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
                    except:
                        self.cap = cv2.VideoCapture(self.source)
                else:
                    self.cap = cv2.VideoCapture(self.source)
            elif self.source.isdigit():
                # Numeric string (camera index)
                source_int = int(self.source)
                self.cap = cv2.VideoCapture(source_int)
            else:
                # Try as string path
                self.cap = cv2.VideoCapture(self.source)
        elif isinstance(self.source, int):
            # Direct integer (camera index)
            self.cap = cv2.VideoCapture(self.source)
        else:
            # Try converting to string
            self.cap = cv2.VideoCapture(str(self.source))
        
        # Check if camera opened successfully
        if self.cap is None:
            raise RuntimeError(f"Failed to create VideoCapture object for: {self.source}")
        
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Failed to open camera: {self.source}. Camera may be in use or not available.")
        
        # Set resolution if specified
        if self.target_width and self.target_height:
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                # Give camera time to adjust
                time.sleep(0.1)
            except Exception as e:
                print(f"Warning: Could not set resolution to {self.target_width}x{self.target_height}: {e}")
        
        # Set camera properties (these may fail on some cameras, so wrap in try/except)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass  # Some backends don't support this
        
        try:
            if isinstance(self.source, int) or (isinstance(self.source, str) and str(self.source).isdigit()):
                self.cap.set(cv2.CAP_PROP_FPS, 30)
        except:
            pass  # Some cameras don't support FPS setting
        
        try:
            if isinstance(self.source, int) or (isinstance(self.source, str) and str(self.source).isdigit()):
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except:
            pass  # Some cameras don't support FOURCC setting
        
        # Verify we can actually read a frame
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            raise RuntimeError(f"Camera opened but cannot read frames from: {self.source}")
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Successfully opened camera {self.source}, resolution: {actual_width}x{actual_height}")
    
    def set_resolution(self, width, height):
        """Change camera resolution on the fly."""
        if self.cap and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                # Give camera time to adjust
                time.sleep(0.1)
                # Read a frame to apply the change
                self.cap.read()
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.target_width = actual_width
                self.target_height = actual_height
                print(f"Resolution changed to: {actual_width}x{actual_height}")
                return True
            except Exception as e:
                print(f"Error changing resolution: {e}")
                return False
        return False
    
    def read(self):
        """Read frame."""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release camera."""
        if self.cap:
            self.cap.release()
            self.cap = None


class ONNXRunner:
    """Direct ONNX inference."""
    def __init__(self, model_path, model_type="coco"):
        if not HAS_ONNX:
            raise RuntimeError("ONNX Runtime not available")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.inter_op_num_threads = 0
        sess_options.intra_op_num_threads = 0
        
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=[('CPUExecutionProvider', {})]
        )
        
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (640, 640)
        if len(input_shape) >= 4:
            # Handle None values and convert to int safely
            try:
                h_val = input_shape[2]
                w_val = input_shape[3]
                # Convert to int, handling None, strings, or negative values
                if h_val is not None:
                    try:
                        h_int = int(h_val)
                        h = h_int if h_int > 0 else 640
                    except (ValueError, TypeError):
                        h = 640
                else:
                    h = 640
                if w_val is not None:
                    try:
                        w_int = int(w_val)
                        w = w_int if w_int > 0 else 640
                    except (ValueError, TypeError):
                        w = 640
                else:
                    w = 640
                self.input_size = (w, h)
            except (ValueError, TypeError, IndexError):
                # Default to 640x640 if parsing fails
                self.input_size = (640, 640)
        
        # Detect classes from model
        self.model_type = model_type
        self.class_names = detect_model_classes(model_path)
        print(f"Model {Path(model_path).name} detected classes: {len(self.class_names)} classes")
    
    def infer(self, frame, conf_threshold=0.25):
        """Run inference on frame."""
        h, w = self.input_size
        img_resized = cv2.resize(frame, (w, h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_norm, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        outputs = self.session.run(None, {self.input_name: img_batch})
        
        detections = []
        if len(outputs) > 0:
            output = outputs[0]
            if output.shape[1] >= 6:
                frame_h, frame_w = frame.shape[:2]
                scale_x = frame_w / w
                scale_y = frame_h / h
                
                for det in output[0]:
                    if det[4] >= conf_threshold:
                        x1, y1, x2, y2 = det[0:4]
                        conf = float(det[4])
                        cls_id = int(det[5])
                        
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "class": cls_name,
                            "class_id": cls_id
                        })
        
        return detections
    
    def get_classes(self):
        """Get available classes for this model."""
        return self.class_names


class DetectionEngine:
    """Direct detection engine."""
    def __init__(self):
        self.models = {}
        self.model_configs = {}  # Store model configs (enabled classes, confidence, etc.)
        self.capture = None
        self.running = False
        self.current_frame = None
        self.current_detections = []
    
    def load_model(self, model_name, model_path, model_type="coco"):
        """Load ONNX model."""
        if model_name not in self.models:
            self.models[model_name] = ONNXRunner(model_path, model_type)
            self.model_configs[model_name] = {
                "enabled": False,
                "conf": 0.35,
                "enabled_classes": {}
            }
    
    def get_model_classes(self, model_name):
        """Get classes for a model."""
        if model_name in self.models:
            return self.models[model_name].get_classes()
        return []
    
    def set_model_config(self, model_name, enabled, conf=None, enabled_classes=None):
        """Configure model."""
        if model_name in self.model_configs:
            if enabled is not None:
                self.model_configs[model_name]["enabled"] = enabled
            if conf is not None:
                self.model_configs[model_name]["conf"] = conf
            if enabled_classes is not None:
                self.model_configs[model_name]["enabled_classes"] = enabled_classes
    
    def start_capture(self, source="/dev/video0", width=None, height=None):
        """Start video capture."""
        self.capture = VideoCapture(source, width, height)
        self.capture.open()
        self.running = True
    
    def set_resolution(self, width, height):
        """Change video capture resolution."""
        if self.capture and self.capture.cap:
            return self.capture.set_resolution(width, height)
        return False
    
    def stop_capture(self):
        """Stop video capture."""
        self.running = False
        if self.capture:
            self.capture.release()
            self.capture = None
    
    def process_frame(self):
        """Process one frame with enabled models."""
        if not self.capture or not self.running:
            return None, []
        
        frame = self.capture.read()
        if frame is None:
            return None, []
        
        all_detections = []
        for model_name, config in self.model_configs.items():
            if not config["enabled"] or model_name not in self.models:
                continue
            
            runner = self.models[model_name]
            conf_threshold = config.get("conf", 0.35)
            enabled_classes = config.get("enabled_classes", {})
            
            detections = runner.infer(frame, conf_threshold)
            
            # Filter by enabled classes
            for det in detections:
                cls_name = det["class"]
                # If no classes specified, allow all; otherwise check enabled_classes
                if enabled_classes and not enabled_classes.get(cls_name, False):
                    continue
                det["model"] = model_name
                all_detections.append(det)
        
        self.current_frame = frame
        self.current_detections = all_detections
        return frame, all_detections

