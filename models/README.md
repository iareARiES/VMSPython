# YOLO Models Directory

Place your ONNX model files here for object detection.

## Supported Models

### Face Detection
- `best.onnx` - Face detection (WIDER FACE dataset)
- `w600k_mbf.onnx` - Face recognition embeddings

### General Detection  
- `yolo11npRETRAINED.onnx` - COCO general detection (all classes)
- Custom YOLO models trained on COCO dataset

### Specialized Detection
- `Fire_Event_best.onnx` - Fire/smoke detection
- Custom models for specific use cases

## Model Requirements

- **Format:** ONNX (.onnx files)
- **Input:** Standard YOLO input format (usually 640x640)
- **Output:** YOLO detection format

## Getting Models

1. **Train your own** using YOLOv11, YOLOv8, etc.
2. **Convert existing models** to ONNX format
3. **Download pre-trained** models from repositories

## Usage

1. Place `.onnx` files in this directory
2. Restart the application or click "Refresh" in Models tab
3. Enable desired models in the GUI
4. Configure confidence thresholds and classes

## File Size Note

ONNX model files are typically large (10MB - 500MB+) and are not included in the repository. You need to obtain them separately based on your detection requirements.
