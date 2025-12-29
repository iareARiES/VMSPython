# Video Management System (VMS) - Python Client

A comprehensive **Video Management System** with AI-powered object detection, face recognition, and intelligent video analysis capabilities. Built with Python, PySide6 (Qt6), and ONNX Runtime for real-time inference.

## 🎯 Overview

This VMS Client is a desktop application that provides:
- **Real-time video detection** from cameras or video files
- **AI-powered object detection** using YOLO-based ONNX models
- **Face recognition** with database management
- **Video playback analysis** with natural language queries
- **Detection database** for storing and querying results
- **SOS alert system** for security events
- **Multi-model support** with configurable detection classes

## 🏗️ System Architecture

### Core Components

```
VMSPython/
├── vms_gui.py                 # Main entry point
├── register_face.py          # Face registration utility
├── requirements.txt           # Python dependencies
│
├── vms_gui/                   # Main application package
│   ├── app.py                # Main application window
│   ├── config.py             # Configuration and constants
│   │
│   ├── detection/            # Detection engine
│   │   ├── engine.py        # DetectionEngine, ONNXRunner, VideoCapture
│   │   ├── face_recognition.py  # FaceRecognizer with embedding extraction
│   │   ├── face_database.py  # Face embeddings database
│   │   └── detection_database.py  # Detection results database
│   │
│   └── gui/                  # GUI components
│       ├── components.py     # TopBar, BottomBar UI components
│       ├── model_config.py   # Model configuration panel
│       ├── video_display.py  # Live video display widget
│       ├── video_player.py   # Video playback widget
│       ├── results_panel.py  # Detection results display
│       ├── chatbot.py        # Natural language query interface
│       └── gemini_parser.py  # Query parser (Gemini API + fallback)
│
├── models/                   # ONNX model files
│   ├── best.onnx            # Face detection model
│   ├── w600k_mbf.onnx       # Face recognition/embedding model
│   ├── yolo11npRETRAINED.onnx  # General object detection (COCO)
│   └── Fire_Event_best.onnx # Fire/smoke detection
│
└── storage/                  # Data storage
    └── db/                   # SQLite databases
        ├── app.db           # Application database
        ├── events.sqlite    # Events database
        ├── face_embeddings.db  # Face recognition database
        └── detection_results.db  # Detection results database
```

### Architecture Flow

1. **Video Input** → `VideoCapture` (supports cameras, video files, V4L2 devices)
2. **Frame Processing** → `DetectionEngine.process_frame()`
3. **Model Inference** → `ONNXRunner.infer()` (multiple models supported)
4. **Face Recognition** → `FaceRecognizer.recognize_face()` (if enabled)
5. **Results Storage** → `DetectionDatabase.save_detection()`
6. **GUI Display** → `VideoDisplay` / `VideoPlayer` with real-time visualization
7. **Query System** → `ChatBot` → `QueryParser` → Database queries

## ✨ Key Features

### 🎥 Live Video Detection
- Real-time camera feed with AI detection overlay
- Support for USB webcams, V4L2 devices (Linux/Raspberry Pi), and video files
- Configurable resolution and frame rate
- Multiple camera source selection
- Bounding box visualization with class labels and confidence scores

### 🤖 Multi-Model Detection
- **Face Detection**: Detect faces in real-time (`best.onnx`)
- **Object Detection**: COCO classes (person, car, etc.) (`yolo11npRETRAINED.onnx`)
- **Fire/Smoke Detection**: Fire and smoke detection (`Fire_Event_best.onnx`)
- **Custom Models**: Support for any ONNX YOLO-based model
- Per-model configuration (confidence threshold, enabled classes)
- Automatic class detection from model metadata

### 👤 Face Recognition
- **Embedding-based recognition** using `w600k_mbf.onnx`
- **Face database** with multiple embeddings per person
- **Registration tool** (`register_face.py`) captures 90 images per person (30 from each angle)
- **Real-time recognition** during live detection
- **Known/Unknown face classification**
- Cosine similarity matching with configurable threshold

### 📹 Video Playback Analysis
- Load and analyze video files
- Frame-by-frame detection processing
- Detection results panel with thumbnails
- Jump to detection timestamps
- Export detection frames

### 💬 Natural Language Query System
- **Chatbot interface** for video analysis queries
- **Gemini API integration** for intelligent query parsing (with fallback parser)
- **Example queries**:
  - "find all humans from 10 min to 15 min and save them"
  - "find tigers in the video"
  - "find all unknown faces from 5:00 to 10:00"
  - "find whatever you see and save to database"
- Query results displayed in results panel
- Save detection images to filesystem

### 🗄️ Detection Database
- **SQLite-based storage** for all detection results
- Stores: timestamp, class, confidence, bounding box, frame image, model name
- **Query interface** with filters (class, time range, recognized name)
- **Statistics** by class and model
- Efficient indexing for fast queries

### 🚨 SOS Alert System
- Configurable triggers based on detection counts
- **Unknown face alerts**: Trigger when unknown faces detected
- **Known face alerts**: Trigger when specific known faces detected
- **Class-based alerts**: Custom thresholds for any detection class
- Visual SOS indicator in top bar
- User confirmation before triggering

### ⚙️ Model Configuration
- Enable/disable models independently
- Adjust confidence thresholds per model
- Filter by detection classes (enable only specific classes)
- Real-time configuration changes
- Model auto-discovery from `models/` directory

## 📋 Requirements

### System Requirements
- **Python**: 3.9 or higher (3.12 recommended)
- **OS**: Windows 10/11, Linux (Ubuntu/Debian/Raspberry Pi OS), macOS 10.15+
- **RAM**: 2GB minimum (4GB+ recommended)
- **Storage**: 1GB free space
- **Camera**: USB webcam, V4L2 device, or video files

### Python Dependencies
All dependencies are listed in `requirements.txt`:
- **PySide6** (Qt6 GUI framework)
- **OpenCV** (computer vision)
- **ONNX Runtime** (AI model inference)
- **NumPy** (numerical computing)
- **SQLite3** (built-in, database)
- **Google Generative AI** (optional, for Gemini query parsing)

## 🚀 Installation

### 1. Clone or Download
```bash
git clone <repository-url>
cd VMSPython
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add ONNX Models
Place your ONNX model files in the `models/` directory:
- `best.onnx` - Face detection model
- `w600k_mbf.onnx` - Face recognition/embedding model
- `yolo11npRETRAINED.onnx` - General object detection (COCO classes)
- `Fire_Event_best.onnx` - Fire/smoke detection

**Note**: Models are auto-discovered on startup. The application will detect model types from filenames and metadata.

## 🎮 Usage

### Starting the Application

```bash
# Activate virtual environment (if using)
source venv/bin/activate

# Run the application
python vms_gui.py
```

### Live View Mode

1. **Select Video Source**:
   - Camera index (0, 1, 2, etc.)
   - Linux device path (`/dev/video0`, `/dev/video1`)
   - Video file path

2. **Configure Models**:
   - Enable/disable models in the left panel
   - Adjust confidence thresholds
   - Select detection classes to filter

3. **Start Detection**:
   - Click "Start" button
   - Detection results appear in real-time
   - Bounding boxes show detected objects

4. **Face Recognition** (if enabled):
   - Load face recognition model in model config
   - Recognized faces show name labels
   - Unknown faces marked as "Unknown"

### Playback Mode

1. **Switch to Playback Tab**:
   - Click "PlayBack" tab in top bar
   - Load video file using file dialog

2. **Analyze Video**:
   - Video plays with detection overlay
   - Detection results saved to database automatically
   - Results panel shows all detections

3. **Query Detections**:
   - Use chatbot to query detections
   - Examples:
     - "find all persons from 0:00 to 5:00"
     - "find unknown faces and save them"
     - "find cars in the video"
   - Results appear in results panel
   - Click results to jump to timestamp

### Face Registration

Register faces for recognition:

```bash
python register_face.py
```

**Process**:
1. Enter person's name
2. Capture 30 images from front angle
3. Capture 30 images from left angle
4. Capture 30 images from right angle
5. Total: 90 images per person for robust recognition

**Requirements**:
- Face detection model (`best.onnx`)
- Face recognition model (`w600k_mbf.onnx`)
- Camera access

## 🔧 Configuration

### Model Configuration

Models are configured in the GUI:
- **Enable/Disable**: Toggle model on/off
- **Confidence Threshold**: Minimum confidence for detections (0.0-1.0)
- **Class Filtering**: Enable/disable specific detection classes
- **Face Recognition**: Load recognition model for face identification

### Camera Configuration

- **Source Selection**: Choose camera index or device path
- **Resolution**: Select preset or custom resolution
- **Live Resolution Change**: Change resolution while camera is running

### SOS Settings

Configure in Model Config panel:
- **Unknown Face SOS**: Enable and set count threshold
- **Known Face SOS**: Enable and set count threshold
- **Class-based SOS**: Set thresholds for any detection class

### Database

Databases are stored in `storage/db/`:
- `face_embeddings.db` - Face recognition database
- `detection_results.db` - Detection results database
- `app.db` - Application database
- `events.sqlite` - Events database

## 📊 Detection Models

### Supported Model Types

1. **Face Detection** (`best.onnx`)
   - Classes: `["face"]`
   - Detects faces in video frames
   - Used with face recognition

2. **Object Detection** (`yolo11npRETRAINED.onnx`)
   - Classes: COCO classes (80 classes)
   - Person, car, bicycle, etc.
   - General purpose detection

3. **Fire/Smoke Detection** (`Fire_Event_best.onnx`)
   - Classes: `["fire", "smoke"]`
   - Fire and smoke detection

4. **Face Recognition** (`w600k_mbf.onnx`)
   - Not a detection model
   - Extracts face embeddings for recognition
   - Used with face detection model

### Adding Custom Models

1. Place `.onnx` file in `models/` directory
2. Model type detected from filename or metadata
3. Classes auto-detected from model output
4. Enable and configure in GUI

## 🗄️ Database Schema

### Detection Results (`detection_results.db`)

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    video_path TEXT,
    timestamp REAL,
    time_string TEXT,
    class_name TEXT,
    recognized_name TEXT,
    confidence REAL,
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 INTEGER,
    model_name TEXT,
    frame_image BLOB,
    num_objects INTEGER,
    created_at TIMESTAMP
)
```

### Face Embeddings (`face_embeddings.db`)

Stores face embeddings for recognition:
- Face ID, name, embedding vector
- Multiple embeddings per person supported

## 🔍 Query System

### Natural Language Queries

The chatbot uses Gemini API (with fallback parser) to understand queries:

**Supported Actions**:
- `find` - Search for detections
- `save` - Save detections to filesystem

**Query Examples**:
```
"find all humans from 10 min to 15 min and save them"
"find tigers in the video"
"find all unknown faces from 5:00 to 10:00"
"find whatever you see and save to database"
"find persons and save them"
```

**Query Parameters**:
- `class_name`: Object class (person, car, face, etc.)
- `recognized_name`: For faces ("Unknown", "known", or specific name)
- `time_start`: Start time in seconds
- `time_end`: End time in seconds
- `save_to_db`: Whether to save images to filesystem

## 🐛 Troubleshooting

### Camera Not Working

**Linux/Raspberry Pi**:
```bash
# Check available cameras
ls /dev/video*

# Test camera permissions
v4l2-ctl --list-devices
```

**Windows**:
- Check Device Manager for camera
- Ensure no other application is using camera
- Try different camera indices (0, 1, 2)

### Models Not Loading

- Verify `.onnx` files are in `models/` directory
- Check file permissions
- Ensure models are valid ONNX format
- Check console for error messages

### Face Recognition Not Working

- Ensure face detection model is enabled
- Load face recognition model in model config
- Register faces using `register_face.py`
- Check face database has registered faces

### Performance Issues

- Reduce camera resolution
- Lower target FPS
- Disable unused models
- Use smaller model files
- Close other applications

### Database Errors

- Check `storage/db/` directory exists
- Verify write permissions
- Check disk space
- Database files are SQLite, can be opened with SQLite tools

## 🔒 Security & Privacy

- **Local Operation**: All processing happens locally
- **No Internet Required**: Core functionality works offline
- **Data Storage**: All data stored locally in SQLite databases
- **Face Privacy**: Face embeddings stored locally, not shared
- **Optional Gemini API**: Only used for query parsing (can use fallback parser)

## 📝 Development

### Project Structure

- `vms_gui/` - Main application package
- `vms_gui/detection/` - Detection engine and models
- `vms_gui/gui/` - GUI components
- `models/` - ONNX model files
- `storage/` - Data storage

### Key Classes

- `DetectionEngine`: Main detection orchestrator
- `ONNXRunner`: ONNX model inference
- `VideoCapture`: Camera/video file handling
- `FaceRecognizer`: Face recognition with embeddings
- `DetectionDatabase`: Detection results storage
- `VMSClientApp`: Main application window

### Extending the System

1. **Add New Model Type**: Update `detect_model_classes()` in `engine.py`
2. **Add New Detection Class**: Update `COCO_CLASSES` in `config.py`
3. **Custom Query Parser**: Extend `QueryParser` in `gemini_parser.py`
4. **New GUI Component**: Add to `vms_gui/gui/`

## 📄 License

[Specify your license here]

## 🤝 Contributing

[Contributing guidelines]

## 📞 Support

For issues and questions:
1. Check this README
2. Review console output for errors
3. Check database files for data integrity
4. Verify model files are valid

## 🎉 Features Summary

✅ Real-time video detection  
✅ Multi-model support (face, object, fire detection)  
✅ Face recognition with database  
✅ Video playback analysis  
✅ Natural language query system  
✅ Detection database with SQLite  
✅ SOS alert system  
✅ Configurable model settings  
✅ Cross-platform support (Windows, Linux, macOS)  
✅ Raspberry Pi compatible  

---

**Built with**: Python, PySide6, OpenCV, ONNX Runtime, SQLite
