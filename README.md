# Intrusion Detection System - Complete Standalone Application

A **completely self-contained** Python application that includes the full intrusion detection system with desktop GUI. This PythonOnly directory contains everything needed to run the system independently.

## 🚀 **ONE-COMMAND STARTUP**

```bash
python run_all.py
```

That's it! This single command starts:
- ✅ Backend API service
- ✅ Detection service with YOLO models
- ✅ Desktop GUI application
- ✅ Database initialization
- ✅ Storage setup

## 📦 **What's Included**

This standalone application contains:

```
PythonOnly/
├── 🎮 main.py              # Desktop GUI application
├── 🚀 run_all.py           # Unified launcher (ONE COMMAND!)
├── 📋 requirements.txt     # All dependencies
├── 📖 README.md            # This file
├── ⚙️  setup_standalone.py  # Setup utility
│
├── 🖥️  gui/                 # Complete GUI application
│   ├── live_view.py        # Live video feed
│   ├── model_manager.py    # Model management
│   ├── zone_editor.py      # Zone drawing/editing
│   ├── event_log.py        # Event history
│   ├── system_status.py    # System monitoring
│   └── settings.py         # Configuration
│
├── 🔗 backend/              # Complete FastAPI backend
│   └── app/
│       ├── main.py         # FastAPI application
│       ├── config.py       # Backend configuration
│       ├── deps.py         # Database dependencies
│       ├── db/             # Database models & repos
│       ├── routers/        # API endpoints
│       ├── services/       # Business logic
│       └── ws/             # WebSocket handlers
│
├── 🤖 detection_service/    # Complete detection service
│   └── detectsvc/
│       ├── main.py         # Detection FastAPI app
│       ├── config.py       # Detection configuration
│       ├── registry.py     # Model registry
│       ├── accel/          # ONNX inference
│       └── pipeline/       # Detection pipeline
│
├── 🗄️  models/              # YOLO model files (.onnx)
├── 💾 storage/             # Data storage
│   ├── db/                 # SQLite database
│   ├── videos/            # Video recordings
│   ├── snaps/             # Snapshots
│   └── clips/             # Event clips
│
└── 🛠️  utils/               # Utilities
    ├── api_client.py       # API communication
    └── config.py           # Configuration management
```

## 🎯 **Quick Start**

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Add YOLO Models**
Place your ONNX model files in the `models/` directory:
- `best.onnx` - Face detection
- `yolo11npRETRAINED.onnx` - General object detection  
- `Fire_Event_best.onnx` - Fire/smoke detection
- `w600k_mbf.onnx` - Face recognition

### 3. **Run Everything**
```bash
python run_all.py
```

### 4. **Use the System**
- **GUI opens automatically** with full functionality
- **Backend API** available at `http://localhost:8000`
- **Detection service** available at `http://localhost:8010`

## 🎮 **Usage Modes**

### **Full System** (Default)
```bash
python run_all.py
```
Starts everything: backend services + GUI

### **Backend Services Only**
```bash
python run_all.py --mode backend
```
Starts only API and detection services (no GUI)

### **GUI Only**
```bash
python run_all.py --mode gui
```
Starts only GUI (requires backend running separately)

### **Dependency Check**
```bash
python run_all.py --check-deps
```
Checks if all required packages are installed

## ✨ **Features**

### 🎥 **Live Video Detection**
- Real-time camera feed with AI detection
- Multiple camera sources (V4L2, USB, IP cameras, files)
- Bounding box visualization
- FPS monitoring

### 🤖 **Model Management**
- Enable/disable YOLO models
- Confidence and IoU threshold adjustment
- Per-class detection filtering
- Real-time model switching

### 📐 **Zone Editor**
- Interactive zone drawing (polygon, rectangle, line)
- Zone-based detection filtering
- Visual zone overlay
- Import/export zone configurations

### 📋 **Event System**
- Real-time event logging
- Advanced filtering and search
- Event export (JSON/CSV)
- Detailed event information

### 💻 **System Monitoring**
- Real-time performance metrics
- Service health monitoring
- Hardware information (CPU, memory, disk)
- Network statistics

### ⚙️ **Configuration**
- Comprehensive settings panel
- Camera configuration
- Detection parameters
- Storage options
- Import/export settings

## 🔧 **Architecture**

This standalone application runs three services internally:

1. **Detection Service** (Port 8010)
   - YOLO model inference
   - Camera capture
   - Detection processing

2. **Backend Service** (Port 8000)  
   - REST API endpoints
   - Database operations
   - WebSocket communication

3. **GUI Application**
   - Desktop interface
   - Real-time visualization
   - Configuration management

All services communicate via HTTP/WebSocket APIs internally.

## 📋 **System Requirements**

### **Minimum Requirements:**
- **Python 3.9+**
- **2GB RAM** (4GB+ recommended)
- **1GB free disk space**
- **Camera** (USB/V4L2) or video files

### **Operating Systems:**
- ✅ **Windows** 10/11
- ✅ **Linux** (Ubuntu, Debian, Raspberry Pi OS)
- ✅ **macOS** 10.15+

### **Hardware:**
- **Any modern computer** (laptop, desktop, Raspberry Pi)
- **Camera support:** USB webcams, V4L2 devices, IP cameras
- **GPU:** Not required (CPU inference with ONNX)

## 🛠️ **Installation**

### **Option 1: From Main Project**
```bash
# From main intrusion-suite directory
cp -r PythonOnly ~/intrusion-detection-standalone
cd ~/intrusion-detection-standalone
pip install -r requirements.txt
python run_all.py
```

### **Option 2: Direct Download**
```bash
# Download just the PythonOnly folder
# (Copy models separately if available)
pip install -r requirements.txt
python run_all.py
```

## 🐛 **Troubleshooting**

### **Dependencies Missing**
```bash
python run_all.py --check-deps
pip install -r requirements.txt
```

### **Camera Not Found**
```bash
# Check available cameras
ls /dev/video*  # Linux
# Or use GUI Settings → Camera → Test Camera
```

### **Models Not Loading**
```bash
# Check models directory
ls models/
# Ensure .onnx files are present
# Use GUI Models tab to enable them
```

### **Services Not Starting**
```bash
# Check if ports are in use
netstat -tulpn | grep 8000
netstat -tulpn | grep 8010

# Kill conflicting processes if needed
pkill -f "uvicorn.*app.main"
```

### **Performance Issues**
- Reduce FPS in Settings → Detection
- Use smaller model files  
- Lower camera resolution
- Close unnecessary applications

## 🔒 **Security**

- **Local operation:** All data stays on your machine
- **No internet required** for core functionality
- **SQLite database:** Local file-based storage
- **Configurable access:** Bind to localhost or network

## 🎨 **Customization**

### **Adding New Models**
1. Place `.onnx` file in `models/` directory
2. Restart application or use Models → Refresh
3. Enable and configure in GUI

### **Custom Zones**
1. Use Zone Editor to draw detection areas
2. Export/import zone configurations
3. Zone-based filtering and alerts

### **Camera Sources**
- **V4L2:** `/dev/video0`, `/dev/video1`
- **USB:** Camera index `0`, `1`, `2`
- **Files:** `/path/to/video.mp4`
- **IP Cameras:** `rtsp://camera.ip/stream`

## 📊 **Performance**

### **Typical Performance:**
- **Detection FPS:** 10-30 FPS (depends on hardware)
- **GUI Responsiveness:** 60 FPS interface
- **Memory Usage:** 200-500MB (depends on models)
- **CPU Usage:** 20-60% (depends on models and FPS)

### **Optimization Tips:**
- Use CPU-optimized ONNX models
- Reduce target FPS for lower CPU usage
- Enable only needed detection classes
- Use appropriate camera resolution

## 🚀 **Advanced Usage**

### **Remote Access**
Configure backend to accept network connections:
```bash
# Edit backend/app/config.py
backend_host = "0.0.0.0"  # Allow network access
```

### **Multiple Instances**
Run different configurations:
```bash
# Change ports in config files
backend_port = 8001
detect_port = 8011
```

### **Custom Configuration**
Create `.env` file for custom settings:
```
BACKEND_PORT=8000
DETECT_PORT=8010
CAMERA_SOURCE=/dev/video0
TARGET_FPS=15
```

## 📞 **Support**

1. **Check this README** for common solutions
2. **Use `--check-deps`** to verify installation
3. **Check GUI Settings** for configuration issues
4. **Monitor System tab** for performance issues

## 🎉 **Success!**

Once running, you have a complete intrusion detection system:
- 🎥 **Live video monitoring**
- 🤖 **AI-powered object detection** 
- 📐 **Custom detection zones**
- 📋 **Event logging and history**
- 💻 **System monitoring**
- ⚙️ **Full configuration control**

**All in one self-contained Python application!**

---

*This standalone application provides the exact same functionality as the full intrusion-suite but in a single, portable directory that can be moved to any system.*