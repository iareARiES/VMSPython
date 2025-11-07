"""
Video player component for playback mode.
"""

import cv2
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSlider, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QImage


class VideoPlayer(QWidget):
    """Video player widget for playback mode."""
    
    # Signals
    frame_processed = Signal(object, list)  # frame, detections
    video_loaded = Signal(str)  # video_path
    
    def __init__(self, detection_engine, parent=None):
        super().__init__(parent)
        self.detection_engine = detection_engine
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.fps = 30
        self.total_frames = 0
        self.current_frame_num = 0
        self.is_playing = False
        self.is_paused = False
        
        # Timer for playback
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup video player UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1a1a1a; color: white; font-size: 14px;")
        self.video_label.setText("No Video Loaded\n\nClick 'Load Video' to select a video file")
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Load video button
        self.load_btn = QPushButton("📁 Load Video")
        self.load_btn.clicked.connect(self.load_video)
        controls_layout.addWidget(self.load_btn)
        
        # Play/Pause button
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        controls_layout.addWidget(self.stop_btn)
        
        # Time slider
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.sliderPressed.connect(self.on_slider_pressed)
        self.time_slider.sliderReleased.connect(self.on_slider_released)
        self.time_slider.valueChanged.connect(self.on_slider_changed)
        controls_layout.addWidget(self.time_slider, 1)
        
        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)
        
        # Speed control
        controls_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QLabel("1x")
        controls_layout.addWidget(self.speed_combo)
        
        layout.addLayout(controls_layout)
    
    def load_video(self):
        """Load video file from device."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.open_video(file_path)
    
    def open_video(self, video_path):
        """Open video file."""
        # Close existing video if any
        if self.cap:
            self.cap.release()
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup slider
        self.time_slider.setMaximum(self.total_frames - 1)
        self.time_slider.setValue(0)
        self.time_slider.setEnabled(True)
        
        # Enable controls
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # Load first frame
        self.current_frame_num = 0
        self.seek_to_frame(0)
        
        # Emit signal
        self.video_loaded.emit(video_path)
        
        print(f"Video loaded: {video_path} ({width}x{height}, {self.total_frames} frames, {self.fps:.2f} fps)")
    
    def seek_to_frame(self, frame_num):
        """Seek to specific frame number."""
        if not self.cap:
            return
        
        self.current_frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # Process frame for detection first, then display with detections
            self.process_current_frame()
            self.update_time_label()
            self.time_slider.setValue(self.current_frame_num)
    
    def process_current_frame(self):
        """Process current frame with detection engine."""
        if self.current_frame is None:
            return
        
        # Run detection on current frame
        detections = []
        try:
            # Get enabled models and run detection
            for model_name, config in self.detection_engine.model_configs.items():
                if not config.get("enabled", False):
                    continue
                
                if model_name in self.detection_engine.models:
                    runner = self.detection_engine.models[model_name]
                    conf_threshold = config.get("conf", 0.35)
                    enabled_classes = config.get("enabled_classes", {})
                    
                    frame_detections = runner.infer(self.current_frame, conf_threshold)
                    
                    # Filter by enabled classes
                    for det in frame_detections:
                        cls_name = det["class"]
                        if enabled_classes and not enabled_classes.get(cls_name, False):
                            continue
                        det["model"] = model_name
                        detections.append(det)
            
            # Apply face recognition if available
            if self.detection_engine.face_recognizer and len(detections) > 0:
                frame_h, frame_w = self.current_frame.shape[:2]
                for det in detections:
                    if det.get("class") == "face":
                        x1, y1, x2, y2 = det["bbox"]
                        x1 = max(0, min(x1, frame_w - 1))
                        y1 = max(0, min(y1, frame_h - 1))
                        x2 = max(x1 + 1, min(x2, frame_w))
                        y2 = max(y1 + 1, min(y2, frame_h))
                        
                        face_roi = self.current_frame[y1:y2, x1:x2]
                        if face_roi.size > 0 and face_roi.shape[0] > 10 and face_roi.shape[1] > 10:
                            try:
                                name, similarity = self.detection_engine.face_recognizer.recognize_face(face_roi)
                                if name:
                                    det["recognized_name"] = name
                                    det["recognition_confidence"] = float(similarity)
                                else:
                                    det["recognized_name"] = "Unknown"
                                    det["recognition_confidence"] = float(similarity) if similarity > 0 else 0.0
                            except:
                                det["recognized_name"] = "Unknown"
                                det["recognition_confidence"] = 0.0
            
            # Display frame with detections
            self.display_frame(self.current_frame, detections)
            
            # Emit signal with frame and detections
            self.frame_processed.emit(self.current_frame.copy(), detections)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Display frame without detections on error
            self.display_frame(self.current_frame, [])
    
    def display_frame(self, frame, detections=None):
        """Display frame with detections."""
        if frame is None:
            return
        
        # Draw detections on frame
        display_frame = frame.copy()
        
        # Draw detections if provided
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                cls_name = det["class"]
                
                # Check if face is recognized
                recognized_name = det.get("recognized_name")
                recognition_conf = det.get("recognition_confidence", 0.0)
                
                # Check if recognition model is active
                recognition_active = recognized_name is not None
                
                # Draw bounding box
                if recognition_active:
                    if recognized_name and recognized_name != "Unknown":
                        color = (0, 255, 0)  # Green for recognized faces
                    else:
                        color = (0, 165, 255)  # Orange for Unknown faces
                elif conf > 0.7:
                    color = (255, 0, 0)  # Red for high confidence
                else:
                    color = (74, 158, 255)  # Blue for normal
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                if recognition_active:
                    if recognized_name and recognized_name != "Unknown":
                        label = f"{recognized_name} ({recognition_conf:.0%})"
                    else:
                        label = f"Unknown ({conf:.0%})"
                else:
                    label = f"{cls_name} {conf:.0%}"
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.rectangle(display_frame, (x1, label_y - label_size[1] - 5), 
                             (x1 + label_size[0], label_y + 5), color, -1)
                cv2.putText(display_frame, label, (x1, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert to QImage
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale and display
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def toggle_play(self):
        """Toggle play/pause."""
        if not self.cap:
            return
        
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """Start playing video."""
        if not self.cap:
            return
        
        self.is_playing = True
        self.is_paused = False
        self.play_btn.setText("⏸ Pause")
        
        # Calculate interval based on fps
        interval = int(1000 / self.fps)
        self.play_timer.start(interval)
    
    def pause(self):
        """Pause video playback."""
        self.is_playing = False
        self.is_paused = True
        self.play_btn.setText("▶ Play")
        self.play_timer.stop()
    
    def stop(self):
        """Stop video playback."""
        self.is_playing = False
        self.is_paused = False
        self.play_btn.setText("▶ Play")
        self.play_timer.stop()
        
        if self.cap:
            self.seek_to_frame(0)
    
    def play_next_frame(self):
        """Play next frame."""
        if not self.cap or self.current_frame_num >= self.total_frames - 1:
            self.pause()
            return
        
        self.current_frame_num += 1
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame
            # Process frame (which will also display it with detections)
            self.process_current_frame()
            self.update_time_label()
            self.time_slider.setValue(self.current_frame_num)
        else:
            self.pause()
    
    def update_time_label(self):
        """Update time label."""
        if not self.cap:
            return
        
        current_time = self.current_frame_num / self.fps
        total_time = self.total_frames / self.fps
        
        current_min = int(current_time // 60)
        current_sec = int(current_time % 60)
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)
        
        self.time_label.setText(f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
    
    def on_slider_pressed(self):
        """Handle slider press - pause if playing."""
        if self.is_playing:
            self.pause()
    
    def on_slider_released(self):
        """Handle slider release - seek to position."""
        frame_num = self.time_slider.value()
        self.seek_to_frame(frame_num)
    
    def on_slider_changed(self, value):
        """Handle slider value change."""
        if not self.is_paused and not self.is_playing:
            self.seek_to_frame(value)
    
    def get_current_time(self):
        """Get current playback time in seconds."""
        if not self.cap:
            return 0
        return self.current_frame_num / self.fps
    
    def get_video_path(self):
        """Get current video path."""
        return self.video_path
    
    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.play_timer.stop()

