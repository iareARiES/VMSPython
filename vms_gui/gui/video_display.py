"""
Video display component with detection overlays.
"""

import cv2
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage


class VideoDisplay(QWidget):
    """Center video display with detection overlays."""
    def __init__(self, detection_engine, parent=None):
        super().__init__(parent)
        self.detection_engine = detection_engine
        self.setup_ui()
        
        # Video update timer
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video)
        # Don't start timer immediately - wait for camera to start
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1a1a1a; color: white; font-size: 14px;")
        self.video_label.setText("No Video Feed\n\nSelect a video source and click 'Start'")
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)
    
    def update_video(self):
        """Update video frame."""
        if not self.detection_engine.running:
            return
        
        try:
            frame, detections = self.detection_engine.process_frame()
            if frame is None:
                return
        except Exception as e:
            # Silently handle errors - camera might be disconnected
            return
        
        # Draw detections
        display_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class"]
            
            # Draw bounding box (blue for normal, red for high confidence)
            color = (255, 0, 0) if conf > 0.7 else (74, 158, 255)  # Red or blue
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
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
    
    def start(self):
        """Start video display."""
        if not self.video_timer.isActive():
            self.video_timer.start(33)  # ~30 FPS
    
    def stop(self):
        """Stop video display."""
        self.video_timer.stop()

