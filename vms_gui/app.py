"""
Main VMS Client application.
"""

import sys
import cv2
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QDialog

from .detection import DetectionEngine
from .gui import TopBar, BottomBar, ModelConfigPanel, VideoDisplay


class VMSClientApp(QMainWindow):
    """Main VMS Client application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VMS Client")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize detection engine
        self.detection_engine = DetectionEngine()
        
        # Setup GUI
        self.setup_ui()
        
        # Don't auto-start camera - let user select source first
    
    def setup_ui(self):
        """Setup the main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top bar
        self.top_bar = TopBar(self)
        main_layout.addWidget(self.top_bar)
        
        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Left sidebar - Model configuration
        self.model_panel = ModelConfigPanel(self.detection_engine, self)
        self.model_panel.setMaximumWidth(300)
        self.model_panel.setMinimumWidth(250)
        content_layout.addWidget(self.model_panel)
        
        # Center - Video display
        self.video_display = VideoDisplay(self.detection_engine, self)
        content_layout.addWidget(self.video_display, 1)
        
        main_layout.addLayout(content_layout)
        
        # Bottom bar
        self.bottom_bar = BottomBar(self)
        main_layout.addWidget(self.bottom_bar)
        
        # Connect bottom bar buttons
        self.bottom_bar.start_btn.clicked.connect(self.start_detection)
        self.bottom_bar.stop_btn.clicked.connect(self.stop_detection)
        self.bottom_bar.stop_all_btn.clicked.connect(self.stop_all)
        self.bottom_bar.refresh_btn.clicked.connect(self.refresh_models)
        # Also refresh video sources when refresh button is clicked (double-click or separate action)
        # For now, we'll add a separate method that can be called
        self.bottom_bar.video_source_combo.currentTextChanged.connect(self.on_video_source_changed)
        self.bottom_bar.resolution_combo.currentTextChanged.connect(self.bottom_bar.on_resolution_changed)
        self.bottom_bar.resolution_combo.currentTextChanged.connect(self.on_resolution_changed_live)
        self.bottom_bar.custom_width_spin.valueChanged.connect(self.on_custom_resolution_changed)
        self.bottom_bar.custom_height_spin.valueChanged.connect(self.on_custom_resolution_changed)
    
    def test_camera_source(self, source):
        """Test if a camera source is available."""
        test_cap = None
        try:
            import platform
            is_linux = platform.system() == "Linux"
            
            source_str = str(source)
            if isinstance(source, str) and source_str.isdigit():
                source = int(source)
            
            # On Linux, try V4L2 backend for /dev/video* devices
            if isinstance(source, str) and source_str.startswith("/dev/video") and is_linux:
                if hasattr(cv2, 'CAP_V4L2'):
                    test_cap = cv2.VideoCapture(source_str, cv2.CAP_V4L2)
                else:
                    test_cap = cv2.VideoCapture(source_str)
            elif isinstance(source, int):
                test_cap = cv2.VideoCapture(source)
            else:
                test_cap = cv2.VideoCapture(str(source))
            
            if test_cap is None or not test_cap.isOpened():
                return False
            
            # Try to read a frame
            ret, frame = test_cap.read()
            test_cap.release()
            return ret and frame is not None
        except:
            if test_cap:
                test_cap.release()
            return False
    
    def start_detection(self):
        """Start detection with selected video source."""
        # Get selected video source
        source_text = self.bottom_bar.video_source_combo.currentText().strip()
        
        # Check if combo box has user data (V4L2 device path)
        current_data = self.bottom_bar.video_source_combo.currentData()
        if current_data:
            source = current_data
        else:
            # Parse source from text
            try:
                if source_text.isdigit():
                    source = int(source_text)
                elif source_text.startswith("/dev/video"):
                    source = source_text
                elif "(" in source_text and "/dev/video" in source_text:
                    # Extract device path from display name like "/dev/video0 (Camera Name)"
                    source = source_text.split("(")[0].strip()
                else:
                    # Try as integer first
                    try:
                        source = int(source_text)
                    except:
                        source = source_text
            except:
                source = 0
        
        # Test camera before attempting to open
        print(f"Testing camera source: {source}")
        if not self.test_camera_source(source):
            QMessageBox.warning(
                self, "Camera Not Available",
                f"Camera source '{source}' is not available or cannot be accessed.\n\n"
                f"Common issues:\n"
                f"- Camera is being used by another application\n"
                f"- Camera index is incorrect (try 0, 1, 2, etc.)\n"
                f"- Camera driver not installed\n"
                f"- On Windows: Check Device Manager for camera\n"
                f"- On Linux: Check /dev/video* devices with: ls /dev/video*"
            )
            return
        
        # Stop existing capture if running
        if self.detection_engine.running:
            self.detection_engine.stop_capture()
            self.video_display.stop()
        
        # Get resolution
        width, height = self.bottom_bar.get_resolution()
        
        # Start new capture
        try:
            print(f"Attempting to open camera source: {source}")
            if width and height:
                print(f"Setting resolution: {width}x{height}")
            self.detection_engine.start_capture(source, width, height)
            self.video_display.start()
            self.bottom_bar.start_btn.setEnabled(False)
            self.bottom_bar.stop_btn.setEnabled(True)
            res_text = f" ({width}x{height})" if width and height else ""
            self.video_display.video_label.setText(f"Camera: {source}{res_text}")
            print(f"Camera {source} opened successfully")
        except Exception as e:
            error_msg = f"Failed to open camera source '{source}':\n{str(e)}\n\nTry a different source:\n- 0, 1, 2 (for camera indices)\n- /dev/video0, /dev/video1 (for Linux devices)\n- Or enter a custom path"
            print(f"Camera error: {e}")
            QMessageBox.critical(self, "Camera Error", error_msg)
            self.video_display.video_label.setText(f"Camera Error: {source}\nClick Start to try again")
            self.bottom_bar.start_btn.setEnabled(True)
            self.bottom_bar.stop_btn.setEnabled(False)
    
    def stop_detection(self):
        """Stop detection."""
        self.detection_engine.stop_capture()
        self.video_display.stop()
        self.video_display.video_label.setText("Detection Stopped")
        self.bottom_bar.start_btn.setEnabled(True)
        self.bottom_bar.stop_btn.setEnabled(False)
    
    def stop_all(self):
        """Stop all operations."""
        self.stop_detection()
    
    def refresh_models(self):
        """Refresh model list."""
        self.model_panel.load_models()
    
    def on_video_source_changed(self, text):
        """Handle video source change."""
        # If camera is running, suggest restarting
        if self.detection_engine.running:
            reply = QMessageBox.question(
                self, "Change Video Source",
                "Camera is currently running. Do you want to stop and switch to the new source?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.stop_detection()
                # Auto-start with new source
                self.start_detection()
    
    def on_resolution_changed_live(self, text):
        """Handle resolution change - apply live if camera is running."""
        if self.detection_engine.running and text != "Custom":  # Don't trigger on "Custom" selection, only on actual change
            width, height = self.bottom_bar.get_resolution()
            if width and height:
                print(f"Changing resolution live to: {width}x{height}")
                if self.detection_engine.set_resolution(width, height):
                    # Update video label to show new resolution
                    actual_width = int(self.detection_engine.capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.detection_engine.capture.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    source_text = self.bottom_bar.video_source_combo.currentText()
                    self.video_display.video_label.setText(f"Camera: {source_text} ({actual_width}x{actual_height})")
                    print(f"Resolution changed successfully to {actual_width}x{actual_height}")
                else:
                    QMessageBox.warning(self, "Resolution Change Failed", 
                                      f"Could not change resolution to {width}x{height}.\nCamera may not support this resolution.")
    
    def on_custom_resolution_changed(self):
        """Handle custom resolution change - apply live if camera is running."""
        if self.detection_engine.running and self.bottom_bar.resolution_combo.currentText() == "Custom":
            width = self.bottom_bar.custom_width_spin.value()
            height = self.bottom_bar.custom_height_spin.value()
            print(f"Changing resolution live to custom: {width}x{height}")
            if self.detection_engine.set_resolution(width, height):
                # Update video label to show new resolution
                actual_width = int(self.detection_engine.capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.detection_engine.capture.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                source_text = self.bottom_bar.video_source_combo.currentText()
                self.video_display.video_label.setText(f"Camera: {source_text} ({actual_width}x{actual_height})")
                print(f"Resolution changed successfully to {actual_width}x{actual_height}")
            else:
                QMessageBox.warning(self, "Resolution Change Failed", 
                                  f"Could not change resolution to {width}x{height}.\nCamera may not support this resolution.")
    
    def closeEvent(self, event):
        """Handle application close."""
        self.detection_engine.stop_capture()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("VMS Client")
    
    window = VMSClientApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

