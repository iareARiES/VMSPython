"""
Main VMS Client application.
"""

import sys
import cv2
import platform
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QDialog, QFileDialog

from .detection import DetectionEngine
from .detection.detection_database import DetectionDatabase
from .gui import TopBar, BottomBar, ModelConfigPanel, VideoDisplay, VideoPlayer, ChatBot, ResultsPanel


class VMSClientApp(QMainWindow):
    """Main VMS Client application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VMS Client")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize detection engine
        self.detection_engine = DetectionEngine()
        
        # Initialize detection database
        self.detection_db = DetectionDatabase()
        
        # SOS tracking
        self.sos_settings = {
            "sos_unknown_face_enabled": False,
            "sos_unknown_face_count": 1,
            "sos_known_face_enabled": False,
            "sos_known_face_count": 1,
            "class_settings": {}  # For other models: {class_name: count_threshold}
        }
        self.unknown_face_count = 0
        self.known_face_count = 0
        self.class_counts = {}  # {class_name: current_count} for class-based SOS
        self.last_known_faces = set()  # Track known faces to avoid duplicate alerts
        
        # Current mode: "liveview" or "playback"
        self.current_mode = "liveview"
        
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
        
        # Connect tab switching
        self.top_bar.liveview_tab.clicked.connect(self.on_liveview_tab_clicked)
        self.top_bar.playback_tab.clicked.connect(self.on_playback_tab_clicked)
        
        # Main content area - will be updated based on mode
        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        
        # Left sidebar - Model configuration (same for both modes)
        self.model_panel = ModelConfigPanel(self.detection_engine, self)
        self.model_panel.setMaximumWidth(300)
        self.model_panel.setMinimumWidth(250)
        self.content_layout.addWidget(self.model_panel)
        
        # Center and Right panels - will be created based on mode
        self.video_display = None
        self.video_player = None
        self.results_panel = None
        self.chatbot = None
        
        # Bottom bar (created before setup_liveview_mode so it can be accessed)
        self.bottom_bar = BottomBar(self)
        main_layout.addWidget(self.bottom_bar)
        
        main_layout.addLayout(self.content_layout)
        
        # Create LiveView mode components (after bottom_bar is created)
        self.setup_liveview_mode()
        
        # Connect bottom bar buttons
        self.bottom_bar.start_btn.clicked.connect(self.start_detection)
        self.bottom_bar.stop_btn.clicked.connect(self.stop_detection)
        self.bottom_bar.stop_all_btn.clicked.connect(self.stop_all)
        self.bottom_bar.refresh_btn.clicked.connect(self.refresh_models)
        self.bottom_bar.video_source_combo.currentTextChanged.connect(self.on_video_source_changed)
        self.bottom_bar.resolution_combo.currentTextChanged.connect(self.bottom_bar.on_resolution_changed)
        self.bottom_bar.resolution_combo.currentTextChanged.connect(self.on_resolution_changed_live)
        self.bottom_bar.custom_width_spin.valueChanged.connect(self.on_custom_resolution_changed)
        
        # Connect SOS settings
        self.model_panel.sos_settings_changed.connect(self.on_sos_settings_changed)
        
        # Connect video display for notifications
        if self.video_display:
            self.video_display.known_face_detected.connect(self.on_known_face_detected)
            self.video_display.parent_app = self  # Set reference for SOS checking
        
        self.bottom_bar.custom_height_spin.valueChanged.connect(self.on_custom_resolution_changed)
    
    def setup_liveview_mode(self):
        """Setup LiveView mode UI."""
        # Remove playback components if they exist
        if self.video_player:
            self.video_player.setParent(None)
            self.video_player = None
        if self.results_panel:
            self.results_panel.setParent(None)
            self.results_panel = None
        if self.chatbot:
            self.chatbot.setParent(None)
            self.chatbot = None
        
        # Create LiveView components
        if not self.video_display:
            self.video_display = VideoDisplay(self.detection_engine, self)
            self.video_display.known_face_detected.connect(self.on_known_face_detected)
            self.video_display.parent_app = self
        
        # Add to layout
        self.content_layout.addWidget(self.video_display, 1)
        
        # Show/hide bottom bar
        self.bottom_bar.setVisible(True)
    
    def setup_playback_mode(self):
        """Setup PlayBack mode UI."""
        # Remove liveview components if they exist
        if self.video_display:
            self.video_display.setParent(None)
            self.video_display = None
        
        # Hide bottom bar
        self.bottom_bar.setVisible(False)
        
        # Create PlayBack components
        if not self.video_player:
            self.video_player = VideoPlayer(self.detection_engine, self)
            self.video_player.frame_processed.connect(self.on_video_frame_processed)
        
        if not self.results_panel:
            self.results_panel = ResultsPanel(self)
        
        if not self.chatbot:
            self.chatbot = ChatBot(self)
            self.chatbot.query_submitted.connect(self.on_chatbot_query)
        
        # Create center container for video player and chatbot
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        
        # Video player (takes most space)
        center_layout.addWidget(self.video_player, 1)
        
        # Chatbot at bottom
        center_layout.addWidget(self.chatbot)
        
        # Add center container and results panel to main layout
        self.content_layout.addWidget(center_container, 1)
        self.content_layout.addWidget(self.results_panel, 0)
        self.results_panel.setMaximumWidth(300)
        self.results_panel.setMinimumWidth(250)
    
    def on_liveview_tab_clicked(self):
        """Handle LiveView tab click."""
        if not self.top_bar.liveview_tab.isChecked():
            # If unchecking, don't allow it - keep it checked
            self.top_bar.liveview_tab.setChecked(True)
            return
        self.top_bar.playback_tab.setChecked(False)
        self.switch_mode("liveview")
    
    def on_playback_tab_clicked(self):
        """Handle PlayBack tab click."""
        if not self.top_bar.playback_tab.isChecked():
            # If unchecking, don't allow it - keep it checked
            self.top_bar.playback_tab.setChecked(True)
            return
        self.top_bar.liveview_tab.setChecked(False)
        self.switch_mode("playback")
    
    def switch_mode(self, mode):
        """Switch between LiveView and PlayBack modes."""
        if mode == self.current_mode:
            return
        
        self.current_mode = mode
        
        # Clear content layout (but keep model panel)
        # Remove all widgets except model panel
        items_to_remove = []
        for i in range(self.content_layout.count()):
            item = self.content_layout.itemAt(i)
            if item and item.widget() and item.widget() != self.model_panel:
                items_to_remove.append(item.widget())
        
        for widget in items_to_remove:
            widget.setParent(None)
        
        if mode == "liveview":
            self.setup_liveview_mode()
        else:  # playback
            self.setup_playback_mode()
    
    def on_video_frame_processed(self, frame, detections):
        """Handle processed video frame."""
        if self.video_player and len(detections) > 0:
            # Add to results panel only if there are detections
            time_seconds = self.video_player.get_current_time()
            video_path = self.video_player.get_video_path()
            
            # Draw detections on frame for saving
            frame_with_detections = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                cls_name = det["class"]
                recognized_name = det.get("recognized_name")
                
                if recognized_name:
                    color = (0, 255, 0) if recognized_name != "Unknown" else (0, 165, 255)
                else:
                    color = (255, 0, 0) if conf > 0.7 else (74, 158, 255)
                
                cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, 2)
                label = f"{recognized_name or cls_name} {conf:.0%}"
                cv2.putText(frame_with_detections, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Get model names used
            model_names = list(set([det.get("model", "unknown") for det in detections]))
            
            # Save to database
            if video_path:
                self.detection_db.save_detection(
                    video_path=video_path,
                    timestamp=time_seconds,
                    detections=detections,
                    frame=frame_with_detections,
                    model_names=model_names
                )
            
            # Add to results panel
            self.results_panel.add_result(frame, time_seconds, detections)
    
    def on_chatbot_query(self, query_text, parsed_query):
        """Handle chatbot query."""
        if not parsed_query.get("valid"):
            self.chatbot.add_response("Could not understand the query. Please try again.")
            return
        
        action = parsed_query.get("action")
        class_name = parsed_query.get("class_name")  # Note: changed from "class" to "class_name"
        time_start = parsed_query.get("time_start")
        time_end = parsed_query.get("time_end")
        recognized_name = parsed_query.get("recognized_name")
        save_to_db = parsed_query.get("save_to_db", False)
        
        # Get current video path
        video_path = None
        if self.video_player:
            video_path = self.video_player.get_video_path()
        
        if not video_path:
            self.chatbot.add_response("No video loaded. Please load a video first.")
            return
        
        # If time_end is None, set to video end
        if time_end is None:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                time_end = total_frames / fps
                cap.release()
        
        if action in ["find", "save"]:
            # Query database for detections
            db_results = self.detection_db.query_detections(
                video_path=video_path,
                class_name=class_name,
                recognized_name=recognized_name,
                time_start=time_start,
                time_end=time_end
            )
            
            if not db_results:
                self.chatbot.add_response(f"No detections found matching your criteria.")
                return
            
            # Convert database results to results panel format
            # Group by timestamp to combine multiple detections in same frame
            results_by_time = {}
            for db_result in db_results:
                timestamp = db_result["timestamp"]
                if timestamp not in results_by_time:
                    results_by_time[timestamp] = {
                        "frame": db_result["frame"],
                        "time": timestamp,
                        "detections": []
                    }
                
                # Add detection to this timestamp's list
                det = {
                    "bbox": db_result["bbox"],
                    "confidence": db_result["confidence"],
                    "class": db_result["class_name"],
                    "recognized_name": db_result["recognized_name"],
                    "model": db_result["model_name"]
                }
                results_by_time[timestamp]["detections"].append(det)
            
            # Convert to list
            results_for_panel = list(results_by_time.values())
            
            # Update results panel
            self.results_panel.detection_results = results_for_panel
            self.results_panel.update_display()
            
            # Save to filesystem if requested
            if save_to_db or action == "save":
                # Save images to filesystem
                saved_count = 0
                save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
                if save_dir:
                    for result in results_for_panel:
                        time_str = f"{int(result['time']//60):02d}_{int(result['time']%60):02d}"
                        filename = f"detection_{time_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        filepath = Path(save_dir) / filename
                        cv2.imwrite(str(filepath), result["frame"])
                        saved_count += 1
                    
                    self.chatbot.add_response(
                        f"Found {len(db_results)} detections. "
                        f"Saved {saved_count} images to: {save_dir}\n"
                        f"All detections are also saved in the database."
                    )
                else:
                    self.chatbot.add_response(
                        f"Found {len(db_results)} detections in database. "
                        f"All detections are saved in the database."
                    )
            else:
                self.chatbot.add_response(f"Found {len(db_results)} detections matching your criteria.")
    
    def test_camera_source(self, source):
        """Test if a camera source is available (RPi 5 compatible)."""
        test_cap = None
        try:
            is_linux = platform.system() == "Linux"
            
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            
            # Use V4L2 on Linux/RPi for better compatibility
            if is_linux:
                if isinstance(source, int):
                    try:
                        test_cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
                        if not test_cap.isOpened():
                            test_cap = cv2.VideoCapture(source)
                    except:
                        test_cap = cv2.VideoCapture(source)
                elif isinstance(source, str) and source.startswith("/dev/video"):
                    try:
                        test_cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
                        if not test_cap.isOpened():
                            test_cap = cv2.VideoCapture(source)
                    except:
                        test_cap = cv2.VideoCapture(source)
                else:
                    test_cap = cv2.VideoCapture(str(source))
            else:
                if isinstance(source, int):
                    test_cap = cv2.VideoCapture(source)
                else:
                    test_cap = cv2.VideoCapture(str(source))
            
            if test_cap is None or not test_cap.isOpened():
                return False
            
            # Try to read a frame (with retry for RPi 5)
            ret = False
            frame = None
            for _ in range(3):
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    break
                import time
                time.sleep(0.1)
            
            test_cap.release()
            return ret and frame is not None
        except Exception as e:
            if test_cap:
                try:
                    test_cap.release()
                except:
                    pass
            return False
    
    def start_detection(self):
        """Start detection with selected video source."""
        # Get selected video source
        source_text = self.bottom_bar.video_source_combo.currentText().strip()
        
        # Parse source
        try:
            if source_text.isdigit():
                source = int(source_text)
            elif source_text.startswith("/dev/video"):
                source = source_text
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
    
    def on_sos_settings_changed(self, settings):
        """Handle SOS settings change."""
        self.sos_settings = settings
        self.unknown_face_count = 0
        self.known_face_count = 0
        # Reset class counts
        if "class_settings" in settings:
            self.class_counts = {class_name: 0 for class_name in settings["class_settings"].keys()}
        else:
            self.class_counts = {}
        self.last_known_faces.clear()
        print(f"SOS settings updated: {settings}")
    
    def on_known_face_detected(self, name):
        """Handle known face detection notification."""
        if name not in self.last_known_faces:
            self.last_known_faces.add(name)
            # Show notification
            QMessageBox.information(
                self, "Known Face Detected",
                f"Known face detected: {name}"
            )
            print(f"Known face detected: {name}")
    
    def check_sos_triggers(self, detections):
        """Check if SOS should be triggered based on detections."""
        unknown_faces = 0
        known_faces = set()
        class_detections = {}  # Count detections by class
        
        for det in detections:
            cls_name = det.get("class")
            recognized_name = det.get("recognized_name")
            
            # Handle face recognition cases
            if cls_name == "face" and recognized_name:
                if recognized_name == "Unknown":
                    unknown_faces += 1
                    # Also count as "unknown face" class if it exists
                    if "unknown face" in class_detections:
                        class_detections["unknown face"] += 1
                    else:
                        class_detections["unknown face"] = 1
                elif recognized_name and recognized_name != "Unknown":
                    known_faces.add(recognized_name)
            else:
                # Regular class detection
                class_detections[cls_name] = class_detections.get(cls_name, 0) + 1
        
        # Check unknown face trigger (for best model + recognition)
        if self.sos_settings.get("sos_unknown_face_enabled", False) and unknown_faces > 0:
            self.unknown_face_count += unknown_faces
            if self.unknown_face_count >= self.sos_settings.get("sos_unknown_face_count", 1):
                if not self.top_bar.sos_active:
                    reply = QMessageBox.warning(
                        self, "SOS Trigger - Unknown Face",
                        f"Unknown face detected {self.unknown_face_count} time(s).\n"
                        f"Trigger SOS alert?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.top_bar.toggle_sos()
                self.unknown_face_count = 0  # Reset after trigger
        
        # Check known face trigger
        if self.sos_settings.get("sos_known_face_enabled", False) and len(known_faces) > 0:
            self.known_face_count += len(known_faces)
            if self.known_face_count >= self.sos_settings.get("sos_known_face_count", 1):
                if not self.top_bar.sos_active:
                    reply = QMessageBox.warning(
                        self, "SOS Trigger - Known Face",
                        f"Known face(s) detected {self.known_face_count} time(s).\n"
                        f"Trigger SOS alert?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.top_bar.toggle_sos()
                self.known_face_count = 0  # Reset after trigger
        
        # Check class-based SOS triggers (for other models)
        class_settings = self.sos_settings.get("class_settings", {})
        for class_name, threshold in class_settings.items():
            if class_name in class_detections:
                count = class_detections[class_name]
                
                # Update running count for this class
                if class_name not in self.class_counts:
                    self.class_counts[class_name] = 0
                self.class_counts[class_name] += count
                
                # Check if threshold reached
                if self.class_counts[class_name] >= threshold:
                    if not self.top_bar.sos_active:
                        reply = QMessageBox.warning(
                            self, f"SOS Trigger - {class_name.title()}",
                            f"{class_name.title()} detected {self.class_counts[class_name]} time(s) "
                            f"(threshold: {threshold}).\n"
                            f"Trigger SOS alert?",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        if reply == QMessageBox.Yes:
                            self.top_bar.toggle_sos()
                    
                    # Reset count after trigger
                    self.class_counts[class_name] = 0
    
    def closeEvent(self, event):
        """Handle application close."""
        self.detection_engine.stop_capture()
        if self.video_player:
            self.video_player.cleanup()
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

