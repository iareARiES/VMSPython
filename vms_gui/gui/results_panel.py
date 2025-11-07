"""
Results panel for displaying video analysis results.
"""

import cv2
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage


class ResultsPanel(QWidget):
    """Right panel for displaying analysis results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detection_results = []  # List of {frame, time, detections, image}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup results panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header = QLabel("Analysis Results")
        header.setStyleSheet("font-weight: bold; font-size: 12px; background-color: #34495e; color: white; padding: 5px;")
        layout.addWidget(header)
        
        # Results count
        self.results_count_label = QLabel("0 results")
        self.results_count_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.results_count_label)
        
        # Scroll area for results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #2c3e50; border: 1px solid #34495e;")
        
        # Results container
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setSpacing(10)
        scroll.setWidget(self.results_widget)
        
        layout.addWidget(scroll, 1)
        
        # Clear button
        clear_btn = QPushButton("Clear Results")
        clear_btn.clicked.connect(self.clear_results)
        layout.addWidget(clear_btn)
    
    def add_result(self, frame, time_seconds, detections):
        """Add a detection result."""
        result = {
            "frame": frame.copy(),
            "time": time_seconds,
            "detections": detections.copy(),
            "image": None  # Will be created when displayed
        }
        self.detection_results.append(result)
        self.update_display()
    
    def update_display(self):
        """Update results display."""
        # Clear existing widgets
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Update count
        self.results_count_label.setText(f"{len(self.detection_results)} results")
        
        # Display results in grid (2 columns)
        for idx, result in enumerate(self.detection_results):
            row = idx // 2
            col = idx % 2
            
            # Create result item widget
            item_widget = self.create_result_item(result)
            self.results_layout.addWidget(item_widget, row, col)
    
    def create_result_item(self, result):
        """Create a widget for a single result item."""
        widget = QWidget()
        widget.setStyleSheet("background-color: #34495e; border: 1px solid #2c3e50; padding: 5px;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Time label
        time_min = int(result["time"] // 60)
        time_sec = int(result["time"] % 60)
        time_label = QLabel(f"Time: {time_min:02d}:{time_sec:02d}")
        time_label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(time_label)
        
        # Thumbnail
        thumbnail_label = QLabel()
        thumbnail_label.setAlignment(Qt.AlignCenter)
        thumbnail_label.setMinimumSize(150, 100)
        thumbnail_label.setMaximumSize(150, 100)
        thumbnail_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #555;")
        
        # Create thumbnail from frame
        frame = result["frame"]
        if frame is not None:
            # Draw detections on thumbnail
            display_frame = frame.copy()
            for det in result["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                color = (0, 255, 0) if det.get("recognized_name") else (255, 0, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Resize for thumbnail
            h, w = display_frame.shape[:2]
            scale = min(150 / w, 100 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            thumbnail = cv2.resize(display_frame, (new_w, new_h))
            
            # Convert to QPixmap
            rgb_image = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            thumbnail_label.setPixmap(pixmap)
        
        layout.addWidget(thumbnail_label)
        
        # Detections info
        detections_text = f"Detections: {len(result['detections'])}"
        if result["detections"]:
            classes = [d.get("class", "unknown") for d in result["detections"]]
            detections_text += f"\nClasses: {', '.join(set(classes))}"
        
        detections_label = QLabel(detections_text)
        detections_label.setStyleSheet("color: #bbb; font-size: 9px;")
        detections_label.setWordWrap(True)
        layout.addWidget(detections_label)
        
        # Save button
        save_btn = QPushButton("Save Image")
        save_btn.setMaximumHeight(25)
        save_btn.clicked.connect(lambda: self.save_result_image(result))
        layout.addWidget(save_btn)
        
        return widget
    
    def save_result_image(self, result):
        """Save a single result image."""
        if result["frame"] is None:
            return
        
        # Get save directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
        
        # Create filename with timestamp
        time_str = f"{int(result['time']//60):02d}_{int(result['time']%60):02d}"
        filename = f"detection_{time_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = Path(save_dir) / filename
        
        # Draw detections on frame
        frame = result["frame"].copy()
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class"]
            recognized_name = det.get("recognized_name")
            
            if recognized_name:
                color = (0, 255, 0) if recognized_name != "Unknown" else (0, 165, 255)
            else:
                color = (255, 0, 0) if conf > 0.7 else (74, 158, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{recognized_name or cls_name} {conf:.0%}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save image
        cv2.imwrite(str(filepath), frame)
        QMessageBox.information(self, "Saved", f"Image saved to:\n{filepath}")
    
    def filter_results(self, class_name=None, time_start=None, time_end=None, recognized_name=None):
        """Filter results based on criteria."""
        filtered = []
        
        for result in self.detection_results:
            # Check time range
            if time_start is not None and result["time"] < time_start:
                continue
            if time_end is not None and result["time"] > time_end:
                continue
            
            # Check if any detection matches criteria
            matches = False
            for det in result["detections"]:
                if class_name and det.get("class") != class_name:
                    continue
                if recognized_name:
                    det_name = det.get("recognized_name")
                    if recognized_name == "known" and (not det_name or det_name == "Unknown"):
                        continue
                    elif recognized_name == "Unknown" and det_name != "Unknown":
                        continue
                    elif recognized_name != "known" and recognized_name != "Unknown" and det_name != recognized_name:
                        continue
                matches = True
                break
            
            if matches:
                filtered.append(result)
        
        return filtered
    
    def save_filtered_images(self, class_name=None, time_start=None, time_end=None, recognized_name=None):
        """Save filtered result images."""
        filtered = self.filter_results(class_name, time_start, time_end, recognized_name)
        
        if not filtered:
            QMessageBox.information(self, "No Results", "No images match the criteria.")
            return
        
        # Get save directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
        
        saved_count = 0
        for result in filtered:
            time_str = f"{int(result['time']//60):02d}_{int(result['time']%60):02d}"
            filename = f"detection_{time_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = Path(save_dir) / filename
            
            # Draw detections on frame
            frame = result["frame"].copy()
            for det in result["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                cls_name = det["class"]
                det_name = det.get("recognized_name")
                
                if det_name:
                    color = (0, 255, 0) if det_name != "Unknown" else (0, 165, 255)
                else:
                    color = (255, 0, 0) if conf > 0.7 else (74, 158, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{det_name or cls_name} {conf:.0%}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imwrite(str(filepath), frame)
            saved_count += 1
        
        QMessageBox.information(self, "Saved", f"Saved {saved_count} images to:\n{save_dir}")
    
    def clear_results(self):
        """Clear all results."""
        self.detection_results.clear()
        self.update_display()

