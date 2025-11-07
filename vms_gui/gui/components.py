"""
GUI components: TopBar, BottomBar, SOSDialog.
"""

import platform
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt, QTimer


class SOSDialog(QDialog):
    """SOS confirmation dialog."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm SOS Alert")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Are you sure you want to trigger an SOS alert?"))
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class TopBar(QWidget):
    """Top bar with info, tabs, and SOS button."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sos_active = False
        self.setup_ui()
        
        # Timers
        self.datetime_timer = QTimer()
        self.datetime_timer.timeout.connect(self.update_datetime)
        self.datetime_timer.start(1000)  # Update every second
        
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self.update_health)
        self.health_timer.start(5000)  # Update every 5 seconds
        
        self.update_datetime()
        self.update_health()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Left section - Info
        left_layout = QHBoxLayout()
        
        title = QLabel("(Version 1.0.0) VMS Client")
        title.setStyleSheet("color: #4a9eff; font-weight: bold;")
        left_layout.addWidget(title)
        
        left_layout.addWidget(QLabel("|"))
        
        self.server_label = QLabel("Server: localhost")
        left_layout.addWidget(self.server_label)
        
        self.user_label = QLabel("User: administrator")
        left_layout.addWidget(self.user_label)
        
        self.datetime_label = QLabel()
        left_layout.addWidget(self.datetime_label)
        
        self.cpu_label = QLabel("CPU: 0%")
        left_layout.addWidget(self.cpu_label)
        
        self.ram_label = QLabel("RAM: 0 MB")
        left_layout.addWidget(self.ram_label)
        
        layout.addLayout(left_layout)
        
        # Center section - Tabs
        center_layout = QHBoxLayout()
        self.liveview_tab = QPushButton("LiveView")
        self.liveview_tab.setCheckable(True)
        self.liveview_tab.setChecked(True)
        self.liveview_tab.setStyleSheet("""
            QPushButton:checked {
                background-color: #4a9eff;
                color: white;
            }
        """)
        center_layout.addWidget(self.liveview_tab)
        
        self.playback_tab = QPushButton("PlayBack")
        self.playback_tab.setCheckable(True)
        self.playback_tab.setStyleSheet("""
            QPushButton:checked {
                background-color: #4a9eff;
                color: white;
            }
        """)
        center_layout.addWidget(self.playback_tab)
        
        layout.addLayout(center_layout)
        
        # Right section
        right_layout = QHBoxLayout()
        
        title_label = QLabel("Intrusion/Loitering Detection")
        right_layout.addWidget(title_label)
        
        self.sos_button = QPushButton("SOS")
        self.sos_button.setStyleSheet("background-color: #ff0000; color: white; font-weight: bold; padding: 5px 15px;")
        self.sos_button.clicked.connect(self.toggle_sos)
        right_layout.addWidget(self.sos_button)
        
        layout.addLayout(right_layout)
    
    def update_datetime(self):
        """Update date/time display."""
        now = datetime.now()
        self.datetime_label.setText(now.strftime("%d/%m/%Y %H:%M:%S"))
    
    def update_health(self):
        """Update CPU and RAM display."""
        if not HAS_PSUTIL:
            self.cpu_label.setText("CPU: N/A")
            self.ram_label.setText("RAM: N/A")
            return
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            ram_mb = ram.used / (1024 * 1024)
            
            self.cpu_label.setText(f"CPU: {cpu_percent:.0f}%")
            self.ram_label.setText(f"RAM: {ram_mb:.0f} MB")
        except:
            self.cpu_label.setText("CPU: N/A")
            self.ram_label.setText("RAM: N/A")
    
    def toggle_sos(self):
        """Toggle SOS alert."""
        if not self.sos_active:
            dialog = SOSDialog(self)
            if dialog.exec() == QDialog.Accepted:
                self.sos_active = True
                self.sos_button.setText("CANCEL SOS")
                self.sos_button.setStyleSheet("background-color: #ff0000; color: white; font-weight: bold; padding: 5px 15px;")
        else:
            self.sos_active = False
            self.sos_button.setText("SOS")
            self.sos_button.setStyleSheet("background-color: #ff0000; color: white; font-weight: bold; padding: 5px 15px;")


class BottomBar(QWidget):
    """Bottom bar with controls."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Progress bar (placeholder)
        self.progress_label = QLabel("00:00 / 00:00")
        layout.addWidget(self.progress_label)
        
        # Video Source Selector
        layout.addWidget(QLabel("Video Source:"))
        self.video_source_combo = QComboBox()
        # Add common camera sources
        self.video_source_combo.addItems(["0", "1", "2", "/dev/video0", "/dev/video1", "/dev/video2"])
        self.video_source_combo.setEditable(True)  # Allow custom input
        self.video_source_combo.setCurrentText("0")
        self.video_source_combo.setMinimumWidth(120)
        layout.addWidget(self.video_source_combo)
        
        # Resolution Selector
        layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        # Common resolutions
        self.resolution_combo.addItems([
            "640x480", "800x600", "1024x768", "1280x720", "1280x960", 
            "1600x1200", "1920x1080", "640x640", "Custom"
        ])
        self.resolution_combo.setCurrentText("640x480")
        self.resolution_combo.setMinimumWidth(120)
        layout.addWidget(self.resolution_combo)
        
        # Custom resolution inputs (hidden by default)
        self.custom_width_spin = QSpinBox()
        self.custom_width_spin.setRange(320, 3840)
        self.custom_width_spin.setValue(640)
        self.custom_width_spin.setMinimumWidth(60)
        self.custom_width_spin.setVisible(False)
        layout.addWidget(self.custom_width_spin)
        
        layout.addWidget(QLabel("x"))
        
        self.custom_height_spin = QSpinBox()
        self.custom_height_spin.setRange(240, 2160)
        self.custom_height_spin.setValue(480)
        self.custom_height_spin.setMinimumWidth(60)
        self.custom_height_spin.setVisible(False)
        layout.addWidget(self.custom_height_spin)
        
        # Start button
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 5px 15px;")
        layout.addWidget(self.start_btn)
        
        # Control buttons
        self.volume_btn = QPushButton("🔊")
        layout.addWidget(self.volume_btn)
        
        self.video_source_btn = QPushButton("Video Source")
        layout.addWidget(self.video_source_btn)
        
        self.stop_all_btn = QPushButton("Stop All")
        layout.addWidget(self.stop_all_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)  # Disabled until camera starts
        layout.addWidget(self.stop_btn)
        
        self.bookmark_btn = QPushButton("Bookmark")
        layout.addWidget(self.bookmark_btn)
        
        self.refresh_btn = QPushButton("Refresh")
        layout.addWidget(self.refresh_btn)
        
        self.settings_btn = QPushButton("Audio Video Settings")
        layout.addWidget(self.settings_btn)
        
        self.mic_btn = QPushButton("🎤")
        layout.addWidget(self.mic_btn)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.setStyleSheet("background-color: #4a9eff; color: white; font-weight: bold;")
        layout.addWidget(self.next_btn)
        
        layout.addStretch()
    
    def on_resolution_changed(self, text):
        """Handle resolution change in bottom bar."""
        if text == "Custom":
            self.custom_width_spin.setVisible(True)
            self.custom_height_spin.setVisible(True)
        else:
            self.custom_width_spin.setVisible(False)
            self.custom_height_spin.setVisible(False)
    
    def get_resolution(self):
        """Get current resolution setting."""
        text = self.resolution_combo.currentText()
        if text == "Custom":
            return self.custom_width_spin.value(), self.custom_height_spin.value()
        elif "x" in text:
            parts = text.split("x")
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except:
                    pass
        return None, None  # Use camera default

