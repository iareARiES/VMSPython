"""
Model configuration panel component.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, QScrollArea, QDoubleSpinBox, QComboBox
)
from PySide6.QtCore import Qt


class ModelConfigPanel(QWidget):
    """Left sidebar with model configuration."""
    def __init__(self, detection_engine, parent=None):
        super().__init__(parent)
        self.detection_engine = detection_engine
        self.selected_models = set()  # Track multiple selected models
        self.model_checkboxes = {}  # Store checkboxes for models
        self.setup_ui()
        self.load_models()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel("Model Configuration")
        header.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #34495e; color: white; padding: 5px;")
        layout.addWidget(header)
        
        # Model Selection - Multiple models with checkboxes
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        model_layout.addWidget(QLabel("Select Models (multiple allowed):"))
        
        # Scrollable model list
        model_scroll = QScrollArea()
        model_scroll_widget = QWidget()
        self.model_list_layout = QVBoxLayout(model_scroll_widget)
        model_scroll.setWidget(model_scroll_widget)
        model_scroll.setWidgetResizable(True)
        model_scroll.setMaximumHeight(150)
        model_layout.addWidget(model_scroll)
        
        # Select All / Deselect All buttons
        model_buttons_layout = QHBoxLayout()
        self.select_all_models_btn = QPushButton("Select All")
        self.select_all_models_btn.clicked.connect(self.select_all_models)
        model_buttons_layout.addWidget(self.select_all_models_btn)
        
        self.deselect_all_models_btn = QPushButton("Deselect All")
        self.deselect_all_models_btn.clicked.connect(self.deselect_all_models)
        model_buttons_layout.addWidget(self.deselect_all_models_btn)
        
        model_layout.addLayout(model_buttons_layout)
        
        layout.addWidget(model_group)
        
        # Class Selection
        self.class_group = QGroupBox("Class Selection")
        class_layout = QVBoxLayout(self.class_group)
        
        self.class_label = QLabel("Select classes to detect (0 selected)")
        class_layout.addWidget(self.class_label)
        
        # Select All / Deselect All buttons for classes
        class_buttons_layout = QHBoxLayout()
        self.select_all_classes_btn = QPushButton("Select All Classes")
        self.select_all_classes_btn.clicked.connect(self.select_all_classes)
        class_buttons_layout.addWidget(self.select_all_classes_btn)
        
        self.deselect_all_classes_btn = QPushButton("Deselect All Classes")
        self.deselect_all_classes_btn.clicked.connect(self.deselect_all_classes)
        class_buttons_layout.addWidget(self.deselect_all_classes_btn)
        
        class_layout.addLayout(class_buttons_layout)
        
        # Scrollable class list
        scroll = QScrollArea()
        scroll_widget = QWidget()
        self.class_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        class_layout.addWidget(scroll)
        
        self.class_checkboxes = {}
        
        layout.addWidget(self.class_group)
        
        # Recognition Model Selection
        recognition_group = QGroupBox("Recognition Model")
        recognition_layout = QVBoxLayout(recognition_group)
        
        recognition_layout.addWidget(QLabel("Embedding Extractor Model:"))
        self.recognition_combo = QComboBox()
        self.recognition_combo.addItem("-- None --")
        self.recognition_combo.currentTextChanged.connect(self.on_recognition_model_changed)
        recognition_layout.addWidget(self.recognition_combo)
        
        self.recognition_status_label = QLabel("No recognition model selected")
        self.recognition_status_label.setStyleSheet("color: #666; font-size: 10px;")
        recognition_layout.addWidget(self.recognition_status_label)
        
        layout.addWidget(recognition_group)
        
        # Confidence threshold
        conf_group = QGroupBox("Confidence Threshold")
        conf_layout = QVBoxLayout(conf_group)
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.35)
        self.conf_spin.valueChanged.connect(self.on_confidence_changed)
        conf_layout.addWidget(self.conf_spin)
        
        layout.addWidget(conf_group)
        
        # Enable/Disable
        self.enable_checkbox = QCheckBox("Enable Detection")
        self.enable_checkbox.stateChanged.connect(self.on_enable_changed)
        layout.addWidget(self.enable_checkbox)
        
        layout.addStretch()
    
    def load_models(self):
        """Load available models and create checkboxes."""
        models_dir = Path("models")
        
        # Clear existing model checkboxes
        for checkbox in self.model_checkboxes.values():
            checkbox.setParent(None)
        self.model_checkboxes.clear()
        self.selected_models.clear()
        
        # Clear recognition combo
        self.recognition_combo.clear()
        self.recognition_combo.addItem("-- None --")
        
        if models_dir.exists():
            # Load recognition models separately
            recognition_models = []
            
            for model_file in models_dir.glob("*.onnx"):
                model_name = model_file.stem  # Name without extension
                
                # Check if it's a recognition/embedding model
                if "w600k" in model_name.lower() or "mbf" in model_name.lower():
                    recognition_models.append((model_name, model_file))
                    continue
                
                # Determine model type from filename
                model_type = "coco"
                if "face" in model_name.lower() or model_name.lower() == "best":
                    model_type = "face"
                elif "fire" in model_name.lower():
                    model_type = "fire"
                
                # Load detection model into engine
                try:
                    self.detection_engine.load_model(model_name, str(model_file), model_type)
                    
                    # Create checkbox for this model
                    checkbox = QCheckBox(model_name)
                    checkbox.setChecked(False)  # Start unchecked
                    # Use a lambda with default argument to capture model_name correctly
                    def make_handler(name):
                        return lambda checked: self.on_model_checkbox_changed(name, checked)
                    checkbox.stateChanged.connect(make_handler(model_name))
                    self.model_checkboxes[model_name] = checkbox
                    self.model_list_layout.addWidget(checkbox)
                except Exception as e:
                    print(f"Failed to load model {model_name}: {e}")
            
            # Load recognition models into combo box
            for model_name, model_file in recognition_models:
                try:
                    # Load recognition model (but don't add to detection engine)
                    # Just store the path for now
                    self.recognition_combo.addItem(model_name, str(model_file))
                    print(f"Loaded recognition model: {model_name}")
                except Exception as e:
                    print(f"Failed to load recognition model {model_name}: {e}")
        
        # Update class display
        self.update_class_display()
    
    def on_model_checkbox_changed(self, model_name, checked):
        """Handle model checkbox change."""
        if checked:
            self.selected_models.add(model_name)
        else:
            self.selected_models.discard(model_name)
        
        # Update class display when models are selected/deselected
        self.update_class_display()
    
    def select_all_models(self):
        """Select all models."""
        for model_name, checkbox in self.model_checkboxes.items():
            checkbox.setChecked(True)
            self.selected_models.add(model_name)
        self.update_class_display()
    
    def deselect_all_models(self):
        """Deselect all models."""
        for checkbox in self.model_checkboxes.values():
            checkbox.setChecked(False)
        self.selected_models.clear()
        self.update_class_display()
    
    def update_class_display(self):
        """Update class checkboxes based on selected models."""
        # Clear existing class checkboxes
        for checkbox in self.class_checkboxes.values():
            checkbox.setParent(None)
        self.class_checkboxes.clear()
        
        if not self.selected_models:
            self.class_group.setEnabled(False)
            self.class_label.setText("Select classes to detect (select models first)")
            return
        
        self.class_group.setEnabled(True)
        
        # Get all unique classes from selected models
        all_classes = set()
        model_info = {}
        for model_name in self.selected_models:
            classes = self.detection_engine.get_model_classes(model_name)
            all_classes.update(classes)
            model_info[model_name] = classes
        
        # Sort classes for consistent display
        sorted_classes = sorted(all_classes)
        
        # Update label with model and class count
        models_text = ", ".join(self.selected_models)
        self.class_label.setText(f"Classes from {len(self.selected_models)} model(s): {len(sorted_classes)} available")
        
        # Create checkboxes for each class (all unchecked initially)
        for class_name in sorted_classes:
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(False)  # Start unchecked as requested
            checkbox.stateChanged.connect(self.on_class_changed)
            self.class_checkboxes[class_name] = checkbox
            self.class_layout.addWidget(checkbox)
        
        # Update all selected models' configs
        self.update_all_model_configs()
        self.update_class_count_label()
    
    def update_all_model_configs(self):
        """Update configuration for all selected models."""
        enabled_classes = {}
        for class_name, checkbox in self.class_checkboxes.items():
            enabled_classes[class_name] = checkbox.isChecked()
        
        conf = self.conf_spin.value()
        enabled = self.enable_checkbox.isChecked()
        
        # Update each selected model
        for model_name in self.selected_models:
            # Get classes specific to this model
            model_classes = self.detection_engine.get_model_classes(model_name)
            model_enabled_classes = {cls: enabled_classes.get(cls, False) for cls in model_classes}
            self.detection_engine.set_model_config(model_name, enabled, conf, model_enabled_classes)
    
    def select_all_classes(self):
        """Select all class checkboxes."""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(True)
        self.update_class_count_label()
    
    def deselect_all_classes(self):
        """Deselect all class checkboxes."""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(False)
        self.update_class_count_label()
    
    def update_class_count_label(self):
        """Update the class label with selection count."""
        if not self.selected_models:
            return
        
        selected_count = sum(1 for cb in self.class_checkboxes.values() if cb.isChecked())
        total_count = len(self.class_checkboxes)
        models_text = ", ".join(self.selected_models)
        self.class_label.setText(f"Classes from {len(self.selected_models)} model(s): {selected_count}/{total_count} selected")
    
    def on_class_changed(self):
        """Handle class checkbox change."""
        self.update_all_model_configs()
        self.update_class_count_label()
    
    def on_confidence_changed(self, value):
        """Handle confidence threshold change."""
        self.update_all_model_configs()
    
    def on_enable_changed(self, state):
        """Handle enable/disable change."""
        self.update_all_model_configs()
    
    def on_recognition_model_changed(self, text):
        """Handle recognition model selection."""
        if text == "-- None --":
            self.recognition_status_label.setText("No recognition model selected")
            return
        
        # Get model file path from combo box data
        model_index = self.recognition_combo.currentIndex()
        model_file = self.recognition_combo.itemData(model_index)
        
        if model_file:
            self.recognition_status_label.setText(f"Recognition model: {text}\n(Embedding extractor - no classes)")
            print(f"Recognition model selected: {text} ({model_file})")
            # Note: We're not loading it into detection engine yet
            # This is just for display/selection for now

