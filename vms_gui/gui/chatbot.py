"""
Chatbot component for querying video analysis.
"""

import re
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from .gemini_parser import QueryParser


class ChatBot(QWidget):
    """Chatbot widget for video analysis queries."""
    
    # Signal emitted when query is submitted
    query_submitted = Signal(str, dict)  # query_text, parsed_query
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query_parser = QueryParser()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup chatbot UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header = QLabel("Video Analysis Chatbot")
        header.setStyleSheet("font-weight: bold; font-size: 12px; background-color: #34495e; color: white; padding: 5px;")
        layout.addWidget(header)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMaximumHeight(150)
        self.chat_history.setStyleSheet("background-color: #2c3e50; color: white; border: 1px solid #34495e;")
        font = QFont("Courier", 9)
        self.chat_history.setFont(font)
        layout.addWidget(self.chat_history)
        
        # Add welcome message
        parser_status = "Gemini API" if self.query_parser.use_gemini else "Fallback Parser"
        self.add_message("System", f"Welcome! Using {parser_status} for query parsing.\n"
                                   "Examples:\n"
                                   "- 'find all humans in the uploaded video from 10 min to 15 min and save them in my database'\n"
                                   "- 'find tigers'\n"
                                   "- 'find whatever you see and save them'\n"
                                   "- 'find all unknown faces from 5:00 to 10:00 and save to database'")
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your query...")
        self.query_input.returnPressed.connect(self.submit_query)
        input_layout.addWidget(self.query_input, 1)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.submit_query)
        input_layout.addWidget(self.send_btn)
        
        layout.addLayout(input_layout)
    
    def add_message(self, sender, message):
        """Add message to chat history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_history.append(f"[{timestamp}] {sender}: {message}")
        # Auto-scroll to bottom
        scrollbar = self.chat_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def submit_query(self):
        """Submit query and parse it."""
        query_text = self.query_input.text().strip()
        if not query_text:
            return
        
        # Add user message to chat
        self.add_message("You", query_text)
        self.query_input.clear()
        
        # Parse query using Gemini API with fallback
        parsed_query = self.query_parser.parse_query(query_text)
        
        # Add system response
        if parsed_query.get("valid"):
            source = parsed_query.get("source", "unknown")
            desc = parsed_query.get("description", "Query parsed successfully")
            self.add_message("System", f"[{source}] {desc}")
        else:
            self.add_message("System", "Could not parse query. Please try rephrasing.\n"
                                      "Examples:\n"
                                      "- 'find all humans from 10 min to 15 min and save them'\n"
                                      "- 'find tigers'\n"
                                      "- 'find whatever you see and save them'")
        
        # Emit signal
        self.query_submitted.emit(query_text, parsed_query)
    
    def add_response(self, message):
        """Add system response to chat."""
        self.add_message("System", message)

