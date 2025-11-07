"""
Face recognition module using embedding extraction and database matching.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import platform

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from .face_database import FaceDatabase


class FaceRecognizer:
    """Face recognition using embedding extraction and database matching."""
    
    def __init__(self, model_path: str, db_path: str = "storage/db/face_embeddings.db", threshold: float = 0.6):
        """
        Initialize face recognizer.
        
        Args:
            model_path: Path to ONNX embedding extraction model (e.g., w600k_mbf.onnx)
            db_path: Path to face embeddings database
            threshold: Similarity threshold for recognition (lower = more strict)
        """
        if not HAS_ONNX:
            raise RuntimeError("ONNX Runtime not available")
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.threshold = threshold
        self.database = FaceDatabase(db_path)
        
        # Load ONNX model
        self.load_model()
        
        # Load all embeddings from database
        self.load_database_embeddings()
    
    def load_model(self):
        """Load ONNX embedding extraction model."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=[('CPUExecutionProvider', {})]
        )
        
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        
        # Get input size (typically 112x112 or 128x128 for face recognition models)
        if len(input_shape) >= 4:
            h = input_shape[2] if input_shape[2] else 112
            w = input_shape[3] if input_shape[3] else 112
        else:
            h, w = 112, 112
        
        self.input_size = (w, h)
        print(f"Loaded face recognition model: {self.model_path.name}, input size: {self.input_size}")
    
    def load_database_embeddings(self):
        """Load all embeddings from database into memory for faster matching."""
        self.known_faces = {}  # {name: [embedding1, embedding2, ...]}
        faces = self.database.get_all_faces()
        
        for face_id, name, embedding in faces:
            # Normalize embedding
            embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
            
            # Store multiple embeddings per name
            if name not in self.known_faces:
                self.known_faces[name] = []
            self.known_faces[name].append(embedding_norm)
        
        total_embeddings = sum(len(embeddings) for embeddings in self.known_faces.values())
        print(f"Loaded {len(self.known_faces)} unique faces with {total_embeddings} total embeddings from database")
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from cropped face image.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Normalized embedding vector
        """
        # Resize to model input size
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] or [-1, 1] depending on model
        # Most face recognition models expect [0, 1] range
        face_norm = face_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        face_transposed = np.transpose(face_norm, (2, 0, 1))
        face_batch = np.expand_dims(face_transposed, axis=0)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: face_batch})
        
        # Get embedding (usually first output)
        embedding = outputs[0][0]
        
        # Normalize embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        return embedding_norm
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face by comparing embedding with database.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Tuple (name, similarity_score) or (None, 0.0) if not recognized
        """
        if len(self.known_faces) == 0:
            return None, 0.0
        
        # Extract embedding
        embedding = self.extract_embedding(face_image)
        
        # Compare with all known faces (check all embeddings for each name)
        best_match = None
        best_similarity = 0.0
        
        for name, known_embeddings in self.known_faces.items():
            # Check all embeddings for this name and use the best match
            for known_embedding in known_embeddings:
                # Cosine similarity
                similarity = np.dot(embedding, known_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
        
        # Check if similarity is above threshold
        if best_similarity >= self.threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity
    
    def register_face(self, name: str, face_image: np.ndarray) -> bool:
        """
        Register a new face in the database.
        
        Args:
            name: Person's name
            face_image: Cropped face image (BGR format)
            
        Returns:
            True if registered successfully
        """
        try:
            # Extract embedding
            embedding = self.extract_embedding(face_image)
            
            # Add to database
            self.database.add_face(name, embedding)
            
            # Update in-memory cache
            embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
            if name not in self.known_faces:
                self.known_faces[name] = []
            self.known_faces[name].append(embedding_norm)
            
            print(f"Registered face: {name}")
            return True
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def reload_database(self):
        """Reload embeddings from database (useful after registration)."""
        self.load_database_embeddings()

