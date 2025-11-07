"""
Face embeddings database for storing and retrieving face recognition data.
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime


class FaceDatabase:
    """Database for storing face embeddings and names."""
    
    def __init__(self, db_path: str = "storage/db/face_embeddings.db"):
        """Initialize face database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create faces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on name for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_name ON faces(name)
        """)
        
        conn.commit()
        conn.close()
    
    def add_face(self, name: str, embedding: np.ndarray) -> int:
        """
        Add a face embedding to the database.
        
        Args:
            name: Person's name
            embedding: Face embedding vector (numpy array)
            
        Returns:
            ID of the inserted face
        """
        # Convert numpy array to bytes
        embedding_bytes = embedding.tobytes()
        embedding_shape = embedding.shape
        embedding_dtype = str(embedding.dtype)
        
        # Store as JSON for shape and dtype info
        embedding_data = {
            "data": embedding_bytes.hex(),
            "shape": embedding_shape,
            "dtype": embedding_dtype
        }
        embedding_json = json.dumps(embedding_data)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO faces (name, embedding)
            VALUES (?, ?)
        """, (name, embedding_json))
        
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return face_id
    
    def get_all_faces(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Get all face embeddings from database.
        
        Returns:
            List of tuples (id, name, embedding)
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, embedding FROM faces")
        rows = cursor.fetchall()
        
        faces = []
        for row in rows:
            face_id, name, embedding_json = row
            embedding_data = json.loads(embedding_json)
            
            # Reconstruct numpy array
            embedding_bytes = bytes.fromhex(embedding_data["data"])
            embedding = np.frombuffer(embedding_bytes, dtype=embedding_data["dtype"])
            embedding = embedding.reshape(embedding_data["shape"])
            
            faces.append((face_id, name, embedding))
        
        conn.close()
        return faces
    
    def get_face_by_name(self, name: str) -> Optional[Tuple[int, np.ndarray]]:
        """
        Get face embedding by name.
        
        Args:
            name: Person's name
            
        Returns:
            Tuple (id, embedding) or None if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, embedding FROM faces WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row is None:
            return None
        
        face_id, embedding_json = row
        embedding_data = json.loads(embedding_json)
        
        # Reconstruct numpy array
        embedding_bytes = bytes.fromhex(embedding_data["data"])
        embedding = np.frombuffer(embedding_bytes, dtype=embedding_data["dtype"])
        embedding = embedding.reshape(embedding_data["shape"])
        
        return (face_id, embedding)
    
    def delete_face(self, face_id: int) -> bool:
        """
        Delete a face from database.
        
        Args:
            face_id: ID of face to delete
            
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def delete_face_by_name(self, name: str) -> bool:
        """
        Delete a face by name.
        
        Args:
            name: Person's name
            
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def get_all_names(self) -> List[str]:
        """Get all registered names."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT name FROM faces ORDER BY name")
        names = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return names
    
    def count_faces(self) -> int:
        """Get total number of registered faces."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM faces")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count

