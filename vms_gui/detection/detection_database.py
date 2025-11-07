"""
Database for storing video detection results.
"""

import sqlite3
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class DetectionDatabase:
    """Database for storing video detection results."""
    
    def __init__(self, db_path: str = "storage/db/detection_results.db"):
        """Initialize detection database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT,
                timestamp REAL NOT NULL,
                time_string TEXT NOT NULL,
                class_name TEXT NOT NULL,
                recognized_name TEXT,
                confidence REAL NOT NULL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                model_name TEXT NOT NULL,
                frame_image BLOB,
                num_objects INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_time ON detections(video_path, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_class ON detections(class_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recognized_name ON detections(recognized_name)
        """)
        
        conn.commit()
        conn.close()
    
    def save_detection(self, video_path: str, timestamp: float, detections: List[Dict], 
                      frame: np.ndarray, model_names: List[str]) -> List[int]:
        """
        Save detection results to database.
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            detections: List of detection dictionaries
            frame: Frame image with detections drawn
            model_names: List of model names used
            
        Returns:
            List of inserted detection IDs
        """
        if not detections or frame is None:
            return []
        
        # Convert frame to bytes
        _, frame_encoded = cv2.imencode('.jpg', frame)
        frame_bytes = frame_encoded.tobytes()
        
        # Format time string
        time_min = int(timestamp // 60)
        time_sec = int(timestamp % 60)
        time_string = f"{time_min:02d}:{time_sec:02d}"
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        inserted_ids = []
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            
            cursor.execute("""
                INSERT INTO detections 
                (video_path, timestamp, time_string, class_name, recognized_name, 
                 confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, model_name, 
                 frame_image, num_objects)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_path,
                timestamp,
                time_string,
                det.get("class", "unknown"),
                det.get("recognized_name"),
                det.get("confidence", 0.0),
                x1, y1, x2, y2,
                det.get("model", "unknown"),
                frame_bytes,
                len(detections)
            ))
            
            inserted_ids.append(cursor.lastrowid)
        
        conn.commit()
        conn.close()
        
        return inserted_ids
    
    def query_detections(self, video_path: Optional[str] = None, 
                        class_name: Optional[str] = None,
                        recognized_name: Optional[str] = None,
                        time_start: Optional[float] = None,
                        time_end: Optional[float] = None,
                        model_name: Optional[str] = None) -> List[Dict]:
        """
        Query detections from database.
        
        Returns:
            List of detection dictionaries
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Build query
        conditions = []
        params = []
        
        if video_path:
            conditions.append("video_path = ?")
            params.append(video_path)
        
        if class_name:
            conditions.append("class_name = ?")
            params.append(class_name)
        
        if recognized_name:
            if recognized_name == "Unknown":
                conditions.append("recognized_name = 'Unknown'")
            elif recognized_name == "known":
                conditions.append("recognized_name IS NOT NULL AND recognized_name != 'Unknown'")
            else:
                conditions.append("recognized_name = ?")
                params.append(recognized_name)
        
        if time_start is not None:
            conditions.append("timestamp >= ?")
            params.append(time_start)
        
        if time_end is not None:
            conditions.append("timestamp <= ?")
            params.append(time_end)
        
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT id, video_path, timestamp, time_string, class_name, recognized_name,
                   confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, model_name,
                   frame_image, num_objects, created_at
            FROM detections
            WHERE {where_clause}
            ORDER BY timestamp ASC
        """
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            (det_id, vid_path, ts, time_str, cls_name, rec_name, conf,
             x1, y1, x2, y2, mod_name, frame_bytes, num_obj, created_at) = row
            
            # Decode frame image
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            results.append({
                "id": det_id,
                "video_path": vid_path,
                "timestamp": ts,
                "time_string": time_str,
                "class_name": cls_name,
                "recognized_name": rec_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "model_name": mod_name,
                "frame": frame,
                "num_objects": num_obj,
                "created_at": created_at
            })
        
        conn.close()
        return results
    
    def get_statistics(self, video_path: Optional[str] = None) -> Dict:
        """Get statistics about detections."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        where_clause = "WHERE video_path = ?" if video_path else ""
        params = [video_path] if video_path else []
        
        # Total detections
        cursor.execute(f"SELECT COUNT(*) FROM detections {where_clause}", params)
        total = cursor.fetchone()[0]
        
        # By class
        cursor.execute(f"""
            SELECT class_name, COUNT(*) 
            FROM detections 
            {where_clause}
            GROUP BY class_name
        """, params)
        by_class = dict(cursor.fetchall())
        
        # By model
        cursor.execute(f"""
            SELECT model_name, COUNT(*) 
            FROM detections 
            {where_clause}
            GROUP BY model_name
        """, params)
        by_model = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total": total,
            "by_class": by_class,
            "by_model": by_model
        }
    
    def delete_detections(self, video_path: Optional[str] = None) -> int:
        """Delete detections (optionally for a specific video)."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if video_path:
            cursor.execute("DELETE FROM detections WHERE video_path = ?", (video_path,))
        else:
            cursor.execute("DELETE FROM detections")
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted

