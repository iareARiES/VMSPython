"""
Gemini API integration for query parsing with fallback to hardcoded parsing.
"""

import re
import json
from typing import Dict, Optional

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("Warning: google-generativeai not installed. Using fallback parser.")


class QueryParser:
    """Parse natural language queries using Gemini API with fallback."""
    
    def __init__(self, api_key: str = "AIzaSyChqw4awZLPeyILxRS5vlQgOECygHlm-aI"):
        """Initialize parser with Gemini API key."""
        self.api_key = api_key
        self.use_gemini = HAS_GEMINI
        
        if HAS_GEMINI:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                print("Gemini API initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Gemini API: {e}. Using fallback parser.")
                self.use_gemini = False
        else:
            print("Using fallback parser (Gemini API not available)")
    
    def parse_query(self, query_text: str) -> Dict:
        """
        Parse natural language query.
        
        Returns:
            Dictionary with parsed query information
        """
        if self.use_gemini:
            result = self._parse_with_gemini(query_text)
            if result.get("valid"):
                return result
        
        # Fallback to hardcoded parser
        return self._parse_fallback(query_text)
    
    def _parse_with_gemini(self, query_text: str) -> Dict:
        """Parse query using Gemini API."""
        try:
            prompt = f"""Parse this video analysis query and return JSON only:
"{query_text}"

Return JSON with these fields:
- action: "find" or "save" 
- class_name: object class to find (e.g., "person", "human", "tiger", "face", etc.) or null for all
- recognized_name: for faces, "Unknown", "known", or specific name, or null
- time_start: start time in seconds (extract from "X min" or "X:XX", convert minutes to seconds) or null
- time_end: end time in seconds (convert minutes to seconds) or null
- save_to_db: true if query mentions "save" or "database", false otherwise
- description: brief description

IMPORTANT: Convert minutes to seconds (multiply by 60). If time is given as "10 min", convert to 600 seconds.

Examples:
- "find all humans from 10 min to 15 min and save them" -> {{"action": "save", "class_name": "person", "time_start": 600, "time_end": 900, "save_to_db": true}}
- "find tigers" -> {{"action": "find", "class_name": "tiger", "save_to_db": false}}
- "find whatever you see and save them" -> {{"action": "save", "class_name": null, "save_to_db": true}}
- "find all unknown faces from 5:00 to 10:00" -> {{"action": "find", "class_name": "face", "recognized_name": "Unknown", "time_start": 300, "time_end": 600, "save_to_db": false}}

Return ONLY valid JSON, no other text:"""

            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Convert time from minutes to seconds if needed
                if parsed.get("time_start") and isinstance(parsed["time_start"], (int, float)) and parsed["time_start"] < 1000:
                    parsed["time_start"] = parsed["time_start"] * 60
                if parsed.get("time_end") and isinstance(parsed["time_end"], (int, float)) and parsed["time_end"] < 1000:
                    parsed["time_end"] = parsed["time_end"] * 60
                
                parsed["valid"] = True
                parsed["source"] = "gemini"
                return parsed
        except Exception as e:
            print(f"Gemini API error: {e}")
        
        return {"valid": False, "source": "gemini_failed"}
    
    def _parse_fallback(self, query_text: str) -> Dict:
        """Fallback hardcoded parser."""
        query_lower = query_text.lower()
        parsed = {
            "valid": False,
            "action": None,
            "class_name": None,
            "recognized_name": None,
            "time_start": None,
            "time_end": None,
            "save_to_db": False,
            "description": "",
            "source": "fallback"
        }
        
        # Check for save/find action
        if "save" in query_lower or "database" in query_lower:
            parsed["action"] = "save"
            parsed["save_to_db"] = True
            parsed["valid"] = True
        elif "find" in query_lower or "show" in query_lower or "search" in query_lower:
            parsed["action"] = "find"
            parsed["save_to_db"] = False
            parsed["valid"] = True
        else:
            return parsed
        
        # Extract class name
        class_keywords = {
            "human": "person",
            "humans": "person",
            "person": "person",
            "people": "person",
            "tiger": "tiger",
            "tigers": "tiger",
            "face": "face",
            "faces": "face",
            "car": "car",
            "cars": "car",
            "fire": "fire",
            "smoke": "smoke"
        }
        
        for keyword, class_name in class_keywords.items():
            if keyword in query_lower:
                parsed["class_name"] = class_name
                break
        
        # Check for "whatever" or "all" - means all classes
        if "whatever" in query_lower or ("all" in query_lower and "class" not in query_lower and "humans" not in query_lower):
            parsed["class_name"] = None  # All classes
        
        # Extract recognized name for faces
        if parsed["class_name"] == "face" or "face" in query_lower:
            if "unknown" in query_lower:
                parsed["recognized_name"] = "Unknown"
            elif "known" in query_lower:
                parsed["recognized_name"] = "known"
        
        # Extract time range
        # Pattern: "X min" or "X:XX" or "from X min to Y min"
        time_patterns = [
            r'from\s+(\d+)\s*min\s+to\s+(\d+)\s*min',
            r'(\d+)\s*min\s+to\s+(\d+)\s*min',
            r'from\s+(\d{1,2}):(\d{2})\s+to\s+(\d{1,2}):(\d{2})',
            r'(\d{1,2}):(\d{2})\s+to\s+(\d{1,2}):(\d{2})',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:  # Minutes format
                    parsed["time_start"] = int(groups[0]) * 60
                    parsed["time_end"] = int(groups[1]) * 60
                elif len(groups) == 4:  # Time format
                    h1, m1, h2, m2 = groups
                    parsed["time_start"] = int(h1) * 3600 + int(m1) * 60
                    parsed["time_end"] = int(h2) * 3600 + int(m2) * 60
                break
        
        # Build description
        desc_parts = []
        if parsed["action"]:
            desc_parts.append(parsed["action"])
        if parsed["class_name"]:
            desc_parts.append(parsed["class_name"])
        else:
            desc_parts.append("all objects")
        if parsed["time_start"] and parsed["time_end"]:
            desc_parts.append(f"from {int(parsed['time_start']//60)}:{int(parsed['time_start']%60):02d} to {int(parsed['time_end']//60)}:{int(parsed['time_end']%60):02d}")
        if parsed["save_to_db"]:
            desc_parts.append("and save to database")
        
        parsed["description"] = " ".join(desc_parts)
        
        return parsed

