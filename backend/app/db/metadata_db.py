"""
Metadata DB connector (Postgres/Elasticsearch)
"""
from typing import Dict, List

class MetadataDB:
    def __init__(self, config: dict):
        """Initialize connection to metadata DB (Postgres/Elasticsearch)."""
        pass

    def insert(self, scene_id: str, metadata: Dict):
        """Insert metadata for a scene."""
        pass

    def filter(self, filters: Dict) -> List[Dict]:
        """Filter scenes by metadata (objects, OCR, transcript, etc)."""
        pass

    def close(self):
        """Close DB connection."""
        pass
