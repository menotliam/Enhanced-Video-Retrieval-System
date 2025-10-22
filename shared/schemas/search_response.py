"""
Pydantic schema for search response
"""
from pydantic import BaseModel
from typing import List, Dict

class SearchResult(BaseModel):
    scene_id: str
    start_ts: str
    end_ts: str
    preview: str
    transcript: str
    ocr_text: str
    objects: List[str]
    scene_label: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
