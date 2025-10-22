"""
Pydantic schema for search request
"""
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    filters: dict = None
