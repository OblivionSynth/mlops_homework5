from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    top_k: Optional[int] = Field(5, description="Number of results to return")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "top_k": 3
            }
        }

class SearchResult(BaseModel):
    index: int = Field(..., description="Index in the corpus")
    score: float = Field(..., description="Similarity score")
    text: str = Field(..., description="Wikipedia excerpt")

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="Search results")