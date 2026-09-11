# Pydantic schemas for the API
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User query string")

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or only whitespace.")
        return v.strip()

class Citation(BaseModel):
    file_name: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    needs_refusal: bool
    retrieved_chunks: List[dict] = Field(..., description="List of retrieved chunks with their scores and metadata")
    latency: Optional[float] = Field(None, description="Time taken to process the query in seconds")

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    llm_model: str