from pydantic import BaseModel, Field
from typing import List

class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"

class TopicRequest(BaseModel):
    text: str

#class TopicResponse(BaseModel):
#    description: str
#    topic: str
#    # Uncomment if you want to include embedding vectors in the response
#    # vector: List[float]


class TopicResponse(BaseModel):
    """Response model for each topic with its confidence score."""
    topic: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")


class CategoryResponse(BaseModel):
    """Response model for the categorization result."""
    description: str
    top_topics: List[TopicResponse]