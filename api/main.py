from fastapi import FastAPI, status, HTTPException
from models import HealthCheck, TopicRequest, CategoryResponse
from services import Services
from importlib.metadata import version

app = FastAPI(title="Topic API") # , version=version("topic_api_v2")

# Set a default model name for encoding
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

@app.on_event("startup")
async def startup_event():
    """Initialize models once on app startup."""
    Services.initialize_models(model_name=DEFAULT_MODEL)

@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a health check. Primarily useful in Docker or other container
    orchestration environments for confirming service availability.
    
    Returns:
        HealthCheck: JSON response with the health status.
    """
    return HealthCheck(status="OK")

@app.post(
    "/get_category",
    tags=["categorization"],
    summary="Categorize Text Based on Confidence Threshold",
    response_description="Return relevant categories with their confidence scores",
    response_model=CategoryResponse,
    status_code=status.HTTP_200_OK
)
async def get_category(body: TopicRequest, threshold: float = 0.2):
    """
    Categorize input text into a predefined topic using SBERT and KNN.

    Args:
        body (TopicRequest): Text to be categorized.

    Returns:
        TopicResponse: JSON response with the original description and topic.
    """
    try:
        # Step 1: Generate embeddings
        embedding_vector = await Services.encode(texts=[body.text])
        
        # Step 2: Get topic based on embedding
        top_topics = await Services.get_topics_by_threshold(embedding_vector, threshold=threshold)
        
        return {"description": body.text, "top_topics": top_topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during categorization: {e}")