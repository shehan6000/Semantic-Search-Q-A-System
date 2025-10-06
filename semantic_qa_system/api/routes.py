from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List

from services.application_service import app_service

app = FastAPI(
    title="Semantic Search Q&A System",
    description="A production-ready semantic search and question answering system",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    query: str
    use_approximate: bool = True
    top_k: int = 1

class SearchResponse(BaseModel):
    success: bool
    query: Optional[str] = None
    answer: Optional[str] = None
    source_document: Optional[dict] = None
    search_method: Optional[str] = None
    latency_ms: float
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    database_info: dict

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        db_info = app_service.get_database_info()
        return {
            "status": "healthy",
            "database_info": db_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Perform semantic search and answer generation"""
    try:
        result = app_service.search_and_answer(
            query=request.query,
            use_approximate=request.use_approximate,
            k=request.top_k
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def semantic_search_get(
    query: str = Query(..., description="Search query"),
    use_approximate: bool = Query(True, description="Use approximate search"),
    top_k: int = Query(1, description="Number of results to return")
):
    """GET endpoint for semantic search"""
    try:
        result = app_service.search_and_answer(
            query=query,
            use_approximate=use_approximate,
            k=top_k
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """Retrieve a specific document by ID"""
    try:
        document = app_service.search_service.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    from config.settings import settings
    
    uvicorn.run(
        "api.routes:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )