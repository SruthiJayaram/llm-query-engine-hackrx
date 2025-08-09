from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Import ultra-fast optimized functions
from utils_ultra_fast import (
    process_document_input_fast, process_questions_parallel, cleanup_cache
)

# Configure minimal logging for speed
logging.basicConfig(level=logging.WARNING)  # Reduced logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRx LLM Query Engine - ULTRA FAST",
    description="Ultra-optimized API for document-based question answering with <30s response time",
    version="3.0.0"
)

class QueryRequest(BaseModel):
    documents: str  # Can be either a PDF URL or direct text content
    questions: list[str]

@app.get("/")
async def root():
    return {"message": "HackRx LLM Query Engine API v3.0 - ULTRA FAST!", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/hackrx/run")
async def run_submission(req: QueryRequest, authorization: str = Header(None)):
    """
    ULTRA-FAST endpoint optimized for <30 second response times
    Supports both PDF URLs and direct text content
    """
    start_time = time.time()
    
    try:
        # Quick auth check
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        # Quick validation
        if not req.documents or not req.documents.strip():
            raise HTTPException(status_code=400, detail="Documents field is required")
        
        if not req.questions or len(req.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Limit questions for speed
        questions = req.questions[:5]  # Max 5 questions
        
        logger.warning(f"Processing {len(questions)} questions with ultra-fast pipeline")
        
        # Process document with aggressive optimizations
        document_text = process_document_input_fast(req.documents)
        
        if not document_text or len(document_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="No meaningful content found in document")
        
        # Process all questions in parallel with speed optimizations
        answers = process_questions_parallel(document_text, questions, max_workers=3)
        
        # Clean up caches periodically
        cleanup_cache()
        
        processing_time = time.time() - start_time
        
        return {
            "answers": answers,
            "processing_time": round(processing_time, 2),
            "questions_processed": len(questions),
            "performance_optimized": True,
            "version": "3.0.0"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        processing_time = time.time() - start_time
        
        # Return fast fallback response
        fallback_answers = []
        for question in req.questions[:5]:
            fallback_answers.append(f"Unable to process: {question}")
        
        return {
            "answers": fallback_answers,
            "processing_time": round(processing_time, 2),
            "error": "Processing failed, returned fallback responses",
            "version": "3.0.0"
        }

@app.get("/performance-stats")
async def get_performance_stats():
    """Get current performance optimizations status"""
    return {
        "optimizations_enabled": [
            "Aggressive caching",
            "Parallel processing", 
            "Fast embedding model",
            "Context truncation",
            "Reduced chunk size",
            "Multiple model fallback",
            "Smart text extraction",
            "Memory management"
        ],
        "target_response_time": "< 30 seconds",
        "accuracy_target": "> 50%",
        "version": "3.0.0"
    }

# Startup event to warm up the model
@app.on_event("startup")
async def startup_event():
    """Warm up the models for faster first response"""
    try:
        from utils_ultra_fast import get_embedding_model
        # Warm up embedding model
        model = get_embedding_model()
        # Test embedding to warm up
        model.encode(["test sentence for warmup"], show_progress_bar=False)
        logger.warning("Ultra-fast model warmed up successfully")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
