from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Import optimized functions
from utils import (
    download_pdf, extract_text_from_pdf, split_text, 
    embed_chunks, get_top_chunks, ask_llm_optimized, process_document_input
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRx LLM Query Engine - Optimized",
    description="Optimized API for document-based question answering",
    version="2.0.0"
)

class QueryRequest(BaseModel):
    documents: str  # Can be either a PDF URL or direct text content
    questions: list[str]

@app.get("/")
async def root():
    return {"message": "HackRx LLM Query Engine API v2.0 - Optimized!", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/hackrx/run")
async def run_submission(req: QueryRequest, authorization: str = Header(None)):
    """
    Optimized endpoint for processing document questions
    Accepts either PDF URLs or direct text content in the documents field
    """
    start_time = time.time()
    
    try:
        # Validate authorization
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        # Validate input
        if not req.documents or not req.questions:
            raise HTTPException(status_code=400, detail="Documents and questions are required")
        
        if len(req.questions) > 5:  # Limit questions for performance
            req.questions = req.questions[:5]
        
        logger.info(f"Processing {len(req.questions)} questions - Start")
        
        # Step 1: Process document input (handles both PDF URLs and text content)
        try:
            full_text = process_document_input(req.documents)
            logger.info(f"Document processed - {len(full_text)} characters extracted")
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text found in document")
        
        # Step 2: Split and embed text
        chunks = split_text(full_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not create meaningful text chunks")
        
        logger.info(f"Created {len(chunks)} text chunks")
        
        # Create embeddings
        try:
            vectors = embed_chunks(chunks)
            logger.info("Embeddings created successfully")
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
        
        # Step 3: Process questions in parallel for better performance
        answers = []
        
        def process_question(question_data):
            question_idx, question = question_data
            try:
                logger.info(f"Processing question {question_idx + 1}: {question[:50]}...")
                
                # Get relevant chunks
                top_chunks = get_top_chunks(question, chunks, vectors, k=3)
                if not top_chunks:
                    return "No relevant information found for this question."
                
                context = "\n\n".join(top_chunks)
                
                # Get answer using optimized LLM
                answer = ask_llm_optimized(question, context)
                
                # Clean and validate answer
                if not answer or len(answer.strip()) < 3:
                    answer = "Unable to find a specific answer to this question in the document."
                
                logger.info(f"Question {question_idx + 1} processed successfully")
                return answer.strip()
                
            except Exception as e:
                logger.error(f"Error processing question {question_idx + 1}: {e}")
                return "An error occurred while processing this question."
        
        # Process questions with controlled concurrency
        with ThreadPoolExecutor(max_workers=2) as executor:
            question_futures = [
                executor.submit(process_question, (i, q)) 
                for i, q in enumerate(req.questions)
            ]
            
            # Collect results with timeout
            for future in question_futures:
                try:
                    answer = future.result(timeout=15)  # 15 second timeout per question
                    answers.append(answer)
                except Exception as e:
                    logger.error(f"Question processing timeout or error: {e}")
                    answers.append("Processing timeout - please try with a shorter question.")
        
        processing_time = time.time() - start_time
        logger.info(f"All questions processed successfully in {processing_time:.2f}s")
        
        return {
            "answers": answers,
            "processing_time": round(processing_time, 2),
            "questions_processed": len(answers)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error after {processing_time:.2f}s: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
