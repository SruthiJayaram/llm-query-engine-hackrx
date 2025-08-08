from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from utils import *
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRx LLM Query Engine",
    description="API for document-based question answering using LLM and semantic search",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@app.get("/")
async def root():
    return {"message": "HackRx LLM Query Engine API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/hackrx/run")
async def run_submission(req: QueryRequest, authorization: str = Header(None)):
    """
    Main endpoint for processing document questions
    """
    try:
        # Check authorization
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        logger.info(f"Processing request with {len(req.questions)} questions")
        
        # Download and extract text from PDF
        pdf_path = download_pdf(req.documents)
        full_text = extract_text_from_pdf(pdf_path)
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Split text into chunks
        chunks = split_text(full_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not create text chunks from PDF")
        
        # Create embeddings for chunks
        vectors = embed_chunks(chunks)
        
        # Process each question
        answers = []
        for i, question in enumerate(req.questions):
            logger.info(f"Processing question {i+1}/{len(req.questions)}")
            
            # Get most relevant chunks
            top_chunks = get_top_chunks(question, chunks, vectors)
            context = "\n\n".join(top_chunks)
            
            # Get answer from LLM
            answer = ask_llm(question, context)
            answers.append(answer)
        
        logger.info("Successfully processed all questions")
        return {"answers": answers}
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
