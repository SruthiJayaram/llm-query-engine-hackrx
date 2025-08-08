import fitz  # PyMuPDF
import requests
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import tempfile
from sentence_transformers import SentenceTransformer
import json

load_dotenv()

# Initialize the embedding model (free, runs locally)
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def download_pdf(url):
    """Download PDF from URL and save to temp file"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    # Clean up temp file
    os.unlink(pdf_path)
    return text

def split_text(text, max_length=500):
    """Split text into chunks of maximum length"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i:i+max_length])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    """Create embeddings for text chunks using local model"""
    model = get_embedding_model()
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return np.array(embeddings, dtype='float32')

def get_top_chunks(query, chunks, chunk_vectors, k=3):
    """Find top-k most similar chunks to the query"""
    model = get_embedding_model()
    query_vector = model.encode([query], convert_to_tensor=False)[0]
    query_vector = np.array(query_vector, dtype='float32')
    
    # Create FAISS index
    dimension = len(query_vector)
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_vectors)
    
    # Search for similar chunks
    distances, indices = index.search(np.array([query_vector]), k)
    
    return [chunks[i] for i in indices[0]]

def ask_llm(question, context):
    """Ask LLM to answer question based on context using Hugging Face API"""
    # Use Hugging Face's free inference API
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    
    # Truncate if too long
    if len(prompt) > 1000:
        prompt = prompt[:1000] + "..."
    
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.7}},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '').replace(prompt, '').strip()
                if answer:
                    return answer
        
        # Fallback to simple text processing if API fails
        return extract_answer_from_context(question, context)
    
    except Exception as e:
        print(f"LLM API error: {e}")
        return extract_answer_from_context(question, context)

def extract_answer_from_context(question, context):
    """Simple fallback method to extract answers from context"""
    # Simple keyword-based answer extraction
    sentences = context.split('.')
    question_lower = question.lower()
    
    # Look for sentences containing question keywords
    keywords = [word.lower() for word in question.split() if len(word) > 3]
    
    best_sentence = ""
    max_matches = 0
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matches = sum(1 for keyword in keywords if keyword in sentence_lower)
        if matches > max_matches:
            max_matches = matches
            best_sentence = sentence.strip()
    
    return best_sentence if best_sentence else "Answer not found in the provided context."
