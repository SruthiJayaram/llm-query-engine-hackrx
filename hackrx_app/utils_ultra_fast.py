import fitz  # PyMuPDF
import requests
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import tempfile
from sentence_transformers import SentenceTransformer
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
from functools import lru_cache
import asyncio

load_dotenv()

# Global cache for embeddings and responses
EMBEDDING_CACHE = {}
RESPONSE_CACHE = {}

# Initialize faster embedding model
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading ultra-fast embedding model...")
        # Use the fastest small model available
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Even faster than L6
        embedding_model.max_seq_length = 256  # Limit sequence length for speed
    return embedding_model

def download_pdf_fast(url):
    """Download PDF with aggressive timeout and size limits"""
    try:
        response = requests.get(url, timeout=8, stream=True)  # Reduced timeout
        response.raise_for_status()
        
        # Limit PDF size to 5MB for speed
        content = b''
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 5 * 1024 * 1024:  # 5MB limit
                break
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")

def process_document_input_fast(document_input):
    """
    Ultra-fast document processing with caching
    """
    # Create cache key
    cache_key = hashlib.md5(document_input.encode()).hexdigest()
    if cache_key in RESPONSE_CACHE:
        return RESPONSE_CACHE[cache_key]
    
    try:
        if (document_input.startswith('http://') or 
            document_input.startswith('https://') or 
            document_input.endswith('.pdf')):
            # Process as PDF URL
            pdf_path = download_pdf_fast(document_input)
            text = extract_text_from_pdf_fast(pdf_path)
            try:
                os.unlink(pdf_path)
            except:
                pass
            RESPONSE_CACHE[cache_key] = text
            return text
        else:
            # Direct text content - just cache and return
            RESPONSE_CACHE[cache_key] = document_input
            return document_input
    except Exception as e:
        if len(document_input) > 100:  # Likely text content
            RESPONSE_CACHE[cache_key] = document_input
            return document_input
        else:
            raise Exception(f"Failed to process document: {str(e)}")

def extract_text_from_pdf_fast(pdf_path):
    """Extract text from PDF with speed optimizations"""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        # Only process first 5 pages for speed
        max_pages = min(len(doc), 5)
        
        for page_num in range(max_pages):
            page = doc[page_num]
            page_text = page.get_text("text")  # Fastest extraction method
            if page_text.strip():
                # Only keep first 1000 chars per page
                text_parts.append(page_text[:1000])
        
        doc.close()
        
        # Clean up temp file
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        
        combined_text = "\n".join(text_parts)
        # Limit total text length to 3000 chars for speed
        return combined_text[:3000] if len(combined_text) > 3000 else combined_text
        
    except Exception as e:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def split_text_fast(text, max_length=200):  # Smaller chunks for speed
    """Ultra-fast text splitting with minimal overhead"""
    if not text or len(text) < 100:
        return [text] if text else []
    
    # Simple sentence-based splitting
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence exceeds max_length, save current chunk
        if len(current_chunk) + len(sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Limit to max 6 chunks for speed
    return chunks[:6]

def embed_chunks_fast(chunks):
    """Create embeddings with caching"""
    if not chunks:
        return np.array([])
    
    # Check cache first
    embeddings = []
    uncached_chunks = []
    uncached_indices = []
    
    for i, chunk in enumerate(chunks):
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
        if chunk_hash in EMBEDDING_CACHE:
            embeddings.append(EMBEDDING_CACHE[chunk_hash])
        else:
            uncached_chunks.append(chunk)
            uncached_indices.append(i)
            embeddings.append(None)  # Placeholder
    
    # Process uncached chunks
    if uncached_chunks:
        model = get_embedding_model()
        new_embeddings = model.encode(
            uncached_chunks, 
            convert_to_tensor=False, 
            show_progress_bar=False,
            batch_size=32  # Process in batches for speed
        )
        
        # Cache new embeddings
        for chunk, embedding in zip(uncached_chunks, new_embeddings):
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            EMBEDDING_CACHE[chunk_hash] = embedding
        
        # Fill in the embeddings
        for i, embedding in enumerate(new_embeddings):
            embeddings[uncached_indices[i]] = embedding
    
    return np.array(embeddings, dtype='float32')

def get_top_chunks_fast(query, chunks, chunk_vectors, k=2):  # Reduce k for speed
    """Ultra-fast similarity search"""
    if not chunks or len(chunk_vectors) == 0:
        return []
    
    # Check query cache
    query_hash = hashlib.md5(query.encode()).hexdigest()
    
    model = get_embedding_model()
    query_vector = model.encode([query], convert_to_tensor=False, show_progress_bar=False)[0]
    query_vector = np.array(query_vector, dtype='float32')
    
    # Simple dot product similarity (faster than FAISS for small datasets)
    if len(chunks) <= 10:
        similarities = np.dot(chunk_vectors, query_vector)
        top_indices = np.argsort(similarities)[::-1][:k]
        return [chunks[i] for i in top_indices]
    
    # Use FAISS for larger datasets
    dimension = len(query_vector)
    index = faiss.IndexFlatIP(dimension)  # Inner product is faster than L2
    index.add(chunk_vectors)
    
    k = min(k, len(chunks))
    scores, indices = index.search(np.array([query_vector]), k)
    
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def ask_llm_ultra_fast(question, context):
    """Ultra-fast LLM with aggressive optimizations"""
    
    # Aggressive context truncation
    context = context[:600] if len(context) > 600 else context
    
    # Check response cache first
    cache_key = hashlib.md5(f"{question}{context}".encode()).hexdigest()
    if cache_key in RESPONSE_CACHE:
        return RESPONSE_CACHE[cache_key]
    
    # Try multiple models with very short timeouts
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit requests to multiple models simultaneously
        futures = [
            executor.submit(try_huggingface_qa_fast, question, context),
            executor.submit(try_distilbert_qa, question, context),
            executor.submit(extract_answer_fast, question, context)  # Fallback
        ]
        
        # Return first successful response (max 5 seconds)
        start_time = time.time()
        while time.time() - start_time < 5:
            for future in as_completed(futures, timeout=0.1):
                try:
                    result = future.result()
                    if result and len(result.strip()) > 3:
                        RESPONSE_CACHE[cache_key] = result
                        return result
                except:
                    continue
            time.sleep(0.05)
    
    # Final fallback
    result = extract_answer_fast(question, context)
    RESPONSE_CACHE[cache_key] = result
    return result

def try_huggingface_qa_fast(question, context):
    """Try Hugging Face QA with minimal timeout"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
        
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "inputs": {
                    "question": question,
                    "context": context[:500]  # Even more aggressive truncation
                }
            },
            timeout=3  # Very short timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and 'answer' in result:
                answer = result['answer'].strip()
                if answer and len(answer) > 2 and result.get('score', 0) > 0.05:
                    return answer
    except:
        pass
    return None

def try_distilbert_qa(question, context):
    """Try DistilBERT for speed"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-distilled-squad"
        
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "inputs": {
                    "question": question,
                    "context": context[:500]
                }
            },
            timeout=3
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and 'answer' in result:
                return result['answer'].strip()
    except:
        pass
    return None

def extract_answer_fast(question, context):
    """Ultra-fast rule-based extraction"""
    if not context.strip():
        return "Information not found."
    
    # Simple keyword matching
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', context) if len(s.strip()) > 10]
    
    # Get question keywords (simple approach)
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'when', 'where', 'why'}
    keywords = [w for w in question_lower.split() if w not in stop_words and len(w) > 2]
    
    if not keywords:
        return sentences[0][:150] if sentences else "No specific answer found."
    
    # Find best matching sentence
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences[:5]:  # Only check first 5 sentences
        sentence_lower = sentence.lower()
        score = sum(1 for keyword in keywords if keyword in sentence_lower)
        
        if score > best_score:
            best_score = score
            best_sentence = sentence
    
    if best_sentence:
        # Truncate if too long
        if len(best_sentence) > 150:
            best_sentence = best_sentence[:150] + "..."
        return best_sentence
    
    # Return first sentence if no good match
    return sentences[0][:150] + "..." if sentences else "Answer not found in document."

def process_questions_parallel(document_text, questions, max_workers=3):
    """Process multiple questions in parallel for speed"""
    if not questions:
        return []
    
    # Split text once
    chunks = split_text_fast(document_text)
    if not chunks:
        return ["No content found in document."] * len(questions)
    
    # Create embeddings once
    chunk_vectors = embed_chunks_fast(chunks)
    if len(chunk_vectors) == 0:
        return ["Unable to process document content."] * len(questions)
    
    def process_single_question(question):
        try:
            # Get relevant chunks
            relevant_chunks = get_top_chunks_fast(question, chunks, chunk_vectors, k=2)
            context = " ".join(relevant_chunks)
            
            if not context.strip():
                return "No relevant information found."
            
            # Get answer
            answer = ask_llm_ultra_fast(question, context)
            return answer if answer else "Unable to determine answer."
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    # Process questions in parallel
    with ThreadPoolExecutor(max_workers=min(max_workers, len(questions))) as executor:
        answers = list(executor.map(process_single_question, questions))
    
    return answers

# Cache cleanup function
def cleanup_cache():
    """Clean up caches to prevent memory issues"""
    global EMBEDDING_CACHE, RESPONSE_CACHE
    if len(EMBEDDING_CACHE) > 100:
        EMBEDDING_CACHE.clear()
    if len(RESPONSE_CACHE) > 50:
        RESPONSE_CACHE.clear()
