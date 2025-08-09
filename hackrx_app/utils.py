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
from concurrent.futures import ThreadPoolExecutor
import time

load_dotenv()

# Initialize the embedding model (free, runs locally)
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        # Use a faster, smaller model for better response times
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def download_pdf(url):
    """Download PDF from URL and save to temp file with timeout"""
    try:
        response = requests.get(url, timeout=15, stream=True)
        response.raise_for_status()
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")

def process_document_input(document_input):
    """
    Process document input - handle both PDF URLs and direct text content
    Returns the extracted text
    """
    try:
        # Check if input looks like a URL
        if (document_input.startswith('http://') or 
            document_input.startswith('https://') or 
            document_input.endswith('.pdf')):
            # Treat as PDF URL
            pdf_path = download_pdf(document_input)
            text = extract_text_from_pdf(pdf_path)
            # Clean up temp file
            try:
                os.unlink(pdf_path)
            except:
                pass
            return text
        else:
            # Treat as direct text content
            return document_input
    except Exception as e:
        # If URL processing fails, try treating as text content
        if len(document_input) > 200:  # Likely text content
            return document_input
        else:
            raise Exception(f"Failed to process document: {str(e)}")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file efficiently"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        # Limit to first 10 pages for speed
        max_pages = min(len(doc), 10)
        
        for page_num in range(max_pages):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():  # Only add pages with content
                text += page_text + "\n"
        
        doc.close()
        
        # Clean up temp file
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        
        return text.strip()
    except Exception as e:
        # Clean up temp file even if error occurs
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def split_text(text, max_length=400):
    """Split text into smaller, overlapping chunks for better context"""
    if not text.strip():
        return []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text.strip())
    words = text.split()
    
    if len(words) <= max_length:
        return [text]
    
    chunks = []
    overlap = max_length // 4  # 25% overlap
    
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i+max_length])
        if chunk.strip() and len(chunk.split()) > 10:  # Only meaningful chunks
            chunks.append(chunk)
        
        # Break if we've covered the text
        if i + max_length >= len(words):
            break
    
    return chunks

def embed_chunks(chunks):
    """Create embeddings for text chunks using local model"""
    if not chunks:
        return np.array([])
    
    model = get_embedding_model()
    embeddings = model.encode(chunks, convert_to_tensor=False, show_progress_bar=False)
    return np.array(embeddings, dtype='float32')

def get_top_chunks(query, chunks, chunk_vectors, k=3):
    """Find top-k most similar chunks to the query efficiently"""
    if not chunks or len(chunk_vectors) == 0:
        return []
    
    model = get_embedding_model()
    query_vector = model.encode([query], convert_to_tensor=False, show_progress_bar=False)[0]
    query_vector = np.array(query_vector, dtype='float32')
    
    # Create FAISS index
    dimension = len(query_vector)
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_vectors)
    
    # Search for similar chunks
    k = min(k, len(chunks))  # Don't ask for more chunks than available
    distances, indices = index.search(np.array([query_vector]), k)
    
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def ask_llm_optimized(question, context):
    """Optimized LLM function with multiple fallback strategies"""
    
    # Truncate context to reasonable size for faster processing
    context = context[:1000] if len(context) > 1000 else context
    
    # Try multiple models in parallel for better success rate
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit multiple API calls
        future1 = executor.submit(try_flan_t5, question, context)
        future2 = executor.submit(try_bart_qa, question, context)
        
        # Wait for first successful response (max 8 seconds total)
        start_time = time.time()
        while time.time() - start_time < 8:
            if future1.done():
                try:
                    result = future1.result()
                    if result and len(result.strip()) > 5:
                        return result
                except:
                    pass
            
            if future2.done():
                try:
                    result = future2.result()
                    if result and len(result.strip()) > 5:
                        return result
                except:
                    pass
            
            time.sleep(0.1)
    
    # Fallback to rule-based extraction
    return extract_answer_from_context(question, context)

def try_flan_t5(question, context):
    """Try Google FLAN-T5 model"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
        prompt = f"Answer based on context:\nContext: {context[:800]}\nQuestion: {question}\nAnswer:"
        
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_length": 100,
                    "temperature": 0.1,
                    "do_sample": False
                }
            },
            timeout=6
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '').strip()
                if answer and len(answer) > 3:
                    return answer
    except:
        pass
    return None

def try_bart_qa(question, context):
    """Try BART model for QA"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
        
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "inputs": {
                    "question": question,
                    "context": context[:800]
                }
            },
            timeout=6
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and 'answer' in result:
                answer = result['answer'].strip()
                if answer and len(answer) > 2 and result.get('score', 0) > 0.1:
                    return answer
    except:
        pass
    return None

def extract_answer_from_context(question, context):
    """Enhanced rule-based answer extraction"""
    if not context.strip():
        return "No relevant information found in the document."
    
    # Clean the context
    context = re.sub(r'\s+', ' ', context.strip())
    question_lower = question.lower().strip('?.')
    
    # Split into sentences more intelligently
    sentences = re.split(r'(?<=[.!?])\s+', context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    # Extract meaningful keywords from question
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                  'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'how', 'when', 'where', 'why'}
    
    question_words = []
    for word in question_lower.split():
        clean_word = re.sub(r'[^\w]', '', word)
        if len(clean_word) > 2 and clean_word not in stop_words:
            question_words.append(clean_word)
    
    if not question_words:
        return sentences[0] if sentences else "Unable to extract specific information."
    
    # Score sentences based on relevance
    scored_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = 0
        
        # Exact word matches
        for word in question_words:
            if word in sentence_lower:
                score += 3
        
        # Partial matches
        for word in question_words:
            if any(word in sent_word for sent_word in sentence_lower.split()):
                score += 1
        
        # Bonus for informative patterns
        if re.search(r'\b(means?|refers? to|defined as|is a|includes?|contains?)\b', sentence_lower):
            score += 2
        
        # Bonus for numerical or specific information
        if re.search(r'\b\d+\b|percentage|%|amount|number', sentence_lower):
            score += 1
        
        # Penalty for very short sentences
        if len(sentence.split()) < 5:
            score -= 1
        
        if score > 0:
            scored_sentences.append((score, sentence))
    
    if scored_sentences:
        # Sort by score
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Get the best sentence
        best_sentence = scored_sentences[0][1].strip()
        
        # If the best sentence is too long, truncate intelligently
        if len(best_sentence) > 200:
            words = best_sentence.split()
            best_sentence = ' '.join(words[:30]) + '...'
        
        return best_sentence
    
    # Final fallback - return first substantial sentence
    for sentence in sentences[:3]:
        if len(sentence.split()) >= 5:
            if len(sentence) > 200:
                sentence = ' '.join(sentence.split()[:30]) + '...'
            return sentence
    
    return "The specific information requested is not clearly available in the provided document."
