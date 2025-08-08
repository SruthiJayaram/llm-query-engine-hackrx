import fitz  # PyMuPDF
import requests
from openai import OpenAI
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    """Create embeddings for text chunks using OpenAI"""
    embeddings = []
    batch_size = 20
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        for embedding in response.data:
            embeddings.append(np.array(embedding.embedding, dtype='float32'))
    
    return np.vstack(embeddings)

def get_top_chunks(query, chunks, chunk_vectors, k=3):
    """Find top-k most similar chunks to the query"""
    # Get query embedding
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_vector = np.array(response.data[0].embedding, dtype='float32')
    
    # Create FAISS index
    dimension = len(query_vector)
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_vectors)
    
    # Search for similar chunks
    distances, indices = index.search(np.array([query_vector]), k)
    
    return [chunks[i] for i in indices[0]]

def ask_llm(question, context):
    """Ask LLM to answer question based on context"""
    prompt = f"""Answer the question based on the following context. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()
