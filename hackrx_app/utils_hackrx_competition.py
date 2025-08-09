"""
HACKRX COMPETITION v16.0 - Complete Competition System
Handles ALL requirements: PDF/DOCX processing, FAISS embeddings, 
clause matching, explainable decisions, multi-domain support
"""
import re
import time
import logging
import requests
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for DOCX processing
from urllib.parse import urlparse
import tempfile
import os

logger = logging.getLogger(__name__)

class HackRxCompetitionEngine:
    def __init__(self):
        # Initialize sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Domain-specific patterns for insurance, legal, HR, compliance
        self.domain_patterns = self._init_domain_patterns()
        
        # FAISS index for semantic search
        self.faiss_index = None
        self.document_chunks = []
        
    def _init_domain_patterns(self):
        """Initialize domain-specific patterns for insurance, legal, HR, compliance"""
        return {
            'insurance': {
                'keywords': ['policy', 'premium', 'coverage', 'claim', 'deductible', 'benefit', 'waiting period', 'grace period', 'maternity', 'pre-existing', 'exclusion'],
                'patterns': [
                    r'(?:waiting|grace)\s+period.*?(\d+)\s*(?:days?|months?|years?)',
                    r'premium.*?(?:rs\.?|₹|\$)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                    r'coverage.*?(?:rs\.?|₹|\$)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                    r'deductible.*?(?:rs\.?|₹|\$)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                ]
            },
            'legal': {
                'keywords': ['contract', 'clause', 'agreement', 'liability', 'terms', 'conditions', 'breach', 'jurisdiction', 'arbitration'],
                'patterns': [
                    r'(?:section|clause|article)\s*(\d+(?:\.\d+)*)',
                    r'liability.*?(?:rs\.?|₹|\$)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                    r'penalty.*?(?:rs\.?|₹|\$)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                ]
            },
            'hr': {
                'keywords': ['employee', 'salary', 'benefits', 'leave', 'policy', 'termination', 'notice period', 'probation'],
                'patterns': [
                    r'notice\s+period.*?(\d+)\s*(?:days?|months?)',
                    r'probation.*?(\d+)\s*(?:days?|months?)',
                    r'salary.*?(?:rs\.?|₹|\$)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                ]
            },
            'compliance': {
                'keywords': ['regulation', 'compliance', 'audit', 'requirement', 'standard', 'certification', 'mandatory'],
                'patterns': [
                    r'compliance.*?(\d+(?:\.\d+)?%)',
                    r'audit.*?(?:every\s+)?(\d+)\s*(?:days?|months?|years?)',
                ]
            }
        }
    
    def download_and_extract_document(self, url_or_text: str) -> str:
        """Download and extract text from PDF/DOCX URL or return text directly"""
        
        # If it's already text content, return it
        if not url_or_text.startswith(('http://', 'https://')):
            return url_or_text
        
        try:
            # Download the document
            logger.info(f"Downloading document from: {url_or_text[:100]}...")
            response = requests.get(url_or_text, timeout=30)
            response.raise_for_status()
            
            # Determine file type from URL or content-type
            parsed_url = urlparse(url_or_text)
            file_path = parsed_url.path.lower()
            content_type = response.headers.get('content-type', '').lower()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            try:
                extracted_text = ""
                
                # Handle PDF files
                if file_path.endswith('.pdf') or 'pdf' in content_type:
                    logger.info("Processing PDF document...")
                    doc = fitz.open(temp_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        extracted_text += page.get_text() + "\n\n"
                    doc.close()
                
                # Handle DOCX files
                elif file_path.endswith('.docx') or 'wordprocessingml' in content_type:
                    logger.info("Processing DOCX document...")
                    doc = docx.Document(temp_path)
                    for paragraph in doc.paragraphs:
                        extracted_text += paragraph.text + "\n"
                
                # Handle other text formats
                else:
                    logger.info("Processing as text document...")
                    extracted_text = response.text
                
                logger.info(f"Extracted {len(extracted_text)} characters from document")
                return extracted_text
            
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            # Fallback to treating as text
            return url_or_text
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 10:  # Only add non-empty chunks
                chunks.append(chunk.strip())
                
        return chunks
    
    def build_faiss_index(self, chunks: List[str]) -> None:
        """Build FAISS index for semantic search"""
        logger.info(f"Building FAISS index for {len(chunks)} chunks...")
        
        # Generate embeddings for all chunks
        embeddings = self.embedder.encode(chunks)
        embeddings = embeddings.astype('float32')
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        self.document_chunks = chunks
        logger.info(f"FAISS index built successfully with {len(chunks)} vectors")
    
    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform semantic search using FAISS"""
        if self.faiss_index is None:
            return []
        
        # Encode query
        query_embedding = self.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.document_chunks):
                results.append((self.document_chunks[idx], float(similarity)))
        
        return results
    
    def classify_domain(self, text: str, query: str) -> str:
        """Classify document domain based on keywords"""
        combined_text = (text + " " + query).lower()
        
        domain_scores = {}
        for domain, config in self.domain_patterns.items():
            score = sum(1 for keyword in config['keywords'] if keyword in combined_text)
            domain_scores[domain] = score
        
        # Return domain with highest score, default to 'general'
        best_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        return best_domain if domain_scores[best_domain] > 0 else 'general'
    
    def extract_domain_specific_info(self, text: str, domain: str, query: str) -> Dict[str, Any]:
        """Extract domain-specific information using patterns"""
        
        if domain not in self.domain_patterns:
            return {}
        
        extracted_info = {}
        patterns = self.domain_patterns[domain]['patterns']
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Determine what type of information this pattern extracts
                if 'period' in pattern:
                    extracted_info['periods'] = matches
                elif any(currency in pattern for currency in ['rs', '₹', '$']):
                    extracted_info['amounts'] = matches
                elif 'section|clause|article' in pattern:
                    extracted_info['references'] = matches
        
        return extracted_info
    
    def generate_explainable_answer(self, query: str, relevant_chunks: List[Tuple[str, float]], 
                                  domain: str, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer with explainable reasoning"""
        
        # Find the most relevant chunk
        if not relevant_chunks:
            return {
                'answer': 'I could not find relevant information in the document to answer this question.',
                'confidence': 0.0,
                'reasoning': 'No semantically relevant content found.',
                'sources': [],
                'domain': domain
            }
        
        best_chunk, confidence = relevant_chunks[0]
        
        # Extract specific answer based on domain and query pattern
        answer = self.extract_specific_answer(query, best_chunk, domain, extracted_info)
        
        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Domain classified as '{domain}' based on content analysis.")
        reasoning_parts.append(f"Found semantically relevant content with {confidence:.2f} similarity score.")
        
        if extracted_info:
            reasoning_parts.append(f"Extracted domain-specific information: {extracted_info}")
        
        # Build sources with evidence
        sources = []
        for i, (chunk, score) in enumerate(relevant_chunks[:3]):
            sources.append({
                'chunk_id': i + 1,
                'similarity_score': float(score),
                'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk
            })
        
        return {
            'answer': answer,
            'confidence': float(confidence),
            'reasoning': ' '.join(reasoning_parts),
            'sources': sources,
            'domain': domain,
            'extracted_info': extracted_info
        }
    
    def extract_specific_answer(self, query: str, context: str, domain: str, extracted_info: Dict[str, Any]) -> str:
        """Extract specific answer from context based on query"""
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Insurance domain specific extractions
        if domain == 'insurance':
            if any(word in query_lower for word in ['grace period', 'grace']):
                pattern = r'grace\s+period.*?(\d+)\s*(days?|months?)'
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return f"The grace period is {match.group(1)} {match.group(2)}."
            
            elif any(word in query_lower for word in ['waiting period', 'waiting']):
                pattern = r'waiting\s+period.*?(\d+)\s*(days?|months?|years?)'
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return f"The waiting period is {match.group(1)} {match.group(2)}."
            
            elif 'maternity' in query_lower:
                if 'maternity' in context_lower:
                    # Extract maternity-related sentences
                    sentences = context.split('.')
                    maternity_sentences = [s for s in sentences if 'maternity' in s.lower()]
                    if maternity_sentences:
                        return maternity_sentences[0].strip() + "."
            
            elif 'cover' in query_lower and any(word in query_lower for word in ['knee', 'surgery', 'medical']):
                # Look for coverage information
                sentences = context.split('.')
                coverage_sentences = [s for s in sentences if any(word in s.lower() for word in ['cover', 'coverage', 'include'])]
                if coverage_sentences:
                    return coverage_sentences[0].strip() + "."
        
        # Legal domain specific extractions
        elif domain == 'legal':
            if 'clause' in query_lower:
                pattern = r'(?:clause|section)\s*(\d+(?:\.\d+)*)'
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return f"Refers to clause/section {match.group(1)}."
        
        # HR domain specific extractions
        elif domain == 'hr':
            if 'notice period' in query_lower:
                pattern = r'notice\s+period.*?(\d+)\s*(days?|months?)'
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return f"The notice period is {match.group(1)} {match.group(2)}."
        
        # Generic extraction - find most relevant sentence
        sentences = context.split('.')
        query_words = [word for word in query_lower.split() if len(word) > 3]
        
        best_sentence = ""
        max_matches = 0
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                sentence_lower = sentence.lower()
                matches = sum(1 for word in query_words if word in sentence_lower)
                if matches > max_matches:
                    max_matches = matches
                    best_sentence = sentence.strip()
        
        if best_sentence:
            return best_sentence + ("." if not best_sentence.endswith('.') else "")
        
        # Fallback to first meaningful sentence
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip() + ("." if not sentence.strip().endswith('.') else "")
        
        return context[:200] + "..." if len(context) > 200 else context

# Global competition engine
hackrx_engine = HackRxCompetitionEngine()

def process_document_hackrx(document: str) -> Dict[str, Any]:
    """Process document for HackRx competition with full pipeline"""
    start_time = time.time()
    
    try:
        # Extract text from URL or use directly
        extracted_text = hackrx_engine.download_and_extract_document(document)
        
        # Chunk document for better retrieval
        chunks = hackrx_engine.chunk_document(extracted_text)
        
        # Build FAISS index for semantic search
        hackrx_engine.build_faiss_index(chunks)
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'document_text': extracted_text,
            'chunks_count': len(chunks),
            'processing_time': processing_time,
            'status': 'HACKRX_READY'
        }
        
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time
        }

def process_questions_hackrx(questions: List[str], document_data: Dict[str, Any]) -> List[str]:
    """Process questions for HackRx competition with explainable answers"""
    if not document_data.get('success'):
        return ["Error: Document processing failed"] * len(questions)
    
    answers = []
    document_text = document_data['document_text']
    
    for question in questions:
        try:
            # Classify domain
            domain = hackrx_engine.classify_domain(document_text, question)
            
            # Semantic search for relevant content
            relevant_chunks = hackrx_engine.semantic_search(question, k=5)
            
            # Extract domain-specific information
            extracted_info = hackrx_engine.extract_domain_specific_info(document_text, domain, question)
            
            # Generate explainable answer
            result = hackrx_engine.generate_explainable_answer(question, relevant_chunks, domain, extracted_info)
            
            # For competition format, return just the answer
            answers.append(result['answer'])
            
            logger.info(f"Q: {question[:50]}... -> Domain: {domain}, Confidence: {result['confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"Question processing error: {e}")
            answers.append(f"Error processing question: {str(e)}")
    
    return answers
