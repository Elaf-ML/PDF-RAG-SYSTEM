"""
RAG Backend using FastAPI and Gemini API

This script provides a backend for a RAG-based chatbot that:
1. Accepts file uploads (PDF, CSV, TXT, JSON, DOCX)
2. Processes and chunks files for semantic search
3. Uses FAISS for vector storage
4. Leverages Gemini API for generating responses
5. Supports multilingual documents and cross-language queries
6. Implements advanced RAG techniques (hybrid search, re-ranking)
7. Maintains chat history for contextual conversations

Setup:
- pip install fastapi uvicorn python-multipart pydantic langchain langchain-google-genai faiss-cpu pypdf pandas python-docx python-dotenv langdetect rank_bm25 sentence-transformers
- Create a .env file with GOOGLE_API_KEY=your_gemini_api_key

Run:
- uvicorn backend.rag_backend:app --reload
"""

import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import numpy as np
import re
from langdetect import detect, LangDetectException
import json
from rank_bm25 import BM25Okapi

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Document processing
import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import google.generativeai as genai

# Custom embedding class that doesn't rely on external APIs
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document

class MultilingualEmbeddings(Embeddings):
    """Enhanced embedding class that handles multiple languages."""
    
    def __init__(self, dim=200):
        """Initialize with dimension."""
        self.dim = dim
        self.word_to_index = {}
        self.next_index = 0
        
        # Language-specific stopwords
        self.stopwords = {
            'en': {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'at'},
            'ar': {'من', 'إلى', 'عن', 'على', 'في', 'هو', 'هي', 'نحن', 'هم', 'ان', 'كان', 'و', 'ثم', 'او', 'أو'},
            'he': {'של', 'את', 'על', 'הוא', 'היא', 'אני', 'אנחנו', 'הם', 'אתה', 'אתם', 'זה', 'זו', 'אלה'}
        }
        
    def _get_word_index(self, word, lang='en'):
        """Get or create index for a word with language context."""
        # Create a language-specific key to avoid collisions across languages
        lang_word = f"{lang}:{word}"
        
        if lang_word not in self.word_to_index:
            if self.next_index < self.dim:
                self.word_to_index[lang_word] = self.next_index
                self.next_index += 1
            else:
                # If we've reached max dimension, hash the word to an existing index
                self.word_to_index[lang_word] = hash(lang_word) % self.dim
        return self.word_to_index[lang_word]
    
    def _detect_language(self, text):
        """Detect the language of the text."""
        try:
            return detect(text)
        except LangDetectException:
            # Default to English if detection fails
            return 'en'
    
    def _normalize_arabic(self, text):
        """Normalize Arabic text by handling different forms of characters."""
        # Replace different forms of alef
        text = re.sub('[إأآا]', 'ا', text)
        # Replace different forms of yeh
        text = re.sub('[ىي]', 'ي', text)
        # Replace different forms of heh
        text = re.sub('[ةه]', 'ه', text)
        return text
    
    def _normalize_hebrew(self, text):
        """Normalize Hebrew text."""
        # Remove niqqud (vowel points)
        text = re.sub('[\u0591-\u05C7]', '', text)
        return text
    
    def _preprocess_text(self, text, lang=None):
        """Preprocess text based on detected language."""
        if not lang:
            lang = self._detect_language(text)
            
        text = text.lower()
        
        # Language-specific preprocessing
        if lang == 'ar':
            text = self._normalize_arabic(text)
        elif lang == 'he':
            text = self._normalize_hebrew(text)
            
        return text, lang
    
    def _text_to_vector(self, text):
        """Convert text to a vector based on word counts with language awareness."""
        # Detect language and preprocess
        text, lang = self._preprocess_text(text)
        
        # Get stopwords for the language (default to English if not available)
        stops = self.stopwords.get(lang, self.stopwords['en'])
        
        vector = np.zeros(self.dim)
        words = text.split()
        
        for word in words:
            if len(word) > 2 and word not in stops:  # Skip very short words and stopwords
                idx = self._get_word_index(word, lang)
                vector[idx] += 1
                
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()
    
    def embed_documents(self, texts):
        """Embed a list of texts."""
        return [self._text_to_vector(text) for text in texts]
    
    def embed_query(self, text):
        """Embed a query."""
        return self._text_to_vector(text)

# Try to use sentence-transformers for better embeddings if available
try:
    from sentence_transformers import SentenceTransformer
    
    class SentenceTransformerEmbeddings(Embeddings):
        """Sentence Transformer embeddings for multilingual support."""
        
        def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2"):
            """Initialize with model name."""
            try:
                # Try to use a more powerful multilingual model if available
                model_options = [
                    "paraphrase-multilingual-mpnet-base-v2",  # Good general multilingual model
                    "LaBSE",  # Language-agnostic BERT Sentence Embedding
                    "distiluse-base-multilingual-cased-v2"  # Lighter multilingual model
                ]
                
                # Try models in order until one works
                for model in model_options:
                    try:
                        print(f"Trying to load SentenceTransformer model: {model}")
                        self.model = SentenceTransformer(model)
                        self.model_name = model
                        print(f"Successfully loaded SentenceTransformer with model: {model}")
                        break
                    except Exception as e:
                        print(f"Failed to load model {model}: {str(e)}")
                        continue
                
                # If no model was loaded, raise exception
                if not hasattr(self, 'model'):
                    raise ValueError("Could not load any SentenceTransformer model")
                    
            except Exception as e:
                print(f"Error loading SentenceTransformer: {str(e)}")
                # Fall back to custom embeddings
                self.model = None
                self.fallback = MultilingualEmbeddings(dim=384)
                print("Falling back to custom MultilingualEmbeddings")
        
        def _preprocess_text(self, text):
            """Preprocess text for better handling of multilingual content."""
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Handle common encoding issues with Arabic and Hebrew
            text = text.replace('\u200f', '')  # Remove Right-to-Left Mark
            text = text.replace('\u200e', '')  # Remove Left-to-Right Mark
            
            return text
        
        def embed_documents(self, texts):
            """Embed a list of texts."""
            if self.model is None:
                return self.fallback.embed_documents(texts)
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Handle empty texts
            processed_texts = [text if text.strip() else " " for text in processed_texts]
            
            try:
                # Use sentence-transformers for embedding
                embeddings = self.model.encode(processed_texts)
                return embeddings.tolist()
            except Exception as e:
                print(f"Error in SentenceTransformer embedding: {str(e)}")
                # Fall back to custom embeddings
                return self.fallback.embed_documents(texts)
        
        def embed_query(self, text):
            """Embed a query."""
            if self.model is None:
                return self.fallback.embed_query(text)
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Handle empty text
            if not processed_text.strip():
                processed_text = " "
                
            try:
                embedding = self.model.encode([processed_text])[0]
                return embedding.tolist()
            except Exception as e:
                print(f"Error in SentenceTransformer query embedding: {str(e)}")
                # Fall back to custom embeddings
                return self.fallback.embed_query(text)
    
    # Try to use the better embeddings
    embeddings = SentenceTransformerEmbeddings()
    print("Using SentenceTransformerEmbeddings for better multilingual support")
    
except ImportError:
    # Fall back to custom embeddings
    embeddings = MultilingualEmbeddings(dim=384)
    print("Using custom MultilingualEmbeddings for vector embeddings")

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please add it to your .env file.")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Test the API directly
try:
    print("Testing Gemini API directly...")
    
    # Try with gemini-1.5-flash
    test_model = "gemini-1.5-flash"
    print(f"Testing model: {test_model}")
    
    # Test the model directly
    response = genai.GenerativeModel(test_model).generate_content("Hello, are you working?")
    print(f"Direct API test successful with {test_model}. Response: {response.text[:50]}...")
    
    # Use the working model
    generation_model = test_model
    
except Exception as e:
    print(f"Direct API test failed: {str(e)}")
    print("Falling back to model discovery...")
    
    # Continue with model discovery
    try:
        # List available models
        print("Attempting to list available models...")
        models = genai.list_models()
        
        # Preferred model name - use the exact format required by the API
        preferred_model = "gemini-1.5-flash"
        
        # Find a model that supports generateContent
        generation_model = None
        print("Available models:")
        
        # First check if our preferred model is available
        for model in models:
            model_name = model.name
            # Extract just the model name without path prefix if present
            if "/" in model_name:
                model_name = model_name.split("/")[-1]
                
            print(f"- Full name: {model.name}, Short name: {model_name}")
            
            if preferred_model in model_name and "generateContent" in model.supported_generation_methods:
                print(f"- {model.name} (PREFERRED, supports generateContent)")
                # Use the full name as returned by the API
                generation_model = model.name
                break
            elif "generateContent" in model.supported_generation_methods:
                print(f"- {model.name} (supports generateContent)")
            else:
                print(f"- {model.name} (does NOT support generateContent)")
        
        # If preferred model not found, try any model with generateContent
        if not generation_model:
            for model in models:
                if "generateContent" in model.supported_generation_methods:
                    print(f"Selected non-preferred model: {model.name}")
                    generation_model = model.name
                    break
        
        if not generation_model:
            # Use the exact format required by the API
            generation_model = "gemini-1.5-flash"
            print(f"No models with generateContent found, falling back to: {generation_model}")
        else:
            print(f"Selected generation model: {generation_model}")
        
    except Exception as e:
        # Use the exact format required by the API
        generation_model = "gemini-1.5-flash"
        print(f"Error listing models, falling back to: {generation_model}")
        print(f"Original error: {str(e)}")

app = FastAPI(title="Advanced Multilingual RAG Chatbot API")

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global storage for user sessions
sessions = {}

class ChatQuery(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    detected_languages: Optional[Dict[str, str]] = None

# Function to create parent-child relationships between chunks
def create_document_hierarchy(documents):
    """Create parent-child relationships between document chunks for better context."""
    parent_docs = {}
    child_docs = []
    
    for i, doc in enumerate(documents):
        # Create a unique parent ID based on the document source
        parent_id = f"{doc.metadata.get('source', 'unknown')}_{i//5}"  # Group every 5 chunks under one parent
        
        if parent_id not in parent_docs:
            # Create parent document
            parent_docs[parent_id] = Document(
                page_content=doc.page_content,  # Start with the first chunk's content
                metadata={
                    **doc.metadata,
                    "is_parent": True,
                    "parent_id": parent_id,
                    "children_ids": []
                }
            )
        else:
            # Append content to parent (limited to avoid too large documents)
            current_len = len(parent_docs[parent_id].page_content)
            if current_len < 5000:  # Limit parent size
                parent_docs[parent_id].page_content += f"\n\n{doc.page_content}"
        
        # Create child document with reference to parent
        child_doc = Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "is_child": True,
                "parent_id": parent_id,
                "chunk_id": i
            }
        )
        
        # Add child ID to parent's children list
        parent_docs[parent_id].metadata["children_ids"].append(i)
        
        # Add to child documents list
        child_docs.append(child_doc)
    
    return list(parent_docs.values()), child_docs

# Function to process a file and return document chunks
def process_file(file_path, temp_dir):
    """Process a file based on its extension and return document chunks"""
    file_extension = Path(file_path).suffix.lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        elif file_extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.json':
            # For JSON files, we'll read and convert to text
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
            # Write to a temporary text file
            json_text_path = os.path.join(temp_dir, "converted.txt")
            with open(json_text_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            loader = TextLoader(json_text_path, encoding='utf-8')
        elif file_extension in ['.md', '.markdown']:
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load documents
        documents = loader.load()
        
        # Check if documents are empty
        if not documents or not documents[0].page_content.strip():
            print(f"Warning: Empty document content from {file_path}")
            return [], [], 'en'
        
        # Detect document language
        doc_language = 'en'  # Default to English
        if documents and documents[0].page_content:
            try:
                # Sample the first 1000 characters for language detection
                sample_text = documents[0].page_content[:1000]
                doc_language = detect(sample_text)
                print(f"Detected document language: {doc_language}")
            except LangDetectException:
                print("Could not detect document language, defaulting to English")
        
        # Create text splitter with appropriate settings for the language
        # For RTL languages like Arabic and Hebrew, we need different chunking parameters
        is_rtl = doc_language in ['ar', 'he', 'fa', 'ur']
        
        # Adjust chunk size and overlap based on language
        chunk_size = 800 if is_rtl else 1000
        chunk_overlap = 150 if is_rtl else 200
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Add common separators for the detected language
            separators=["\n\n", "\n", ". ", " ", ""] if not is_rtl else 
                      ["\n\n", "\n", ".", " ", "،", "؛", ":", ""]
        )
        
        # Split documents into chunks
        document_chunks = text_splitter.split_documents(documents)
        
        # Add source filename to metadata if not present
        for chunk in document_chunks:
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = file_path
        
        # Create parent-child relationships
        parent_docs, child_docs = create_document_hierarchy(document_chunks)
        
        return parent_docs, child_docs, doc_language
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty results
        return [], [], 'en'

# Function to create BM25 index for keyword search
def create_bm25_index(documents):
    """Create a BM25 index for keyword search"""
    # Extract document texts
    texts = [doc.page_content for doc in documents]
    
    # Tokenize texts
    tokenized_texts = [text.lower().split() for text in texts]
    
    # Create BM25 index
    bm25_index = BM25Okapi(tokenized_texts)
    
    return bm25_index

# Function to perform hybrid search (combining semantic and keyword search)
def hybrid_search(query, vector_store, bm25_index, documents, k=5):
    """Perform hybrid search combining semantic and keyword search"""
    # Semantic search
    semantic_results = vector_store.similarity_search_with_score(query, k=k)
    
    # Keyword search
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Normalize BM25 scores
    max_bm25_score = max(bm25_scores) if bm25_scores.any() else 1.0
    normalized_bm25_scores = bm25_scores / max_bm25_score if max_bm25_score > 0 else bm25_scores
    
    # Combine scores (weighted sum)
    hybrid_results = []
    for i, (doc, semantic_score) in enumerate(semantic_results):
        # Safely find the document index, or use a default score if not found
        try:
            doc_idx = documents.index(doc)
            bm25_score = normalized_bm25_scores[doc_idx] if doc_idx < len(normalized_bm25_scores) else 0
        except ValueError:
            # Document not in list, use default BM25 score
            print(f"Warning: Document {doc.metadata.get('chunk_id', 'unknown')} not found in document list")
            bm25_score = 0
        
        # Combine scores (70% semantic, 30% keyword)
        combined_score = (0.7 * (1 - semantic_score)) + (0.3 * bm25_score)
        
        hybrid_results.append((doc, combined_score))
    
    # Sort by combined score (descending)
    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return [doc for doc, _ in hybrid_results[:k]]

# Function to re-rank search results based on query relevance
def rerank_results(query, docs, parent_docs_map=None):
    """Re-rank search results based on query relevance and parent document context"""
    if not docs:
        return []
    
    # If we have parent documents, include them for context
    if parent_docs_map:
        enhanced_docs = []
        added_parents = set()
        
        for doc in docs:
            # Add the document itself
            enhanced_docs.append(doc)
            
            # Check if it's a child document and has a parent
            if doc.metadata.get("is_child") and "parent_id" in doc.metadata:
                parent_id = doc.metadata["parent_id"]
                
                # Add parent if not already added
                if parent_id in parent_docs_map and parent_id not in added_parents:
                    enhanced_docs.append(parent_docs_map[parent_id])
                    added_parents.add(parent_id)
        
        # Use the enhanced document set
        docs = enhanced_docs
    
    # Simple query term matching for re-ranking
    query_terms = set(query.lower().split())
    
    # Calculate relevance scores
    scored_docs = []
    for doc in docs:
        content = doc.page_content.lower()
        
        # Count query terms in document
        term_matches = sum(1 for term in query_terms if term in content)
        
        # Calculate term density (matches per length)
        term_density = term_matches / (len(content) + 1)  # Add 1 to avoid division by zero
        
        # Check for exact phrase match
        exact_match = 1.5 if query.lower() in content else 1.0
        
        # Calculate final score
        score = term_density * exact_match
        
        # Add parent document bonus
        if doc.metadata.get("is_parent", False):
            score *= 1.2
        
        scored_docs.append((doc, score))
    
    # Sort by score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return re-ranked documents
    return [doc for doc, _ in scored_docs]

@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(None)):
    """
    Upload one or more files to be processed by the RAG system
    """
    # Validate files
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create a unique session ID
    session_id = str(uuid.uuid4())
    
    # Create temp directory for file processing
    temp_dir = tempfile.mkdtemp()
    
    try:
        all_parent_docs = []
        all_child_docs = []
        file_names = []
        languages = []
        
        # Process each file
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process the file
            try:
                parent_docs, child_docs, doc_language = process_file(file_path, temp_dir)
                all_parent_docs.extend(parent_docs)
                all_child_docs.extend(child_docs)
                file_names.append(file.filename)
                languages.append(doc_language)
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                # Continue with other files if one fails
        
        if not all_child_docs:
            shutil.rmtree(temp_dir)
            raise HTTPException(status_code=400, detail="No valid files were processed")
        
        # Create parent document map for quick lookup
        parent_docs_map = {doc.metadata["parent_id"]: doc for doc in all_parent_docs}
        
        # Create vector store using our embeddings
        vector_store = FAISS.from_documents(all_child_docs, embeddings)
        
        # Create BM25 index for keyword search
        bm25_index = create_bm25_index(all_child_docs)
        
        # Store session data
        sessions[session_id] = {
            "vector_store": vector_store,
            "bm25_index": bm25_index,
            "documents": all_child_docs,
            "parent_docs_map": parent_docs_map,
            "file_names": file_names,
            "temp_dir": temp_dir,
            "chat_history": [],  # Store chat history
            "document_languages": languages
        }
        
        # Use the first detected language as the primary one for response
        primary_language = languages[0] if languages else "en"
        
        return {
            "session_id": session_id, 
            "message": f"Successfully processed {len(file_names)} files", 
            "file_names": file_names,
            "detected_language": primary_language
        }
        
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/ask", response_model=ChatResponse)
async def ask_question(chat_query: ChatQuery):
    """
    Ask a question about the uploaded document
    """
    session_id = chat_query.session_id
    query = chat_query.query
    
    # Check if session exists
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")
    
    try:
        # Get session data
        session_data = sessions[session_id]
        vector_store = session_data["vector_store"]
        bm25_index = session_data["bm25_index"]
        documents = session_data["documents"]
        parent_docs_map = session_data["parent_docs_map"]
        chat_history = session_data["chat_history"]
        document_languages = session_data.get("document_languages", ["en"])
        primary_language = document_languages[0] if document_languages else "en"
        
        # Detect query language
        query_language = 'en'  # Default to English
        try:
            query_language = detect(query)
            print(f"Detected query language: {query_language}")
        except LangDetectException:
            print("Could not detect query language, defaulting to English")
        
        # Perform hybrid search with better error handling
        try:
            retrieved_docs = hybrid_search(query, vector_store, bm25_index, documents, k=8)
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            # Fall back to simple vector search
            retrieved_docs = vector_store.similarity_search(query, k=5)
        
        # Re-rank results with error handling
        try:
            ranked_docs = rerank_results(query, retrieved_docs, parent_docs_map)
            # Limit to top 5 after re-ranking
            top_docs = ranked_docs[:5]
        except Exception as e:
            print(f"Error in re-ranking: {str(e)}")
            # Use retrieved docs directly if re-ranking fails
            top_docs = retrieved_docs[:5]
        
        # Extract content from documents
        context = "\n\n".join([doc.page_content for doc in top_docs])
        
        # Format chat history for context
        formatted_history = ""
        if chat_history:
            for entry in chat_history:
                role = entry["role"]
                content = entry["content"]
                formatted_history += f"{role.upper()}: {content}\n\n"
        
        # Append to chat history
        chat_history.append({"role": "user", "content": query})
        
        # Create a system prompt that handles multilingual context and includes chat history
        system_prompt = f"""
        You are an advanced multilingual AI assistant that answers questions based on provided documents.
        
        The documents are in multiple languages, primarily {primary_language}.
        The user query is in {query_language} language.
        
        When answering:
        1. If the document and query languages differ, first understand the document content, then provide your answer in the same language as the user's query.
        2. Pay special attention to names, dates, numbers, and technical terms, preserving them accurately across languages.
        3. If you encounter text in Arabic, Hebrew, or other right-to-left languages, ensure proper handling of these scripts.
        4. If you're unsure about a translation, include both the original term and your best translation.
        5. Always base your answers solely on the provided document content.
        6. IMPORTANT: Consider the chat history to maintain context between questions.
        7. If the answer is not in the documents, say so politely. Do not make up information.
        8. Be concise and direct in your answers.
        """
        
        # Create prompt with context, chat history, and query
        user_prompt = f"""
        Previous conversation:
        {formatted_history}
        
        Context information is below:
        {context}
        
        Given the context information and the conversation history, answer the question: {query}
        """
        
        # Call Gemini API directly
        print(f"Calling Gemini API with model: {generation_model}")
        gemini_model = genai.GenerativeModel(generation_model)
        
        # Combine system prompt and user prompt since system role is not supported
        combined_prompt = f"""{system_prompt}

        {user_prompt}"""
        
        try:
            # Use combined prompt instead of separate system and user messages
            response = gemini_model.generate_content(combined_prompt)
            answer = response.text
        except Exception as api_error:
            print(f"Gemini API error: {str(api_error)}")
            # Try with a simpler prompt if the first attempt fails
            try:
                simplified_prompt = f"""
                Answer this question based on the provided context:
                
                Context: {context}
                
                Question: {query}
                
                Answer in the same language as the question.
                """
                response = gemini_model.generate_content(simplified_prompt)
                answer = response.text
            except Exception as fallback_error:
                print(f"Fallback prompt also failed: {str(fallback_error)}")
                # Return a generic error message in the detected query language
                if query_language == 'ar':
                    answer = "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى بصياغة مختلفة."
                elif query_language == 'he':
                    answer = "סליחה, אירעה שגיאה בעיבוד הבקשה שלך. אנא נסה שוב עם ניסוח אחר."
                else:
                    answer = "Sorry, an error occurred while processing your request. Please try again with a different phrasing."
        
        # Append AI response to chat history
        chat_history.append({"role": "assistant", "content": answer})
        
        # Save updated chat history
        sessions[session_id]["chat_history"] = chat_history
        
        # Extract sources
        sources = []
        for doc in top_docs:
            # Get parent info if this is a child document
            parent_info = {}
            if doc.metadata.get("is_child") and "parent_id" in doc.metadata:
                parent_id = doc.metadata["parent_id"]
                if parent_id in parent_docs_map:
                    parent = parent_docs_map[parent_id]
                    parent_info = {
                        "has_parent": True,
                        "parent_id": parent_id,
                        "parent_excerpt": parent.page_content[:100] + "..." if len(parent.page_content) > 100 else parent.page_content
                    }
            
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": {**doc.metadata, **parent_info}
            })
        
        return ChatResponse(
            answer=answer, 
            sources=sources,
            detected_languages={
                "document": primary_language,
                "query": query_language
            }
        )
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        error_message = f"Error generating answer: {str(e)}"
        # Truncate error message if too long
        if len(error_message) > 500:
            error_message = error_message[:500] + "..."
        raise HTTPException(status_code=500, detail=error_message)

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and clean up temporary files
    """
    if session_id in sessions:
        # Clean up temp directory
        temp_dir = sessions[session_id].get("temp_dir")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Remove session from memory
        del sessions[session_id]
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_backend:app", host="0.0.0.0", port=8000, reload=True) 