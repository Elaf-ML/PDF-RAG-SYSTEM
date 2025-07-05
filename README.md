# Advanced Document AI Chat - RAG Chatbot

A full-stack advanced RAG-based chatbot web application that allows users to upload files (PDF, CSV, TXT, JSON, DOCX), analyze and index the content using a state-of-the-art RAG (Retrieval-Augmented Generation) system powered by Gemini API, and ask questions in a chat interface with persistent conversation history.

## Features

- Upload and process multiple file types (PDF, CSV, TXT, JSON, DOCX)
- Advanced RAG techniques:
  - Hybrid search (combining semantic and keyword search)
  - Parent-child document relationships for better context
  - Re-ranking of search results for improved relevance
  - Persistent chat history for contextual conversations
- Intelligent document chunking for semantic search
- Vector storage using FAISS
- Gemini API integration for AI responses
- Modern, responsive UI with Tailwind CSS and Shadcn/UI
- Dark/light mode support
- Animated chat interface with framer-motion
- Session persistence (chat history is saved between browser sessions)
- Multilingual support for documents and queries

## Project Structure

```
.
├── backend/
│   ├── env.example        # Example environment variables
│   └── rag_backend.py     # FastAPI backend with advanced RAG
├── frontend/
│   ├── public/            # Public assets
│   └── src/
│       ├── components/    # React components
│       │   ├── ChatInterface.jsx
│       │   ├── ChatMessage.jsx
│       │   ├── FileUpload.jsx
│       │   └── ui/        # UI components
│       ├── App.jsx        # Main React component
│       ├── App.css        # Styles
│       └── main.jsx       # React entry point
└── README.md              # This file
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install fastapi uvicorn python-multipart pydantic langchain langchain-google-genai faiss-cpu pypdf pandas python-docx python-dotenv langdetect rank_bm25 sentence-transformers
   ```

4. Create a `.env` file in the backend directory with your Gemini API key (copy from env.example):
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ```

5. Run the backend server:
   ```
   uvicorn rag_backend:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

4. Open your browser and navigate to http://localhost:5173

## Usage

1. Upload one or more documents (PDF, CSV, TXT, JSON, or DOCX)
2. Wait for processing to complete (progress bar will indicate status)
3. Ask questions about the documents in the chat interface
4. View AI-generated answers with source references
5. Click on source references to view detailed excerpts from the documents
6. Your chat history will persist even if you close the browser

## Advanced RAG Features Explained

### Hybrid Search
The system combines traditional semantic search (using vector embeddings) with keyword search (using BM25 algorithm) to find the most relevant document chunks, ensuring better results than either method alone.

### Parent-Child Document Relationships
Documents are chunked for efficient retrieval, but the system maintains relationships between chunks and their parent documents, allowing the AI to access broader context when needed.

### Re-ranking
After initial retrieval, documents are re-ranked based on multiple factors including:
- Query term density
- Exact phrase matching
- Parent document relevance

### Chat History
The system maintains conversation history and uses it to provide contextual responses. This allows follow-up questions without repeating context.

## Technologies Used

- **Backend**: Python, FastAPI, LangChain, FAISS, Gemini API, BM25
- **Frontend**: React, Vite, Tailwind CSS, Shadcn/UI, Framer Motion
- **Document Processing**: PyPDF, pandas, python-docx
- **Embeddings**: SentenceTransformers (with fallback to custom embeddings)
- **Session Management**: LocalStorage for client-side persistence

## Notes

- The application stores session data in memory on the server and in localStorage on the client
- For production use, consider adding a database for persistent storage
- The Gemini API key is required and must be obtained from https://aistudio.google.com/app/apikey
- For best results with multilingual documents, ensure the sentence-transformers package is installed 