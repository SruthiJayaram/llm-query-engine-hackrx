# HackRx LLM Query Engine

A FastAPI-based document question answering system that uses semantic search and OpenAI's GPT models to answer questions based on PDF documents.

## Features

- 🔗 **Working API Endpoint**: Public HTTPS URL that receives documents + questions, returns answers
- ⚙️ **Correct Endpoint Path**: `/hackrx/run` using POST method
- 🧠 **LLM-based Logic**: Semantic search with embeddings + GPT-4 for intelligent answers
- 📜 **Proper JSON Response**: Returns structured JSON with answers array
- ✅ **Bearer Auth Header**: Reads `Authorization: Bearer <token>` header
- ⏱ **Under 30s Response Time**: Optimized for fast response times

## Tech Stack

- **FastAPI**: Web framework for building APIs
- **OpenAI API**: For embeddings and LLM responses (GPT-3.5-turbo)
- **FAISS**: Vector search for semantic similarity
- **PyMuPDF**: PDF text extraction
- **Python-dotenv**: Environment variable management

## Local Setup

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd llm-query-engine-hackrx
```

### 2. Set up Python Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
cd hackrx_app
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the `hackrx_app` directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run Locally
```bash
uvicorn main:app --reload --port 8000
```

### 6. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Test main endpoint
curl -X POST http://localhost:8000/hackrx/run \
  -H "Authorization: Bearer test-token-123" \
  -H "Content-Type: application/json" \
  -d '{"documents":"https://example.com/sample.pdf","questions":["What is this document about?"]}'
```

## Deployment on Render.com

### 1. Push to GitHub
```bash
git add .
git commit -m "Add HackRx LLM Query Engine"
git push origin main
```

### 2. Deploy on Render
1. Go to [render.com](https://render.com)
2. Create a new **Web Service**
3. Connect your GitHub repository
4. Use these settings:
   - **Build Command**: `cd hackrx_app && pip install -r requirements.txt`
   - **Start Command**: `cd hackrx_app && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**:
     - `OPENAI_API_KEY`: Your OpenAI API key

### 3. Get Your Public URL
After deployment, you'll get a URL like:
```
https://your-app-name.onrender.com
```

Your API endpoint will be:
```
https://your-app-name.onrender.com/hackrx/run
```

## API Usage

### Endpoint
- **URL**: `/hackrx/run`
- **Method**: POST
- **Headers**: 
  - `Authorization: Bearer <your-token>`
  - `Content-Type: application/json`

### Request Format
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key findings?",
    "What is the conclusion?"
  ]
}
```

### Response Format
```json
{
  "answers": [
    "The main topic is...",
    "The key findings are...",
    "The conclusion states..."
  ]
}
```

## Project Structure
```
hackrx_app/
├── main.py          # FastAPI app and endpoints
├── utils.py         # Core processing functions
├── requirements.txt # Python dependencies
├── .env            # Environment variables
└── server.log      # Server logs (when running)
```

## How It Works

1. **Document Processing**: Downloads PDF from URL and extracts text
2. **Text Chunking**: Splits document into manageable chunks (500 words each)
3. **Embedding Creation**: Uses OpenAI's text-embedding-3-small to create vector embeddings
4. **Semantic Search**: Uses FAISS to find most relevant chunks for each question
5. **LLM Processing**: Sends relevant context + question to GPT-3.5-turbo for answer generation
6. **Response**: Returns structured JSON with all answers

## Testing
Use the included `test_api.sh` script:
```bash
./test_api.sh
```

## Error Handling
- ✅ Invalid/missing authorization tokens
- ✅ PDF download failures
- ✅ Text extraction errors
- ✅ OpenAI API errors
- ✅ Timeout handling (30s limit)

## Next Steps for Production
- [ ] Add caching for frequently accessed documents
- [ ] Implement async processing for multiple documents
- [ ] Add support for more document formats (DOCX, TXT)
- [ ] Upgrade to GPT-4 for better accuracy
- [ ] Add rate limiting and request validation
- [ ] Implement proper logging and monitoring
