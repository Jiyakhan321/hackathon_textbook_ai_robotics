# RAG Chatbot Backend

This is the backend service for the RAG (Retrieval-Augmented Generation) chatbot that answers questions about your book content.

## Features

- `/health` - Health check endpoint
- `/chat` - General questions about the book
- `/chat/selected` - Questions based only on user-selected text

## Tech Stack

- FastAPI - Web framework
- Qdrant Cloud - Vector database for embeddings
- Neon Serverless Postgres - Metadata and session storage
- OpenAI/Cohere - LLM for responses

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
5. Fill in your environment variables in `.env`

## Running Locally

```bash
uvicorn main:app --reload
```

The server will start on `http://localhost:8000`

## Deployment to Railway

1. Connect your GitHub repository to Railway
2. Set the following build configuration:
   - Root Directory: `chatbot-backend`
3. Add environment variables in Railway dashboard:
   - DATABASE_URL
   - QDRANT_URL
   - QDRANT_API_KEY
   - OPENAI_API_KEY
   - COHERE_API_KEY
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## API Endpoints

### GET /health
Health check endpoint that returns `{"status":"ok"}`

### POST /chat
Accepts a JSON body:
```json
{
  "query": "Your question here",
  "context": "Optional additional context"
}
```

### POST /chat/selected
Accepts a JSON body:
```json
{
  "query": "Your question about the selected text",
  "selected_text": "The text that was selected by the user"
}
```