# Example API Requests for RAG Chatbot

## Testing the API

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

Response:
```json
{
  "status": "ok"
}
```

### General Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type": "application/json" \
  -d '{
    "query": "What are the main topics covered in this book?",
    "context": "Optional additional context for the query"
  }'
```

Response:
```json
{
  "response": "The book covers topics such as...",
  "sources": [
    {
      "title": "Chapter 1: Introduction",
      "source": "book_content.md"
    }
  ],
  "query": "What are the main topics covered in this book?"
}
```

### Selected Text Chat
```bash
curl -X POST http://localhost:8000/chat/selected \
  -H "Content-Type": "application/json" \
  -d '{
    "query": "Explain this concept in simpler terms?",
    "selected_text": "The complex concept of embodied intelligence involves..."
  }'
```

Response:
```json
{
  "response": "In simpler terms, embodied intelligence means...",
  "sources": [
    {
      "title": "Selected Text",
      "source": "User Selection"
    }
  ],
  "query": "Explain this concept in simpler terms?"
}
```

## Environment Variables for Frontend

For the frontend to connect to your backend, you'll need to set the backend URL. In your Docusaurus environment:

```bash
# In your .env file or environment
REACT_APP_BACKEND_URL=https://your-railway-app-name.railway.app
```

## Deployment to Railway

1. Create a new Railway project
2. Connect your GitHub repository
3. Set the build directory to `chatbot-backend`
4. Add environment variables:
   - DATABASE_URL
   - QDRANT_URL
   - QDRANT_API_KEY
   - OPENAI_API_KEY
   - COHERE_API_KEY
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Your API will be available at: `https://your-railway-app-name.railway.app`