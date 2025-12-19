# Quickstart: RAG Chatbot for AI Textbook

## Overview
This guide helps you quickly set up and start using the RAG Chatbot for AI Textbook. The implementation follows all constitutional requirements including zero hallucination, faithfulness to source material, and exclusive use of Cohere APIs.

## Prerequisites
- Python 3.11+
- Pip package manager
- Git
- Access to Cohere API (https://dashboard.cohere.ai/)
- Access to Qdrant Cloud account
- Access to Neon Serverless Postgres account

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rag-chatbot-ai-textbook
```

### 2. Set Up Backend Environment
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # Or use your preferred editor
```

Required environment variables:
```
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DB_URL=your_neon_postgres_connection_string
SECRET_KEY=your_secret_key_for_fastapi
```

### 4. Set Up Vector Database (Qdrant)
```bash
# Run the book ingestion script
python scripts/ingest_book.py --book-path path/to/your/book/content
```

### 5. Start the Backend Server
```bash
# From the backend directory
uvicorn src.api.main:app --reload --port 8000
```

## Basic Usage

### 1. Query Full Book Content
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "question": "What are attention mechanisms in transformers?"
  }'
```

### 2. Query Selected Text Only
```bash
curl -X POST http://localhost:8000/query/selected-text \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "question": "What does this text explain about attention?",
    "selected_text": "Attention mechanisms allow the model to focus on different parts of the input sequence when producing an output. This is accomplished through a weighted sum of the values, where the weights are computed by a compatibility function of the query with the corresponding key."
  }'
```

### 3. Expected Response Format
```json
{
  "answer": "Attention mechanisms allow the model to focus on different parts of the input sequence...",
  "sources": [
    {
      "chunk_id": "chunk-001",
      "chapter": "Transformer Architecture",
      "section": "Attention Mechanisms",
      "page": 45
    }
  ],
  "confidence": 0.92,
  "context_mode": "full_book",
  "timestamp": "2025-12-18T10:30:00Z"
}
```

## Frontend Integration

### 1. Using the JavaScript SDK
```html
<!DOCTYPE html>
<html>
<head>
    <title>Textbook with Chatbot</title>
</head>
<body>
    <div id="textbook-content">
        <!-- Your textbook content -->
    </div>
    
    <!-- Include the chatbot widget -->
    <script src="path/to/chatbot-sdk.js"></script>
    <div id="chatbot-container"></div>
    
    <script>
        // Initialize the chatbot
        ChatbotWidget.init({
            containerId: 'chatbot-container',
            apiUrl: 'http://localhost:8000',
            apiKey: 'your-api-key'
        });
    </script>
</body>
</html>
```

### 2. Using an iFrame
```html
<iframe 
    src="http://localhost:8000/widget" 
    width="400" 
    height="600"
    frameborder="0">
</iframe>
```

## Development

### Running Tests
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/
```

### Adding Book Content
```bash
# Re-ingest book with updated content
python scripts/ingest_book.py --book-path path/to/updated/book --force-reindex
```

## Production Deployment

### Backend Deployment
The application is designed to run on platforms like:
- Railway
- Render
- Vercel (functions)
- AWS/Azure cloud run

Remember to set environment variables in your deployment platform.

### API Rate Limits
- Default: 10 requests per minute per IP
- Configurable in settings

## Troubleshooting

### Common Issues
1. **"This information is not available in the book" for all queries**
   - Check if the book content was properly ingested
   - Verify similarity thresholds in the configuration

2. **High response times**
   - Check Cohere API response times
   - Verify Qdrant Cloud connectivity

3. **Cohere API errors**
   - Validate your API key in environment variables
   - Check your Cohere account limits

### Logging
- Application logs are written to standard output
- Query metadata is stored in Neon Postgres (QueryLog table)
- Raw book content is never logged (per constitutional requirements)

## Compliance Verification
This implementation complies with all constitutional requirements:
✓ Responses grounded only in book content
✓ Zero hallucination with explicit fallback responses
✓ Selected text mode isolated from global corpus
✓ Cohere API only (no OpenAI)
✓ Proper source attribution
✓ No raw book text in logs