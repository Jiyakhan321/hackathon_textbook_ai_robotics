---
title: Hackathon Chatbot Backend
emoji: 🤖
colorFrom: purple
colorTo: red
sdk: docker
secrets:
  - OPENROUTER_API_KEY
  - OPENAI_API_KEY
  - QDRANT_API_KEY
  - QDRANT_URL
  - DATABASE_URL
persistence: true
---

# RAG Chatbot Backend

This is the backend service for the RAG (Retrieval-Augmented Generation) chatbot that answers questions about your book content.

## Features

- `/health` - Health check endpoint
- `/chat` - General questions about the book
- `/chat/selected` - Questions based only on user-selected text

## Tech Stack

- FastAPI - Web framework
- Qdrant Cloud - Vector database for embeddings
- OpenRouter/OpenAI - LLM for responses
- Sentence Transformers - Qwen-like embeddings

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

## Deployment to HuggingFace Spaces

This backend is designed to run on HuggingFace Spaces with the following configuration:
- Port: 7860 (configured in main.py)
- Dockerfile included for containerization
- Environment variables set as HuggingFace Secrets

## API Endpoints

### GET /health
Health check endpoint that returns `{"status":"ok"}`

### POST /chat
Accepts a JSON body:
```json
{
  "query": "Your question here",
  "context": null,
  "selected_text": null
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