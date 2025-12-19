---
id: 1
title: create-rag-chatbot-constitution
stage: constitution
date_iso: 2025-12-18
surface: agent
model: Qwen Code
feature: rag-chatbot
branch: main
user: user
command: /sp.constitution
labels: ["constitution", "rag", "chatbot", "cohere", "backend"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - ".specify/memory/constitution.md"
tests: []
outcome: "Successfully created the project constitution for the Integrated RAG Chatbot project with specific principles for faithfulness, zero hallucination, context priority, transparency, modularity, and LLM provider constraints."
evaluation: "Constitution document created successfully with all specified principles and constraints. Template consistency verified across dependent artifacts."
---

# Constitution Creation for Integrated RAG Chatbot

## User Input

Project: Integrated RAG Chatbot for an AI Textbook (Cohere-based)

Purpose:
Design and implement a production-grade Retrieval-Augmented Generation (RAG) chatbot
embedded directly inside a published book website. The chatbot must answer user questions
strictly using the book’s content and optional user-selected text, without relying on
external or general knowledge.

Core Principles:
- Faithfulness: Responses must be grounded ONLY in retrieved book content
- Zero Hallucination: If information is not present in the retrieved context, explicitly say
  "This information is not available in the book"
- Context Priority: User-selected text always overrides global book context
- Transparency: Clearly indicate when an answer is based on selected text vs full book
- Modularity: Each component (LLM, retrieval, storage) must be swappable

LLM & AI Constraints:
- LLM Provider: Cohere ONLY (Command / Command-R / Embed models)
- OpenAI APIs are strictly forbidden
- Use Cohere Embeddings for vector generation
- Use Cohere Chat / Generate for response synthesis
- Temperature must be kept low (≤ 0.3) for factual accuracy

RAG Architecture Rules:
- Book content must be chunked with semantic boundaries (headings, paragraphs)
- Every chunk must store:
  - source (chapter, section, page)
  - chunk_id
- Vector storage must use Qdrant Cloud
- Metadata (users, queries, logs) must be stored in Neon Serverless Postgres
- Retrieval must be top-k with similarity threshold enforcement

Selected Text Mode:
- When user provides selected text:
  - Ignore the rest of the book corpus
  - Answer strictly from the selected text
  - Do NOT enrich with other chapters or external info
- If the answer cannot be derived, respond with:
  "The selected text does not contain enough information to answer this question."

Backend Standards:
- Backend framework: FastAPI
- Async-first design
- Clear API separation:
  - /query (normal RAG)
  - /query/selected-text (restricted RAG)
- Input validation and response schema required

Security & Reliability:
- API keys must be stored in environment variables
- No secrets hardcoded
- Graceful handling of:
  - empty retrieval
  - low similarity scores
  - malformed input

User Experience Rules:
- Responses must be:
  - Clear
  - Concise
  - Structured (bullet points when useful)
- No speculative or creative writing
- No emojis or casual tone
- Academic / technical clarity

Logging & Observability:
- Log:
  - user query
  - retrieved chunk IDs
  - similarity scores
- Do NOT log raw book text in plaintext logs

Success Criteria:
- 100% answers traceable to retrieved book chunks
- Zero hallucinated facts
- Selected-text questions never reference outside content
- System is deployable on free tiers (Qdrant + Neon)
- Chatbot can be embedded in a book website via iframe or JS SDK

## Response

Successfully created the Integrated RAG Chatbot Constitution with all specified principles. The constitution includes:

- Core Principles: Faithfulness, Zero Hallucination, Context Priority, Transparency, Modularity, and LLM Provider Constraint
- Additional Constraints: RAG Architecture Requirements, Selected Text Mode Operation, Backend Standards, Security & Reliability, User Experience Requirements, and Logging & Observability
- Development Workflow and Success Criteria
- Governance section with compliance requirements

The constitution is stored at `.specify/memory/constitution.md` with version 1.0.0, ratified on 2025-12-18.