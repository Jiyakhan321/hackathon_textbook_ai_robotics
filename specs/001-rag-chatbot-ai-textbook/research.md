# Research: RAG Chatbot for AI Textbook

## Overview
This research document details the technical decisions, best practices, and patterns needed for implementing the RAG chatbot for an AI textbook. The implementation must adhere to all constitutional requirements including zero hallucination, faithfulness to source material, and use of Cohere APIs exclusively.

## Decision: Chunking Strategy
**Decision**: Use semantic chunking based on document structure (chapters, sections, paragraphs)
**Rationale**: Semantic chunking maintains context and meaning better than fixed-length chunks, which is critical for the Zero Hallucination requirement. This approach aligns with the constitutional requirement that "Book content must be chunked with semantic boundaries (headings, paragraphs)".
**Alternatives considered**:
- Fixed-length token chunks: Risk breaking up related concepts
- Sentence-level chunks: Might create too many disconnected pieces
- Page-level chunks: Might be too large and dilute relevance

## Decision: Embedding Model Selection
**Decision**: Use Cohere's multilingual-22-12 embedding model
**Rationale**: Cohere embeddings are required by the constitution, and the multilingual model will handle diverse textbook content effectively. This ensures compliance with "Use Cohere Embeddings for vector generation".
**Alternatives considered**:
- Sentence-transformers models: Not compliant with LLM Provider Constraint
- OpenAI embeddings: Forbidden by constitution

## Decision: Vector Database Configuration
**Decision**: Configure Qdrant with cosine similarity and enforce minimum similarity thresholds
**Rationale**: Cosine similarity is ideal for semantic search, and configurable thresholds directly support the "similarity threshold enforcement" requirement from the constitution.
**Implementation details**: Set threshold to 0.7 minimum for results to be considered relevant

## Decision: Retrieval Strategy
**Decision**: Implement top-k retrieval (k=5) with re-ranking approach
**Rationale**: Top-k provides a balance between performance and accuracy, while re-ranking ensures the most relevant chunks are used first, supporting the Faithfulness requirement.
**Alternatives considered**:
- Simple top-k without re-ranking: Might miss more relevant content
- Dynamic k based on query: More complex without clear benefit

## Decision: Cohere Model Selection
**Decision**: Use Cohere Command-R+ for generation
**Rationale**: Command-R+ offers strong reasoning capabilities needed for complex textbook questions while maintaining factual accuracy, supporting Zero Hallucination requirement.
**Alternatives considered**:
- Command-light: Less reasoning capability
- OpenAI models: Forbidden by constitution

## Decision: Response Validation
**Decision**: Implement strict content validation that compares generated responses to source chunks
**Rationale**: Essential for meeting Zero Hallucination requirement by ensuring all information in responses is directly supported by retrieved content.
**Implementation**: Use text overlap metrics and semantic similarity between response and source chunks

## Decision: API Design Pattern
**Decision**: Use RESTful API design with clear separation between full-book and selected-text endpoints
**Rationale**: Clean separation ensures Context Priority principle is enforced at the API level.
**Endpoints**:
- `POST /query` - Full book RAG
- `POST /query/selected-text` - Selected text only RAG

## Decision: Frontend Integration Method
**Decision**: Provide both iframe and JavaScript SDK embedding options
**Rationale**: Offers flexibility for different textbook website implementations while meeting the requirement that "Chatbot can be embedded in a book website via iframe or JS SDK".
**Implementation**: Offer lightweight widget that can be embedded via script tag or iframe

## Decision: Rate Limiting Strategy
**Decision**: Implement token-based rate limiting at 10 requests per minute per IP
**Rationale**: Prevents abuse while allowing reasonable usage for textbook readers.
**Implementation**: Use in-memory sliding window for simplicity in initial implementation

## Decision: Error Handling for External Dependencies
**Decision**: Implement graceful degradation when Cohere API is unavailable
**Rationale**: Ensures system remains usable even when external services have issues.
**Implementation**: Return informative error messages and cache previous successful responses where appropriate

## Decision: Data Privacy and Logging
**Decision**: Log query metadata without content, implement strict data retention
**Rationale**: Complies with the logging requirement to "NOT log raw book text in plaintext logs" while still maintaining observability.
**Implementation**: Log only query timestamps, user IDs, chunk IDs, and similarity scores - never the actual chunk content or book text