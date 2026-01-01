# Feature Specification: RAG Chatbot for AI Textbook

**Feature Branch**: `001-rag-chatbot-ai-textbook`
**Created**: 2025-12-18
**Status**: Draft
**Input**: User description: "Integrated RAG Chatbot Embedded in AI Textbook (Cohere-based) Objective: Specify and build a high-quality, production-grade Retrieval-Augmented Generation (RAG) chatbot embedded inside a published AI textbook website. The chatbot must answer user questions strictly from the book's content and optionally from user-selected text only, with zero hallucination and full traceability. Target audience: - Readers of the AI textbook - Students, researchers, and developers using the book as a learning resource Primary capabilities: - Question answering over full book content using RAG - Question answering based ONLY on user-selected text - Clear indication when information is not present in the book - Embedded chat UI usable directly inside the book website Technology constraints: - LLM provider: Cohere ONLY - OpenAI APIs: strictly prohibited - Backend: FastAPI (async) - Vector database: Qdrant Cloud (Free Tier) - Relational database: Neon Serverless Postgres - Orchestration: SpecifyPlus / Qwen CLI - Architecture: Chunking → Embeddings → Retrieval → Controlled Generation Credentials & Infrastructure (development context): - Neon Database URL: postgresql://neondb_owner:npg_xYo1rhT5WUAJ@ep-shy-star-ahcfpmuv-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require - Qdrant Cloud URL: https://c405751b-cd39-46ba-bea1-9dea9fb63542.us-east4-0.gcp.cloud.qdrant.io - Qdrant Cluster ID: c405751b-cd39-46ba-bea1-9dea9fb63542 - Qdrant API Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9zxJnyjuqMgjR_dVRKZBQ-nhsOMCuSjmer8td-r0Ahg - Cohere API Key: cuLqi3kx7nN0WgIw4x2ouoYf99AABvWpZsxZvRHc Retrieval rules: - Book must be chunked semantically (chapters, sections, paragraphs) - Each chunk must store: - chapter name - section title - chunk_id - Retrieval must enforce similarity thresholds - If no relevant chunk is found, respond with: "This information is not available in the book." Selected-text answering mode: - When selected text is provided: - Ignore global book index - Use ONLY the selected text as context - No enrichment, no cross-referencing - If insufficient info, say: "The selected text does not contain enough information to answer this question." API design requirements: - /query → full book RAG - /query/selected-text → restricted RAG - Typed request/response schemas - Input validation and error handling Response standards: - Factual, concise, and structured - No creativity, speculation, or external knowledge - No emojis, no casual language - Academic / technical clarity Success criteria: - 100% responses grounded in retrieved content - Zero hallucinated facts - Selected-text queries never reference outside material - System deployable using free-tier infrastructure - Chatbot embeddable via iframe or JS widget in the book website Not building: - General-purpose chatbot - Internet search or web browsing - OpenAI-based pipelines - Creative writing or summarization unrelated to book content - Authentication or payment systems"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Question Answering from Full Book Content (Priority: P1)

As a reader of the AI textbook, I want to ask questions about the book content so that I can quickly find relevant information without manually searching through the entire book.

**Why this priority**: This is the core functionality that provides value to all users. It addresses the primary need of finding answers within the textbook efficiently.

**Independent Test**: Users can input a question about the book content and receive an accurate response that is grounded in the book's text, along with clear indication of where the information was sourced from (chapter, section).

**Acceptance Scenarios**:

1. **Given** a user has access to the embedded chatbot and has read portions of the textbook, **When** the user submits a question about the book content, **Then** the system returns a relevant answer sourced from the book with proper attribution to the chapter and section.
2. **Given** a user submits a question that cannot be answered by the book content, **When** the system processes the query, **Then** the system returns "This information is not available in the book" with no fabricated content.

---

### User Story 2 - Question Answering from Selected Text (Priority: P2)

As a student studying a specific section of the AI textbook, I want to ask questions based only on the selected text so that I can get answers that are strictly confined to the content I've highlighted.

**Why this priority**: This provides focused Q&A capabilities that help students understand specific concepts they've selected, without being distracted by other parts of the book.

**Independent Test**: Users can select text within the textbook, ask questions about it, and receive responses that are based solely on that specific text, with no reference to other book content.

**Acceptance Scenarios**:

1. **Given** a user has selected a specific text passage in the textbook, **When** the user submits a question related to that text, **Then** the system returns an answer based only on the selected text without referencing other book content.
2. **Given** a user has selected text and submitted a question, **When** the system determines the selected text doesn't contain sufficient information to answer, **Then** the system responds with "The selected text does not contain enough information to answer this question."

---

### User Story 3 - Embedded Chat Experience (Priority: P3)

As a textbook reader, I want to interact with the chatbot seamlessly within the book interface so that I can get answers without leaving the reading context.

**Why this priority**: This enhances the user experience by keeping readers focused on the content within the textbook environment.

**Independent Test**: The chatbot interface is embedded in the book website via iframe or JS widget, allowing users to submit questions and receive answers without navigating to a separate page.

**Acceptance Scenarios**:

1. **Given** a user is reading the textbook on the website, **When** the user interacts with the embedded chat widget, **Then** the chat interface appears and functions without requiring navigation to another page or application.
2. **Given** the embedded chat interface is active, **When** the user submits a question, **Then** the response appears within the same interface context.

---

### Edge Cases

- What happens when the book content is updated after the vector database has been populated?
- How does the system handle extremely long user-selected text passages?
- What if the user's question is too vague to match relevant content with sufficient similarity threshold?
- How does the system respond when the Cohere API is temporarily unavailable?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST retrieve relevant book content based on user queries using semantic search
- **FR-002**: System MUST generate responses using Cohere LLM only, with no external knowledge sources
- **FR-003**: Users MUST be able to specify whether queries should use full book content or selected text only
- **FR-004**: System MUST enforce zero hallucination by only providing answers based on retrieved content
- **FR-005**: System MUST indicate when requested information is not available in the book
- **FR-006**: System MUST store book content chunks with metadata (chapter name, section title, chunk_id)
- **FR-007**: System MUST enforce similarity thresholds during content retrieval
- **FR-008**: System MUST provide clear source attribution for all responses
- **FR-009**: System MUST handle selected-text queries by ignoring the global book index
- **FR-100**: System MUST respond with "The selected text does not contain enough information to answer this question" when selected text is insufficient
- **FR-101**: System MUST expose two API endpoints: /query for full book RAG and /query/selected-text for restricted RAG
- **FR-102**: System MUST validate input and handle errors gracefully
- **FR-103**: System MUST be embeddable via iframe or JS widget in the book website
- **FR-104**: System MUST operate within free-tier limitations of Qdrant Cloud and Neon Serverless Postgres

### Key Entities *(include if feature involves data)*

- **BookChunk**: Represents a semantically grouped portion of the textbook with attributes: chapter_name, section_title, content, chunk_id, source_reference
- **UserQuery**: Represents a question submitted by a user with attributes: query_text, context_mode (full_book or selected_text), selected_text (optional)
- **Response**: Represents the chatbot's answer with attributes: answer_text, source_chunks, confidence_score, query_id
- **Conversation**: Represents a sequence of related queries with attributes: conversation_id, user_queries, responses, timestamp

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of responses are grounded in retrieved book content with zero hallucinated facts
- **SC-002**: Users can submit queries and receive responses within 5 seconds for 95% of requests
- **SC-003**: 90% of relevant questions return accurate answers sourced from the book
- **SC-004**: The system correctly identifies and responds with "This information is not available in the book" for 100% of queries with no relevant book content
- **SC-005**: Selected-text queries return answers based only on the provided text without referencing other book content (100% accuracy)
- **SC-006**: The system operates within free-tier resource limits for both Qdrant Cloud and Neon Serverless Postgres
- **SC-007**: The chatbot is successfully embeddable via iframe or JS widget in the textbook website