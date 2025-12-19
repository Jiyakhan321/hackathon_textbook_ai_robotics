<!--
SYNC IMPACT REPORT
- Version change: N/A → 1.0.0
- Modified principles: N/A (new principles added)
- Added sections: Core Principles (I-VI), Additional Constraints, Development Workflow, Success Criteria
- Removed sections: N/A
- Templates requiring updates:
  - ✅ plan-template.md: Constitution Check section will now reference new principles
  - ✅ spec-template.md: Requirements section aligns with new constraints
  - ✅ tasks-template.md: Task categorization reflects new principle-driven requirements
- Follow-up TODOs: None
-->

# Integrated RAG Chatbot Constitution

## Core Principles

### I. Faithfulness
Responses must be grounded ONLY in retrieved book content; all claims must be traceable to specific book passages.

### II. Zero Hallucination
If information is not present in the retrieved context, explicitly say "This information is not available in the book"; No creative interpretation or assumption of missing information.

### III. Context Priority
User-selected text always overrides global book context; When provided, selected text becomes the exclusive source for answering queries.

### IV. Transparency
Clearly indicate when an answer is based on selected text vs full book; All responses must be structured to show the source of information.

### V. Modularity
Each component (LLM, retrieval, storage) must be swappable without affecting other system components; Clear interfaces between modules.

### VI. LLM Provider Constraint
Cohere ONLY (Command / Command-R / Embed models) - OpenAI APIs are strictly forbidden; Use Cohere Embeddings for vector generation and Cohere Chat/Generate for response synthesis.

## Additional Constraints

### RAG Architecture Requirements
Book content must be chunked with semantic boundaries (headings, paragraphs); Every chunk must store source (chapter, section, page) and chunk_id; Vector storage must use Qdrant Cloud; Metadata (users, queries, logs) must be stored in Neon Serverless Postgres; Retrieval must be top-k with similarity threshold enforcement.

### Selected Text Mode Operation
When user provides selected text: ignore the rest of the book corpus, answer strictly from the selected text, Do NOT enrich with other chapters or external info; If the answer cannot be derived, respond with: "The selected text does not contain enough information to answer this question."

### Backend Standards
Backend framework: FastAPI; Async-first design; Clear API separation (/query for normal RAG, /query/selected-text for restricted RAG); Input validation and response schema required.

### Security & Reliability
API keys must be stored in environment variables; No secrets hardcoded; Graceful handling of empty retrieval, low similarity scores, and malformed input.

### User Experience Requirements
Responses must be clear, concise, and structured (bullet points when useful); No speculative or creative writing; No emojis or casual tone; Academic/technical clarity maintained.

### Logging & Observability
Log user query, retrieved chunk IDs, and similarity scores; Do NOT log raw book text in plaintext logs; All logs must be structured for debugging and analysis.

## Development Workflow
All implementations must follow the production-ready standard; Code must be deployed on free tiers (Qdrant + Neon); All features must be testable with automated tests; Performance requirements must be met for real-time response.

## Success Criteria
100% answers traceable to retrieved book chunks; Zero hallucinated facts; Selected-text questions never reference outside content; System is deployable on free tiers (Qdrant + Neon); Chatbot can be embedded in a book website via iframe or JS SDK.

## Governance
This constitution supersedes all other development practices; All pull requests and code reviews must verify compliance with these principles; Amendments require explicit documentation, approval, and migration plan.

**Version**: 1.0.0 | **Ratified**: 2025-12-18 | **Last Amended**: 2025-12-18
