# Implementation Plan: RAG Chatbot for AI Textbook

**Branch**: `001-rag-chatbot-ai-textbook` | **Date**: 2025-12-18 | **Spec**: [link]
**Input**: Feature specification from `/specs/001-rag-chatbot-ai-textbook/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This plan outlines the implementation of a production-grade Retrieval-Augmented Generation (RAG) chatbot for an AI textbook. The system will enable users to ask questions about textbook content and receive accurate, traceable answers grounded only in the book's content. It supports two distinct modes: (1) answering from the full book corpus and (2) answering exclusively from user-selected text passages. The implementation prioritizes zero hallucination, clear source attribution, and embeddability in textbook websites while operating within free-tier cloud resource limits.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: FastAPI, Cohere SDK, Qdrant Client, asyncpg, Pydantic, NumPy
**Storage**: Qdrant Cloud (vector database), Neon Serverless Postgres (metadata/logs)
**Testing**: pytest with integration and unit test frameworks
**Target Platform**: Linux server (cloud deployment)
**Project Type**: Web application (backend API with embeddable frontend)
**Performance Goals**: <5 second response time for 95% of requests, support hundreds of concurrent users
**Constraints**: Free-tier resource limits of Qdrant Cloud and Neon Serverless Postgres, zero hallucination requirement
**Scale/Scope**: AI textbook Q&A system supporting semantic search on book content

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Verification

**I. Faithfulness**: ✅ VERIFIED - System will retrieve and ground responses only in book content with traceability to specific passages
**II. Zero Hallucination**: ✅ VERIFIED - Implementation includes explicit fallback response ("This information is not available in the book") when content is not found
**III. Context Priority**: ✅ VERIFIED - System will have two distinct modes: full book RAG and selected-text only RAG with selected text overriding global context
**IV. Transparency**: ✅ VERIFIED - Responses will include clear source attribution indicating if answer is from selected text or full book
**V. Modularity**: ✅ VERIFIED - Components (LLM, retrieval, storage) designed with clear interfaces to enable swapping
**VI. LLM Provider Constraint**: ✅ VERIFIED - Only Cohere APIs will be used (Command R+/Embed models) as required

### Additional Constraints Verified
- ✅ RAG Architecture: Chunking with semantic boundaries, metadata storage with source/chunk_id
- ✅ Selected Text Mode: Implementation of isolated selected-text answering
- ✅ Backend Standards: FastAPI with async design and proper API separation (/query vs /query/selected-text)
- ✅ Security & Reliability: API keys via env vars, no hardcoded secrets
- ✅ User Experience: Structured, factual responses without creativity/emojis
- ✅ Logging: Query logs with chunk IDs and similarity scores, no raw book text logged
- ✅ Free-tier deployable: Architecture designed for Qdrant Cloud and Neon Serverless Postgres

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-chatbot-ai-textbook/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: Web application with separate backend and frontend directories. The backend will handle the RAG logic and API, while the frontend will contain the embeddable chat widget.

```text
backend/
├── src/
│   ├── models/
│   │   ├── book_chunk.py
│   │   ├── user_query.py
│   │   ├── response.py
│   │   └── conversation.py
│   ├── services/
│   │   ├── rag_service.py
│   │   ├── chunking_service.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   └── cohere_service.py
│   └── api/
│       ├── main.py
│       ├── endpoints/
│       │   ├── query.py
│       │   └── query_selected_text.py
│       └── middleware/
│           └── auth_middleware.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py
└── scripts/
    ├── ingest_book.py
    └── vectorize_chunks.py

frontend/
├── src/
│   ├── components/
│   │   ├── ChatWidget.jsx
│   │   ├── Message.jsx
│   │   └── InputField.jsx
│   ├── services/
│   │   └── apiService.js
│   └── utils/
│       └── textSelection.js
├── public/
│   └── index.html
├── package.json
└── build/
    └── bundle.js

```

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
