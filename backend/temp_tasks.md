---

description: "Task list for RAG Chatbot for AI Textbook implementation"
---

# Tasks: RAG Chatbot for AI Textbook

**Input**: Design documents from `/specs/001-rag-chatbot-ai-textbook/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests should be implemented for all core functionality to ensure zero hallucination and proper source attribution.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `backend/src/`, `frontend/src/`
- Paths shown below assume web app structure - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan in backend/
- [ ] T002 Initialize Python 3.11 project with FastAPI, Cohere SDK, Qdrant Client, asyncpg, Pydantic, NumPy dependencies in backend/requirements.txt
- [ ] T003 [P] Configure linting and formatting tools (black, flake8, mypy) in backend/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [ ] T004 Setup database schema and migrations framework for Neon Postgres in backend/src/config/database.py
- [ ] T005 [P] Configure Qdrant vector database connection in backend/src/config/vector_db.py
- [ ] T006 [P] Setup API routing and middleware structure in backend/src/api/main.py
- [ ] T007 Create base models/entities that all stories depend on in backend/src/models/
- [ ] T008 Configure error handling and logging infrastructure in backend/src/utils/
- [ ] T009 Setup environment configuration management with .env.example in backend/
- [ ] T010 Create configuration settings module in backend/src/config/settings.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Question Answering from Full Book Content (Priority: P1) 🎯 MVP

**Goal**: Enable users to ask questions about the book content and receive accurate, traceable answers grounded only in the book's content.

**Independent Test**: Users can input a question about the book content and receive an accurate response that is grounded in the book's text, along with clear indication of where the information was sourced from (chapter, section).

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T011 [P] [US1] Contract test for /query endpoint in backend/tests/contract/test_query_endpoint.py
- [ ] T012 [P] [US1] Integration test for full book RAG user journey in backend/tests/integration/test_full_book_rag.py

### Implementation for User Story 1

- [ ] T013 [P] [US1] Create BookChunk model in backend/src/models/book_chunk.py
- [ ] T014 [P] [US1] Create UserQuery model in backend/src/models/user_query.py
- [ ] T015 [P] [US1] Create Response model in backend/src/models/response.py
- [ ] T016 [US1] Implement Chunking service in backend/src/services/chunking_service.py
- [ ] T017 [US1] Implement Embedding service using Cohere in backend/src/services/embedding_service.py
- [ ] T018 [US1] Implement Retrieval service for Qdrant in backend/src/services/retrieval_service.py
- [ ] T019 [US1] Implement Cohere service for text generation in backend/src/services/cohere_service.py
- [ ] T020 [US1] Implement RAG service that orchestrates the flow in backend/src/services/rag_service.py
- [ ] T021 [US1] Implement /query endpoint in backend/src/api/endpoints/query.py
- [ ] T022 [US1] Add input validation and response schema to /query endpoint
- [ ] T023 [US1] Add error handling for empty retrieval and low similarity scores
- [ ] T024 [US1] Add logging for user story 1 operations with chunk IDs and similarity scores

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Question Answering from Selected Text (Priority: P2)

**Goal**: Enable students to ask questions based only on selected text and receive responses that are strictly confined to the content they've highlighted.

**Independent Test**: Users can select text within the textbook, ask questions about it, and receive responses that are based solely on that specific text, with no reference to other book content.

### Tests for User Story 2 ⚠️

- [ ] T025 [P] [US2] Contract test for /query/selected-text endpoint in backend/tests/contract/test_selected_text_endpoint.py
- [ ] T026 [P] [US2] Integration test for selected text RAG user journey in backend/tests/integration/test_selected_text_rag.py

### Implementation for User Story 2

- [ ] T027 [P] [US2] Create Conversation model in backend/src/models/conversation.py
- [ ] T028 [US2] Extend RAG service to support selected-text mode in backend/src/services/rag_service.py
- [ ] T029 [US2] Implement selected-text chunking and embedding in backend/src/services/chunking_service.py
- [ ] T030 [US2] Implement /query/selected-text endpoint in backend/src/api/endpoints/query_selected_text.py
- [ ] T031 [US2] Add validation to ensure selected-text mode ignores global book index
- [ ] T032 [US2] Add response logic for "The selected text does not contain enough information to answer this question."
- [ ] T033 [US2] Add input validation and response schema to /query/selected-text endpoint

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Embedded Chat Experience (Priority: P3)

**Goal**: Provide a seamless chat interface within the book website that allows users to get answers without navigating to a separate page.

**Independent Test**: The chatbot interface is embedded in the book website via iframe or JS widget, allowing users to submit questions and receive answers without navigating to a separate page.

### Tests for User Story 3 ⚠️

- [ ] T034 [P] [US3] Contract test for chat widget endpoint in backend/tests/contract/test_widget_endpoint.py
- [ ] T035 [P] [US3] Integration test for embedded chat experience in frontend/tests/integration/test_embedded_chat.js

### Implementation for User Story 3

- [ ] T036 [P] [US3] Create QueryLog model for analytics and debugging in backend/src/models/query_log.py
- [ ] T037 [US3] Implement API service for frontend in backend/src/api/endpoints/widget.py
- [ ] T038 [US3] Create ChatWidget component in frontend/src/components/ChatWidget.jsx
- [ ] T039 [US3] Create Message component in frontend/src/components/Message.jsx
- [ ] T040 [US3] Create InputField component in frontend/src/components/InputField.jsx
- [ ] T041 [US3] Implement API service for frontend in frontend/src/services/apiService.js
- [ ] T042 [US3] Implement text selection utility in frontend/src/utils/textSelection.js
- [ ] T043 [US3] Implement widget embedding via iframe in backend/src/api/endpoints/widget.py
- [ ] T044 [US3] Create JavaScript SDK for embedding in frontend/build/bundle.js
- [ ] T045 [US3] Add loading and error state handling in frontend components

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T046 [P] Documentation updates in docs/
- [ ] T047 Code cleanup and refactoring
- [ ] T048 Performance optimization across all stories (token limits, embedding caching)
- [ ] T049 [P] Additional unit tests in backend/tests/unit/ and frontend/tests/unit/
- [ ] T050 Security hardening (rate limiting, input sanitization)
- [ ] T051 Run quickstart.md validation
- [ ] T052 Create environment variable validation
- [ ] T053 Implement rate limiting at 10 requests per minute per IP
- [ ] T054 Add monitoring and health check endpoints
- [ ] T055 Create book ingestion script in backend/scripts/ingest_book.py
- [ ] T056 Create vectorization script in backend/scripts/vectorize_chunks.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
T011 [P] [US1] Contract test for /query endpoint in backend/tests/contract/test_query_endpoint.py
T012 [P] [US1] Integration test for full book RAG user journey in backend/tests/integration/test_full_book_rag.py

# Launch all models for User Story 1 together:
T013 [P] [US1] Create BookChunk model in backend/src/models/book_chunk.py
T014 [P] [US1] Create UserQuery model in backend/src/models/user_query.py
T015 [P] [US1] Create Response model in backend/src/models/response.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence