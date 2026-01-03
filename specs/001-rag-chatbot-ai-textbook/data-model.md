# Data Model: RAG Chatbot for AI Textbook

## Overview
This document defines the data models for the RAG chatbot system, designed to support the constitutional requirements of faithfulness, zero hallucination, and clear source attribution.

## Core Entities

### BookChunk
Represents a semantically grouped portion of the textbook.

**Fields:**
- `chunk_id`: (String, required) Unique identifier for the chunk
- `content`: (Text, required) The actual text content of the chunk
- `chapter_name`: (String, required) Name of the chapter containing this chunk
- `section_title`: (String, required) Title of the section containing this chunk
- `page_number`: (Integer, optional) Page number where this chunk appears (if applicable)
- `source_reference`: (String, required) Complete reference including chapter, section, and page
- `embedding_vector`: (Array<Float>, required) Vector representation from Cohere embedding
- `created_at`: (DateTime, required) Timestamp when chunk was created
- `updated_at`: (DateTime, optional) Timestamp of last update

**Validation rules:**
- `chunk_id` must be unique across all chunks
- `content` must not exceed 1000 tokens to maintain semantic coherence
- `source_reference` must follow format: "Chapter {name}, Section {title}, Page {number}"

### UserQuery
Represents a question submitted by a user to the system.

**Fields:**
- `query_id`: (String, required) Unique identifier for the query
- `query_text`: (Text, required) The actual question text submitted by the user
- `context_mode`: (Enum, required) Either "full_book" or "selected_text"
- `selected_text`: (Text, optional) Provided text if context_mode is "selected_text"
- `user_id`: (String, optional) ID of the user who made the query
- `session_id`: (String, optional) ID to group related queries in a session
- `timestamp`: (DateTime, required) When the query was submitted

**Validation rules:**
- `context_mode` must be one of the allowed values
- When `context_mode` is "selected_text", `selected_text` must be provided
- When `context_mode` is "full_book", `selected_text` must be null

### Response
Represents the chatbot's answer to a user query.

**Fields:**
- `response_id`: (String, required) Unique identifier for the response
- `answer_text`: (Text, required) The text of the chatbot's answer
- `source_chunks`: (Array<String>, required) IDs of chunks used to generate the response
- `confidence_score`: (Float, required) Confidence level of the response (0.0-1.0)
- `query_id`: (String, required) Reference to the original query
- `timestamp`: (DateTime, required) When the response was generated
- `context_mode`: (Enum, required) Context mode used to generate this response

**Validation rules:**
- `confidence_score` must be between 0.0 and 1.0
- Each `source_chunks` ID must exist in the BookChunk table
- `context_mode` must match the original query's context mode

### Conversation
Represents a sequence of related queries and responses.

**Fields:**
- `conversation_id`: (String, required) Unique identifier for the conversation
- `user_id`: (String, optional) ID of the user involved in the conversation
- `session_id`: (String, required) ID to group queries in a session
- `created_at`: (DateTime, required) When the conversation started
- `updated_at`: (DateTime, required) When the conversation was last modified
- `is_active`: (Boolean, required) Whether the conversation is still active

**Validation rules:**
- `conversation_id` must be unique
- `is_active` helps with conversation cleanup after inactivity

### QueryLog
Represents logged metadata about queries for analytics and debugging.

**Fields:**
- `log_id`: (String, required) Unique identifier for the log entry
- `query_id`: (String, required) Reference to the original query
- `response_id`: (String, required) Reference to the generated response
- `chunk_ids`: (Array<String>, required) IDs of all retrieved chunks (not just used ones)
- `similarity_scores`: (Array<Float>, required) Similarity scores for retrieved chunks
- `response_time_ms`: (Integer, required) Time taken to generate the response in milliseconds
- `has_sufficient_context`: (Boolean, required) Whether sufficient context was available to answer
- `timestamp`: (DateTime, required) When the log entry was created

**Validation rules:**
- No raw text content is stored in this table (compliance with logging constraint)
- `similarity_scores` and `chunk_ids` must be arrays of the same length

## Relationships

- `UserQuery` to `Response`: One-to-one (each query generates one response)
- `Response` to `BookChunk`: Many-to-many through source_chunks field
- `UserQuery` to `QueryLog`: One-to-one (each query has one log entry)
- `Conversation` to `UserQuery`: One-to-many (one conversation can contain multiple queries)

## Indexes

- BookChunk table: Index on `chunk_id` (primary), `chapter_name`, `section_title`
- UserQuery table: Index on `query_id` (primary), `timestamp`, `context_mode`
- Response table: Index on `response_id` (primary), `query_id`, `source_chunks`
- QueryLog table: Index on `log_id` (primary), `query_id`, `timestamp`

## Constraints & Validation

1. **Zero Hallucination Compliance**: Responses must only reference content from chunks in the `source_chunks` list
2. **Source Attribution**: Every response must include the `source_chunks` that were used to generate it
3. **Context Mode Isolation**: When `context_mode` is "selected_text", retrieval must only consider the provided text chunks, not the full book corpus
4. **Content Privacy**: QueryLog does not store actual content from chunks or book, only IDs and metadata
5. **Semantic Boundaries**: BookChunk content must respect chapter and section boundaries as specified in the constitution