# Task List: OpenAI API Compatibility

## miLLM Feature 2

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `002_FPRD|OpenAI_API.md`
- Feature TDD: `002_FTDD|OpenAI_API.md`
- Feature TID: `002_FTID|OpenAI_API.md`

---

## Relevant Files

### Backend - Routes

- `millm/api/routes/openai/__init__.py` - Router aggregation, exports openai_router
- `millm/api/routes/openai/chat.py` - POST /v1/chat/completions endpoint
- `millm/api/routes/openai/completions.py` - POST /v1/completions endpoint
- `millm/api/routes/openai/models.py` - GET /v1/models and /v1/models/{id} endpoints
- `millm/api/routes/openai/embeddings.py` - POST /v1/embeddings endpoint
- `millm/api/routes/openai/errors.py` - OpenAI error format helpers

### Backend - Schemas

- `millm/api/schemas/openai.py` - All Pydantic models for OpenAI API

### Backend - Services

- `millm/services/inference_service.py` - Core generation logic for completions
- `millm/services/request_queue.py` - Concurrent request management

### Backend - ML

- `millm/ml/generation_config.py` - Generation parameter mapping from OpenAI to Transformers

### Backend - Dependencies

- `millm/api/dependencies.py` - FastAPI dependencies (get_inference_service)
- `millm/main.py` - App setup with OpenAI routes mounted

### Tests - Unit

- `tests/unit/api/test_openai_schemas.py` - Schema validation tests
- `tests/unit/api/test_openai_errors.py` - Error formatting tests
- `tests/unit/api/test_generation_config.py` - Parameter mapping tests

### Tests - Integration

- `tests/integration/api/test_chat_completions.py` - Full chat endpoint tests
- `tests/integration/api/test_streaming.py` - SSE streaming tests
- `tests/integration/api/test_completions.py` - Text completions tests
- `tests/integration/api/test_embeddings.py` - Embeddings tests
- `tests/integration/api/test_models.py` - Models endpoint tests

### Tests - Compatibility

- `tests/compatibility/test_openai_client.py` - OpenAI Python library tests
- `tests/compatibility/conftest.py` - Shared fixtures for compatibility tests

### Notes

- Unit tests should be placed alongside source files or in corresponding test directories
- Use `pytest` to run tests: `pytest tests/unit/` or `pytest tests/integration/`
- Compatibility tests require a running server with loaded model
- SSE streaming requires `sse-starlette>=1.6.0`

---

## Tasks

### Phase 1: Core Infrastructure

- [x] 1.0 Set up OpenAI API route structure
  - [x] 1.1 Create `millm/api/routes/openai/` directory structure
  - [x] 1.2 Create `millm/api/routes/openai/__init__.py` with router aggregation
  - [x] 1.3 Create placeholder route files (chat.py, completions.py, models.py, embeddings.py)
  - [x] 1.4 Mount OpenAI router in main app with `/v1` prefix
  - [x] 1.5 Verify routes are accessible (GET /v1/models should return empty list)

- [x] 2.0 Implement OpenAI request/response schemas
  - [x] 2.1 Create `millm/api/schemas/openai.py` file
  - [x] 2.2 Implement ChatMessage schema with role validation
  - [x] 2.3 Implement ChatCompletionRequest with all parameters and validation
  - [x] 2.4 Implement ChatCompletionResponse and ChatCompletionChoice schemas
  - [x] 2.5 Implement streaming chunk schemas (ChatCompletionChunk, ChunkDelta, ChunkChoice)
  - [x] 2.6 Implement TextCompletionRequest schema
  - [x] 2.7 Implement EmbeddingRequest and EmbeddingResponse schemas
  - [x] 2.8 Implement ModelObject and ModelListResponse schemas
  - [x] 2.9 Implement OpenAIError and OpenAIErrorResponse schemas
  - [x] 2.10 Implement Usage schema with token counting

- [x] 3.0 Implement OpenAI error handling
  - [x] 3.1 Create `millm/api/routes/openai/errors.py` file
  - [x] 3.2 Implement `create_openai_error()` helper function
  - [x] 3.3 Create error code to HTTP status mapping
  - [x] 3.4 Implement `openai_exception_handler` for MiLLMError
  - [x] 3.5 Register exception handler in main app
  - [x] 3.6 Test error responses match OpenAI format exactly

- [x] 4.0 Implement generation configuration mapping
  - [x] 4.1 Create `millm/ml/generation_config.py` file
  - [x] 4.2 Implement GenerationConfig dataclass
  - [x] 4.3 Implement `from_request()` class method for OpenAI parameters
  - [x] 4.4 Implement `to_generate_kwargs()` for Transformers conversion
  - [x] 4.5 Handle temperature=0 → do_sample=False mapping
  - [x] 4.6 Map frequency_penalty to repetition_penalty approximation

### Phase 2: Request Queue

- [x] 5.0 Implement request queue for concurrent requests
  - [x] 5.1 Create `millm/services/request_queue.py` file
  - [x] 5.2 Implement QueueFullError exception class
  - [x] 5.3 Implement RequestQueue class with semaphore and pending counter
  - [x] 5.4 Implement async context manager `acquire()` method
  - [x] 5.5 Add timeout support for queue acquisition
  - [x] 5.6 Implement `pending_count` and `is_available` properties
  - [ ] 5.7 Write unit tests for queue behavior (concurrent, full, timeout)

### Phase 3: InferenceService Core

- [x] 6.0 Implement InferenceService base structure
  - [x] 6.1 Create `millm/services/inference_service.py` file
  - [x] 6.2 Implement constructor with model_service and steering_service dependencies
  - [x] 6.3 Implement `is_model_loaded()` method
  - [x] 6.4 Implement `get_loaded_model_info()` method
  - [x] 6.5 Add private properties for model and tokenizer access
  - [x] 6.6 Integrate RequestQueue into InferenceService

- [x] 7.0 Implement chat message formatting
  - [x] 7.1 Implement `_format_chat_messages()` method
  - [x] 7.2 Use tokenizer's `apply_chat_template()` if available
  - [x] 7.3 Implement fallback simple concatenation format
  - [x] 7.4 Handle system, user, and assistant roles
  - [x] 7.5 Add generation prompt to end of formatted string
  - [ ] 7.6 Write unit tests for message formatting

### Phase 4: Non-Streaming Completions

- [x] 8.0 Implement non-streaming chat completions
  - [x] 8.1 Implement `create_chat_completion()` async method
  - [x] 8.2 Generate unique completion ID (chatcmpl-{24 hex})
  - [x] 8.3 Format messages and tokenize input
  - [x] 8.4 Count prompt tokens for usage stats
  - [x] 8.5 Build generation config from request parameters
  - [x] 8.6 Call model.generate() with torch.no_grad()
  - [x] 8.7 Decode generated tokens, skipping prompt and special tokens
  - [x] 8.8 Build and return ChatCompletionResponse
  - [x] 8.9 Integrate steering service hooks (if active)

- [x] 9.0 Implement chat completions route
  - [x] 9.1 Complete `millm/api/routes/openai/chat.py` implementation
  - [x] 9.2 Add model-loaded check at start of handler
  - [x] 9.3 Add request logging with model, message count
  - [x] 9.4 Call InferenceService for non-streaming requests
  - [x] 9.5 Return proper JSON response
  - [ ] 9.6 Write integration tests for non-streaming chat

### Phase 5: Streaming Completions

- [x] 10.0 Implement SSE streaming infrastructure
  - [x] 10.1 Add sse-starlette dependency to requirements
  - [x] 10.2 Import EventSourceResponse in chat route
  - [x] 10.3 Configure proper media_type for SSE responses

- [x] 11.0 Implement streaming chat completions
  - [x] 11.1 Implement `stream_chat_completion()` async generator
  - [x] 11.2 Set up TextIteratorStreamer from transformers
  - [x] 11.3 Implement `_generate_in_thread()` for blocking generation
  - [x] 11.4 Start generation thread with streamer
  - [x] 11.5 Yield first chunk with role="assistant"
  - [x] 11.6 Yield content chunks for each token from streamer
  - [x] 11.7 Yield final chunk with finish_reason="stop"
  - [x] 11.8 Yield "data: [DONE]\n\n" at end
  - [x] 11.9 Handle asyncio.CancelledError for client disconnect
  - [x] 11.10 Ensure thread cleanup in finally block
  - [x] 11.11 Update chat route to use EventSourceResponse for stream=true

- [ ] 12.0 Test streaming implementation
  - [ ] 12.1 Write unit tests for chunk format
  - [ ] 12.2 Write integration tests for full streaming flow
  - [ ] 12.3 Test client disconnect handling
  - [ ] 12.4 Verify SSE format (data: {json}\n\n)
  - [ ] 12.5 Test with curl and verify format

### Phase 6: Models Endpoint

- [x] 13.0 Implement models listing endpoint
  - [x] 13.1 Complete `millm/api/routes/openai/models.py` implementation
  - [x] 13.2 Implement GET /v1/models handler
  - [x] 13.3 Return empty list if no model loaded (not error)
  - [x] 13.4 Return loaded model with id, created, owned_by fields
  - [x] 13.5 Implement GET /v1/models/{model_id} handler
  - [x] 13.6 Return 404 with OpenAI error format for unknown models
  - [ ] 13.7 Write integration tests for models endpoints

### Phase 7: Text Completions

- [x] 14.0 Implement text completions endpoint
  - [x] 14.1 Complete `millm/api/routes/openai/completions.py` implementation
  - [x] 14.2 Handle prompt as string or list (use first item)
  - [x] 14.3 Implement non-streaming text completion helper
  - [ ] 14.4 Implement streaming text completion helper (deferred - legacy endpoint)
  - [x] 14.5 Format response with "text_completion" object type
  - [x] 14.6 Support all generation parameters (temperature, max_tokens, etc.)
  - [ ] 14.7 Write integration tests for completions endpoint

### Phase 8: Embeddings Endpoint

- [x] 15.0 Implement embeddings endpoint
  - [x] 15.1 Complete `millm/api/routes/openai/embeddings.py` implementation
  - [x] 15.2 Handle input as string or array of strings
  - [x] 15.3 Tokenize and run through model with output_hidden_states=True
  - [x] 15.4 Extract last hidden layer
  - [x] 15.5 Apply mean pooling over sequence dimension
  - [x] 15.6 Convert to list and return in EmbeddingResponse
  - [x] 15.7 Calculate usage tokens correctly
  - [ ] 15.8 Write integration tests for embeddings endpoint

### Phase 9: Dependencies and App Integration

- [x] 16.0 Set up FastAPI dependencies
  - [x] 16.1 Update `millm/api/dependencies.py` with get_inference_service
  - [x] 16.2 Implement get_model_service dependency
  - [x] 16.3 Use app.state for service storage

- [x] 17.0 Update main app with OpenAI integration
  - [x] 17.1 Update lifespan context manager for InferenceService initialization
  - [x] 17.2 Mount openai_router on app
  - [x] 17.3 Register openai_exception_handler
  - [x] 17.4 Verify all routes are accessible
  - [ ] 17.5 Test full app startup and shutdown

### Phase 10: Unit Tests

- [x] 18.0 Implement schema unit tests
  - [x] 18.1 Create `tests/unit/api/test_openai_schemas.py`
  - [x] 18.2 Test ChatCompletionRequest validation (valid, invalid temperature, stop limits)
  - [x] 18.3 Test ChatCompletionResponse serialization matches OpenAI format
  - [x] 18.4 Test streaming chunk serialization
  - [x] 18.5 Test extra fields are ignored (not errored)
  - [x] 18.6 Test Usage auto-computation of total_tokens

- [x] 19.0 Implement error handling unit tests
  - [x] 19.1 Create `tests/unit/api/test_openai_errors.py`
  - [x] 19.2 Test create_openai_error returns correct format
  - [x] 19.3 Test error code to status mapping
  - [x] 19.4 Test exception handler converts MiLLMError correctly

- [x] 20.0 Implement generation config unit tests
  - [x] 20.1 Create `tests/unit/api/test_generation_config.py`
  - [x] 20.2 Test from_request() with various parameter combinations
  - [x] 20.3 Test temperature=0 produces do_sample=False
  - [x] 20.4 Test stop sequence normalization
  - [x] 20.5 Test to_generate_kwargs() output

### Phase 11: Integration Tests

- [x] 21.0 Implement chat completions integration tests
  - [x] 21.1 Create `tests/integration/api/test_chat_completions.py`
  - [x] 21.2 Test non-streaming returns valid response
  - [x] 21.3 Test no model loaded returns 503
  - [x] 21.4 Test invalid parameters return 422
  - [x] 21.5 Test with various parameter combinations

- [ ] 22.0 Implement streaming integration tests (deferred - requires loaded model)
  - [ ] 22.1 Create `tests/integration/api/test_streaming.py`
  - [ ] 22.2 Test SSE response has correct content-type
  - [ ] 22.3 Test first chunk has role
  - [ ] 22.4 Test middle chunks have content
  - [ ] 22.5 Test final chunk has finish_reason
  - [ ] 22.6 Test ends with [DONE]

- [x] 23.0 Implement other endpoint integration tests
  - [x] 23.1 Create `tests/integration/api/test_models.py`
  - [x] 23.2 Create `tests/integration/api/test_completions.py`
  - [x] 23.3 Create `tests/integration/api/test_embeddings.py`
  - [x] 23.4 Test all endpoints with loaded and unloaded states

### Phase 12: Compatibility Tests

- [ ] 24.0 Implement OpenAI Python client compatibility tests
  - [ ] 24.1 Create `tests/compatibility/test_openai_client.py`
  - [ ] 24.2 Create `tests/compatibility/conftest.py` with fixtures
  - [ ] 24.3 Test client.models.list() works
  - [ ] 24.4 Test client.chat.completions.create() non-streaming
  - [ ] 24.5 Test client.chat.completions.create() streaming
  - [ ] 24.6 Test client.embeddings.create() works
  - [ ] 24.7 Document test requirements (running server, loaded model)

- [ ] 25.0 Manual compatibility testing
  - [ ] 25.1 Test with Open WebUI
  - [ ] 25.2 Test with LibreChat (if available)
  - [ ] 25.3 Test with Continue.dev (if available)
  - [ ] 25.4 Document any compatibility issues found

### Phase 13: Documentation and Polish

- [x] 26.0 Create OpenAPI documentation
  - [x] 26.1 Ensure all endpoints have proper docstrings
  - [x] 26.2 Add response_model to all routes
  - [x] 26.3 Add responses dict for error status codes
  - [ ] 26.4 Verify /docs endpoint shows complete documentation (requires running server)

- [x] 27.0 Final review and cleanup
  - [x] 27.1 Review all error messages for clarity
  - [x] 27.2 Ensure logging is consistent and useful
  - [x] 27.3 Remove any debug code or print statements
  - [x] 27.4 Verify all tests pass
  - [ ] 27.5 Update feature documentation if needed (optional)

---

## Notes

### Development Order Recommendation

1. Start with schemas (Task 2.0) - foundation for everything
2. Then route structure (Task 1.0) - basic scaffolding
3. Then error handling (Task 3.0) - needed for all routes
4. Then models endpoint (Task 13.0) - simplest endpoint, validates setup
5. Then non-streaming chat (Tasks 6-9) - core functionality
6. Then streaming (Tasks 10-12) - builds on non-streaming
7. Then other endpoints (Tasks 14-15) - follow established patterns
8. Finally tests and polish (Tasks 18-27)

### Key Dependencies

- Requires ModelService from Feature 1 to be complete
- SteeringService integration is optional (Feature 4)
- `sse-starlette>=1.6.0` required for streaming

### Testing Notes

- Unit tests can run without model
- Integration tests use mocked model/service
- Compatibility tests require running server with loaded model
- Mark compatibility tests with `@pytest.mark.integration`

---

**Document Status:** Complete
**Total Tasks:** 27 parent tasks, 130+ sub-tasks
**Estimated Timeline:** 1.5-2 weeks
**Next Feature:** Feature 3 - SAE Management
