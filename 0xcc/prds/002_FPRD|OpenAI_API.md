# Feature PRD: OpenAI API Compatibility

## miLLM Feature 2

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**Feature Priority:** Core/MVP (P1)
**References:**
- Project PRD: `000_PPRD|miLLM.md`
- ADR: `000_PADR|miLLM.md`
- BRD: `0xcc/docs/miLLM_BRD_v1.0.md`

---

## 1. Feature Overview

### Feature Name
**OpenAI API Compatibility** - Full implementation of OpenAI v1 API endpoints for seamless client integration.

### Brief Description
OpenAI API Compatibility provides a drop-in replacement for the OpenAI API, enabling any OpenAI-compatible client (Open WebUI, LibreChat, Continue.dev, etc.) to connect to miLLM without modification. This includes chat completions, text completions, embeddings, and model listing endpoints with full streaming support.

### Problem Statement
Users who want to leverage SAE steering capabilities need to integrate miLLM into their existing workflows:
- Many tools and applications are built around the OpenAI API
- Switching inference backends should not require code changes
- Streaming responses are essential for real-time chat experiences
- Users expect familiar API behavior and response formats

### Feature Goals
1. **100% Client Compatibility:** Work with any OpenAI API client without modification
2. **Streaming Support:** Real-time token streaming via SSE for responsive chat
3. **Familiar Behavior:** Response formats match OpenAI exactly
4. **Transparent Steering:** SAE steering applied automatically when configured
5. **Error Parity:** Error responses match OpenAI format for client handling

### User Value Proposition
Users can point any existing OpenAI-compatible tool at miLLM and immediately benefit from SAE steering capabilities without changing their workflows or code.

### Connection to Project Objectives
- **BO-3:** 100% compatibility with OpenAI API clients
- **NFR-1.3:** Time to first token <500ms
- **FR-5.1 through FR-5.6:** Direct implementation requirements

---

## 2. User Stories & Scenarios

### Primary User Stories

#### US-2.1: Chat Completions with Streaming
**As a** developer using Open WebUI
**I want to** connect to miLLM as my OpenAI backend
**So that** I can have steered conversations with streaming responses

**Acceptance Criteria:**
- [ ] Configure Open WebUI with miLLM URL (http://localhost:8000/v1)
- [ ] Chat messages sent to `/v1/chat/completions`
- [ ] Responses stream in real-time (SSE)
- [ ] Active steering profile applied to responses
- [ ] Conversation context maintained across messages

#### US-2.2: Text Completions
**As a** developer building a code completion tool
**I want to** use the completions endpoint
**So that** I can generate text completions with steering

**Acceptance Criteria:**
- [ ] Send prompts to `/v1/completions`
- [ ] Receive generated text with steering applied
- [ ] Support for max_tokens, temperature, top_p parameters
- [ ] Support for stop sequences
- [ ] Both streaming and non-streaming modes work

#### US-2.3: List Available Models
**As a** user setting up a client
**I want to** query available models
**So that** I can select the correct model for my application

**Acceptance Criteria:**
- [ ] GET `/v1/models` returns list of available models
- [ ] Response format matches OpenAI exactly
- [ ] Only loaded model appears in list (miLLM serves one model at a time)
- [ ] Model metadata includes id, created, owned_by fields

#### US-2.4: Generate Embeddings
**As a** developer building a RAG application
**I want to** generate embeddings via the API
**So that** I can build semantic search with steering capabilities

**Acceptance Criteria:**
- [ ] POST `/v1/embeddings` generates vector embeddings
- [ ] Support single string or array of strings
- [ ] Response includes embedding vectors
- [ ] Embeddings can be monitored for feature activations

#### US-2.5: Non-Streaming Chat Response
**As a** developer using a simple HTTP client
**I want to** receive complete responses without streaming
**So that** I can process responses in batch workflows

**Acceptance Criteria:**
- [ ] Set `stream: false` in request
- [ ] Receive complete response in single JSON object
- [ ] Response includes usage statistics
- [ ] Steering applied to full response

### Secondary User Scenarios

#### US-2.6: Error Handling Compatibility
**Scenario:** Client sends invalid request
- System returns OpenAI-format error response
- HTTP status codes match OpenAI behavior
- Error messages are actionable
- Client error handling code works unchanged

#### US-2.7: Request with All Parameters
**Scenario:** Client sends request with all optional parameters
- temperature, top_p, max_tokens respected
- stop sequences halt generation correctly
- frequency_penalty, presence_penalty applied
- n parameter for multiple completions (n=1 only in v1.0)

### Edge Cases and Error Scenarios

#### EC-2.1: No Model Loaded
- **Trigger:** API request when no model is loaded
- **Behavior:** Return 503 Service Unavailable
- **Message:** `{"error": {"message": "No model loaded. Load a model via admin UI.", "type": "service_unavailable", "code": "model_not_loaded"}}`

#### EC-2.2: Invalid Model ID
- **Trigger:** Request specifies model not matching loaded model
- **Behavior:** Return 404 or use loaded model (configurable)
- **Message:** `{"error": {"message": "Model 'gpt-4' not found. Available: gemma-2-2b", "type": "invalid_request_error", "code": "model_not_found"}}`

#### EC-2.3: Context Length Exceeded
- **Trigger:** Input tokens exceed model's context length
- **Behavior:** Return 400 Bad Request
- **Message:** `{"error": {"message": "Context length exceeded. Max: 8192 tokens, got: 10000", "type": "invalid_request_error", "code": "context_length_exceeded"}}`

#### EC-2.4: Generation Timeout
- **Trigger:** Generation takes longer than timeout
- **Behavior:** Return partial response with finish_reason "timeout"
- **Message:** Include `finish_reason: "timeout"` in response

#### EC-2.5: OOM During Generation
- **Trigger:** Out of memory during inference
- **Behavior:** Gracefully fail, disable SAE if enabled, retry
- **Message:** Return error with suggestion to use quantized model

---

## 3. Functional Requirements

### Chat Completions (FR-5.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| OA-C1 | System shall implement POST /v1/chat/completions | Must |
| OA-C2 | System shall support messages array with role/content | Must |
| OA-C3 | System shall support streaming via SSE (stream: true) | Must |
| OA-C4 | System shall support non-streaming responses (stream: false) | Must |
| OA-C5 | System shall apply active steering profile to generation | Must |
| OA-C6 | System shall support system, user, assistant message roles | Must |
| OA-C7 | System shall return usage statistics (prompt_tokens, completion_tokens) | Must |
| OA-C8 | System shall support temperature parameter (0.0-2.0) | Should |
| OA-C9 | System shall support top_p parameter (0.0-1.0) | Should |
| OA-C10 | System shall support max_tokens parameter | Must |
| OA-C11 | System shall support stop sequences | Should |

### Text Completions (FR-5.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| OA-T1 | System shall implement POST /v1/completions | Must |
| OA-T2 | System shall support prompt string input | Must |
| OA-T3 | System shall support streaming responses | Must |
| OA-T4 | System shall apply active steering to generation | Must |
| OA-T5 | System shall support same parameters as chat completions | Should |

### Models Endpoint (FR-5.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| OA-M1 | System shall implement GET /v1/models | Must |
| OA-M2 | System shall return currently loaded model | Must |
| OA-M3 | System shall match OpenAI response format exactly | Must |
| OA-M4 | System shall implement GET /v1/models/{model_id} | Should |

### Embeddings Endpoint (FR-5.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| OA-E1 | System shall implement POST /v1/embeddings | Must |
| OA-E2 | System shall support single string input | Must |
| OA-E3 | System shall support array of strings input | Must |
| OA-E4 | System shall return embedding vectors | Must |
| OA-E5 | System shall trigger feature monitoring if enabled | Must |

### Streaming (FR-5.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| OA-S1 | System shall stream tokens via Server-Sent Events | Must |
| OA-S2 | System shall send [DONE] marker at stream end | Must |
| OA-S3 | System shall format chunks per OpenAI streaming spec | Must |
| OA-S4 | System shall handle client disconnect gracefully | Must |

### Input/Output Specifications

#### Chat Completions Request
```typescript
interface ChatCompletionRequest {
  model: string;                    // Model ID (matched against loaded model)
  messages: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
  stream?: boolean;                 // Default: false
  temperature?: number;             // 0.0-2.0, default: 1.0
  top_p?: number;                   // 0.0-1.0, default: 1.0
  max_tokens?: number;              // Max generation length
  stop?: string | string[];         // Stop sequences
  frequency_penalty?: number;       // -2.0 to 2.0
  presence_penalty?: number;        // -2.0 to 2.0
  user?: string;                    // User identifier (logged, not used)
  // miLLM extension
  profile?: string;                 // Steering profile name (optional override)
}
```

#### Chat Completions Response (Non-Streaming)
```typescript
interface ChatCompletionResponse {
  id: string;                       // "chatcmpl-xxx"
  object: "chat.completion";
  created: number;                  // Unix timestamp
  model: string;                    // Model used
  choices: Array<{
    index: number;
    message: {
      role: "assistant";
      content: string;
    };
    finish_reason: "stop" | "length" | "timeout";
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}
```

#### Streaming Chunk
```typescript
interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: "assistant";           // Only in first chunk
      content?: string;             // Token content
    };
    finish_reason: null | "stop" | "length";
  }>;
}
```

---

## 4. User Experience Requirements

### API Developer Experience

#### Documentation
- OpenAPI/Swagger documentation at `/docs`
- Clear examples for each endpoint
- Error response documentation
- Parameter descriptions and defaults

#### Error Messages
- Match OpenAI error format exactly
- Include actionable information
- Reference miLLM-specific solutions where applicable

### Client Compatibility Requirements

#### Tested Clients
| Client | Version | Status |
|--------|---------|--------|
| Open WebUI | Latest | Required |
| LibreChat | Latest | Required |
| Continue.dev | Latest | Should |
| Python openai library | ^1.0 | Required |
| OpenAI Node.js | ^4.0 | Should |

#### Compatibility Checklist
- [ ] Standard chat conversation works
- [ ] Streaming displays tokens in real-time
- [ ] System prompts are respected
- [ ] Temperature affects randomness
- [ ] Stop sequences halt generation
- [ ] Error handling works unchanged

---

## 5. Data Requirements

### Request Logging (Optional)

```sql
CREATE TABLE request_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(50) NOT NULL,
    endpoint VARCHAR(50) NOT NULL,
    model VARCHAR(100),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    steering_profile VARCHAR(100),
    duration_ms INTEGER,
    status_code INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Data Validation
| Field | Validation |
|-------|------------|
| model | Non-empty string |
| messages | Non-empty array, valid roles |
| temperature | 0.0-2.0 |
| top_p | 0.0-1.0 |
| max_tokens | Positive integer, <= model max |
| stop | String or array of strings, max 4 |

---

## 6. Technical Constraints

### From ADR
- **Backend:** Python 3.11+ / FastAPI
- **Streaming:** Server-Sent Events (SSE)
- **Async:** All inference operations must be async-compatible
- **Queue:** Request queue for concurrent requests

### Performance Requirements

| Metric | Target |
|--------|--------|
| Time to first token | <500ms |
| Token throughput | Model-dependent |
| Concurrent requests | Queue up to 5 |
| Response format overhead | <1ms |

### Compatibility Requirements
- Full OpenAI v1 API compatibility
- No client-side modifications required
- Standard HTTP/SSE protocols

---

## 7. API Specifications

### Endpoints

#### POST /v1/chat/completions
```
Request:
  Content-Type: application/json
  Body: ChatCompletionRequest

Response (non-streaming):
  Content-Type: application/json
  Body: ChatCompletionResponse

Response (streaming):
  Content-Type: text/event-stream
  Body: data: {ChatCompletionChunk}\n\n (repeated)
        data: [DONE]\n\n
```

#### POST /v1/completions
```
Request:
  Content-Type: application/json
  Body: {
    model: string,
    prompt: string,
    max_tokens?: number,
    temperature?: number,
    stream?: boolean,
    ...
  }

Response:
  Content-Type: application/json
  Body: {
    id: string,
    object: "text_completion",
    choices: [{ text: string, finish_reason: string }],
    usage: { prompt_tokens, completion_tokens, total_tokens }
  }
```

#### GET /v1/models
```
Response:
  {
    "object": "list",
    "data": [
      {
        "id": "gemma-2-2b",
        "object": "model",
        "created": 1706627200,
        "owned_by": "miLLM"
      }
    ]
  }
```

#### POST /v1/embeddings
```
Request:
  {
    "model": "gemma-2-2b",
    "input": "Text to embed" | ["Text 1", "Text 2"]
  }

Response:
  {
    "object": "list",
    "data": [
      {
        "object": "embedding",
        "index": 0,
        "embedding": [0.123, -0.456, ...]
      }
    ],
    "model": "gemma-2-2b",
    "usage": { "prompt_tokens": 5, "total_tokens": 5 }
  }
```

### Error Response Format
```json
{
  "error": {
    "message": "Human-readable error message",
    "type": "invalid_request_error" | "authentication_error" | "rate_limit_error" | "server_error",
    "param": "field_name",  // Optional
    "code": "error_code"    // Optional
  }
}
```

---

## 8. Non-Functional Requirements

### Performance

| Requirement | Target |
|-------------|--------|
| Time to first token | <500ms after model loaded |
| Streaming latency | <50ms between tokens |
| Request queue | 5 concurrent requests |
| Response parsing overhead | <1ms |

### Reliability

| Requirement | Target |
|-------------|--------|
| Request success rate | >99% (when model loaded) |
| Graceful degradation | Continue without SAE on OOM |
| Request timeout | 300s default, configurable |

### Scalability
- Single-user focus for v1.0
- Request queue prevents overload
- Memory management via quantization

---

## 9. Feature Boundaries (Non-Goals)

### Explicitly NOT Included in v1.0

| Non-Goal | Rationale |
|----------|-----------|
| Multi-turn function calling | Complexity, limited steering benefit |
| Vision/multimodal inputs | Model-dependent, future feature |
| Multiple completions (n>1) | Limited use case, resource intensive |
| Log probabilities (logprobs) | Implementation complexity |
| Fine-tuning endpoints | Out of scope for inference server |
| Files/Assistants API | OpenAI-specific features |
| Authentication | Trusted local network assumption |

### Future Enhancements (Post v1.0)
- API key authentication
- Function calling support
- Logprobs output option
- Multi-user rate limiting
- Request prioritization

---

## 10. Dependencies

### Feature Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| Model Management | Internal | Feature 1 (Required) |
| Feature Steering | Internal | Feature 4 (Optional - enhances) |
| Profile Management | Internal | Feature 6 (Optional - enables profile param) |

### Library Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| FastAPI | >=0.109 | API framework |
| sse-starlette | >=1.6 | SSE streaming |
| transformers | >=4.36 | Tokenization, generation |
| torch | >=2.0 | Inference |

---

## 11. Success Criteria

### Quantitative Metrics

| Metric | Target |
|--------|--------|
| OpenAI client compatibility | 100% for tested clients |
| Time to first token | <500ms |
| Streaming chunk latency | <50ms |
| API response format accuracy | 100% match |

### User Satisfaction Indicators
- Users can configure Open WebUI without documentation
- Existing OpenAI code works without modification
- Error messages enable self-service troubleshooting
- Streaming feels responsive and natural

### Completion Criteria
- [ ] All endpoints implemented per specification
- [ ] Open WebUI integration tested and working
- [ ] LibreChat integration tested and working
- [ ] Python openai library compatibility confirmed
- [ ] Streaming performance targets met
- [ ] Error responses match OpenAI format
- [ ] Documentation complete

---

## 12. Testing Requirements

### Unit Testing
- Request parsing and validation
- Response formatting
- Token counting
- Error response generation

### Integration Testing
- End-to-end chat completion flow
- Streaming response handling
- Embeddings generation
- Models endpoint

### Compatibility Testing
```python
# Example: OpenAI Python library test
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # miLLM doesn't require auth
)

response = client.chat.completions.create(
    model="gemma-2-2b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### Performance Testing
- Time to first token benchmark
- Streaming throughput measurement
- Concurrent request handling

---

## 13. Implementation Considerations

### Complexity Assessment

| Component | Complexity | Risk |
|-----------|------------|------|
| Chat completions | Medium | Response format accuracy |
| Streaming (SSE) | Medium | Client compatibility |
| Tokenization | Low | Library handles |
| Error handling | Low | Format matching |
| Embeddings | Medium | Model support varies |

### Recommended Implementation Order
1. Models endpoint (simple, validates setup)
2. Non-streaming chat completions
3. Streaming chat completions (SSE)
4. Text completions
5. Embeddings
6. Error handling polish

### Technical Challenges
- SSE streaming with proper chunk formatting
- Accurate token counting for usage stats
- Graceful handling of inference errors
- Steering integration without breaking format

---

## 14. Open Questions

### Resolved
| Question | Resolution |
|----------|------------|
| Model ID mismatch behavior? | Return error with available model name |
| Unsupported parameters? | Ignore silently, log warning |

### Questions for TDD
1. Should we implement a model alias system (e.g., "gpt-3.5-turbo" → loaded model)?
2. How to handle context length for different models?
3. Should embeddings use same model or separate embedding model?

---

**Document Status:** Complete
**Next Document:** `002_FTDD|OpenAI_API.md` (Technical Design Document)
**Instruction File:** `@0xcc/instruct/004_create-tdd.md`
