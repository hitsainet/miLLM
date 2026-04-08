---
sidebar_position: 1
title: OpenAI-Compatible API
---

# OpenAI-Compatible API

miLLM exposes an OpenAI-compatible API at `/v1`, making it a drop-in replacement for the OpenAI SDK.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming and non-streaming) |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |

## Usage with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://millm.hitsai.local/v1",
    api_key="not-needed"  # miLLM doesn't require auth
)

response = client.chat.completions.create(
    model="gemma-2-2b-it",  # Must match loaded model name
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

## Streaming

```python
stream = client.chat.completions.create(
    model="gemma-2-2b-it",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Profile-Based Steering via API

Use the `profile` parameter to apply a saved steering profile per-request:

```python
response = client.chat.completions.create(
    model="gemma-2-2b-it",
    messages=[{"role": "user", "content": "What is truth?"}],
    extra_body={"profile": "honesty-amplification"},
)
```

:::tip Integration with Other Tools
- **Open WebUI:** Set the OpenAI API base URL to your miLLM instance
- **miStudio Labeling:** Use "OpenAI Compatible" method with miLLM's `/v1` endpoint
- **LangChain/LlamaIndex:** Use the OpenAI provider pointed at miLLM
:::

## Steered Inference

When steering is enabled via the admin UI, **all** API requests are steered. The steering affects the model's residual stream at the attached SAE layer. To use unsteered inference while steering is configured, disable steering from the UI or use a profile-less request.
