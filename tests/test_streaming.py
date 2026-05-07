from unittest.mock import patch

import pytest

from app.schemas.chat import ChatRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect(agen):
    """Drain an async generator into a list."""
    items = []
    async for item in agen:
        items.append(item)
    return items


@pytest.fixture
def streaming_service(llm_service_factory):
    """LLMService with Redis disabled so is_provider_healthy always passes."""
    service = llm_service_factory()
    service._get_redis_safe = lambda: None
    return service


# ---------------------------------------------------------------------------
# Test 1: no fallback after first byte is sent to the client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_fallback_after_first_stream_byte(streaming_service):
    """Once response_started=True, an exception must NOT fall back to the
    next provider.  The stream simply stops and returns."""
    service = streaming_service
    gemini_stream_called = False

    async def ollama_stream(*args, **kwargs):
        yield "hello."          # ends with punct → buffer flushes → response_started=True
        raise RuntimeError("mid-stream failure")

    async def gemini_stream(*args, **kwargs):
        nonlocal gemini_stream_called
        gemini_stream_called = True
        yield "should not appear"

    service.ollama.stream = ollama_stream
    service.gemini.stream = gemini_stream

    chunks = await _collect(
        service.stream_response(ChatRequest(message="hi", preferred_provider="auto"))
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == "hello."
    assert chunks[0]["provider"] == "ollama"
    assert not gemini_stream_called, "fallback must not happen after first byte was sent"


# ---------------------------------------------------------------------------
# Test 2: buffer flushes when a chunk ends with punctuation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_buffer_flush_on_punctuation(streaming_service):
    """Small chunks that don't individually exceed 40 chars must be held in
    the buffer and released together when a punctuation boundary is hit."""
    service = streaming_service

    async def ollama_stream(*args, **kwargs):
        yield "hi"   # no punctuation, buffer holds
        yield "!"    # punctuation → should_flush=True → emit "hi!"

    service.ollama.stream = ollama_stream

    chunks = await _collect(
        service.stream_response(ChatRequest(message="hello", preferred_provider="auto"))
    )

    assert len(chunks) == 1, f"expected one flushed chunk, got {chunks}"
    assert chunks[0]["content"] == "hi!"
    assert chunks[0]["provider"] == "ollama"


# ---------------------------------------------------------------------------
# Test 3: elapsed-time guard breaks the loop on timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_timeout_breaks_loop(streaming_service):
    """The inner async-for loop must break (no exception raised to caller)
    when the elapsed time since stream_start exceeds timeout_s."""
    service = streaming_service

    async def ollama_infinite(*args, **kwargs):
        yield "."       # first chunk: punctuation → flush immediately
        while True:
            yield "x"   # subsequent chunks; time guard should stop these

    service.ollama.stream = ollama_infinite

    # time.monotonic call order inside stream_response for auto/rules path:
    #   1-2  _resolve_route (start, observe)
    #   3    last_flush = time.monotonic()
    #   4    stream_start = time.monotonic()
    #   5    chunk "." timeout check  (0.0 - 0.0 = 0 < 30 → ok)
    #   6    now = time.monotonic()   (flush timing for ".")
    #   7    chunk "x" timeout check  (35.0 - 0.0 = 35 > 30 → break)
    monotonic_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.0] + [35.0] * 10

    with patch("app.services.llm_services.time.monotonic", side_effect=monotonic_values):
        chunks = await _collect(
            service.stream_response(
                ChatRequest(message="hi", preferred_provider="auto", timeout_ms=30000)
            )
        )

    # At minimum the first flushed chunk ("." from ollama) must be yielded.
    assert len(chunks) >= 1
    assert all(c["provider"] == "ollama" for c in chunks)
