import inspect
import logging
import os
import time

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("[OPENAI SMOKE] Invalid int for %s='%s', using default=%s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("[OPENAI SMOKE] Invalid float for %s='%s', using default=%s", name, value, default)
        return default


OPENAI_SMOKE_DEFAULT_MODEL = os.environ.get("OPENAI_SMOKE_MODEL", "gpt-4.1-mini")
OPENAI_SMOKE_MAX_RETRIES = _env_int("OPENAI_SMOKE_MAX_RETRIES", 2)
OPENAI_SMOKE_TIMEOUT_SECONDS = _env_float("OPENAI_SMOKE_TIMEOUT_SECONDS", 30.0)


class OpenAISmokeRequest(BaseModel):
    model: str = Field(default=OPENAI_SMOKE_DEFAULT_MODEL, description="OpenAI model to test")
    prompt: str = Field(
        default="Reply with exactly OPENAI_SMOKE_OK.",
        description="Simple user prompt for connectivity test",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=64, ge=1, le=1024)
    response_format_json: bool = Field(
        default=False,
        description="If true, request JSON output mode from chat completions",
    )
    include_raw_response: bool = Field(
        default=True,
        description="If true, include full chat completion payload in the response",
    )


def _error_chain(exc: Exception, max_depth: int = 4) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    current = exc.__cause__ or exc.__context__
    depth = 0
    while current is not None and depth < max_depth:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
        depth += 1
    return " | caused by: ".join(parts)


async def _close_client(client: AsyncOpenAI) -> None:
    close_fn = getattr(client, "close", None)
    if close_fn is None:
        return
    try:
        maybe_awaitable = close_fn()
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
    except Exception as exc:
        logger.warning("[OPENAI SMOKE] Failed to close AsyncOpenAI client cleanly: %s", exc)


async def run_openai_async_smoke(request: OpenAISmokeRequest) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY_VALUE", "").strip()
    base_result = {
        "ok": False,
        "provider": "openai",
        "api_key_present": bool(api_key),
        "service": os.environ.get("K_SERVICE", ""),
        "revision": os.environ.get("K_REVISION", ""),
        "configuration": os.environ.get("K_CONFIGURATION", ""),
        "model": request.model,
    }

    if not api_key:
        base_result["error"] = "OPENAI_API_KEY_VALUE is missing or empty"
        return base_result

    client = AsyncOpenAI(
        api_key=api_key,
        max_retries=OPENAI_SMOKE_MAX_RETRIES,
        timeout=OPENAI_SMOKE_TIMEOUT_SECONDS,
    )
    started = time.perf_counter()
    try:
        request_kwargs = {}
        if request.response_format_json:
            request_kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a connectivity smoke-test assistant.",
                },
                {
                    "role": "user",
                    "content": request.prompt,
                },
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            **request_kwargs,
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        content = response.choices[0].message.content if response.choices else ""
        return {
            **base_result,
            "ok": True,
            "latency_ms": latency_ms,
            "request_id": getattr(response, "_request_id", None),
            "finish_reason": response.choices[0].finish_reason if response.choices else None,
            "response_text": content or "",
            "response_preview": (content or "")[:500],
            "chat_completion": response.model_dump() if request.include_raw_response else None,
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            **base_result,
            "ok": False,
            "latency_ms": latency_ms,
            "error_type": type(exc).__name__,
            "error": _error_chain(exc),
        }
    finally:
        await _close_client(client)
