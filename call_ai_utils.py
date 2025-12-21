from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random
import time
from typing import Any, Optional

from groq import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    Groq,
    InternalServerError,
    RateLimitError,
)


@dataclass(frozen=True)
class _Message:
    role: str
    content: str


@dataclass(frozen=True)
class _Choice:
    index: int
    message: _Message


@dataclass(frozen=True)
class ChatLikeResponse:
    """
    Minimal adapter so the rest of the codebase can keep using:
      resp.choices[0].message.content
    while we call the OpenAI Responses API under the hood.
    """

    choices: list[_Choice]
    raw: Any


def _make_client() -> Groq:
    return Groq()


def call_ai(
    prompt: str,
    concurrent_calls: int = 1,
    temperature: float = 1.0,
    *,
    model: str = "moonshotai/kimi-k2-instruct-0905",#moonshotai/kimi-k2-instruct-0905, openai/gpt-oss-120b
    system_prompt: str = "",
    max_completion_tokens: Optional[int] = 8192,
    top_p: Optional[float] = None,
    reasoning_effort: Optional[str] = "medium",
    max_retries: int = 6,
    initial_backoff_s: float = 1.0,
    max_backoff_s: float = 30.0,
) -> list[ChatLikeResponse]:
    if concurrent_calls < 1:
        raise ValueError("`concurrent_calls` must be >= 1")

    def _retry_after_s(err: BaseException) -> float | None:
        try:
            if isinstance(err, APIStatusError):
                ra = err.response.headers.get("retry-after")
                if ra is None:
                    return None
                # Retry-After can be seconds or a HTTP date; we only handle seconds.
                secs = float(ra)
                return secs if secs >= 0 else None
        except Exception:
            return None
        return None

    def _is_retryable(err: BaseException) -> bool:
        if isinstance(err, (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError)):
            return True
        if isinstance(err, APIStatusError):
            return err.status_code in (500, 502, 503, 504)
        return False

    def _one_call() -> ChatLikeResponse:
        client = _make_client()
        attempts = 0
        while True:
            try:
                raw = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    top_p=top_p,
                    # reasoning_effort=reasoning_effort,
                    stream=False,
                )
                content = raw.choices[0].message.content or ""
                return ChatLikeResponse(
                    choices=[_Choice(index=0, message=_Message(role="assistant", content=content))],
                    raw=raw,
                )
            except Exception as e:
                attempts += 1
                if attempts > max(0, int(max_retries)) or not _is_retryable(e):
                    raise
                retry_after = _retry_after_s(e)
                if retry_after is not None:
                    sleep_s = min(max_backoff_s, max(0.0, retry_after))
                else:
                    base = float(initial_backoff_s) * (2.0 ** (attempts - 1))
                    # Small jitter to avoid thundering herd when running concurrently.
                    sleep_s = min(max_backoff_s, base * random.uniform(0.8, 1.2))
                time.sleep(max(0.0, sleep_s))

    with ThreadPoolExecutor(max_workers=concurrent_calls) as executor:
        futures = [executor.submit(_one_call) for _ in range(concurrent_calls)]
        return [future.result() for future in futures]


if __name__ == "__main__":
    raise SystemExit("Import `call_ai` from this module; this file is not meant to be run directly.")
