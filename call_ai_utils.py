from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from groq import Groq


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
) -> list[ChatLikeResponse]:
    if concurrent_calls < 1:
        raise ValueError("`concurrent_calls` must be >= 1")

    def _one_call() -> ChatLikeResponse:
        client = _make_client()
        raw = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            #reasoning_effort=reasoning_effort,
            stream=False,
        )
        content = raw.choices[0].message.content or ""
        return ChatLikeResponse(
            choices=[_Choice(index=0, message=_Message(role="assistant", content=content))],
            raw=raw,
        )

    with ThreadPoolExecutor(max_workers=concurrent_calls) as executor:
        futures = [executor.submit(_one_call) for _ in range(concurrent_calls)]
        return [future.result() for future in futures]


if __name__ == "__main__":
    raise SystemExit("Import `call_ai` from this module; this file is not meant to be run directly.")
