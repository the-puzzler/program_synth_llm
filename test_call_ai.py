from __future__ import annotations

from pprint import pprint

from call_ai_utils import call_ai


def _extract_text(response) -> str:
    try:
        return response.choices[0].message.content  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return response.choices[0].text  # type: ignore[attr-defined]
    except Exception:
        pass
    return str(response)


def main() -> None:
    prompt = "Say 'ok' and then give a 1-sentence summary of what 2+2 is."
    responses = call_ai(prompt, concurrent_calls=2, temperature=0.7)

    print(f"got {len(responses)} responses")
    for i, r in enumerate(responses, start=1):
        print(f"\n--- response {i} ---")
        print(_extract_text(r))

    print("\n--- raw object 1 (debug) ---")
    print(responses[0])


if __name__ == "__main__":
    main()

