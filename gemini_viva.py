"""Optional Gemini API helper for broad-knowledge, four-option viva question generation."""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lab_companion import retrieve

MODEL = "gemini-3.5-flash-lite"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def generate_with_gemini(
    api_key: str,
    document: str,
    topic: str,
    count: int,
    difficulty: str = "medium",
    include_document: bool = False,
):
    """Generate real four-choice viva questions from model knowledge, with optional manual context."""
    if not topic.strip():
        raise ValueError("Enter a topic first, such as meter bridge, enthalpy of neutralisation, or Experiment 3.2.")

    matches = retrieve(topic, document, limit=4) if include_document and document.strip() else []
    excerpts = "\n\n".join(
        f"Source: {item['source']}\n{item['text'][:1800]}" for item in matches
    )[:6000]
    if include_document and not excerpts:
        excerpts = "No relevant passage was found in the uploaded manual. Use standard course knowledge and do not pretend the manual supports an answer."

    context_note = (
        f"Treat the uploaded passages as optional context to tailor the questions; use standard course knowledge too.\n{excerpts}"
        if include_document
        else "No uploaded document is being sent. Use standard textbook-level and practical-viva knowledge for this topic."
    )
    prompt = f"""You are an experienced examiner preparing a realistic engineering/physics/chemistry practical viva quiz.
Create exactly {count} concise multiple-choice viva questions on this topic: {topic!r}.
Difficulty: {difficulty}. Make the quiz useful for a student preparing for an oral exam, not a reading-comprehension test.
Use broad, reliable course knowledge; the uploaded manual is optional context, not the only source. Ask a mix of principle/concept, formula or units, apparatus or method reasoning, observations, and common sources of error when relevant. Keep safety questions general and conservative.
Each question must have exactly four short, plausible answer options and exactly one clearly correct answer. Avoid silly distractors, repeated questions, unsupported exact values, and unsafe procedural advice. Give a brief explanation of the correct answer. Do not claim to browse or cite sources you did not consult.
{context_note}
Return only valid JSON in this shape (correct_index is zero-based): {{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."], "correct_index": 0, "explanation": "..."}}]}}"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.35,
            "maxOutputTokens": 2600,
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key.strip()},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:700]
        if exc.code in (401, 403):
            raise RuntimeError("Google rejected the API key or project access. Check the key in Google AI Studio.") from None
        if exc.code == 404:
            raise RuntimeError("Google could not find the selected Gemini model for this account. Use offline questions or check the app version.") from None
        if exc.code == 429:
            raise RuntimeError("Gemini's free request limit may have been reached. Wait and retry, or use offline questions.") from None
        if exc.code == 503:
            raise RuntimeError("Google's Gemini service is busy right now. Wait a little and retry, or use offline questions.") from None
        raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail}") from None
    except URLError:
        raise RuntimeError("Could not reach Google. Check your internet connection, or use offline questions.") from None
    except TimeoutError:
        raise RuntimeError("Gemini took too long to reply. Retry, or use offline questions.") from None

    try:
        text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        questions = json.loads(text)["questions"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        raise RuntimeError("Gemini returned an unexpected response. Try again or use offline questions.") from None

    cleaned = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        options = item.get("options")
        answer_index = item.get("correct_index")
        if (not question or not isinstance(options, list) or len(options) != 4
                or not isinstance(answer_index, int) or isinstance(answer_index, bool)
                or answer_index not in range(4)):
            continue
        cleaned.append({
            "question": question,
            "options": [str(option).strip() for option in options],
            "correct_index": answer_index,
            "explanation": str(item.get("explanation", "Review this concept in your course material.")).strip(),
            "source": "Gemini general knowledge" + (" + uploaded manual context" if include_document else ""),
            "reason": "AI-generated from broad course knowledge; not a live web search.",
        })
        if len(cleaned) == count:
            break
    if len(cleaned) < count:
        raise RuntimeError("Gemini did not return enough valid four-choice questions. Retry or use offline questions.")
    return cleaned
