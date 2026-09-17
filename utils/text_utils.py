"""
utils/text_utils.py

Responsible for cleaning, deduplicating, and truncating search
results before they are sent to Gemini. This keeps token usage
low and removes noisy/duplicate data.
"""

from config import MAX_SEARCH_RESULTS, MAX_CONTENT_LENGTH


def deduplicate_results(results: list[dict]) -> list[dict]:
    """Remove results that share the same URL (or near-identical titles)."""
    seen_urls = set()
    seen_titles = set()
    unique_results = []

    for result in results:
        url = result.get("url", "").strip().rstrip("/")
        title = result.get("title", "").strip().lower()

        if url in seen_urls or (title and title in seen_titles):
            continue

        seen_urls.add(url)
        if title:
            seen_titles.add(title)
        unique_results.append(result)

    return unique_results


def truncate_content(text: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Collapse whitespace and cut long text down to a max length."""
    if not text:
        return ""

    cleaned = " ".join(text.split())

    if len(cleaned) <= max_length:
        return cleaned

    # Cut at the last full word before the limit, then add an ellipsis.
    truncated = cleaned[:max_length].rsplit(" ", 1)[0]
    return truncated + "..."


def prepare_sources_for_gemini(results: list[dict]) -> list[dict]:
    """
    Full pipeline: dedupe -> limit count -> truncate content -> number sources.

    Returns a list of dicts ready to be formatted into the Gemini prompt:
        {"number": 1, "title": ..., "url": ..., "content": ...}
    """
    deduped = deduplicate_results(results)
    limited = deduped[:MAX_SEARCH_RESULTS]

    prepared = []
    for index, result in enumerate(limited, start=1):
        prepared.append({
            "number": index,
            "title": result.get("title", "Untitled source"),
            "url": result.get("url", ""),
            "content": truncate_content(result.get("content", "")),
        })

    return prepared


def split_followups(analysis_text: str) -> tuple[str, list[str]]:
    """
    Split Gemini's Markdown output into:
        (main_analysis_text, [follow_up_question_1, question_2, question_3])

    Looks for the "### Suggested Follow-Up Questions" heading that the
    Gemini prompt asks for. If it's missing (e.g. the model skipped it),
    the full text is returned with an empty question list — nothing breaks.
    """
    marker = "### Suggested Follow-Up Questions"

    if marker not in analysis_text:
        return analysis_text.strip(), []

    main_part, _, followup_part = analysis_text.partition(marker)

    questions = []
    for line in followup_part.splitlines():
        line = line.strip()
        line = line.lstrip("-*•").strip()
        # Also strip a leading "1. " / "2)" style numbering, if present.
        for sep in [". ", ") "]:
            if sep in line[:4]:
                line = line.split(sep, 1)[-1].strip()
                break
        if line:
            questions.append(line)

    return main_part.strip(), questions[:3]


def format_sources_block(prepared_sources: list[dict]) -> str:
    """
    Turn the prepared sources into a clearly separated text block
    that gets inserted into the Gemini prompt.
    """
    blocks = []
    for source in prepared_sources:
        block = (
            f"Source {source['number']}: {source['title']}\n"
            f"{source['content']}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)