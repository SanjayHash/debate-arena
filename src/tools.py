from ddgs import DDGS
from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for concise facts, statistics, or counter-evidence on a debate topic."""
    try:
        results = DDGS().text(query, max_results=3)
    except Exception as exc:
        return f"Search failed: {exc}"

    if not results:
        return f"No web results found for: {query}"

    formatted: list[str] = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "No title")
        body = (item.get("body", "") or "")[:240]
        formatted.append(f"[{i}] {title}: {body}")

    return "\n".join(formatted)