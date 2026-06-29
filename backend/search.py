"""nChat - web search providers.

Local-first by default. The reference provider is a self-hosted SearXNG
instance, so a search turn never depends on a third-party API key and the
query never leaves your box. Brave is offered as a keyed fallback for people
who'd rather not run another container, and DuckDuckGo (via the optional
`ddgs` package) as a no-setup tinkering option.

The contract is deliberately small so the provider is a config swap, not a
rewrite: the rest of the app only ever sees a `list[SearchResult]`. Same
seed-artifact / no-runtime-coupling idea as the topology import elsewhere in
the stack -- downstream code couples to the shape, never the source.

Configuration (env):
    NCHAT_SEARCH_PROVIDER   "searxng" | "brave" | "ddg"   (optional; inferred
                            from whichever of the below is set)
    SEARXNG_URL             e.g. http://localhost:8888
    BRAVE_API_KEY           Brave Search API subscription token
"""
from __future__ import annotations

import os
import re
import asyncio
from dataclasses import dataclass, asdict

import httpx


# Search engines (DDG especially) sometimes concatenate related-link text onto
# the tail of a result snippet -- e.g. "...iBGP peers, ...Understanding BGP ·"
# or "...not actually ...BGP route reflectors ...More results from reddit.com".
# Two tells: a literal "More results from ..." tail, and an ellipsis glued
# directly to a word with no following space ("...Word"), which is how DDG
# splices related-link titles on. Legitimate in-snippet ellipsis keeps its
# spaces ("... word"), so keying on the no-space form is safe. Provider-
# agnostic: a no-op on already-clean snippets (Brave/SearXNG).
_MORE_RESULTS_RE = re.compile(r"\s*More results from\b.*$", re.IGNORECASE | re.DOTALL)
_GLUE_RE = re.compile(r"(?:\.\.\.|\u2026)(?=\S)")  # "...Word" / "…Word" run-on


def clean_snippet(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    s = " ".join(text.split())
    s = _MORE_RESULTS_RE.sub("", s)
    m = _GLUE_RE.search(s)
    if m:
        s = s[: m.start()]
    s = s.rstrip(" ,.;:\u00b7-\u2026").strip()
    if len(s) <= max_len:
        return s
    # Length backstop: cut on a word boundary, add an ellipsis.
    return s[:max_len].rsplit(" ", 1)[0].rstrip(" ,.;:\u00b7-") + "\u2026"


# ---------------------------------------------------------------------------
# Result shape + failure signal
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def __post_init__(self):
        # Clean at the boundary so every consumer -- the injected context, the
        # SSE `sources` payload, the persisted row -- sees normalized text.
        self.snippet = clean_snippet(self.snippet)

    def as_dict(self) -> dict:
        return asdict(self)


class SearchError(Exception):
    """Provider/transport failure. Caught by the caller so the model is told
    the web was UNREACHABLE and answers accordingly, rather than silently
    producing unsourced claims under a 'searched the web' banner."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class WebSearchProvider:
    name = "base"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class SearxngProvider(WebSearchProvider):
    """Self-hosted SearXNG. Run one next to Ollama:

        docker run -d --name searxng -p 8888:8080 \
            -e "BASE_URL=http://localhost:8888/" searxng/searxng

    Then enable JSON output in its settings.yml (search.formats: [html, json]).
    No API key; the query never leaves your network.
    """
    name = "searxng"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        params = {"q": query, "format": "json"}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(
                    f"{self.base_url}/search",
                    params=params,
                    headers={"Accept": "application/json"},
                )
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001 - normalize to one signal
            raise SearchError(f"SearXNG request failed: {e}") from e

        out: list[SearchResult] = []
        for item in (data.get("results") or [])[:max_results]:
            out.append(SearchResult(
                title=(item.get("title") or "").strip(),
                url=(item.get("url") or "").strip(),
                snippet=(item.get("content") or "").strip(),
            ))
        return out


class BraveProvider(WebSearchProvider):
    """Brave Search API. One subscription token, independent index."""
    name = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": max_results}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(self.ENDPOINT, params=params, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            raise SearchError(f"Brave request failed: {e}") from e

        web = (data.get("web") or {}).get("results") or []
        out: list[SearchResult] = []
        for item in web[:max_results]:
            out.append(SearchResult(
                title=(item.get("title") or "").strip(),
                url=(item.get("url") or "").strip(),
                snippet=(item.get("description") or "").strip(),
            ))
        return out


class DDGSProvider(WebSearchProvider):
    """DuckDuckGo via the optional `ddgs` package. No key, no setup. ToS-grey
    and rate-limited -- fine for tinkering, not what you'd ship under your
    name. The import is deferred to search() so a missing dependency surfaces
    as a clean 'search unavailable' notice through the caller's error handling,
    never as a 500 out of the provider factory."""
    name = "ddg"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError as e:
            raise SearchError("ddgs not installed: pip install ddgs") from e

        def _run():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        try:
            rows = await asyncio.to_thread(_run)
        except Exception as e:  # noqa: BLE001
            raise SearchError(f"DDG request failed: {e}") from e

        out: list[SearchResult] = []
        for item in rows[:max_results]:
            out.append(SearchResult(
                title=(item.get("title") or "").strip(),
                url=(item.get("href") or item.get("url") or "").strip(),
                snippet=(item.get("body") or "").strip(),
            ))
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider() -> WebSearchProvider | None:
    """Pick a provider from the environment. Returns None when search is not
    configured so the caller can degrade cleanly (toggle becomes a no-op with
    a visible notice rather than an error)."""
    name = os.getenv("NCHAT_SEARCH_PROVIDER", "").strip().lower()
    searx = os.getenv("SEARXNG_URL", "").strip()
    brave = os.getenv("BRAVE_API_KEY", "").strip()

    if not name:
        # Infer from whatever credentials/URLs are present.
        if searx:
            name = "searxng"
        elif brave:
            name = "brave"
        else:
            return None

    if name == "searxng":
        return SearxngProvider(searx) if searx else None
    if name == "brave":
        return BraveProvider(brave) if brave else None
    if name in ("ddg", "ddgs", "duckduckgo"):
        return DDGSProvider()
    return None


# ---------------------------------------------------------------------------
# Query shaping + result framing + grounding contract
# ---------------------------------------------------------------------------

def shape_query(message: str, max_len: int = 320) -> str:
    """v1 query shaping: the raw user turn, whitespace-collapsed and trimmed.
    Good enough for most questions. The obvious upgrade is a one-shot LLM
    rewrite against the same Ollama model -- left as a hook rather than baked
    in, to keep a search turn to a single round trip."""
    return " ".join(message.split())[:max_len]


def format_results_block(query: str, results: list[SearchResult]) -> str:
    """Render results as a numbered, citeable context block. The number is the
    citation key the model is told to use ([1], [2], ...)."""
    lines = [f"WEB SEARCH RESULTS for: {query}", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    URL: {r.url}")
        if r.snippet:
            lines.append(f"    {r.snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


# Appended to the system message on a search turn. The abstention clause is the
# point: distinguish what the sources support from what the model is inferring.
SEARCH_SYSTEM_ADDENDUM = (
    "You have been given WEB SEARCH RESULTS below the user's question. Ground "
    "your answer in them and cite sources inline as [1], [2], etc., matching "
    "the numbered results. Paraphrase; quote sparingly and briefly. If the "
    "results do not actually contain what is needed to answer, say so plainly "
    "and do not fill the gap with unsourced claims -- clearly separate what the "
    "sources support from what you are inferring."
)

# Used when a search was requested but the provider failed or returned nothing.
SEARCH_UNAVAILABLE_ADDENDUM = (
    "Note: a web search was attempted for this turn but no results were "
    "available. Answer from your own knowledge, and state that live results "
    "were unavailable rather than implying you searched the web."
)