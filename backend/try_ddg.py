#!/usr/bin/env python3
"""Standalone probe for the DDG search path.

Run this from backend/ (next to search.py) to confirm DuckDuckGo returns
results through the *exact* provider code nChat will use -- before touching
main.py. Proves the transport in isolation; if this prints results, the
toggle will too.

    pip install ddgs
    python try_ddg.py "latest arista eos release"
"""
import os
import sys
import asyncio

# Force the DDG provider regardless of any other env you've set.
os.environ["NCHAT_SEARCH_PROVIDER"] = "ddg"

import search as websearch  # noqa: E402


async def main():
    query = " ".join(sys.argv[1:]) or "what is BGP route reflection"

    provider = websearch.get_provider()
    if provider is None:
        print("No provider resolved. Is ddgs installed? `pip install ddgs`")
        return 1
    print(f"provider: {provider.name}")
    print(f"query:    {query}\n")

    try:
        results = await provider.search(websearch.shape_query(query), max_results=5)
    except websearch.SearchError as e:
        # This is the same path main.py catches -> model is told the web was
        # unreachable. DDG rate-limits aggressively, so a block surfaces here.
        print(f"SearchError (handled gracefully in nChat): {e}")
        return 2

    if not results:
        print("Provider returned 0 results (handled as 'no results' in nChat).")
        return 3

    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.title}")
        print(f"    {r.url}")
        if r.snippet:
            print(f"    {r.snippet[:140]}")
        print()

    print("--- exactly what gets injected into the turn ---\n")
    print(websearch.format_results_block(query, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))