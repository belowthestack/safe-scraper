r"""
MCP server wrapping the personal scraper as a Claude Code native tool.

Exposes three tools:
  - scrape_url: fetches a single URL and returns clean AI-readable plain text
  - scrape_batch: scrapes multiple URLs in parallel using a thread pool
  - scrape_url_with_credibility: fetches a URL and runs a credibility-risk
    triage on the returned content — scores and flags signals for human review
    without blocking access to the content

The credibility layer is advisory.  The scraper always returns text; the
credibility score is a risk signal to guide prioritisation, not a gate.
See KNOWN_LIMITS.md for failure modes before using this in automated pipelines.

Run with: python mcp_server.py
Register with Claude Code: claude mcp add scraper -- python C:\Users\Arfa\Desktop\tools\scraper\mcp_server.py
"""

import asyncio
import concurrent.futures
from functools import partial

from mcp.server.fastmcp import FastMCP

from credibility import assess_credibility
from scraper import scrape, scrape_with_raw

mcp = FastMCP("scraper")


@mcp.tool()
async def scrape_url(url: str, force_dynamic: bool = False, force_static: bool = False) -> str:
    """Fetch a single URL and return clean AI-readable plain text.

    Uses a hybrid static-first then Playwright-fallback strategy by default.
    Set force_dynamic=True to skip the static attempt and always use headless
    Chromium. Set force_static=True to disable the Playwright fallback entirely
    and only use the fast HTTP path.

    Returns the scraped plain text on success, or a string beginning with
    'ERROR: ' describing the failure if anything goes wrong.

    Args:
        url: The URL to fetch.
        force_dynamic: If True, skip static attempt and use Playwright directly.
        force_static: If True, use requests only — no browser fallback.

    Returns:
        Scraped plain text, or an 'ERROR: <type>: <message>' string on failure.
    """
    try:
        return await asyncio.to_thread(
            scrape, url, force_dynamic=force_dynamic, force_static=force_static
        )
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@mcp.tool()
async def scrape_batch(
    urls: list[str],
    force_dynamic: bool = False,
    force_static: bool = False,
    max_workers: int = 5,
) -> dict[str, str]:
    """Scrape multiple URLs in parallel and return a mapping of URL to content.

    Uses a thread pool to fetch all URLs concurrently. Each URL is scraped with
    the same hybrid static-first / Playwright-fallback strategy as scrape_url.
    Failed URLs are stored as 'ERROR: <type>: <message>' strings rather than
    raising exceptions, so partial results are always returned.

    max_workers is clamped between 1 and 10 to avoid overwhelming target sites.
    The default of 5 is a safe balance between speed and politeness.

    Args:
        urls: List of URLs to scrape.
        force_dynamic: If True, use Playwright for every URL (skips static attempt).
        force_static: If True, use requests only for every URL (no browser fallback).
        max_workers: Number of parallel threads. Clamped to [1, 10]. Default 5.

    Returns:
        Dict mapping each URL to its scraped plain text or an 'ERROR: ...' string.
    """
    max_workers = max(1, min(10, max_workers))

    def _run_batch() -> dict[str, str]:
        results: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url: dict[concurrent.futures.Future[str], str] = {
                executor.submit(
                    scrape, url, force_dynamic=force_dynamic, force_static=force_static
                ): url
                for url in urls
            }
            for future in concurrent.futures.as_completed(future_to_url):
                u = future_to_url[future]
                try:
                    results[u] = future.result()
                except Exception as exc:
                    results[u] = f"ERROR: {type(exc).__name__}: {exc}"
        return results

    return await asyncio.to_thread(_run_batch)


@mcp.tool()
async def scrape_url_with_credibility(
    url: str,
    force_dynamic: bool = False,
    force_static: bool = False,
) -> dict:
    """Fetch a URL and return scraped text plus a credibility-risk triage.

    Combines the scraper with the credibility-risk layer to give a structured
    four-component result every time:
      1. Raw content      — "text" key: scraped plain text, always present
      2. Extracted signals — "credibility.red_flags" (per-tactic evidence) and
                             "credibility.positive_signals"
      3. Assessment       — "credibility.credibility_score" (0–100),
                             "credibility.risk_level", "credibility.confidence",
                             "credibility.assessment_summary"
      4. Recommendation   — "credibility.recommendation":
                             "use_freely" | "use_with_caution" | "avoid"

    The credibility layer is advisory.  The scraper always returns text
    regardless of the score — the assessment flags signals for human review,
    it does not block access.  Confidence degrades for short or ambiguous
    content (see credibility.py and KNOWN_LIMITS.md for details).

    Scoring: baseline 70, -8 to -12 per detected tactic, +5 per positive
    signal, -15 extra penalty when 3+ tactics detected.  Trusted healthcare/
    news domains (HIMSS, AHA, CMS, Reuters, NYT, Fierce Healthcare, etc.)
    are protected by a 70-point score floor.  No external API is called.

    Args:
        url: The URL to fetch and assess.
        force_dynamic: If True, skip static attempt and use Playwright directly.
        force_static: If True, use requests only — no browser fallback.

    Returns:
        A dict with two top-level keys:
          "text"        — scraped plain text (or "ERROR: ..." string on failure)
          "credibility" — CredibilityReport dict (all four components always
                          present — never just a pass/fail)
    """
    try:
        text, raw_html = await asyncio.to_thread(
            scrape_with_raw, url, force_dynamic=force_dynamic, force_static=force_static
        )
    except Exception as exc:
        text = f"ERROR: {type(exc).__name__}: {exc}"
        raw_html = ""

    credibility = assess_credibility(text, url, raw_html=raw_html)

    return {"text": text, "credibility": credibility}


if __name__ == "__main__":
    mcp.run()
