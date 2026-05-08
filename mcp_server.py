r"""
MCP server wrapping the credibility scraper as a Claude Code native tool.

Exposes three tools:
  - scrape_url: fetches a single URL and returns clean AI-readable plain text
  - scrape_batch: scrapes multiple URLs in parallel using a thread pool
  - scrape_url_with_credibility: fetches a URL and runs a credibility assessment
    on the returned content, flagging manipulation tactics without blocking access

Run with: python mcp_server.py
Register with Claude Code: claude mcp add scraper -- python /path/to/mcp_server.py
"""

import concurrent.futures

from mcp.server.fastmcp import FastMCP

from credibility import assess_credibility
from scraper import scrape

mcp = FastMCP("scraper")


@mcp.tool()
def scrape_url(url: str, force_dynamic: bool = False, force_static: bool = False) -> str:
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
        return scrape(url, force_dynamic=force_dynamic, force_static=force_static)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@mcp.tool()
def scrape_batch(
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
    results: dict[str, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url: dict[concurrent.futures.Future[str], str] = {
            executor.submit(
                scrape, url, force_dynamic=force_dynamic, force_static=force_static
            ): url
            for url in urls
        }

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as exc:
                results[url] = f"ERROR: {type(exc).__name__}: {exc}"

    return results


@mcp.tool()
def scrape_url_with_credibility(
    url: str,
    force_dynamic: bool = False,
    force_static: bool = False,
) -> dict:
    """Fetch a URL and return both the scraped text and a credibility assessment.

    Combines the scraper with the credibility module to give a richer result.
    The credibility assessment analyses content-only signals — it does NOT call
    any external API (no domain-age lookup, no backlink API).  Scoring starts at
    65, deducts 8-12 points per detected manipulation tactic, adds 5 points per
    positive credibility signal, and applies an additional -15 penalty when 3 or
    more tactics are detected.  Trusted healthcare/news domains (HIMSS, AHA,
    CMS, Reuters, NYT, Fierce Healthcare, etc.) are protected by a 70-point
    score floor.

    The scraper ALWAYS returns text regardless of the credibility score.  The
    assessment is advisory — it flags signals, it does not block access.

    Args:
        url: The URL to fetch and assess.
        force_dynamic: If True, skip static attempt and use Playwright directly.
        force_static: If True, use requests only — no browser fallback.

    Returns:
        A dict with two keys:
          "text"        — scraped plain text (or "ERROR: ..." string on failure)
          "credibility" — CredibilityReport dict with score, risk_level,
                          detected_tactics, red_flags, positive_signals,
                          assessment_summary, recommendation, and confidence
    """
    try:
        text = scrape(url, force_dynamic=force_dynamic, force_static=force_static)
    except Exception as exc:
        text = f"ERROR: {type(exc).__name__}: {exc}"

    credibility = assess_credibility(text, url)

    return {"text": text, "credibility": credibility}


if __name__ == "__main__":
    mcp.run()
