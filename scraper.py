"""
Personal Scraper — Lightweight Firecrawl Alternative

Hybrid static-first / dynamic-fallback web scraper designed for
content research, competitor analysis, and prospect website intake.
Tries a fast requests-based fetch first; falls back to a headless
Playwright browser with stealth patching when the static result is
thin or the server refuses the plain request.

Usage (CLI):
    python scraper.py <url> [--dynamic | --static] [--output FILE] [--quiet]

Usage (module):
    from scraper import scrape
    text = scrape("https://www.fiercehealthcare.com")
"""

import argparse
import random
import re
import sys
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    # Keep static responses simple. Some sites return compressed bytes that
    # requests cannot decode cleanly in this environment when br is advertised.
    "Accept-Encoding": "gzip, deflate",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
}

# Minimum character threshold — results shorter than this trigger fallback.
_STATIC_MIN_CHARS: int = 300

# Tags that add noise without informational value.
_NOISE_TAGS: tuple[str, ...] = (
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "noscript",
)


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------


def clean_html(html: str) -> str:
    """Parse raw HTML and return clean, readable plain text.

    Removes script, style, nav, footer, header, aside, and noscript
    elements before extracting text.  Collapses runs of three or more
    consecutive blank lines down to a single blank line.

    Args:
        html: Raw HTML string as returned by a browser or requests.

    Returns:
        Cleaned plain-text string suitable for feeding to an LLM.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in _NOISE_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    text: str = soup.get_text(separator="\n", strip=True)

    # Collapse 3+ consecutive newlines to exactly 2.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


# ---------------------------------------------------------------------------
# Static fetcher
# ---------------------------------------------------------------------------


def scrape_static(url: str) -> str:
    """Fetch a URL with a plain HTTP request and return cleaned text.

    Uses a realistic Chrome 124 User-Agent header set.  Raises an
    exception on any HTTP error status (4xx / 5xx).

    Args:
        url: Fully-qualified URL to fetch.

    Returns:
        Cleaned plain-text content of the page.

    Raises:
        requests.HTTPError: On 4xx or 5xx responses.
        requests.RequestException: On connection or timeout failures.
    """
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return clean_html(response.text)


# ---------------------------------------------------------------------------
# Dynamic fetcher (Playwright — lazy import)
# ---------------------------------------------------------------------------


def scrape_dynamic(url: str) -> str:
    """Fetch a URL using a headless Chromium browser with stealth patching.

    Playwright and playwright_stealth are imported lazily inside this
    function so that the module can be used for static-only scraping
    without those packages installed.

    Navigates to the URL, waits for the network to go idle, then waits
    an additional random 1.5–3.0 seconds to allow JS rendering to
    complete before capturing page content.

    Args:
        url: Fully-qualified URL to fetch.

    Returns:
        Cleaned plain-text content of the rendered page.

    Raises:
        playwright._impl._errors.TimeoutError: If page load exceeds 30 s.
        Exception: Any Playwright-level launch or navigation error.
    """
    # Lazy imports so static-only usage doesn't require playwright installed.
    from playwright.sync_api import sync_playwright  # noqa: PLC0415
    from playwright_stealth import stealth_sync  # noqa: PLC0415

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(extra_http_headers=HEADERS)
        page = context.new_page()
        stealth_sync(page)

        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(random.uniform(1.5, 3.0))

        html = page.content()
        browser.close()

    return clean_html(html)


# ---------------------------------------------------------------------------
# Hybrid entry point
# ---------------------------------------------------------------------------


def scrape(
    url: str,
    force_dynamic: bool = False,
    force_static: bool = False,
) -> str:
    """Scrape a URL and return clean AI-readable plain text.

    Hybrid logic:
    - If force_dynamic is True, always use the Playwright browser path.
    - If force_static is True, always use the plain requests path.
    - Otherwise, attempt the fast static fetch first.  If the result is
      fewer than 300 characters (thin/blocked content) OR any exception
      is raised, fall back to the dynamic browser path and print a
      diagnostic message to stderr.

    Args:
        url: Fully-qualified URL to scrape.
        force_dynamic: Skip static attempt and go straight to Playwright.
        force_static: Skip dynamic fallback and only use requests.

    Returns:
        Cleaned plain-text content of the page.

    Raises:
        Exception: Propagates any exception from the chosen scraper when
            the fallback path is also disabled (force_static=True) or
            when the dynamic path itself fails.
    """
    if force_dynamic:
        return scrape_dynamic(url)

    if force_static:
        return scrape_static(url)

    # Hybrid: static first, dynamic fallback.
    fallback_reason: Optional[str] = None

    try:
        result = scrape_static(url)
        if len(result.strip()) < _STATIC_MIN_CHARS:
            fallback_reason = (
                f"static result too short ({len(result.strip())} chars "
                f"< {_STATIC_MIN_CHARS} threshold)"
            )
        else:
            return result
    except Exception as exc:  # noqa: BLE001
        fallback_reason = f"static fetch failed ({type(exc).__name__}: {exc})"

    print(
        f"[fallback] Switching to dynamic scraper — {fallback_reason}",
        file=sys.stderr,
    )
    return scrape_dynamic(url)


# ---------------------------------------------------------------------------
# Raw HTML fetcher (for credibility layer)
# ---------------------------------------------------------------------------


def fetch_raw_html(
    url: str,
    force_dynamic: bool = False,
    force_static: bool = False,
) -> str:
    """Fetch a URL and return the unprocessed HTML string.

    Uses the same hybrid logic as scrape() but returns raw HTML instead of
    cleaned plain text.  Intended to be called alongside scrape() when the
    caller needs both the cleaned text and the raw HTML for the credibility
    hidden-injection scanner.

    The raw HTML is not cleaned or filtered in any way — it contains all
    hidden elements, HTML comments, meta tags, and style attributes that
    clean_html() would strip.

    Args:
        url:           Fully-qualified URL to fetch.
        force_dynamic: Skip static attempt and use Playwright directly.
        force_static:  Use requests only — no browser fallback.

    Returns:
        Raw HTML string.  Empty string on any fetch failure.
    """
    if force_dynamic:
        return _fetch_raw_dynamic(url)

    if force_static:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception:
            return ""

    # Hybrid: static first
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        if len(response.text.strip()) >= _STATIC_MIN_CHARS:
            return response.text
    except Exception:
        pass

    return _fetch_raw_dynamic(url)


def _fetch_raw_dynamic(url: str) -> str:
    """Fetch raw HTML via Playwright without cleaning."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
        from playwright_stealth import stealth_sync  # noqa: PLC0415

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(extra_http_headers=HEADERS)
            page = context.new_page()
            stealth_sync(page)
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(random.uniform(1.5, 3.0))
            html = page.content()
            browser.close()
        return html
    except Exception:
        return ""


def scrape_with_raw(
    url: str,
    force_dynamic: bool = False,
    force_static: bool = False,
) -> tuple[str, str]:
    """Scrape a URL and return both cleaned text and raw HTML.

    Runs a single fetch (respecting force flags and hybrid fallback logic)
    and returns both outputs without a second network request.

    Args:
        url:           Fully-qualified URL to scrape.
        force_dynamic: Skip static attempt and use Playwright directly.
        force_static:  Use requests only — no browser fallback.

    Returns:
        (clean_text, raw_html) tuple.  On failure, clean_text begins with
        'ERROR: ' and raw_html is an empty string.
    """
    if force_dynamic:
        raw = _fetch_raw_dynamic(url)
        if not raw:
            return "ERROR: dynamic fetch returned empty", ""
        return clean_html(raw), raw

    if force_static:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return clean_html(response.text), response.text
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}", ""

    # Hybrid
    fallback_reason: Optional[str] = None
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        raw = response.text
        cleaned = clean_html(raw)
        if len(cleaned.strip()) >= _STATIC_MIN_CHARS:
            return cleaned, raw
        fallback_reason = (
            f"static result too short ({len(cleaned.strip())} chars "
            f"< {_STATIC_MIN_CHARS} threshold)"
        )
    except Exception as exc:
        fallback_reason = f"static fetch failed ({type(exc).__name__}: {exc})"

    print(
        f"[fallback] Switching to dynamic scraper — {fallback_reason}",
        file=sys.stderr,
    )
    raw = _fetch_raw_dynamic(url)
    return (clean_html(raw) if raw else "ERROR: dynamic fetch returned empty"), raw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scraper",
        description=(
            "Hybrid static-first / dynamic-fallback web scraper. "
            "Returns clean AI-readable plain text from any public URL."
        ),
    )
    parser.add_argument("url", help="URL to scrape.")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dynamic",
        action="store_true",
        help="Force headless Playwright browser (skip static attempt).",
    )
    mode_group.add_argument(
        "--static",
        action="store_true",
        help="Force plain requests fetch (no browser fallback).",
    )

    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write output to FILE instead of stdout.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress fallback diagnostic messages on stderr.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # When --quiet is set, redirect stderr to /dev/null equivalent by
    # temporarily replacing it so the fallback print inside scrape()
    # goes nowhere.
    if args.quiet:
        sys.stderr = open(  # noqa: WPS515
            "nul" if sys.platform == "win32" else "/dev/null", "w"
        )

    text = scrape(
        url=args.url,
        force_dynamic=args.dynamic,
        force_static=args.static,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        # Restore stderr before printing confirmation so the user sees it
        # even with --quiet.
        if args.quiet:
            sys.stderr = sys.__stderr__
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(text)
