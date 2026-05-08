"""
Credibility Risk Layer — Content-Only Signal Scorer for Scraped Text

Analyses already-scraped text + a URL and returns a structured credibility-risk
assessment WITHOUT calling any external API.  All scoring is based on content
patterns, writing signals, and URL structure only.

This module is a credibility-RISK TRIAGE layer, not a truth engine.  It scores
and flags signals so you can prioritise human review — it does not block access
to content, and it cannot guarantee that flagged content is malicious or that
clean-scoring content is trustworthy.  See KNOWN_LIMITS.md for a full list of
known failure modes before relying on this in any automated pipeline.

Output structure (every call returns all four components):
  1. source_url        — the URL being assessed (raw, unmodified)
  2. extracted signals — red_flags (per-tactic evidence) + positive_signals
  3. credibility assessment — credibility_score, risk_level, confidence,
                              assessment_summary
  4. recommendation    — "use_freely" | "use_with_caution" | "avoid"

Scoring:
  Baseline: 65 (elevated above 50 to compensate for the absence of domain-age
  and backlink metadata that a full check would use).
  Red flags deduct 8–12 points each depending on severity.
  Positive signals add 5 points each.
  3+ tactics detected triggers an additional -15 penalty.
  Score is clamped to [0, 100].

Confidence score:
  Starts at 80.  Drops to 30 for content under 100 words, 50 for under 300,
  65 for under 500.  When content is short or ambiguous, the assessment is
  explicitly less reliable — the confidence field encodes this so callers can
  weight results appropriately.  When input is ambiguous and no strong signal
  fires, the recommendation defaults to "use_with_caution" (medium-risk default)
  rather than a binary pass/fail.

Calibration notes:
  Established healthcare/news sources (HIMSS, AHA, MGMA, CMS, Reuters, AP,
  NYT, Fierce Healthcare, Modern Healthcare, Health Affairs) score 70+ even
  when press-release language is present.
  Reddit is NOT penalised as a platform — only specific astroturfing patterns
  within Reddit content are flagged.
  Medium and Substack are NOT automatic red flags — only when combined with
  shallow "top-10 parasite" content patterns.
  Press-release language alone (without syndication boilerplate) is a minor
  flag, not a disqualifier.

Usage (module):
    from credibility import assess_credibility
    result = assess_credibility(text, "https://www.fiercehealthcare.com/...")

Usage (CLI):
    python credibility.py <url> <text_file>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import TypedDict
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


class RedFlag(TypedDict):
    tactic: str
    evidence: str


class CredibilityReport(TypedDict):
    source_url: str
    credibility_score: int
    risk_level: str           # "low" | "medium" | "high"
    detected_tactics: list[str]
    red_flags: list[RedFlag]
    positive_signals: list[str]
    assessment_summary: str
    recommendation: str       # "use_freely" | "use_with_caution" | "avoid"
    confidence: int           # 0-100


# ---------------------------------------------------------------------------
# Trusted-source registry
# Domains / hostname fragments that represent established healthcare,
# journalism, or government sources.  These receive a protected floor.
# ---------------------------------------------------------------------------

_TRUSTED_DOMAINS: frozenset[str] = frozenset(
    [
        "himss.org",
        "aha.org",
        "mgma.com",
        "cms.gov",
        "hhs.gov",
        "cdc.gov",
        "nih.gov",
        "reuters.com",
        "apnews.com",
        "nytimes.com",
        "washingtonpost.com",
        "wsj.com",
        "fiercehealthcare.com",
        "modernhealthcare.com",
        "healthaffairs.org",
        "statnews.com",
        "healthcareitnews.com",
        "beckersspine.com",
        "beckershospitalreview.com",
        "medscape.com",
        "pubmed.ncbi.nlm.nih.gov",
        "jamanetwork.com",
        "nejm.org",
        "thelancet.com",
        "kff.org",
        "commonwealthfund.org",
        "rand.org",
        "gao.gov",
    ]
)

# UGC/blog platforms — not automatic red flags, but context matters.
_UGC_PLATFORMS: frozenset[str] = frozenset(
    ["medium.com", "substack.com", "dev.to", "linkedin.com", "notion.so"]
)

_REDDIT_DOMAINS: frozenset[str] = frozenset(["reddit.com", "old.reddit.com"])


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_BASELINE_SCORE: int = 65
_POSITIVE_SIGNAL_BONUS: int = 5
_MULTI_TACTIC_PENALTY: int = 15  # extra penalty when 3+ tactics detected

# (tactic_id, severity_deduction)
_TACTIC_SEVERITY: dict[str, int] = {
    "FAKE_SOCIAL_PROOF": 10,
    "REDDIT_ASTROTURFING": 12,
    "SEO_PARASITE_HOSTING": 9,
    "PROGRAMMATIC_CONTENT": 9,
    "COMPARISON_MANIPULATION": 10,
    "FAKE_PR_SYNDICATION": 8,
    "PBN_BACKLINK_SIGNALS": 8,
    "KNOWLEDGE_GRAPH_MANIPULATION": 8,
    "PROMPT_INJECTION": 12,
    "SCRIPTED_INFLUENCER": 9,
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _hostname(url: str) -> str:
    """Return the lowercase hostname from a URL, or empty string on failure."""
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _is_trusted(url: str) -> bool:
    """Return True if the URL belongs to a known-reputable domain."""
    host = _hostname(url)
    return any(host == td or host.endswith("." + td) for td in _TRUSTED_DOMAINS)


def _is_ugc_platform(url: str) -> bool:
    """Return True if the URL is a user-generated-content hosting platform."""
    host = _hostname(url)
    return any(host == p or host.endswith("." + p) for p in _UGC_PLATFORMS)


def _is_reddit(url: str) -> bool:
    """Return True if the URL is a Reddit page."""
    host = _hostname(url)
    return any(host == r or host.endswith("." + r) for r in _REDDIT_DOMAINS)


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


# ---------------------------------------------------------------------------
# Individual tactic detectors
# Each detector returns (flagged: bool, evidence: str).
# ---------------------------------------------------------------------------


def _detect_fake_social_proof(text: str, url: str) -> tuple[bool, str]:
    """FAKE_SOCIAL_PROOF — inflated, generic review language.

    Looks for high density of short praise phrases (≤20-word chunks),
    no specific feature or outcome mentions, and suspiciously uniform
    superlative language typical of bulk fake reviews.
    """
    praise_phrases = re.findall(
        r"\b(amazing|best ever|love it|highly recommend|five stars?|"
        r"absolutely (fantastic|wonderful|perfect|great)|couldn'?t be happier|"
        r"exceeded (my )?expectations)\b",
        text,
        flags=re.IGNORECASE,
    )
    if len(praise_phrases) < 4:
        return False, ""

    # Count short sentences (≤20 words) relative to total sentences
    sentences = re.split(r"[.!?]+", text)
    short_sentences = [s for s in sentences if 0 < len(s.split()) <= 20]
    short_ratio = len(short_sentences) / max(1, len(sentences))

    # Absence of specific outcomes or features
    specifics = re.findall(
        r"\b(reduced|improved|saved|increased|decreased|"
        r"\d+%|\$\d+|CPT|HIPAA|EHR|EMR|hours? per week)\b",
        text,
        flags=re.IGNORECASE,
    )

    if len(praise_phrases) >= 6 and short_ratio > 0.7 and len(specifics) < 3:
        evidence = (
            f"{len(praise_phrases)} generic praise phrases, "
            f"{short_ratio:.0%} short sentences, "
            f"only {len(specifics)} specific outcome mentions"
        )
        return True, evidence

    return False, ""


def _detect_reddit_astroturfing(text: str, url: str) -> tuple[bool, str]:
    """REDDIT_ASTROTURFING — coordinated brand promotion in Reddit content.

    Only fires on Reddit URLs.  Looks for: brand name appearing 5+ times
    unnaturally, "I tested X tools" narrative with obvious bias, or
    multiple identical story-arc structures.
    """
    if not _is_reddit(url):
        return False, ""

    # Detect "I tested N tools" with brand domination
    tested_pattern = re.search(
        r"I (tested|tried|compared|reviewed)\s+\d+\s+(tools?|products?|apps?|"
        r"platforms?|services?|solutions?|options?)",
        text,
        flags=re.IGNORECASE,
    )

    # Find any word that repeats 5+ times (potential brand name injection)
    # Focus on capitalized multi-character words (likely proper nouns)
    capitalized_words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    word_freq: dict[str, int] = {}
    for w in capitalized_words:
        word_freq[w] = word_freq.get(w, 0) + 1

    # Exclude common stopwords / normal words
    common = {
        "The", "This", "That", "These", "Those", "With", "From", "Have",
        "Been", "When", "Will", "They", "Your", "More", "What", "Some",
        "About", "Also", "Like", "Just", "Very", "Than", "Only", "Into",
    }
    dominant = {
        w: c for w, c in word_freq.items() if w not in common and c >= 5
    }

    if tested_pattern and dominant:
        brands = ", ".join(f"{w}({c}x)" for w, c in list(dominant.items())[:3])
        return True, (
            f"'I tested N tools' narrative detected; "
            f"dominant brand-like terms: {brands}"
        )

    if tested_pattern and len(text.split()) < 600:
        return True, (
            "'I tested N tools' narrative in short post — high bias signal"
        )

    return False, ""


def _detect_seo_parasite_hosting(text: str, url: str) -> tuple[bool, str]:
    """SEO_PARASITE_HOSTING — thin "top-N" listicle on a UGC platform.

    Fires when: URL is a UGC platform AND content is a shallow "best X"
    or "top N" list with no firsthand evidence (screenshots, personal
    experience, data).
    """
    if not _is_ugc_platform(url):
        return False, ""

    list_pattern = re.search(
        r"\b(top\s+\d+|best\s+\d+|\d+\s+best|\d+\s+top|\d+\s+(tools?|"
        r"platforms?|apps?|services?|solutions?|options?)\s+(for|to))\b",
        text,
        flags=re.IGNORECASE,
    )
    if not list_pattern:
        return False, ""

    # Check for absence of firsthand evidence
    firsthand = re.findall(
        r"\b(screenshot|I (personally|actually|directly)|"
        r"in my experience|I found that|when I (tested|used|tried)|"
        r"our (team|data|analysis))\b",
        text,
        flags=re.IGNORECASE,
    )

    # Check for shallow comparison structure (product name + superlative)
    comparison_items = re.findall(
        r"\b(is (the )?best|#\d+[:\.]|number \d+[:\.])\b",
        text,
        flags=re.IGNORECASE,
    )

    if len(firsthand) == 0 and len(comparison_items) >= 3:
        platform = _hostname(url)
        return True, (
            f"UGC platform ({platform}) + 'top N' list structure + "
            f"no firsthand evidence ({len(firsthand)} mentions)"
        )

    return False, ""


def _detect_programmatic_content(text: str, url: str) -> tuple[bool, str]:
    """PROGRAMMATIC_CONTENT — templated AI-generated or MFA content.

    Looks for: city-substitution templates ("Best X in [City]"), identical
    intro/conclusion patterns, very low unique-word ratio, and lack of
    any unique insight or original framing.
    """
    # City-template pattern
    city_template = re.search(
        r"\b(best|top|leading|premier)\s+\w[\w\s]{0,30}?\s+in\s+"
        r"[A-Z][a-zA-Z\s]+,?\s*(GA|FL|TX|CA|NY|OH|IL|PA|WA|CO)\b",
        text,
        flags=re.IGNORECASE,
    )

    # Identical boilerplate phrase density
    boilerplate_phrases = re.findall(
        r"\b(whether you('?re| are)|regardless of (your|the)|"
        r"when it comes to|in (today'?s|the modern)|"
        r"it'?s (no secret|important to note|worth noting))\b",
        text,
        flags=re.IGNORECASE,
    )

    # Low unique-word ratio (repetitive padding)
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    unique_ratio = len(set(words)) / max(1, len(words))

    flags: list[str] = []
    if city_template:
        flags.append("city-substitution template detected")
    if len(boilerplate_phrases) >= 5:
        flags.append(f"{len(boilerplate_phrases)} boilerplate filler phrases")
    if unique_ratio < 0.35 and _word_count(text) > 300:
        flags.append(f"low unique-word ratio ({unique_ratio:.2f})")

    if len(flags) >= 2:
        return True, "; ".join(flags)

    return False, ""


def _detect_comparison_manipulation(text: str, url: str) -> tuple[bool, str]:
    """COMPARISON_MANIPULATION — "X vs Y" content rigged to favour one brand.

    Looks for: multiple competitor comparisons where one brand consistently
    wins, no genuine tradeoffs, no screenshots cited, and selective negative
    framing of alternatives.
    """
    vs_pattern = re.findall(
        r"\b\w[\w\s]{0,20}?\s+vs\.?\s+\w[\w\s]{0,20}?\b",
        text,
        flags=re.IGNORECASE,
    )
    if len(vs_pattern) < 2:
        return False, ""

    # No tradeoffs or honest cons
    tradeoffs = re.findall(
        r"\b(however|on the other hand|disadvantage|downside|con[s]?[:\.]|"
        r"limitation|not (ideal|suitable|great)|trade.?off|drawback)\b",
        text,
        flags=re.IGNORECASE,
    )

    # No screenshots cited (a real comparison usually references visuals)
    screenshots = re.findall(r"\b(screenshot|image|figure|see (above|below))\b",
                             text, flags=re.IGNORECASE)

    # "Always wins" pattern — winner language without balance
    always_wins = re.findall(
        r"\b(clearly (better|superior|the winner)|hands.down|"
        r"no contest|obviously (better|the best))\b",
        text,
        flags=re.IGNORECASE,
    )

    if (len(vs_pattern) >= 2 and len(tradeoffs) == 0
            and len(screenshots) == 0 and len(always_wins) >= 1):
        return True, (
            f"{len(vs_pattern)} vs-comparisons, "
            f"0 tradeoffs mentioned, "
            f"0 screenshots cited, "
            f"{len(always_wins)} 'always wins' phrases"
        )

    return False, ""


def _detect_fake_pr_syndication(text: str, url: str) -> tuple[bool, str]:
    """FAKE_PR_SYNDICATION — press release dressed as editorial coverage.

    Looks for: high density of superlative/PR adjectives + no journalist
    byline + no independent quote + syndication boilerplate footer.

    Press-release language ALONE is only a minor flag.  This tactic fires
    only when multiple signals stack.
    """
    pr_adjectives = re.findall(
        r"\b(revolutionary|award-winning|industry-leading|cutting-edge|"
        r"state-of-the-art|game-changing|groundbreaking|unprecedented|"
        r"best-in-class|world-class|disruptive|innovative leader)\b",
        text,
        flags=re.IGNORECASE,
    )

    # Syndication boilerplate
    syndication_boilerplate = re.search(
        r"(distributed by|this (press release|release) was|"
        r"for immediate release|contact:|media contact:|"
        r"about [A-Z][a-zA-Z\s]{2,30}[:—])",
        text,
        flags=re.IGNORECASE,
    )

    # Independent quote (from someone other than the company)
    independent_quote = re.search(
        r"according to (an? )?(independent|third.party|analyst|researcher|"
        r"professor|doctor|dr\.|study|report)",
        text,
        flags=re.IGNORECASE,
    )

    journalist_byline = re.search(
        r"\bby\s+[A-Z][a-z]+ [A-Z][a-z]+\b",
        text,
    )

    pr_density = len(pr_adjectives) / max(1, _word_count(text) / 100)

    if (pr_density >= 2 and syndication_boilerplate
            and not independent_quote and not journalist_byline):
        return True, (
            f"{len(pr_adjectives)} PR superlatives "
            f"({pr_density:.1f} per 100 words), "
            f"syndication boilerplate present, "
            f"no journalist byline or independent quote"
        )

    return False, ""


def _detect_pbn_backlink_signals(text: str, url: str) -> tuple[bool, str]:
    """PBN_BACKLINK_SIGNALS — over-optimised anchor text patterns.

    Cannot fully detect PBN networks without external API, but content-side
    signals include: unnatural exact-match anchor text clusters, "as seen on"
    irrelevant-domain name-drops, and keyword-stuffed resource lists.
    """
    # "As seen on" with irrelevant/generic domain lists
    as_seen = re.search(
        r"\bas seen (on|in)\b.{0,200}(Forbes|NBC|ABC|CNN|USA Today|"
        r"Entrepreneur|Business Insider)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Keyword-stuffed anchor patterns (exact match repeated 3+ times)
    anchor_candidates = re.findall(
        r"\[([^\]]{5,60})\]\(https?://[^\)]+\)",
        text,
    )
    anchor_freq: dict[str, int] = {}
    for a in anchor_candidates:
        key = a.strip().lower()
        anchor_freq[key] = anchor_freq.get(key, 0) + 1
    over_optimised = {k: v for k, v in anchor_freq.items() if v >= 3}

    # Irrelevant domain name-drop clusters
    domain_namedrop = re.findall(
        r"\b(featured in|covered by|as seen on)\b",
        text,
        flags=re.IGNORECASE,
    )

    if as_seen and len(domain_namedrop) >= 2:
        return True, (
            '"As seen on" major-outlet name-drops without verifiable links'
        )

    if over_optimised:
        items = ", ".join(
            f'"{k}"({v}x)' for k, v in list(over_optimised.items())[:3]
        )
        return True, f"Over-optimised repeated anchor text: {items}"

    return False, ""


def _detect_knowledge_graph_manipulation(text: str, url: str) -> tuple[bool, str]:
    """KNOWLEDGE_GRAPH_MANIPULATION — synthetic entity descriptions.

    Looks for: identical boilerplate bio phrasing, overly corporate
    "entity description" language with no verifiable specifics, and
    suspiciously uniform structured-data style paragraphs.
    """
    # Boilerplate entity description language
    entity_boilerplate = re.findall(
        r"\b([A-Z][a-zA-Z\s]{2,30})\s+is (a leading|an? established|"
        r"a premier|a globally recognized|the leading|a trusted)\s+"
        r"(provider|company|firm|organization|platform|solution|agency)\b",
        text,
        flags=re.IGNORECASE,
    )

    # No verifiable specifics (founded year, HQ city, employee count, etc.)
    verifiable = re.findall(
        r"\b(founded in \d{4}|headquartered in|employees?|"
        r"\d+\s+years? (of )?experience|located in|based in)\b",
        text,
        flags=re.IGNORECASE,
    )

    # Uniform structured paragraphs (same sentence-length pattern)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        lengths = [len(p.split()) for p in paragraphs[:6]]
        avg = sum(lengths) / len(lengths)
        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        suspiciously_uniform = variance < 10 and avg < 40

        if len(entity_boilerplate) >= 2 and len(verifiable) == 0:
            return True, (
                f"{len(entity_boilerplate)} boilerplate entity descriptions, "
                f"no verifiable specifics"
            )

        if len(entity_boilerplate) >= 1 and suspiciously_uniform:
            return True, (
                f"Boilerplate entity description + uniform paragraph structure "
                f"(variance={variance:.1f})"
            )

    return False, ""


def _detect_prompt_injection(text: str, url: str) -> tuple[bool, str]:
    """PROMPT_INJECTION — embedded AI instructions in scraped content.

    Looks for directive language, LLM instruction patterns, and attempts
    to override AI behaviour that should never appear in a genuine article.
    Checks plain text only — see scan_raw_html_for_injection() for hidden
    HTML vectors (display:none, HTML comments, etc.).
    """
    injection_patterns = [
        r"ignore (previous|prior|all|your) (instructions?|context|rules?)",
        r"when (asked|prompted) (about|to)\s+.{0,50}(always|never|say|respond)",
        r"you (are now|must|should|will) (act as|pretend|ignore|disregard)",
        r"(new|updated) (instructions?|system (prompt|message|context))[:—]",
        r"<(system|instruction|context|prompt)>",
        r"\[INST\]|\[SYS\]|\[SYSTEM\]",
        r"do not reveal|do not mention|keep (this|these) (secret|hidden|confidential)",
        r"override (safety|content|moderation|guidelines?)",
    ]

    hits: list[str] = []
    for pattern in injection_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            snippet = match.group(0)[:80]
            hits.append(snippet)

    if hits:
        return True, f"Injection patterns found: {'; '.join(hits[:2])}"

    return False, ""


# ---------------------------------------------------------------------------
# Hidden HTML injection scanner (raw HTML — runs BEFORE clean_html strips it)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[str] = [
    r"ignore (previous|prior|all|your) (instructions?|context|rules?)",
    r"when (asked|prompted) (about|to)\s+.{0,50}(always|never|say|respond)",
    r"you (are now|must|should|will) (act as|pretend|ignore|disregard)",
    r"(new|updated) (instructions?|system (prompt|message|context))[:—]",
    r"<(system|instruction|context|prompt)>",
    r"\[INST\]|\[SYS\]|\[SYSTEM\]",
    r"do not reveal|do not mention|keep (this|these) (secret|hidden|confidential)",
    r"override (safety|content|moderation|guidelines?)",
    r"(assistant|ai|llm|model)[,:]?\s+(you (must|should|will|are to)|please)\b",
    r"\bprompt[:\s]+.{0,80}(ignore|forget|disregard|override)\b",
]


def scan_raw_html_for_injection(raw_html: str) -> tuple[bool, list[str]]:
    """Scan raw HTML for prompt-injection patterns before text cleaning.

    clean_html() calls soup.get_text() which destroys hidden HTML — so
    injection payloads embedded in display:none elements, HTML comments,
    zero-opacity containers, off-screen divs, and aria-hidden nodes are
    invisible to the plain-text detector.  This function checks the raw
    HTML before any cleaning happens.

    Checks:
      - HTML comments (<!-- ... -->) containing injection directives
      - Elements with display:none / visibility:hidden / opacity:0 style
      - Elements with aria-hidden="true" containing instruction-like text
      - Elements positioned off-screen (left:-9999px or top:-9999px)
      - Elements with font-size:0 or color matching background (white-on-white)
      - <meta> tags with suspicious content attributes
      - Any element whose text content matches injection patterns above

    Args:
        raw_html: Unprocessed HTML string from requests or Playwright.

    Returns:
        (flagged, findings) where flagged is True if any hidden injection
        pattern was found, and findings is a list of human-readable
        descriptions of what was detected.
    """
    if not raw_html:
        return False, []

    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415
        soup = BeautifulSoup(raw_html, "lxml")
    except Exception:
        return False, []

    findings: list[str] = []

    # 1. HTML comments containing injection directives
    from bs4 import Comment  # noqa: PLC0415
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment_text = str(comment).strip()
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, comment_text, flags=re.IGNORECASE):
                snippet = comment_text[:120].replace("\n", " ")
                findings.append(f"HTML comment injection: «{snippet}»")
                break

    # 2. Elements hidden via CSS — display:none, visibility:hidden,
    #    opacity:0, font-size:0, or off-screen positioning
    _HIDDEN_STYLE_PATTERNS = re.compile(
        r"(display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0"
        r"|font-size\s*:\s*0|left\s*:\s*-\d{3,}px|top\s*:\s*-\d{3,}px"
        r"|position\s*:\s*absolute[^;]*left\s*:\s*-)",
        re.IGNORECASE,
    )
    for el in soup.find_all(style=True):
        style_val = el.get("style", "")
        if _HIDDEN_STYLE_PATTERNS.search(style_val):
            el_text = el.get_text(separator=" ", strip=True)
            if not el_text:
                continue
            for pattern in _INJECTION_PATTERNS:
                if re.search(pattern, el_text, flags=re.IGNORECASE):
                    snippet = el_text[:120].replace("\n", " ")
                    findings.append(
                        f"Hidden element injection (style='{style_val[:60]}'): «{snippet}»"
                    )
                    break

    # 3. aria-hidden="true" elements with instruction-like text
    for el in soup.find_all(attrs={"aria-hidden": "true"}):
        el_text = el.get_text(separator=" ", strip=True)
        if not el_text:
            continue
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, el_text, flags=re.IGNORECASE):
                snippet = el_text[:120].replace("\n", " ")
                findings.append(f"aria-hidden injection: «{snippet}»")
                break

    # 4. <meta> tags with suspicious content attributes
    for meta in soup.find_all("meta", content=True):
        content_val = meta.get("content", "")
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, content_val, flags=re.IGNORECASE):
                snippet = content_val[:120].replace("\n", " ")
                findings.append(f"<meta> tag injection: «{snippet}»")
                break

    # 5. White-on-white / same-color text: color and background-color match
    #    common camouflage pattern (#fff on white, #ffffff, rgb(255,255,255))
    _WHITE_COLOR = re.compile(
        r"color\s*:\s*(#fff{1,3}|#ffffff|white|rgb\(255\s*,\s*255\s*,\s*255\))",
        re.IGNORECASE,
    )
    for el in soup.find_all(style=True):
        style_val = el.get("style", "")
        if _WHITE_COLOR.search(style_val):
            el_text = el.get_text(separator=" ", strip=True)
            if not el_text:
                continue
            for pattern in _INJECTION_PATTERNS:
                if re.search(pattern, el_text, flags=re.IGNORECASE):
                    snippet = el_text[:120].replace("\n", " ")
                    findings.append(
                        f"White-on-white text injection: «{snippet}»"
                    )
                    break

    return bool(findings), findings


def _detect_scripted_influencer(text: str, url: str) -> tuple[bool, str]:
    """SCRIPTED_INFLUENCER — coordinated talking-point content.

    Looks for: identical talking points across what reads like multiple
    creator voices, generic praise without specific demos or outcomes,
    and sudden "everyone is saying" social proof framing.
    """
    # "Everyone is saying" mass-opinion manufacturing
    mass_opinion = re.findall(
        r"\b(everyone (is|seems to be|I know is)|"
        r"all (my friends|the creators|the community)|"
        r"the (whole|entire) (industry|community|space) (is talking|agrees)|"
        r"I'?ve (heard|seen) (from|a lot of) (people|creators|coaches))\b",
        text,
        flags=re.IGNORECASE,
    )

    # Generic praise without specific demos
    generic_praise = re.findall(
        r"\b(changed (my|the) (life|business|game)|"
        r"you (need|have) to (try|use|get) (this|it)|"
        r"literally (the best|amazing|incredible)|"
        r"no (BS|fluff|nonsense),? (just|straight))\b",
        text,
        flags=re.IGNORECASE,
    )

    # Absence of specific feature or outcome demos
    specific_demo = re.findall(
        r"\b(for example|specifically|in (my|our) case|"
        r"when I (clicked|opened|set up|configured)|"
        r"the (dashboard|feature|workflow|integration) (shows?|lets?|allows?))\b",
        text,
        flags=re.IGNORECASE,
    )

    if (len(mass_opinion) >= 1 and len(generic_praise) >= 3
            and len(specific_demo) == 0):
        return True, (
            f"{len(mass_opinion)} mass-opinion phrases, "
            f"{len(generic_praise)} generic-praise phrases, "
            f"0 specific demo mentions"
        )

    return False, ""


# ---------------------------------------------------------------------------
# Positive signal detectors
# ---------------------------------------------------------------------------


def _detect_positive_signals(text: str, url: str) -> list[str]:
    """Scan text for credibility-boosting positive signals.

    Returns a list of human-readable signal descriptions.  Each unique
    signal found contributes +5 to the credibility score.
    """
    signals: list[str] = []

    # 1. Author name + verifiable role/credentials
    if re.search(
        r"\bby\s+[A-Z][a-z]+ [A-Z][a-z]+\b.{0,200}"
        r"\b(MD|DO|RN|MBA|MHA|PhD|CPA|CAHIMS|CPHIMS|Director|VP|Manager|"
        r"Analyst|Editor|Journalist|Correspondent)\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        signals.append("Author name with verifiable credentials/role present")

    # 2. Specific citations or sources linked within content
    citation_count = len(re.findall(
        r"\b(according to|cited by|source[s]?:|reference[s]?:|"
        r"study (by|from|published)|research (by|from)|"
        r"reported by [A-Z]|data from [A-Z])\b",
        text,
        flags=re.IGNORECASE,
    ))
    if citation_count >= 2:
        signals.append(
            f"Multiple in-content citations or source attributions ({citation_count})"
        )

    # 3. Honest discussion of limitations or downsides
    if re.search(
        r"\b(however|limitation|downside|caveat|drawback|"
        r"not (ideal|suitable|perfect|for everyone)|"
        r"on the other hand|may not|doesn'?t (work|apply)|trade.?off)\b",
        text,
        flags=re.IGNORECASE,
    ):
        signals.append("Honest discussion of limitations or downsides")

    # 4. Specific outcomes with numbers
    if re.search(
        r"\b(\d+%|\$[\d,]+|\d+x|saved \d+|reduced by \d+|"
        r"increased (by )?\d+|cut .{0,20} by \d+)\b",
        text,
        flags=re.IGNORECASE,
    ):
        signals.append("Specific quantified outcomes or metrics mentioned")

    # 5. Named organisation with contact details
    if re.search(
        r"\b(contact|email|phone|address|reach (us|out)|"
        r"get in touch)\b.{0,100}\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|"
        r"\+?[\d\s\-\(\)]{7,15})\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        signals.append("Named organisation with contact details present")

    # 6. Unique insight (not templated)
    unique_insight = re.findall(
        r"\b(our (proprietary|internal|original) (data|research|analysis|survey)|"
        r"we (found|discovered|analysed|tested|built)|"
        r"in our (experience|practice|workflow)|"
        r"unlike (most|other|common))\b",
        text,
        flags=re.IGNORECASE,
    )
    if unique_insight:
        signals.append(
            f"Original or unique insight language ({len(unique_insight)} instances)"
        )

    # 7. Healthcare-specific domain knowledge
    healthcare_terms = re.findall(
        r"\b(HIPAA|CPT code|ICD-10|EHR|EMR|prior authorization|"
        r"revenue cycle|clinical workflow|billing (cycle|code)|"
        r"payer mix|credentialing|NPI|formulary|prior auth)\b",
        text,
        flags=re.IGNORECASE,
    )
    if len(healthcare_terms) >= 3:
        signals.append(
            f"Healthcare-specific domain knowledge demonstrated "
            f"({len(healthcare_terms)} relevant terms)"
        )

    return signals


# ---------------------------------------------------------------------------
# Master assessor
# ---------------------------------------------------------------------------


# Map tactic IDs to their detector functions
_DETECTORS: list[
    tuple[str, object]
] = [
    ("FAKE_SOCIAL_PROOF",           _detect_fake_social_proof),
    ("REDDIT_ASTROTURFING",         _detect_reddit_astroturfing),
    ("SEO_PARASITE_HOSTING",        _detect_seo_parasite_hosting),
    ("PROGRAMMATIC_CONTENT",        _detect_programmatic_content),
    ("COMPARISON_MANIPULATION",     _detect_comparison_manipulation),
    ("FAKE_PR_SYNDICATION",         _detect_fake_pr_syndication),
    ("PBN_BACKLINK_SIGNALS",        _detect_pbn_backlink_signals),
    ("KNOWLEDGE_GRAPH_MANIPULATION", _detect_knowledge_graph_manipulation),
    ("PROMPT_INJECTION",            _detect_prompt_injection),
    ("SCRIPTED_INFLUENCER",         _detect_scripted_influencer),
]


def assess_credibility(text: str, url: str, raw_html: str = "") -> CredibilityReport:
    """Score and triage credibility risk for already-scraped content.

    This is the primary entry point for the credibility-risk layer.  It runs
    all 10 tactic detectors and all positive signal detectors, computes a
    weighted score with a per-flag confidence model, and returns a structured
    CredibilityReport containing all four output components:

      1. Raw source reference  — source_url (unchanged, always returned)
      2. Extracted signals     — red_flags (list of {tactic, evidence} dicts)
                                 positive_signals (list of human-readable strings)
      3. Credibility assessment — credibility_score (0–100), risk_level
                                  ("low"/"medium"/"high"), confidence (0–100),
                                  assessment_summary
      4. Recommendation        — "use_freely" | "use_with_caution" | "avoid"

    The scraper always returns text.  This assessment is advisory — it flags
    signals for human review, it does not block content access.

    Confidence model:
      Confidence starts at 80 and degrades for short or ambiguous content:
        < 100 words → confidence 30
        < 300 words → confidence 50
        < 500 words → confidence 65
      Trusted-domain URLs receive a +10 confidence bonus.
      When confidence is low, treat the score as directional, not definitive.
      When content is short or no strong signal fires, risk_level defaults to
      "medium" and recommendation to "use_with_caution" rather than a false
      pass or fail.

    Args:
        text:     The plain-text content returned by the scraper.
        url:      The source URL from which the text was scraped.
        raw_html: Optional raw HTML string from the fetcher.  When supplied,
                  scan_raw_html_for_injection() checks hidden elements, HTML
                  comments, aria-hidden nodes, meta tags, and white-on-white
                  text for injection payloads that clean_html() would destroy.
                  Pass the unprocessed HTML from requests.text or
                  playwright page.content() before calling clean_html().

    Returns:
        A CredibilityReport TypedDict.  All four output components are always
        present — never just a pass/fail.  See module docstring for details.
    """
    trusted = _is_trusted(url)
    word_count = _word_count(text)

    # --- Run tactic detectors ---
    red_flags: list[RedFlag] = []
    detected_tactics: list[str] = []

    for tactic_id, detector in _DETECTORS:
        flagged, evidence = detector(text, url)
        if flagged:
            detected_tactics.append(tactic_id)
            red_flags.append({"tactic": tactic_id, "evidence": evidence})

    # --- Hidden HTML injection scan (raw HTML path) ---
    # Must run BEFORE clean_html() strips invisible elements.  Only adds to
    # PROMPT_INJECTION if it hasn't already been caught in plain text.
    if raw_html and "PROMPT_INJECTION" not in detected_tactics:
        html_flagged, html_findings = scan_raw_html_for_injection(raw_html)
        if html_flagged:
            detected_tactics.append("PROMPT_INJECTION")
            red_flags.append({
                "tactic": "PROMPT_INJECTION",
                "evidence": "Hidden HTML injection: " + "; ".join(html_findings[:2]),
            })
    elif raw_html and "PROMPT_INJECTION" in detected_tactics:
        # Already flagged in plain text — augment evidence with HTML details
        _, html_findings = scan_raw_html_for_injection(raw_html)
        if html_findings:
            for flag in red_flags:
                if flag["tactic"] == "PROMPT_INJECTION":
                    flag["evidence"] += (
                        " + hidden HTML vectors: " + "; ".join(html_findings[:2])
                    )

    # --- Run positive signal detectors ---
    positive_signals = _detect_positive_signals(text, url)

    # --- Compute score ---
    score = _BASELINE_SCORE

    # Deduct for red flags
    for tactic_id in detected_tactics:
        score -= _TACTIC_SEVERITY.get(tactic_id, 10)

    # Multi-tactic penalty
    if len(detected_tactics) >= 3:
        score -= _MULTI_TACTIC_PENALTY

    # Add positive signals
    score += len(positive_signals) * _POSITIVE_SIGNAL_BONUS

    # Trusted-domain floor: reputable sources never score below 70
    if trusted:
        score = max(score, 70)

    # Clamp to [0, 100]
    score = max(0, min(100, score))

    # --- Confidence (how reliable is this assessment?) ---
    # Confidence degrades for short or ambiguous content.  When confidence
    # is low the score is directional at best — callers should weight it
    # accordingly and not treat it as a definitive pass or fail.
    confidence = 80
    if word_count < 100:
        confidence = 30
    elif word_count < 300:
        confidence = 50
    elif word_count < 500:
        confidence = 65
    if trusted:
        confidence = min(100, confidence + 10)

    # --- Risk level and recommendation ---
    # When confidence is low and no clear signals fired in either direction,
    # default to medium-risk rather than a false low or high — the scorer
    # doesn't have enough signal to commit.
    # Exception: trusted domains always earn at least low/use_freely because
    # their domain-level credibility is established externally.
    no_signals = not detected_tactics and not positive_signals
    if trusted:
        # Trusted-domain floor overrides the low-confidence ambiguity penalty
        risk_level = "low"
        recommendation = "use_freely"
    elif no_signals and confidence < 65:
        # Insufficient signal: force medium regardless of baseline score
        risk_level = "medium"
        recommendation = "use_with_caution"
    elif score >= 70:
        risk_level = "low"
        recommendation = "use_freely"
    elif score >= 40:
        risk_level = "medium"
        recommendation = "use_with_caution"
    else:
        risk_level = "high"
        recommendation = "avoid"

    # --- Summary ---
    if no_signals and confidence < 65:
        summary = (
            f"Content is too short or ambiguous to score reliably "
            f"({word_count} words, confidence {confidence}). "
            "Defaulting to medium risk — human review required."
        )
    elif not detected_tactics and not positive_signals:
        summary = (
            "Content shows no strong credibility signals in either direction. "
            "Assessment is inconclusive — treat as medium confidence."
        )
    elif not detected_tactics:
        summary = (
            f"No manipulation tactics detected. "
            f"{len(positive_signals)} positive credibility signal(s) present."
        )
    elif trusted:
        summary = (
            f"Source domain is in the trusted-reputable registry. "
            f"{len(detected_tactics)} minor flag(s) detected but floor score applied."
        )
    else:
        tactic_list = ", ".join(detected_tactics[:3])
        suffix = f" + {len(detected_tactics) - 3} more" if len(detected_tactics) > 3 else ""
        summary = (
            f"{len(detected_tactics)} manipulation tactic(s) detected "
            f"({tactic_list}{suffix}). "
            f"Verify independently before citing."
        )

    return CredibilityReport(
        source_url=url,
        credibility_score=score,
        risk_level=risk_level,
        detected_tactics=detected_tactics,
        red_flags=red_flags,
        positive_signals=positive_signals,
        assessment_summary=summary,
        recommendation=recommendation,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credibility",
        description=(
            "Assess the credibility of scraped web content. "
            "Pass the source URL and a plain-text file to analyse."
        ),
    )
    parser.add_argument("url", help="Source URL of the scraped content.")
    parser.add_argument(
        "text_file",
        help="Path to a plain-text file containing the scraped content.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent level for output (default: 2).",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    try:
        with open(args.text_file, encoding="utf-8") as fh:
            content = fh.read()
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.text_file}", file=sys.stderr)
        sys.exit(1)

    report = assess_credibility(content, args.url)
    print(json.dumps(report, indent=args.indent))
