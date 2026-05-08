# credibility-scraper

**A web scraper with built-in blackhat signal detection**

Most scrapers pull whatever is indexed and hand you the text. This one also tells you whether what you just pulled is worth trusting. Built for researchers, analysts, and AI pipelines that need to distinguish real editorial content from fake reviews, astroturfed Reddit threads, SEO parasite posts, and prompt-injection traps embedded in HTML.

---

## What it does

`credibility-scraper` is a hybrid static/Playwright web scraper with an attached credibility module that runs 10 manipulation-signal detectors against the scraped text and returns a scored JSON assessment alongside the content. No external APIs required — all detection is content-signal-only.

**The scraper always returns the text.** The credibility score is advisory. It helps you make better decisions about what to cite; it does not block access to anything.

---

## Features

### Hybrid fetch engine
- Static-first (`requests` + `BeautifulSoup`) for speed
- Automatic Playwright fallback when static returns thin/blocked content
- Stealth patching via `playwright-stealth` to reduce bot-detection friction
- Force flags: `--dynamic` or `--static` to override the hybrid logic

### 10 blackhat signal detectors

| Tactic | What it catches | Severity |
|---|---|---|
| `FAKE_SOCIAL_PROOF` | Generic praise density, no specific outcomes, short-review flooding | -10 |
| `REDDIT_ASTROTURFING` | "I tested N tools" bias, brand-name dominance in short threads | -12 |
| `SEO_PARASITE_HOSTING` | Shallow "top-10" listicles on Medium/Substack/LinkedIn with no firsthand evidence | -9 |
| `PROGRAMMATIC_CONTENT` | City-substitution templates, boilerplate filler, low unique-word ratio | -9 |
| `COMPARISON_MANIPULATION` | "X vs Y" content with no tradeoffs, no screenshots, and an obvious winner | -10 |
| `FAKE_PR_SYNDICATION` | PR superlative density + syndication boilerplate + no journalist byline | -8 |
| `PBN_BACKLINK_SIGNALS` | "As seen on" name-drops, over-optimised repeated anchor text | -8 |
| `KNOWLEDGE_GRAPH_MANIPULATION` | Boilerplate entity descriptions, no verifiable specifics, uniform paragraph structure | -8 |
| `PROMPT_INJECTION` | LLM instruction patterns embedded in article content | -12 |
| `SCRIPTED_INFLUENCER` | "Everyone is saying" framing, generic praise with zero specific demos | -9 |

### 7 positive credibility signals

Each adds +5 to the score:

- Author name with verifiable role/credentials
- Multiple in-content citations or source attributions
- Honest discussion of limitations or downsides
- Specific quantified outcomes or metrics
- Named organisation with contact details
- Original or unique insight language
- Healthcare-specific domain knowledge (HIPAA, CPT codes, EHR, revenue cycle, etc.)

### Scoring system

| Parameter | Value |
|---|---|
| Baseline | 65 |
| Per red flag | -8 to -12 |
| 3+ tactics detected | additional -15 |
| Per positive signal | +5 |
| Floor for trusted domains | 70 (never drops below) |
| Range | 0–100 |

**Trusted domains** (protected by a 70-point floor): HIMSS, AHA, MGMA, CMS, HHS, CDC, NIH, Reuters, AP, NYT, Washington Post, WSJ, Fierce Healthcare, Modern Healthcare, Health Affairs, STAT News, Healthcare IT News, Becker's, Medscape, PubMed, JAMA, NEJM, The Lancet, KFF, Commonwealth Fund, RAND, GAO.

**Calibration rules:**
- Reddit as a platform is not a red flag. Only specific astroturfing patterns within Reddit content trigger the detector.
- Medium and Substack are not automatic red flags. Only when combined with shallow "top-10" parasite-hosting patterns.
- Press-release language alone does not trigger `FAKE_PR_SYNDICATION`. Multiple signals must stack.

### MCP server

Exposes all functionality as Claude Code tools via FastMCP:

| Tool | Returns |
|---|---|
| `scrape_url` | Plain text |
| `scrape_batch` | `dict[url, text]` |
| `scrape_url_with_credibility` | `{"text": str, "credibility": CredibilityReport}` |

---

## Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

---

## Usage

### Python module

```python
from scraper import scrape
from credibility import assess_credibility

# Scrape
text = scrape("https://www.fiercehealthcare.com/some-article")

# Assess
report = assess_credibility(text, "https://www.fiercehealthcare.com/some-article")
print(report["credibility_score"])   # e.g. 85
print(report["risk_level"])          # "low"
print(report["recommendation"])      # "use_freely"
```

### CLI — scraper

```bash
# Hybrid mode (static first, Playwright fallback)
python scraper.py https://example.com

# Force Playwright
python scraper.py https://example.com --dynamic

# Save to file
python scraper.py https://example.com --output output.txt
```

### CLI — credibility assessor

```bash
# Scrape first, then assess
python scraper.py https://example.com --output scraped.txt
python credibility.py https://example.com scraped.txt
```

### MCP server (Claude Code)

```bash
# Register
claude mcp add scraper -- python /path/to/mcp_server.py

# Run
python mcp_server.py
```

Once registered, use `scrape_url_with_credibility` in any Claude Code session to get text + credibility in one call.

---

## Sample credibility output

```json
{
  "source_url": "https://medium.com/@unknown/top-10-crm-tools-2024",
  "credibility_score": 37,
  "risk_level": "high",
  "detected_tactics": [
    "SEO_PARASITE_HOSTING",
    "COMPARISON_MANIPULATION",
    "PROGRAMMATIC_CONTENT"
  ],
  "red_flags": [
    {
      "tactic": "SEO_PARASITE_HOSTING",
      "evidence": "UGC platform (medium.com) + 'top N' list structure + no firsthand evidence (0 mentions)"
    },
    {
      "tactic": "COMPARISON_MANIPULATION",
      "evidence": "4 vs-comparisons, 0 tradeoffs mentioned, 0 screenshots cited, 2 'always wins' phrases"
    },
    {
      "tactic": "PROGRAMMATIC_CONTENT",
      "evidence": "city-substitution template detected; 7 boilerplate filler phrases"
    }
  ],
  "positive_signals": [
    "Specific quantified outcomes or metrics mentioned"
  ],
  "assessment_summary": "3 manipulation tactic(s) detected (SEO_PARASITE_HOSTING, COMPARISON_MANIPULATION, PROGRAMMATIC_CONTENT). Verify independently before citing.",
  "recommendation": "avoid",
  "confidence": 80
}
```

---

## Detection patterns reference

### Red flags

**FAKE_SOCIAL_PROOF** — Fires when 6+ generic praise phrases ("amazing", "best ever", "highly recommend") appear alongside >70% short sentences and fewer than 3 specific outcome mentions. Targets bulk review-flooding tactics.

**REDDIT_ASTROTURFING** — Reddit-only detector. Fires on "I tested N tools" narratives combined with a single brand name appearing 5+ times unnaturally in a short post.

**SEO_PARASITE_HOSTING** — Fires on Medium/Substack/Dev.to/LinkedIn "top N" or "best X" list content with zero firsthand evidence markers (no screenshots mentioned, no "in my experience", no "our data").

**PROGRAMMATIC_CONTENT** — Fires when two or more of: city-substitution template detected, 5+ filler boilerplate phrases, low unique-word ratio (<0.35) in content over 300 words.

**COMPARISON_MANIPULATION** — Fires on "X vs Y" content where 2+ comparisons appear but zero tradeoffs are acknowledged, zero screenshots are cited, and at least one "clearly better / hands down / no contest" phrase appears.

**FAKE_PR_SYNDICATION** — Fires when PR superlative density exceeds 2 per 100 words AND syndication boilerplate is present AND there is no journalist byline or independent third-party quote.

**PBN_BACKLINK_SIGNALS** — Content-side only (no domain API). Fires on "as seen on" major-outlet name-drops paired with 2+ "featured in / covered by" phrases, or on exact-match anchor text repeated 3+ times in markdown links.

**KNOWLEDGE_GRAPH_MANIPULATION** — Fires when 2+ "X is a leading/established/trusted provider" boilerplate phrases appear with no verifiable specifics (no founding year, HQ, employee count), or when one boilerplate phrase combines with suspiciously uniform paragraph structure.

**PROMPT_INJECTION** — Fires on any of: "ignore previous instructions", "you are now", "[INST]", "<system>", "override safety/guidelines", or "do not reveal" — patterns that have no place in genuine editorial content.

**SCRIPTED_INFLUENCER** — Fires when mass-opinion manufacturing phrases ("everyone is saying", "the whole community agrees") combine with 3+ generic-praise phrases and zero specific demo mentions.

### Positive signals

Each signal below adds 5 points:

- `Author name with verifiable credentials/role present` — byline found near a credential like MD, MHA, Editor, VP, Director
- `Multiple in-content citations or source attributions (N)` — 2+ "according to", "study by", "data from" phrases
- `Honest discussion of limitations or downsides` — "however", "limitation", "caveat", "not ideal for everyone", "tradeoff"
- `Specific quantified outcomes or metrics mentioned` — percentage, dollar figure, X multiplier, or explicit "reduced by N"
- `Named organisation with contact details present` — contact/email/phone near an identifiable address
- `Original or unique insight language (N instances)` — "our internal data", "we found", "in our experience", "unlike most"
- `Healthcare-specific domain knowledge demonstrated (N relevant terms)` — HIPAA, CPT code, ICD-10, EHR, prior authorization, revenue cycle, payer mix, NPI

---

## Architecture

```
scraper.py          — Hybrid static/Playwright fetch + HTML cleaning
credibility.py      — 10 tactic detectors + 7 positive-signal detectors + scorer
mcp_server.py       — FastMCP server exposing scrape_url, scrape_batch, scrape_url_with_credibility
requirements.txt    — Dependencies
```

---

## Contributing

Pull requests welcome. When adding a new detector:

1. Add a `_detect_<tactic_name>(text, url) -> tuple[bool, str]` function
2. Add the tactic to `_TACTIC_SEVERITY` with a severity value (8–12)
3. Register it in the `_DETECTORS` list in order
4. Add a calibration note in the docstring if the tactic risks over-flagging legitimate sources
5. Test against at least one known-good (HIMSS, CMS) and one known-bad (obvious spam) URL

---

## License

MIT
