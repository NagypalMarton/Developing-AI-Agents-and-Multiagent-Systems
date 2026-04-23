from __future__ import annotations

import argparse
import re
import unicodedata
from datetime import datetime
from html import unescape
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from fastmcp import FastMCP
from pydantic import BaseModel, Field, HttpUrl


mcp = FastMCP(
    name="news-content-mcp",
    instructions=(
        "MCP server for AI-assisted university news processing and "
        "platform-specific social media content generation for n8n workflows."
    ),
)


PlatformName = Literal["linkedin", "facebook", "x"]
DEFAULT_PLATFORMS: tuple[PlatformName, PlatformName, PlatformName] = ("linkedin", "facebook", "x")


class NewsIngestRequest(BaseModel):
    html: str = Field(..., description="Full HTML source of a news page")
    source_url: HttpUrl | None = Field(
        default=None,
        description="Optional canonical source URL of the news article",
    )
    language: str = Field(default="hu", description="Primary language code")


class NewsEvent(BaseModel):
    name: str
    start_date: str | None = None
    location: str | None = None
    registration_url: HttpUrl | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str | None = None


class NewsArticle(BaseModel):
    title: str
    lead: str
    body_text: str
    published_at: str | None = None
    source_url: HttpUrl | None = None
    language: str = "hu"
    tags: list[str] = Field(default_factory=list)
    events: list[NewsEvent] = Field(default_factory=list)


class GenerationOptions(BaseModel):
    include_event_cta: bool = True
    hashtag_limit: int = Field(default=3, ge=0, le=8)


class GeneratePostsRequest(BaseModel):
    article: NewsArticle
    platforms: list[PlatformName] = Field(default_factory=lambda: list(DEFAULT_PLATFORMS))
    options: GenerationOptions = Field(default_factory=GenerationOptions)


class PlatformPost(BaseModel):
    platform: PlatformName
    text: str
    char_count: int
    truncated: bool


class PromptPack(BaseModel):
    platform: PlatformName
    prompt: str


class CrawlRootsRequest(BaseModel):
    root_urls: list[HttpUrl] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Exactly two root URLs: typically university and department homepages.",
    )
    max_pages_per_root: int = Field(default=6, ge=1, le=20)
    language: str = Field(default="hu")


class CrawledNewsItem(BaseModel):
    title: str
    summary: str | None = None
    published_at: str | None = None
    url: HttpUrl
    tags: list[str] = Field(default_factory=list)
    image_url: HttpUrl | None = None


class CrawledEventItem(BaseModel):
    title: str
    description: str | None = None
    event_time: str | None = None
    start_at: str | None = None
    end_at: str | None = None
    location: str | None = None
    url: HttpUrl


class SourceCrawlResult(BaseModel):
    source_root: str
    visited_urls: list[str] = Field(default_factory=list)
    news: list[CrawledNewsItem] = Field(default_factory=list)
    events: list[CrawledEventItem] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


EVENT_KEYWORDS = [
    "rendezveny",
    "rendezvény",
    "esemeny",
    "esemény",
    "konferencia",
    "workshop",
    "szeminarium",
    "szeminárium",
    "seminar",
    "webinar",
    "meetup",
    "eloadas",
    "előadás",
    "vedes",
    "védés",
    "open day",
    "nyilt nap",
    "nyílt nap",
    "jelentkezes",
    "jelentkezés",
    "regisztracio",
    "regisztráció",
    "registration",
]

DATE_REGEXES = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{4}\.\d{1,2}\.\d{1,2}\.)\b"),
    re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b"),
    re.compile(r"\b(\d{1,2}\.\d{1,2}\.)\b"),
]


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _deaccent(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_for_match(text: str) -> str:
    return _deaccent(_clean_whitespace(text)).lower()


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_hu_date_to_iso(value: str) -> str | None:
    clean = _clean_whitespace(value)

    iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", clean)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{year}-{month}-{day}"

    dot_match = re.search(r"\b(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\b", clean)
    if dot_match:
        year, month, day = dot_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    day_first_match = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", clean)
    if day_first_match:
        day, month, year = day_first_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    return None


def _normalize_hu_datetime_ranges(value: str) -> tuple[str | None, str | None]:
    clean = _clean_whitespace(value)
    date_iso = _normalize_hu_date_to_iso(clean)
    if not date_iso:
        return None, None

    times = re.findall(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", clean)
    if not times:
        return f"{date_iso}T00:00:00", None
    if len(times) == 1:
        hh, mm = times[0]
        return f"{date_iso}T{int(hh):02d}:{mm}:00", None

    start_hh, start_mm = times[0]
    end_hh, end_mm = times[1]
    return (
        f"{date_iso}T{int(start_hh):02d}:{start_mm}:00",
        f"{date_iso}T{int(end_hh):02d}:{end_mm}:00",
    )


def _decode_html(raw: bytes, content_type_header: str | None, fallback_charset: str | None) -> str:
    encodings: list[str] = []

    if fallback_charset:
        encodings.append(fallback_charset)

    if content_type_header:
        header_match = re.search(r"charset=([a-zA-Z0-9_\-]+)", content_type_header, flags=re.IGNORECASE)
        if header_match:
            encodings.append(header_match.group(1).strip())

    # Hungarian pages are often UTF-8, but cp1250 and iso-8859-2 still appear.
    encodings.extend(["utf-8", "cp1250", "iso-8859-2", "latin-1"])

    seen: set[str] = set()
    ordered = [enc for enc in encodings if not (enc.lower() in seen or seen.add(enc.lower()))]

    for encoding in ordered:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


def _fetch_html(url: str, timeout_seconds: int = 20) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset()
        content_type = response.headers.get("Content-Type")
        raw = response.read()
        return _decode_html(raw=raw, content_type_header=content_type, fallback_charset=charset)


def _safe_fetch_html(url: str, timeout_seconds: int = 20) -> tuple[str | None, str | None]:
    last_error: Exception | None = None
    for _ in range(2):
        try:
            return _fetch_html(url, timeout_seconds=timeout_seconds), None
        except (HTTPError, URLError, ValueError, TimeoutError) as exc:
            last_error = exc

    if last_error is None:
        return None, f"{url}: unknown fetch error"
    return None, f"{url}: {type(last_error).__name__}: {last_error}"


def _is_same_host(url_a: str, url_b: str) -> bool:
    return urlparse(url_a).netloc.lower() == urlparse(url_b).netloc.lower()


def _normalize_href(base_url: str, href: str) -> str:
    return urljoin(base_url, href.strip())


def _extract_title(soup: BeautifulSoup) -> str:
    candidates = [
        soup.find("meta", property="og:title"),
        soup.find("meta", attrs={"name": "twitter:title"}),
        soup.find("title"),
        soup.find("h1"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate.name == "meta":
            value = candidate.get("content")
        else:
            value = candidate.get_text(" ", strip=True)
        if value:
            return _clean_whitespace(value)
    return "Untitled news"


def _extract_lead(soup: BeautifulSoup, fallback_text: str) -> str:
    meta_description = soup.find("meta", attrs={"name": "description"})
    if meta_description and meta_description.get("content"):
        return _clean_whitespace(meta_description["content"])

    first_paragraph = soup.find("p")
    if first_paragraph:
        lead = _clean_whitespace(first_paragraph.get_text(" ", strip=True))
        if lead:
            return lead

    return _clean_whitespace(fallback_text[:240])


def _extract_published_at(soup: BeautifulSoup) -> str | None:
    time_node = soup.find("time")
    if time_node:
        if time_node.get("datetime"):
            return _clean_whitespace(time_node["datetime"])
        text = _clean_whitespace(time_node.get_text(" ", strip=True))
        if text:
            return text

    for attr_name in ["article:published_time", "publish_date", "date", "dc.date"]:
        node = soup.find("meta", property=attr_name) or soup.find("meta", attrs={"name": attr_name})
        if node and node.get("content"):
            return _clean_whitespace(node["content"])

    return None


def _extract_tags(soup: BeautifulSoup) -> list[str]:
    tags: list[str] = []
    keywords_node = soup.find("meta", attrs={"name": "keywords"})
    if keywords_node and keywords_node.get("content"):
        for item in keywords_node["content"].split(","):
            cleaned = _clean_whitespace(item).lower()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)

    for link in soup.find_all("a"):
        rel = link.get("rel")
        if not rel:
            continue
        rel_joined = " ".join(rel) if isinstance(rel, list) else str(rel)
        if "tag" in rel_joined:
            cleaned = _clean_whitespace(link.get_text(" ", strip=True)).lower()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)

    return tags[:12]


def _extract_bme_news_cards(soup: BeautifulSoup, base_url: str) -> list[CrawledNewsItem]:
    cards = soup.select("div.bme_news_card")
    results: list[CrawledNewsItem] = []

    for card in cards:
        container_link = card.find_parent("a", href=True)
        if not container_link:
            sibling_link = card.find("a", href=True)
            container_link = sibling_link
        if not container_link:
            continue

        href = container_link.get("href", "").strip()
        if not href:
            continue
        full_url = _normalize_href(base_url, href)
        if not _is_http_url(full_url):
            continue

        title_node = card.select_one("h4.bme_news_card-title, h3.bme_news_card-title, h4, h3")
        if not title_node:
            continue
        title = _clean_whitespace(title_node.get_text(" ", strip=True))
        if not title:
            continue

        summary_node = card.select_one("div.bme_news_card-body p, div.bme_news_card-body")
        summary = _clean_whitespace(summary_node.get_text(" ", strip=True)) if summary_node else None

        date_node = card.select_one("span.field--name-created")
        raw_published_at = _clean_whitespace(date_node.get_text(" ", strip=True)) if date_node else None
        published_at = _normalize_hu_date_to_iso(raw_published_at or "") or raw_published_at

        tag_nodes = card.select("div.bme_news_card-tags li")
        tags = []
        for tag_node in tag_nodes:
            tag_text = _clean_whitespace(tag_node.get_text(" ", strip=True))
            if tag_text and tag_text not in tags:
                tags.append(tag_text)

        image_node = card.select_one("img")
        image_url: str | None = None
        if image_node:
            src = (image_node.get("src") or "").strip()
            if src:
                candidate_image_url = _normalize_href(base_url, src)
                if _is_http_url(candidate_image_url):
                    image_url = candidate_image_url

        results.append(
            CrawledNewsItem(
                title=title,
                summary=summary,
                published_at=published_at,
                url=full_url,
                tags=tags,
                image_url=image_url,
            )
        )

    return results


def _extract_department_news_articles(soup: BeautifulSoup, base_url: str) -> list[CrawledNewsItem]:
    articles = soup.select("article.node-hir, article.node.node-hir")
    results: list[CrawledNewsItem] = []

    for article in articles:
        title_link = article.select_one("h2.node__title a, h2.node-title a, h2 a")
        if not title_link:
            continue

        href = (title_link.get("href") or "").strip()
        if not href:
            href = (article.get("about") or "").strip()
        if not href:
            continue
        full_url = _normalize_href(base_url, href)
        if not _is_http_url(full_url):
            continue

        title = _clean_whitespace(title_link.get_text(" ", strip=True))
        if not title:
            continue

        summary_node = article.select_one(
            "div.field-name-body p, div.field--name-body p, div.field-name-body, div.field--name-body"
        )
        summary = _clean_whitespace(summary_node.get_text(" ", strip=True)) if summary_node else None
        if not summary:
            fallback_summary_node = article.select_one("p")
            summary = _clean_whitespace(fallback_summary_node.get_text(" ", strip=True)) if fallback_summary_node else None

        date_node = article.select_one(
            "time, span.field--name-created, span.field-name-created, span[property='dc:date'], meta[property='article:published_time']"
        )
        raw_published_at = None
        if date_node:
            if date_node.name == "meta":
                raw_published_at = _clean_whitespace(date_node.get("content", ""))
            else:
                raw_published_at = _clean_whitespace(date_node.get_text(" ", strip=True))
        published_at = _normalize_hu_date_to_iso(raw_published_at or "") or raw_published_at

        tags: list[str] = []
        for tag_node in article.select("a[rel='tag'], .field-name-field-tags a, .field--name-field-tags a"):
            tag_text = _clean_whitespace(tag_node.get_text(" ", strip=True))
            if tag_text and tag_text not in tags:
                tags.append(tag_text)

        image_node = article.select_one("div.field-name-field-bevezto-kep img, div.field--name-field-bevezto-kep img, img")
        image_url: str | None = None
        if image_node:
            src = (image_node.get("src") or "").strip()
            if src:
                candidate_image_url = _normalize_href(base_url, src)
                if _is_http_url(candidate_image_url):
                    image_url = candidate_image_url

        results.append(
            CrawledNewsItem(
                title=title,
                summary=summary,
                published_at=published_at,
                url=full_url,
                tags=tags,
                image_url=image_url,
            )
        )

    return results


def _extract_bme_event_cards(soup: BeautifulSoup, base_url: str) -> list[CrawledEventItem]:
    anchors = soup.select("a[href]")
    results: list[CrawledEventItem] = []

    for anchor in anchors:
        title_node = anchor.select_one("h4.bme_event_card-title")
        if not title_node:
            continue

        href = (anchor.get("href") or "").strip()
        if not href:
            continue

        full_url = _normalize_href(base_url, href)
        if not _is_http_url(full_url):
            continue
        title = _clean_whitespace(title_node.get_text(" ", strip=True))
        if not title:
            continue

        date_node = anchor.select_one("div.bme_event_card-date")
        raw_event_time = _clean_whitespace(date_node.get_text(" ", strip=True)) if date_node else None
        start_at, end_at = _normalize_hu_datetime_ranges(raw_event_time or "")
        if start_at and end_at:
            event_time = f"{start_at}/{end_at}"
        else:
            event_time = start_at or raw_event_time

        location_node = anchor.select_one("p.bme_event_card-location")
        location = _clean_whitespace(location_node.get_text(" ", strip=True)) if location_node else None

        body_node = anchor.select_one("div.bme_event_card-body p, div.bme_event_card-body")
        description = _clean_whitespace(body_node.get_text(" ", strip=True)) if body_node else None

        results.append(
            CrawledEventItem(
                title=title,
                description=description,
                event_time=event_time,
                start_at=start_at,
                end_at=end_at,
                location=location,
                url=full_url,
            )
        )

    return results


def _discover_news_event_links(soup: BeautifulSoup, root_url: str, max_links: int) -> list[str]:
    keywords = [
        "hirek",
        "hírek",
        "news",
        "esemeny",
        "esemény",
        "esemenyek",
        "események",
        "event",
        "events",
        "rendezveny",
        "rendezvény",
    ]
    negative_keywords = [
        "kapcsolat",
        "contact",
        "impresszum",
        "adatvedelem",
        "adatvédelem",
        "privacy",
        "search",
        "kereses",
        "keresés",
        "felveteli",
        "felvételi",
        "about",
        "bemutatkozas",
        "bemutatkozás",
    ]

    normalized_keywords = [_normalize_for_match(keyword) for keyword in keywords]
    normalized_negative = [_normalize_for_match(keyword) for keyword in negative_keywords]

    scored_candidates: list[tuple[int, str]] = []
    seen_urls: set[str] = set()
    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue

        full_url = _normalize_href(root_url, href)
        if not _is_same_host(root_url, full_url):
            continue

        raw_text = _clean_whitespace(anchor.get_text(" ", strip=True))
        normalized_marker = _normalize_for_match(f"{full_url} {raw_text}")

        score = 0
        if any(keyword in normalized_marker for keyword in normalized_keywords):
            score += 4
        if any(path_kw in _normalize_for_match(full_url) for path_kw in ["/hirek", "/esemeny", "/esemenyek", "/node/"]):
            score += 5
        class_marker = _normalize_for_match(" ".join(anchor.get("class", [])))
        if "news" in class_marker or "event" in class_marker or "hir" in class_marker:
            score += 2
        if any(neg in normalized_marker for neg in normalized_negative):
            score -= 8

        if score <= 0:
            continue

        dedupe_key = full_url.lower()
        if dedupe_key in seen_urls:
            continue
        seen_urls.add(dedupe_key)
        scored_candidates.append((score, full_url))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return [url for _, url in scored_candidates[:max_links]]


def _dedupe_news_items(items: list[CrawledNewsItem]) -> list[CrawledNewsItem]:
    deduped: dict[str, CrawledNewsItem] = {}
    for item in items:
        key = item.url.lower()
        if key not in deduped:
            deduped[key] = item
    return list(deduped.values())


def _dedupe_event_items(items: list[CrawledEventItem]) -> list[CrawledEventItem]:
    deduped: dict[str, CrawledEventItem] = {}
    for item in items:
        key = item.url.lower()
        if key not in deduped:
            deduped[key] = item
    return list(deduped.values())


def _extract_text(soup: BeautifulSoup) -> str:
    for node in soup(["script", "style", "noscript"]):
        node.decompose()

    paragraph_texts: list[str] = []
    for paragraph in soup.find_all("p"):
        text = _clean_whitespace(paragraph.get_text(" ", strip=True))
        if len(text) > 20:
            paragraph_texts.append(text)

    if paragraph_texts:
        return "\n\n".join(paragraph_texts)

    return _clean_whitespace(soup.get_text(" ", strip=True))


def _extract_registration_urls(soup: BeautifulSoup) -> list[str]:
    urls: list[str] = []
    trigger_words = ["register", "registration", "jelentke", "eventbrite", "forms.gle"]
    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        if not href:
            continue
        text = _clean_whitespace(anchor.get_text(" ", strip=True)).lower()
        marker = f"{href.lower()} {text}"
        if any(trigger in marker for trigger in trigger_words):
            urls.append(href)

    unique_urls: list[str] = []
    for url in urls:
        if url not in unique_urls:
            unique_urls.append(url)
    return unique_urls


def _extract_dates(text: str) -> list[str]:
    found: list[str] = []

    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        iso = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        if iso not in found:
            found.append(iso)

    for match in re.finditer(r"\b(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\b", text):
        iso = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        if iso not in found:
            found.append(iso)

    for match in re.finditer(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text):
        iso = f"{int(match.group(3)):04d}-{int(match.group(2)):02d}-{int(match.group(1)):02d}"
        if iso not in found:
            found.append(iso)

    for pattern in DATE_REGEXES:
        for match in pattern.finditer(text):
            value = match.group(1)
            if value not in found:
                found.append(value)
    return found[:3]


def _normalize_event_name(sentence: str) -> str:
    clean = _clean_whitespace(sentence)
    if len(clean) <= 90:
        return clean
    return clean[:87].rstrip() + "..."


def _detect_events(text: str, registration_urls: list[str] | None = None) -> list[NewsEvent]:
    registration_urls = registration_urls or []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    events: list[NewsEvent] = []
    normalized_keywords = [_normalize_for_match(keyword) for keyword in EVENT_KEYWORDS]

    for sentence in sentences:
        normalized = _normalize_for_match(sentence)
        keyword_hits = [kw for kw in normalized_keywords if kw in normalized]
        if not keyword_hits:
            continue

        dates = _extract_dates(sentence)
        registration_in_text = any(
            marker in normalized for marker in ["jelentkez", "regisztr", "registration", "register"]
        )

        possible_location_match = re.search(
            (
                r"\b(Budapest|Debrecen|Szeged|Miskolc|Gyor|Gyorben|online|campus|epulet|"
                r"building|terem|hall|audit[oó]rium|d[íi]szterem|k\. ?[ée]p[üu]let)\b"
            ),
            sentence,
            flags=re.IGNORECASE,
        )
        location = possible_location_match.group(1) if possible_location_match else None

        registration_url = registration_urls[0] if registration_urls else None
        has_date = bool(dates)
        has_location = bool(location)
        has_registration = bool(registration_url or registration_in_text)

        if not (has_date or has_location or has_registration):
            continue

        confidence = 0.25
        confidence += min(0.3, 0.08 * len(keyword_hits))
        if has_date:
            confidence += 0.2
        if has_location:
            confidence += 0.15
        if has_registration:
            confidence += 0.15
        confidence = min(confidence, 0.95)

        if confidence < 0.5:
            continue

        event = NewsEvent(
            name=_normalize_event_name(sentence),
            start_date=dates[0] if dates else None,
            location=location,
            registration_url=registration_url,
            confidence=confidence,
            evidence=f"keywords: {', '.join(keyword_hits)}",
        )
        events.append(event)

    # De-duplicate by event name while keeping the most confident one.
    deduped: dict[str, NewsEvent] = {}
    for event in events:
        key = event.name.lower()
        if key not in deduped or event.confidence > deduped[key].confidence:
            deduped[key] = event

    return list(deduped.values())[:5]


def _to_hashtags(tags: list[str], limit: int) -> str:
    selected: list[str] = []
    for tag in tags:
        candidate = re.sub(r"[^a-zA-Z0-9_\-]", "", tag.replace(" ", ""))
        if not candidate:
            continue
        selected.append(f"#{candidate}")
        if len(selected) >= limit:
            break
    return " ".join(selected)


def _ensure_char_limit(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False

    trimmed = text[: max_chars - 3].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed + "...", True


def _build_event_cta(events: list[NewsEvent]) -> str:
    if not events:
        return ""

    primary = events[0]
    if primary.registration_url:
        return f"\nJelentkezes: {primary.registration_url}"

    return "\nCsatlakozz az esemenyhez, es oszd meg a hiret!"


def _platform_limit(platform: PlatformName) -> int:
    return {
        "linkedin": 1100,
        "facebook": 1800,
        "x": 280,
    }[platform]


def _platform_header(platform: PlatformName) -> str:
    return {
        "linkedin": "Szakmai frissites",
        "facebook": "Friss tanszeki hir",
        "x": "Uj hir",
    }[platform]


def _build_post_text(article: NewsArticle, platform: PlatformName, options: GenerationOptions) -> str:
    summary_base = article.lead or article.body_text[:240]
    summary_base = _clean_whitespace(summary_base)

    url_line = f"\nForras: {article.source_url}" if article.source_url else ""
    hashtags = _to_hashtags(article.tags, options.hashtag_limit)
    hashtags_line = f"\n\n{hashtags}" if hashtags else ""

    event_cta = ""
    if options.include_event_cta and article.events:
        event_cta = _build_event_cta(article.events)

    header = _platform_header(platform)

    if platform == "x":
        return f"{header}: {article.title}. {summary_base}{event_cta}{hashtags_line}"

    return (
        f"{header}\n\n"
        f"{article.title}\n"
        f"{summary_base}"
        f"{event_cta}"
        f"{url_line}"
        f"{hashtags_line}"
    )


def _prompt_for_platform(article: NewsArticle, platform: PlatformName, include_event_cta: bool) -> str:
    platform_rules = {
        "linkedin": "professional tone, 3-5 short paragraphs, clear call-to-action",
        "facebook": "accessible and engaging tone, 1-3 emojis optional, concise community focus",
        "x": "single concise post under 280 characters",
    }[platform]

    event_instructions = ""
    if include_event_cta and article.events:
        first_event = article.events[0]
        event_instructions = (
            "\nIf the article references an event, include a registration call-to-action."
            f" Event data: {first_event.model_dump_json()}"
        )

    return (
        "You are a social media editor for a university department. "
        f"Rewrite the news for {platform}. Follow these platform rules: {platform_rules}."
        f"\nTitle: {article.title}"
        f"\nLead: {article.lead}"
        f"\nBody: {article.body_text[:1600]}"
        f"\nTags: {article.tags}"
        f"\nLanguage: {article.language}"
        f"{event_instructions}"
        "\nReturn only the final post text."
    )


@mcp.tool
def parse_news_html(request: NewsIngestRequest) -> dict:
    """Parse raw HTML news into a structured article model with detected events."""
    soup = BeautifulSoup(request.html, "html.parser")

    body_text = _extract_text(soup)
    article = NewsArticle(
        title=_extract_title(soup),
        lead=_extract_lead(soup, body_text),
        body_text=body_text,
        published_at=_extract_published_at(soup),
        source_url=request.source_url,
        language=request.language,
        tags=_extract_tags(soup),
        events=_detect_events(body_text, _extract_registration_urls(soup)),
    )
    return article.model_dump(mode="json")


@mcp.tool
def detect_events(article_text: str, registration_urls: list[str] | None = None) -> list[dict]:
    """Detect event candidates from article text and optional registration URLs."""
    events = _detect_events(article_text, registration_urls)
    return [event.model_dump(mode="json") for event in events]


@mcp.tool
def generate_platform_posts(request: GeneratePostsRequest) -> list[dict]:
    """Generate platform-specific draft posts with optional event registration CTA insertion."""
    outputs: list[PlatformPost] = []

    for platform in request.platforms:
        raw_text = _build_post_text(request.article, platform, request.options)
        limited_text, truncated = _ensure_char_limit(raw_text, _platform_limit(platform))
        outputs.append(
            PlatformPost(
                platform=platform,
                text=limited_text,
                char_count=len(limited_text),
                truncated=truncated,
            )
        )

    return [output.model_dump(mode="json") for output in outputs]


@mcp.tool
def build_llm_prompt_pack(article: NewsArticle, include_event_cta: bool = True) -> list[dict]:
    """Build platform-specific prompts for external LLM nodes in n8n."""
    packs = [
        PromptPack(
            platform=platform,
            prompt=_prompt_for_platform(article, platform, include_event_cta),
        )
        for platform in DEFAULT_PLATFORMS
    ]
    return [pack.model_dump(mode="json") for pack in packs]


@mcp.tool
def news_workflow_bundle(request: GeneratePostsRequest) -> dict:
    """Return a single n8n-friendly bundle with generated posts and LLM prompt pack."""
    posts = generate_platform_posts(request)
    prompts = build_llm_prompt_pack(request.article, request.options.include_event_cta)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "article": request.article.model_dump(mode="json"),
        "posts": posts,
        "prompt_pack": prompts,
    }


@mcp.tool
def health() -> dict:
    """Simple health endpoint for workflow diagnostics."""
    return {
        "status": "ok",
        "service": "news-content-mcp",
        "time_utc": datetime.utcnow().isoformat() + "Z",
    }


@mcp.tool
def crawl_news_and_events_from_roots(request: CrawlRootsRequest) -> dict:
    """Crawl exactly two root URLs and collect news/event cards through discovered subpages."""
    source_results: list[SourceCrawlResult] = []

    for root_url_obj in request.root_urls:
        root_url = str(root_url_obj)
        visited_urls: list[str] = []
        errors: list[str] = []

        root_html, root_error = _safe_fetch_html(root_url)
        if root_error:
            source_results.append(
                SourceCrawlResult(
                    source_root=root_url,
                    visited_urls=visited_urls,
                    news=[],
                    events=[],
                    errors=[root_error],
                )
            )
            continue

        visited_urls.append(root_url)
        if root_html is None:
            source_results.append(
                SourceCrawlResult(
                    source_root=root_url,
                    visited_urls=visited_urls,
                    news=[],
                    events=[],
                    errors=errors + [f"{root_url}: empty response body"],
                )
            )
            continue
        root_soup = BeautifulSoup(root_html, "html.parser")

        all_news = _extract_bme_news_cards(root_soup, root_url)
        all_news.extend(_extract_department_news_articles(root_soup, root_url))
        all_events = _extract_bme_event_cards(root_soup, root_url)

        candidate_links = _discover_news_event_links(
            root_soup,
            root_url=root_url,
            max_links=request.max_pages_per_root,
        )

        for link in candidate_links:
            if link in visited_urls:
                continue

            page_html, page_error = _safe_fetch_html(link)
            visited_urls.append(link)

            if page_error:
                errors.append(page_error)
                continue

            if page_html is None:
                errors.append(f"{link}: empty response body")
                continue
            page_soup = BeautifulSoup(page_html, "html.parser")
            all_news.extend(_extract_bme_news_cards(page_soup, link))
            all_news.extend(_extract_department_news_articles(page_soup, link))
            all_events.extend(_extract_bme_event_cards(page_soup, link))

        source_results.append(
            SourceCrawlResult(
                source_root=root_url,
                visited_urls=visited_urls,
                news=_dedupe_news_items(all_news),
                events=_dedupe_event_items(all_events),
                errors=errors,
            )
        )

    total_news = sum(len(result.news) for result in source_results)
    total_events = sum(len(result.events) for result in source_results)

    return {
        "language": request.language,
        "sources": [result.model_dump(mode="json") for result in source_results],
        "summary": {
            "source_count": len(source_results),
            "total_news": total_news,
            "total_events": total_events,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FastMCP news content server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/mcp")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # Compatibility fallback across FastMCP versions.
    try:
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)
        return
    except TypeError:
        pass

    try:
        mcp.run(host=args.host, port=args.port, path=args.path)
        return
    except TypeError:
        pass

    mcp.run()


if __name__ == "__main__":
    main()
