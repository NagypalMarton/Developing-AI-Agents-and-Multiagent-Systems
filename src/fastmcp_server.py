from __future__ import annotations

import os
import re
from typing import Literal
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, FeatureNotFound
from fastmcp import FastMCP
from pydantic import BaseModel, Field, HttpUrl, model_validator


DEFAULT_NEWS_KEYWORDS = ["hir", "news", "article", "cikk"]
DEFAULT_EVENT_KEYWORDS = [
    "esemeny",
    "events",
    "event",
    "program",
    "calendar",
    "koncert",
    "workshop",
    "felhivas",
    "börze",
    "borze",
    "konferencia",
]
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\\s+")
DATE_TIME_RE = re.compile(
    r"(\d{4}[./-]\d{1,2}[./-]\d{1,2}(?:\s+\d{1,2}:\d{2})?|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:\s+\d{1,2}:\d{2})?)"
)


class UrlDiscoveryInput(BaseModel):
    page_url: HttpUrl | None = Field(
        default=None,
        description="A forras oldal URL-je, ahonnan a linkeket gyujtjuk.",
    )
    html_content: str | None = Field(
        default=None,
        description="Kozvetlenul megadott HTML tartalom. Akkor hasznald, ha nincs page_url.",
    )
    base_url: HttpUrl | None = Field(
        default=None,
        description="Relativ URL-ek feloldasahoz hasznalt bazis URL.",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "UrlDiscoveryInput":
        if not self.page_url and not self.html_content:
            raise ValueError("Adj meg page_url vagy html_content erteket.")
        return self


class UrlDiscoveryResult(BaseModel):
    source_url: HttpUrl | None
    total_links_scanned: int
    discovered_urls: list[HttpUrl]


class UrlSummarizeInput(BaseModel):
    detected_pages: list["DetectedPage"] = Field(
        description="A discover_text_and_detection_from_url tool kimenete.")
    max_items: int = Field(default=10, ge=1, le=100)
    summary_sentence_count: int = Field(default=3, ge=1, le=10)


class UrlTextDetectionInput(BaseModel):
    urls: list[HttpUrl] = Field(description="Feldolgozando oldalak URL-jei.")
    max_items: int = Field(default=20, ge=1, le=200)
    news_keywords: list[str] = Field(
        default_factory=lambda: DEFAULT_NEWS_KEYWORDS.copy(),
        description="Kulcsszavak a hir tartalmak felismeresehez.",
    )
    event_keywords: list[str] = Field(
        default_factory=lambda: DEFAULT_EVENT_KEYWORDS.copy(),
        description="Kulcsszavak az esemeny tartalmak felismeresehez.",
    )


class DetectedPage(BaseModel):
    source_url: HttpUrl
    detected_type: Literal["news", "event", "unknown"]
    detected_title: str
    detected_author: str | None = None
    detected_text: str
    detected_datetime: str | None = None
    detected_location: str | None = None
    detected_guests_list: list[str] = Field(default_factory=list)
    detected_registration: HttpUrl | None = None


class UrlTextDetectionResult(BaseModel):
    processed_count: int
    detected_pages: list[DetectedPage]
    errors: list[str]


class NewsItem(BaseModel):
    news_title: str
    news_auther: str
    news_content: str
    news_url: HttpUrl
    news_image: HttpUrl | None = None


class NewsBatchResult(BaseModel):
    processed_count: int
    news_items: list[NewsItem]
    errors: list[str]


class EventItem(BaseModel):
    events_title: str
    events_datetime: str
    events_location: str
    events_description: str
    events_guests_list: list[str]
    events_registration: HttpUrl | None = None


class EventBatchResult(BaseModel):
    processed_count: int
    events_items: list[EventItem]
    errors: list[str]


def _fetch_html(url: str, timeout: int = 15) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    return response.text


def _parse_html(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        # Fallback keeps tools working even if lxml is unavailable.
        return BeautifulSoup(html, "html.parser")


def _normalize_url(href: str | None, base_url: str | None) -> str | None:
    if not href:
        return None

    href = href.strip()
    if href.startswith("javascript:") or href.startswith("#"):
        return None

    if base_url:
        return urljoin(base_url, href)

    parsed = urlparse(href)
    if parsed.scheme in {"http", "https"}:
        return href
    return None


def _classify_link(url: str, anchor_text: str, news_keywords: list[str], event_keywords: list[str]) -> str | None:
    url_haystack = f"{url} {urlparse(url).path}".lower()
    anchor_haystack = anchor_text.lower()

    if any(keyword.lower() in url_haystack for keyword in event_keywords):
        return "event"

    if any(keyword.lower() in url_haystack for keyword in news_keywords):
        return "news"

    if any(keyword.lower() in anchor_haystack for keyword in event_keywords):
        return "event"

    if any(keyword.lower() in anchor_haystack for keyword in news_keywords):
        return "news"

    return None


def _classify_page_content(
    url: str,
    title: str | None,
    text: str | None,
    news_keywords: list[str],
    event_keywords: list[str],
) -> Literal["news", "event", "unknown"]:
    text_sample = (text or "")[:4000]
    haystack = f"{url} {title or ''} {text_sample}".lower()

    event_score = sum(1 for keyword in event_keywords if keyword.lower() in haystack)
    news_score = sum(1 for keyword in news_keywords if keyword.lower() in haystack)

    if event_score > news_score and event_score > 0:
        return "event"

    if news_score > event_score and news_score > 0:
        return "news"

    tie_breaker = _classify_link(
        url=url,
        anchor_text=title or "",
        news_keywords=news_keywords,
        event_keywords=event_keywords,
    )
    if tie_breaker == "news":
        return "news"
    if tie_breaker == "event":
        return "event"

    return "unknown"


def _extract_main_text(soup: BeautifulSoup) -> str | None:
    candidates = [
        soup.find("article"),
        soup.find("main"),
        soup.find("div", class_=re.compile("article|content|post|entry", re.I)),
    ]

    for candidate in candidates:
        if candidate:
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in candidate.find_all("p")
                if p.get_text(strip=True)
            ]
            if paragraphs:
                text = "\\n".join(paragraphs)
                if len(text) >= 120:
                    return text

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    if paragraphs:
        return "\\n".join(paragraphs)
    return None


def _extract_title(soup: BeautifulSoup) -> str | None:
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)

    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(" ", strip=True)

    return None


def _extract_author(soup: BeautifulSoup) -> str | None:
    author_meta = soup.find("meta", attrs={"name": "author"})
    if author_meta and author_meta.get("content"):
        return author_meta["content"].strip()

    article_author_meta = soup.find("meta", attrs={"property": "article:author"})
    if article_author_meta and article_author_meta.get("content"):
        return article_author_meta["content"].strip()

    author_pattern = re.compile(r"author|byline|szerzo", re.I)
    for tag in soup.find_all(True):
        class_attr = tag.get("class")
        class_text = " ".join(class_attr) if isinstance(class_attr, list) else str(class_attr or "")
        id_text = str(tag.get("id") or "")
        if author_pattern.search(class_text) or author_pattern.search(id_text):
            text = tag.get_text(" ", strip=True)
            if text and len(text) <= 120:
                return text

    return None


def _extract_event_datetime(soup: BeautifulSoup) -> str | None:
    time_tag = soup.find("time")
    if time_tag:
        if time_tag.get("datetime"):
            return str(time_tag["datetime"]).strip()
        text = time_tag.get_text(" ", strip=True)
        if text:
            return text

    datetime_pattern = re.compile(r"date|time|datum|ido|kezd", re.I)
    for tag in soup.find_all(True):
        class_attr = tag.get("class")
        class_text = " ".join(class_attr) if isinstance(class_attr, list) else str(class_attr or "")
        id_text = str(tag.get("id") or "")
        if datetime_pattern.search(class_text) or datetime_pattern.search(id_text):
            text = tag.get_text(" ", strip=True)
            if text:
                matched = DATE_TIME_RE.search(text)
                return matched.group(1) if matched else text

    whole_page = soup.get_text(" ", strip=True)
    matched = DATE_TIME_RE.search(whole_page)
    return matched.group(1) if matched else None


def _extract_event_location(soup: BeautifulSoup) -> str | None:
    location_pattern = re.compile(r"location|venue|place|helyszin|site", re.I)
    for tag in soup.find_all(True):
        class_attr = tag.get("class")
        class_text = " ".join(class_attr) if isinstance(class_attr, list) else str(class_attr or "")
        id_text = str(tag.get("id") or "")
        if location_pattern.search(class_text) or location_pattern.search(id_text):
            text = tag.get_text(" ", strip=True)
            if text:
                return text

    location_label_pattern = re.compile(r"(location|venue|helyszin)\s*[:\-]\s*(.+)", re.I)
    whole_page = soup.get_text(" ", strip=True)
    matched = location_label_pattern.search(whole_page)
    return matched.group(2).strip() if matched else None


def _extract_event_guests(soup: BeautifulSoup) -> list[str]:
    guests: list[str] = []
    guest_pattern = re.compile(r"guest|speaker|eload|vendeg", re.I)

    for tag in soup.find_all(True):
        class_attr = tag.get("class")
        class_text = " ".join(class_attr) if isinstance(class_attr, list) else str(class_attr or "")
        id_text = str(tag.get("id") or "")
        if guest_pattern.search(class_text) or guest_pattern.search(id_text):
            for child in tag.find_all(["li", "p", "span", "a"]):
                text = child.get_text(" ", strip=True)
                if text and len(text) <= 120:
                    guests.append(text)
            text = tag.get_text(" ", strip=True)
            if text and len(text) <= 120:
                guests.append(text)

    unique_guests = sorted({guest for guest in guests if guest})
    return unique_guests


def _extract_event_registration(soup: BeautifulSoup, page_url: str) -> str | None:
    registration_pattern = re.compile(r"register|registration|signup|ticket|book|jelentkez", re.I)
    for a_tag in soup.find_all("a"):
        href = a_tag.get("href")
        if not href:
            continue
        anchor_text = a_tag.get_text(" ", strip=True)
        haystack = f"{href} {anchor_text}"
        if registration_pattern.search(haystack):
            return urljoin(page_url, href.strip())

    return None


def _make_summary(text: str | None, max_sentences: int) -> str | None:
    if not text:
        return None

    normalized = re.sub(r"\\s+", " ", text).strip()
    if not normalized:
        return None

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
    if not sentences:
        return normalized[:300]

    return " ".join(sentences[:max_sentences])


def _discover_urls(payload: UrlDiscoveryInput) -> UrlDiscoveryResult:
    html = payload.html_content
    if payload.page_url:
        html = _fetch_html(str(payload.page_url))

    if not html:
        raise ValueError("Nem talalhato HTML tartalom a feldolgozashoz.")

    base = str(payload.base_url or payload.page_url) if (payload.base_url or payload.page_url) else None
    soup = _parse_html(html)

    scanned_links = 0
    discovered_urls: list[str] = []

    for a_tag in soup.find_all("a"):
        scanned_links += 1
        href = a_tag.get("href")
        absolute_url = _normalize_url(href, base)
        if not absolute_url:
            continue
        discovered_urls.append(absolute_url)

    unique_urls = sorted(set(discovered_urls))

    return UrlDiscoveryResult(
        source_url=payload.page_url,
        total_links_scanned=scanned_links,
        discovered_urls=unique_urls,
    )


def _discover_text_and_detection(payload: UrlTextDetectionInput) -> UrlTextDetectionResult:
    selected = payload.urls[: payload.max_items]

    detected_pages: list[DetectedPage] = []
    errors: list[str] = []

    for url in selected:
        url_str = str(url)
        try:
            html = _fetch_html(url_str)
            soup = _parse_html(html)

            title = _extract_title(soup) or "N/A"
            text = _extract_main_text(soup) or ""
            author = _extract_author(soup)
            event_datetime = _extract_event_datetime(soup)
            event_location = _extract_event_location(soup)
            event_guests = _extract_event_guests(soup)
            registration = _extract_event_registration(soup, page_url=url_str)

            detected_type = _classify_page_content(
                url=url_str,
                title=title,
                text=text,
                news_keywords=payload.news_keywords,
                event_keywords=payload.event_keywords,
            )

            detected_pages.append(
                DetectedPage(
                    source_url=url,
                    detected_type=detected_type,
                    detected_title=title,
                    detected_author=author,
                    detected_text=text,
                    detected_datetime=event_datetime,
                    detected_location=event_location,
                    detected_guests_list=event_guests,
                    detected_registration=registration,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")

    return UrlTextDetectionResult(
        processed_count=len(detected_pages),
        detected_pages=detected_pages,
        errors=errors,
    )


def _summarize_news_urls(payload: UrlSummarizeInput) -> NewsBatchResult:
    selected = [page for page in payload.detected_pages if page.detected_type == "news"][: payload.max_items]

    news_items: list[NewsItem] = []
    errors: list[str] = []

    for page in selected:
        try:
            summary = _make_summary(page.detected_text, payload.summary_sentence_count) or page.detected_text
            item = NewsItem(
                news_title=page.detected_title or "N/A",
                news_auther=page.detected_author or "N/A",
                news_content=summary or "N/A",
                news_url=page.source_url,
                news_image=None,
            )
            news_items.append(item)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{page.source_url} -> {exc}")

    return NewsBatchResult(
        processed_count=len(news_items),
        news_items=news_items,
        errors=errors,
    )


def _summarize_event_urls(payload: UrlSummarizeInput) -> EventBatchResult:
    selected = [page for page in payload.detected_pages if page.detected_type == "event"][: payload.max_items]

    events_items: list[EventItem] = []
    errors: list[str] = []

    for page in selected:
        try:
            summary = _make_summary(page.detected_text, payload.summary_sentence_count) or page.detected_text
            item = EventItem(
                events_title=page.detected_title or "N/A",
                events_datetime=page.detected_datetime or "N/A",
                events_location=page.detected_location or "N/A",
                events_description=summary or "N/A",
                events_guests_list=page.detected_guests_list,
                events_registration=page.detected_registration,
            )
            events_items.append(item)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{page.source_url} -> {exc}")

    return EventBatchResult(
        processed_count=len(events_items),
        events_items=events_items,
        errors=errors,
    )


mcp = FastMCP(name="NewsEventsAgent")


@mcp.tool()
def discover_news_event_urls(input_data: UrlDiscoveryInput) -> UrlDiscoveryResult:
    """HTML oldalon URL-ek kigyujtese osztalyozas nelkul."""
    return _discover_urls(input_data)


@mcp.tool()
def discover_text_and_detection_from_url(input_data: UrlTextDetectionInput) -> UrlTextDetectionResult:
    """URL-ekrol szovegkinyeres es besorolas (hir, esemeny, unknown)."""
    return _discover_text_and_detection(input_data)


@mcp.tool()
def summarize_news_urls(input_data: UrlSummarizeInput) -> NewsBatchResult:
    """Hir URL-ek letoltese, tartalomkinyerese es rovid osszefoglalasa."""
    return _summarize_news_urls(input_data)


@mcp.tool()
def summarize_event_urls(input_data: UrlSummarizeInput) -> EventBatchResult:
    """Esemeny URL-ek letoltese, tartalomkinyerese es rovid osszefoglalasa."""
    return _summarize_event_urls(input_data)


if __name__ == "__main__":
    host = os.getenv("FASTMCP_HOST", "0.0.0.0")
    port = int(os.getenv("FASTMCP_PORT", "8000"))
    path = os.getenv("FASTMCP_PATH", "/mcp")

    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        path=path,
    )
