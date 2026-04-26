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
DEFAULT_EVENT_KEYWORDS = ["esemeny", "event", "program", "calendar", "koncert"]
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\\s+")


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
    news_keywords: list[str] = Field(
        default_factory=lambda: DEFAULT_NEWS_KEYWORDS.copy(),
        description="Kulcsszavak a hir linkek felismeresehez.",
    )
    event_keywords: list[str] = Field(
        default_factory=lambda: DEFAULT_EVENT_KEYWORDS.copy(),
        description="Kulcsszavak az esemeny linkek felismeresehez.",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "UrlDiscoveryInput":
        if not self.page_url and not self.html_content:
            raise ValueError("Adj meg page_url vagy html_content erteket.")
        return self


class UrlDiscoveryResult(BaseModel):
    source_url: HttpUrl | None
    total_links_scanned: int
    news_urls: list[HttpUrl]
    event_urls: list[HttpUrl]


class UrlSummarizeInput(BaseModel):
    urls: list[HttpUrl] = Field(description="Feldolgozando oldalak URL-jei.")
    max_items: int = Field(default=10, ge=1, le=100)
    summary_sentence_count: int = Field(default=3, ge=1, le=10)


class ContentItem(BaseModel):
    kind: Literal["news", "event"]
    url: HttpUrl
    title: str | None = None
    text: str | None = None
    image_url: HttpUrl | None = None
    summary: str | None = None


class ContentBatchResult(BaseModel):
    processed_count: int
    items: list[ContentItem]
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


def _extract_image_url(soup: BeautifulSoup, page_url: str) -> str | None:
    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        return urljoin(page_url, og_image["content"].strip())

    first_image = soup.find("img")
    if first_image and first_image.get("src"):
        return urljoin(page_url, first_image["src"].strip())

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


def _extract_item_from_url(url: str, kind: Literal["news", "event"], sentence_count: int) -> ContentItem:
    html = _fetch_html(url)
    soup = _parse_html(html)

    title = _extract_title(soup)
    text = _extract_main_text(soup)
    image_url = _extract_image_url(soup, url)
    summary = _make_summary(text, sentence_count)

    return ContentItem(
        kind=kind,
        url=url,
        title=title,
        text=text,
        image_url=image_url,
        summary=summary,
    )


def _discover_urls(payload: UrlDiscoveryInput) -> UrlDiscoveryResult:
    html = payload.html_content
    if payload.page_url:
        html = _fetch_html(str(payload.page_url))

    if not html:
        raise ValueError("Nem talalhato HTML tartalom a feldolgozashoz.")

    base = str(payload.base_url or payload.page_url) if (payload.base_url or payload.page_url) else None
    soup = _parse_html(html)

    scanned_links = 0
    news_urls: list[str] = []
    event_urls: list[str] = []

    for a_tag in soup.find_all("a"):
        scanned_links += 1
        href = a_tag.get("href")
        absolute_url = _normalize_url(href, base)
        if not absolute_url:
            continue

        haystack = f"{absolute_url} {a_tag.get_text(' ', strip=True)}".lower()

        if any(keyword.lower() in haystack for keyword in payload.news_keywords):
            news_urls.append(absolute_url)

        if any(keyword.lower() in haystack for keyword in payload.event_keywords):
            event_urls.append(absolute_url)

    unique_news = sorted(set(news_urls))
    unique_events = sorted(set(event_urls))

    return UrlDiscoveryResult(
        source_url=payload.page_url,
        total_links_scanned=scanned_links,
        news_urls=unique_news,
        event_urls=unique_events,
    )


def _summarize_urls(payload: UrlSummarizeInput, kind: Literal["news", "event"]) -> ContentBatchResult:
    selected = payload.urls[: payload.max_items]

    items: list[ContentItem] = []
    errors: list[str] = []

    for url in selected:
        try:
            item = _extract_item_from_url(str(url), kind=kind, sentence_count=payload.summary_sentence_count)
            items.append(item)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")

    return ContentBatchResult(
        processed_count=len(items),
        items=items,
        errors=errors,
    )


mcp = FastMCP(name="NewsEventsAgent")


@mcp.tool()
def discover_news_event_urls(input_data: UrlDiscoveryInput) -> UrlDiscoveryResult:
    """HTML oldalon hir es esemeny URL-ek kigyujtese."""
    return _discover_urls(input_data)


@mcp.tool()
def summarize_news_urls(input_data: UrlSummarizeInput) -> ContentBatchResult:
    """Hir URL-ek letoltese, tartalomkinyerese es rovid osszefoglalasa."""
    return _summarize_urls(input_data, kind="news")


@mcp.tool()
def summarize_event_urls(input_data: UrlSummarizeInput) -> ContentBatchResult:
    """Esemeny URL-ek letoltese, tartalomkinyerese es rovid osszefoglalasa."""
    return _summarize_urls(input_data, kind="event")


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
