from __future__ import annotations

import json
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


class PageClassificationPayload(BaseModel):
    item_type: Literal["news", "event"]


class PageItem(BaseModel):
    source_url: HttpUrl
    source_unit: Literal["BME", "tmit", "aut", "ttk", "other"]
    item_type: Literal["news", "event"]
    title: str
    content: str


class PageBatchResult(BaseModel):
    processed_count: int
    page_items: list[PageItem]
    errors: list[str]


LLM_MODEL_ENV = "NEWS_EVENTS_LLM_MODEL"
LLM_BASE_URL_ENV = "NEWS_EVENTS_LLM_BASE_URL"
LLM_API_KEY_ENV = "NEWS_EVENTS_LLM_API_KEY"


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


def _infer_source_unit(url: str, soup: BeautifulSoup) -> Literal["BME", "tmit", "aut", "ttk", "other"]:
    url_haystack = f"{url} {urlparse(url).netloc} {urlparse(url).path}".lower()
    page_text = soup.get_text(" ", strip=True).lower()
    haystack = f"{url_haystack} {page_text[:4000]}"

    if "bme" in haystack:
        return "BME"
    if "tmit" in haystack:
        return "tmit"
    if "aut" in haystack:
        return "aut"
    if "ttk" in haystack:
        return "ttk"
    return "other"


def _build_llm_prompt(url: str, title: str | None, content: str | None, source_unit: str) -> str:
    return (
        "Döntsd el, hogy az alábbi oldal hírt vagy eseményt ír le. "
        "Csak JSON-t adj vissza, pontosan ebben a formában: {\"item_type\": \"news\"|\"event\"}. "
        "Ha bizonytalan vagy, az oldal eseménynek számítson csak akkor, ha programot, időpontot, helyszínt vagy regisztrációt tartalmaz. "
        f"Forrás URL: {url}\n"
        f"Forrás egység: {source_unit}\n"
        f"Cím: {title or 'N/A'}\n"
        f"Kinyert tartalom: {content or 'N/A'}"
    )


def _classify_page_with_llm(url: str, title: str | None, content: str | None, source_unit: str) -> Literal["news", "event"]:
    model_name = os.getenv(LLM_MODEL_ENV, "")
    if not model_name:
        raise RuntimeError(
            f"Hiányzik az LLM modell konfigurációja. Állítsd be a {LLM_MODEL_ENV} környezeti változót."
        )

    base_url = os.getenv(LLM_BASE_URL_ENV)
    api_key = os.getenv(LLM_API_KEY_ENV)
    if not base_url and model_name.startswith("openai:"):
        base_url = "https://api.openai.com/v1"

    if not base_url:
        raise RuntimeError(
            f"Hiányzik az LLM base URL konfigurációja. Állítsd be a {LLM_BASE_URL_ENV} környezeti változót."
        )

    prompt = _build_llm_prompt(url=url, title=title, content=content, source_unit=source_unit)
    payload = {
        "model": model_name.replace("openai:", ""),
        "messages": [
            {
                "role": "system",
                "content": "Te egy szigorú osztályozó vagy. Csak érvényes JSON-t adj vissza.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    response_json = response.json()

    try:
        message_content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(message_content)
        item_type = parsed.get("item_type")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Nem sikerült LLM választ JSON-ként értelmezni: {exc}") from exc

    if item_type not in {"news", "event"}:
        raise RuntimeError(f"Az LLM érvénytelen item_type értéket adott vissza: {item_type!r}")

    return item_type


def _extract_page_item_from_url(url: str, sentence_count: int) -> PageItem:
    html = _fetch_html(url)
    soup = _parse_html(html)

    title = _extract_title(soup)
    text = _extract_main_text(soup)
    summary = _make_summary(text, sentence_count) or text
    source_unit = _infer_source_unit(url, soup)
    item_type = _classify_page_with_llm(url=url, title=title, content=summary, source_unit=source_unit)

    return PageItem(
        source_url=url,
        source_unit=source_unit,
        item_type=item_type,
        title=title or "N/A",
        content=summary or "N/A",
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

        anchor_text = a_tag.get_text(" ", strip=True)
        classification = _classify_link(
            url=absolute_url,
            anchor_text=anchor_text,
            news_keywords=payload.news_keywords,
            event_keywords=payload.event_keywords,
        )

        if classification == "news":
            news_urls.append(absolute_url)

        if classification == "event":
            event_urls.append(absolute_url)

    unique_news = sorted(set(news_urls))
    unique_events = sorted(set(event_urls))

    return UrlDiscoveryResult(
        source_url=payload.page_url,
        total_links_scanned=scanned_links,
        news_urls=unique_news,
        event_urls=unique_events,
    )


def _extract_pages(payload: UrlSummarizeInput) -> PageBatchResult:
    selected = payload.urls[: payload.max_items]

    page_items: list[PageItem] = []
    errors: list[str] = []

    for url in selected:
        try:
            item = _extract_page_item_from_url(str(url), sentence_count=payload.summary_sentence_count)
            page_items.append(item)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")

    return PageBatchResult(
        processed_count=len(page_items),
        page_items=page_items,
        errors=errors,
    )


mcp = FastMCP(name="NewsEventsAgent")


@mcp.tool()
def discover_news_event_urls(input_data: UrlDiscoveryInput) -> UrlDiscoveryResult:
    """HTML oldalon hir es esemeny URL-ek kigyujtese."""
    return _discover_urls(input_data)


@mcp.tool()
def extract_page_content(input_data: UrlSummarizeInput) -> PageBatchResult:
    """Oldalak letoltese, tartalomkinyerese, BME/tanszek besorolas es LLM-alapu hír/esemény címkézés."""
    return _extract_pages(input_data)


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
