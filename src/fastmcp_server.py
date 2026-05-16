from __future__ import annotations

import os
import re
from typing import Literal, cast
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup, FeatureNotFound
from fastmcp import FastMCP
from pydantic import BaseModel, Field, HttpUrl, ValidationError, model_validator, TypeAdapter

# Module-level HttpUrl adapter to validate/coerce URL strings without
# recreating the adapter on each call.
URL_ADAPTER = TypeAdapter(HttpUrl)


EVENT_CARD_SELECTORS = [
    "div.views-view-responsive-grid__item-inner",
    "div.event",
]
EVENT_TITLE_SELECTORS = [
    ".bme_event_card-title",
    ".event-title",
    "h4.bme_event_card-title",
]
EVENT_TEXT_SELECTORS = [
    ".bme_event_card-body",
    ".event-body",
]
EVENT_DATE_SELECTORS = [
    ".bme_event_card-date .nowrap",
    ".event-date",
]
EVENT_LOCATION_SELECTORS = [
    ".bme_event_card-location",
]
NEWS_CARD_SELECTORS = [
    "div.bme_news_card",
    "article.node-hir",
    "div.news-item",
]
NEWS_TITLE_SELECTORS = [
    ".bme_news_card-title",
    "h2.node__title a",
    "h2.news-title-important a",
    ".news-title-important a",
    "h4.bme_news_card-title",
]
NEWS_TEXT_SELECTORS = [
    ".bme_news_card-body",
    ".news-excerpt",
    ".field--name-body",
    ".field-name-body",
]
NEWS_DATE_SELECTORS = [
    "datetime .field--name-created",
    "span.field--name-created",
    ".news-date",
]
NEWS_IMAGE_SELECTORS = [
    ".bme_news_card-thumbnail img",
    ".news-content img",
    ".field-name-field-bevezto-kep img",
]
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
DATE_TIME_RE = re.compile(
    r"(\d{4}[./-]\s*\d{1,2}[./-]\s*\d{1,2}(?:\s+\d{1,2}:\d{2})?|\d{1,2}[./-]\s*\d{1,2}[./-]\s*\d{2,4}(?:\s+\d{1,2}:\d{2})?)"
)
ENGLISH_MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}


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
    min_date: str | None = Field(
        default=None,
        description="Minimum datum (YYYY.MM.DD vagy YYYY-MM-DD formatumban). Csak az ennél nem regebbi (URL-ben vagy a talalt HTML blokkban datalhato) linkeket adja vissza.",
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
    filtered_by_date: bool = False
    applied_min_date: str | None = None
    errors: list["ToolError"] = Field(default_factory=list)


class UrlSummarizeInput(BaseModel):
    detected_pages: list["DetectedPage"] = Field(
        description="A discover_text_and_detection_from_url tool kimenete.")
    max_items: int = Field(default=10, ge=1, le=100)
    summary_sentence_count: int = Field(default=3, ge=1, le=10)
    min_date: str | None = Field(
        default=None,
        description="Minimum datum (YYYY.MM.DD vagy YYYY-MM-DD formatumban). A megadott dátumnál régebbi elemeket kihagyja az összefoglalásból.",
    )


class UrlTextDetectionInput(BaseModel):
    urls: list[HttpUrl] = Field(description="Feldolgozando oldalak URL-jei.")
    max_items: int = Field(default=20, ge=1, le=200)
    min_date: str | None = Field(
        default=None,
        description="Minimum datum (YYYY.MM.DD vagy YYYY-MM-DD formatumban). Csak az ennél nem régebbi oldalakat dolgozza fel.",
    )


class DetectedPage(BaseModel):
    source_url: HttpUrl
    detected_type: Literal["news", "event", "unknown"]
    detected_title: str
    detected_author: str | None = None
    detected_text: str
    detected_image: HttpUrl | None = None
    detected_datetime: str | None = None
    detected_european_date: str | None = None
    detected_location: str | None = None
    detected_guests_list: list[str] = Field(default_factory=list)
    detected_registration: HttpUrl | None = None


class UrlTextDetectionResult(BaseModel):
    processed_count: int
    detected_pages: list[DetectedPage]
    errors: list["ToolError"] = Field(default_factory=list)



class NewsItem(BaseModel):
    news_title: str
    news_author: str
    news_content_summary: str
    news_url: HttpUrl
    news_image: HttpUrl | None = None


class NewsBatchResult(BaseModel):
    processed_count: int
    news_items: list[NewsItem]
    errors: list["ToolError"] = Field(default_factory=list)


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
    errors: list["ToolError"] = Field(default_factory=list)


class ToolError(BaseModel):
    stage: str
    error_type: str
    message: str
    url: str | None = None


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


def _make_tool_error(stage: str, exc: Exception, url: str | None = None) -> ToolError:
    return ToolError(
        stage=stage,
        error_type=exc.__class__.__name__,
        message=str(exc),
        url=url,
    )


def _coerce_attribute_text(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    text = str(value).strip()
    return text or None


def _parse_html(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        # Fallback keeps tools working even if lxml is unavailable.
        return BeautifulSoup(html, "html.parser")


def _normalize_url(href: str | None, base_url: str | None) -> HttpUrl | None:
    if not href:
        return None

    href = href.strip()
    if href.startswith("javascript:") or href.startswith("#"):
        return None

    # Resolve relative URLs when a base URL is provided, otherwise use href as-is
    candidate = urljoin(base_url, href) if base_url else href
    parsed = urlparse(candidate)
    # Only accept http/https schemes for HttpUrl fields
    if parsed.scheme and parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        # Use Pydantic TypeAdapter to validate and coerce the URL into HttpUrl.
        try:
            validated = URL_ADAPTER.validate_python(candidate)
            return validated
        except ValidationError:
            return None
    return None



def _extract_first_selector_text(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            text = element.get_text(" ", strip=True)
            if text:
                return text
    return None


def _extract_first_selector_url(soup: BeautifulSoup, selectors: list[str], page_url: str) -> HttpUrl | None:
    for selector in selectors:
        element = soup.select_one(selector)
        if not element:
            continue

        source = element.get("src") or element.get("href")
        source_text = _coerce_attribute_text(source)
        if source_text:
            candidate = urljoin(page_url, source_text)
            parsed = urlparse(candidate)
            if parsed.scheme and parsed.scheme.lower() in {"http", "https"}:
                return cast(HttpUrl, candidate)
    return None


def _extract_date_from_link_context(a_tag) -> str | None:
    containers = [
        a_tag,
        a_tag.find_parent("article"),
        a_tag.find_parent("li"),
        a_tag.find_parent("div"),
    ]

    seen_ids: set[int] = set()
    for container in containers:
        if container is None:
            continue
        container_id = id(container)
        if container_id in seen_ids:
            continue
        seen_ids.add(container_id)

        time_tag = container.find("time")
        if time_tag:
            if time_tag.get("datetime"):
                return str(time_tag["datetime"]).strip()
            time_text = time_tag.get_text(" ", strip=True)
            if time_text:
                return time_text

        for selector in NEWS_DATE_SELECTORS + EVENT_DATE_SELECTORS:
            node = container.select_one(selector)
            if node:
                node_text = node.get_text(" ", strip=True)
                if node_text:
                    return node_text

        container_text = container.get_text(" ", strip=True)
        if container_text:
            matched = DATE_TIME_RE.search(container_text)
            if matched:
                return matched.group(1)

    return None


def _has_news_structure(soup: BeautifulSoup) -> bool:
    return any(soup.select_one(selector) for selector in NEWS_CARD_SELECTORS)


def _has_event_structure(soup: BeautifulSoup) -> bool:
    return any(soup.select_one(selector) for selector in EVENT_CARD_SELECTORS)


def _extract_news_title(soup: BeautifulSoup) -> str | None:
    title = _extract_first_selector_text(soup, NEWS_TITLE_SELECTORS)
    if title:
        return title
    return _extract_title(soup)


def _extract_news_text(soup: BeautifulSoup) -> str | None:
    text = _extract_first_selector_text(soup, NEWS_TEXT_SELECTORS)
    if text:
        return text

    article = soup.find("article", class_=re.compile(r"node-hir|node-promoted|node-teaser", re.I))
    if article:
        paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p") if p.get_text(strip=True)]
        if paragraphs:
            return "\n".join(paragraphs)

    return _extract_main_text(soup)


def _extract_news_datetime(soup: BeautifulSoup) -> str | None:
    date_text = _extract_first_selector_text(soup, NEWS_DATE_SELECTORS)
    if date_text:
        return date_text
    return _extract_event_datetime(soup)


def _extract_news_image(soup: BeautifulSoup, page_url: str) -> HttpUrl | None:
    return _extract_first_selector_url(soup, NEWS_IMAGE_SELECTORS, page_url)


def _extract_page_image(soup: BeautifulSoup, page_url: str) -> HttpUrl | None:
    og_image = soup.find("meta", attrs={"property": "og:image"})
    og_image_content = _coerce_attribute_text(og_image.get("content")) if og_image else None
    if og_image_content:
        return _normalize_url(og_image_content, page_url)

    image = soup.find("img")
    if image:
        source_text = _coerce_attribute_text(image.get("src"))
        if source_text:
            return _normalize_url(source_text, page_url)

    return None


def _extract_event_title(soup: BeautifulSoup) -> str | None:
    title = _extract_first_selector_text(soup, EVENT_TITLE_SELECTORS)
    if title:
        return title
    return _extract_title(soup)


def _extract_event_text(soup: BeautifulSoup) -> str | None:
    text = _extract_first_selector_text(soup, EVENT_TEXT_SELECTORS)
    if text:
        return text
    return _extract_main_text(soup)


def _extract_event_date_text(soup: BeautifulSoup) -> str | None:
    date_text = _extract_first_selector_text(soup, EVENT_DATE_SELECTORS)
    if date_text:
        return date_text
    return _extract_event_datetime(soup)


def _extract_event_card_location(soup: BeautifulSoup) -> str | None:
    location = _extract_first_selector_text(soup, EVENT_LOCATION_SELECTORS)
    if location:
        return location
    return _extract_event_location(soup)


def _extract_news_page_data(soup: BeautifulSoup, page_url: str) -> tuple[str | None, str | None, HttpUrl | None, str | None, str | None]:
    return (
        _extract_news_title(soup),
        _extract_news_text(soup),
        _extract_news_image(soup, page_url=page_url),
        _extract_news_datetime(soup),
        None,
    )


def _extract_event_page_data(soup: BeautifulSoup) -> tuple[str | None, str | None, HttpUrl | None, str | None, str | None]:
    return (
        _extract_event_title(soup),
        _extract_event_text(soup),
        None,
        _extract_event_date_text(soup),
        _extract_event_card_location(soup),
    )


def _classify_page_content(soup: BeautifulSoup | None = None) -> Literal["news", "event", "unknown"]:
    if soup and _has_event_structure(soup):
        return "event"

    if soup and _has_news_structure(soup):
        return "news"

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
                text = "\n".join(paragraphs)
                if len(text) >= 120:
                    return text

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    if paragraphs:
        return "\n".join(paragraphs)
    return None


def _extract_title(soup: BeautifulSoup) -> str | None:
    og_title = soup.find("meta", attrs={"property": "og:title"})
    og_title_content = _coerce_attribute_text(og_title.get("content")) if og_title else None
    if og_title_content:
        return og_title_content

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)

    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(" ", strip=True)

    return None


def _extract_author(soup: BeautifulSoup) -> str | None:
    author_meta = soup.find("meta", attrs={"name": "author"})
    author_meta_content = _coerce_attribute_text(author_meta.get("content")) if author_meta else None
    if author_meta_content:
        return author_meta_content

    article_author_meta = soup.find("meta", attrs={"property": "article:author"})
    article_author_content = (
        _coerce_attribute_text(article_author_meta.get("content")) if article_author_meta else None
    )
    if article_author_content:
        return article_author_content

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


def _extract_event_registration(soup: BeautifulSoup, page_url: str) -> HttpUrl | None:
    registration_pattern = re.compile(r"register|registration|signup|ticket|book|jelentkez", re.I)
    for a_tag in soup.find_all("a"):
        href = a_tag.get("href")
        href_text = _coerce_attribute_text(href)
        if not href_text:
            continue
        anchor_text = a_tag.get_text(" ", strip=True)
        haystack = f"{href_text} {anchor_text}"
        if registration_pattern.search(haystack):
            candidate = urljoin(page_url, href_text)
            parsed = urlparse(candidate)
            if parsed.scheme and parsed.scheme.lower() in {"http", "https"}:
                return cast(HttpUrl, candidate)

    return None


def _extract_date_candidate(text: str) -> str:
    matched = DATE_TIME_RE.search(text)
    candidate = matched.group(1) if matched else text
    return re.sub(r"\s+", "", candidate).rstrip(".-/")


def _normalize_date_text(text: str) -> str:
    normalized_text = text.strip().lower()
    for month_name, month_number in sorted(ENGLISH_MONTHS.items(), key=lambda item: len(item[0]), reverse=True):
        normalized_text = re.sub(rf"\b{re.escape(month_name)}\b", month_number, normalized_text)
    return re.sub(r"\s+", " ", normalized_text)


def _parse_date_candidate(candidate: str) -> str | None:
    loose_match = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", candidate)
    if loose_match:
        year, month, day = loose_match.groups()
        try:
            dt = datetime.strptime(f"{year}.{int(month):02d}.{int(day):02d}", "%Y.%m.%d")
            return dt.strftime("%Y.%m.%d")
        except ValueError:
            pass

    fmts = [
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y.%m.%d",
        "%Y/%m/%d",
        "%Y.%b.%d",
        "%Y.%B.%d",
        "%d.%b.%Y",
        "%d.%B.%Y",
        "%Y-%b-%d",
        "%Y-%B-%d",
        "%d-%b-%Y",
        "%d-%B-%Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.strftime("%Y.%m.%d")
        except ValueError:
            continue

    compact_match = re.fullmatch(r"(\d{4})(\d{2})(\d{1,2})", candidate)
    if compact_match:
        year, month, day = compact_match.groups()
        try:
            dt = datetime.strptime(f"{year}.{month}.{day}", "%Y.%m.%d")
            return dt.strftime("%Y.%m.%d")
        except ValueError:
            pass

    return None


def _parse_to_yyyy_mm_dd(s: str | None) -> str | None:
    if not s:
        return None
    normalized_text = _normalize_date_text(s)
    for candidate in (
        normalized_text,
        _extract_date_candidate(normalized_text),
    ):
        parsed = _parse_date_candidate(candidate)
        if parsed:
            return parsed
    return None


def _is_date_after_min_date(extracted_date: str | None, min_date: str | None) -> bool:
    """Return True only if extracted_date is parseable and >= min_date.

    Behavior:
    - If min_date is missing, filtering is disabled and returns True.
    - If min_date is present but invalid, raises ValueError.
    - If extracted_date is missing/unparseable, returns False (fail-closed).
    """
    if not min_date:
        return True
    min_normalized = _parse_to_yyyy_mm_dd(min_date)
    if not min_normalized:
        raise ValueError("Invalid min_date. Expected YYYY.MM.DD or YYYY-MM-DD.")

    if not extracted_date:
        return False

    extracted_normalized = _parse_to_yyyy_mm_dd(extracted_date)
    if not extracted_normalized:
        return False

    # Lexicographic compare is safe for YYYY.MM.DD.
    return extracted_normalized >= min_normalized


def _is_any_date_after_min_date(candidate_dates: list[str | None], min_date: str | None) -> bool:
    if not min_date:
        return True
    return any(
        _is_date_after_min_date(candidate_date, min_date)
        for candidate_date in candidate_dates
        if candidate_date
    )


def _make_summary(text: str | None, max_sentences: int) -> str | None:
    if not text:
        return None

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return None

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
    if not sentences:
        return normalized[:300]

    return " ".join(sentences[:max_sentences])


def _discover_urls(payload: UrlDiscoveryInput) -> UrlDiscoveryResult:
    html = payload.html_content
    if payload.page_url:
        try:
            html = _fetch_html(str(payload.page_url))
        except requests.RequestException as exc:
            return UrlDiscoveryResult(
                source_url=payload.page_url,
                total_links_scanned=0,
                discovered_urls=[],
                filtered_by_date=bool(payload.min_date),
                applied_min_date=payload.min_date,
                errors=[_make_tool_error("discover_news_event_urls", exc, url=str(payload.page_url))],
            )

    if not html:
        return UrlDiscoveryResult(
            source_url=payload.page_url,
            total_links_scanned=0,
            discovered_urls=[],
            filtered_by_date=bool(payload.min_date),
            applied_min_date=payload.min_date,
            errors=[_make_tool_error("discover_news_event_urls", ValueError("Nem talalhato HTML tartalom a feldolgozashoz."))],
        )

    base = str(payload.base_url or payload.page_url) if (payload.base_url or payload.page_url) else None
    soup = _parse_html(html)

    scanned_links = 0
    discovered_urls: list[HttpUrl] = []

    for a_tag in soup.find_all("a"):
        scanned_links += 1
        href = a_tag.get("href")
        absolute_url = _normalize_url(_coerce_attribute_text(href), base)
        if not absolute_url:
            continue

        if payload.min_date:
            if not _is_any_date_after_min_date(
                [str(absolute_url), _extract_date_from_link_context(a_tag)],
                payload.min_date,
            ):
                continue

        discovered_urls.append(absolute_url)

    unique_urls = list(dict.fromkeys(discovered_urls))

    return UrlDiscoveryResult(
        source_url=payload.page_url,
        total_links_scanned=scanned_links,
        discovered_urls=unique_urls,
        filtered_by_date=bool(payload.min_date),
        applied_min_date=payload.min_date,
    )


def _discover_text_and_detection(payload: UrlTextDetectionInput) -> UrlTextDetectionResult:
    selected = payload.urls[: payload.max_items]

    detected_pages: list[DetectedPage] = []
    errors: list[ToolError] = []

    for url in selected:
        url_str = str(url)
        try:
            html = _fetch_html(url_str)
            soup = _parse_html(html)

            is_news_structure = _has_news_structure(soup)
            is_event_structure = _has_event_structure(soup)

            if is_news_structure:
                title, text, page_image, page_datetime, page_location = _extract_news_page_data(soup, url_str)
                title = title or "N/A"
                text = text or ""
            elif is_event_structure:
                title, text, page_image, page_datetime, page_location = _extract_event_page_data(soup)
                title = title or "N/A"
                text = text or ""
            else:
                title = _extract_title(soup) or "N/A"
                text = _extract_main_text(soup) or ""
                page_image = _extract_page_image(soup, url_str)
                page_datetime = _extract_event_datetime(soup)
                page_location = _extract_event_location(soup)

            author = _extract_author(soup)
            event_datetime = page_datetime
            event_eu = _parse_to_yyyy_mm_dd(event_datetime)
            event_location = page_location
            event_guests = _extract_event_guests(soup)
            registration = _extract_event_registration(soup, page_url=url_str)

            # Only apply date filter if detected_type would be "unknown"
            # For news/event, we skip the date filter since we don't reliably extract dates
            if payload.min_date and not (is_news_structure or is_event_structure):
                if not _is_any_date_after_min_date([event_datetime, url_str], payload.min_date):
                    continue

            detected_type = _classify_page_content(soup=soup)

            if is_news_structure:
                detected_type = "news"
            elif is_event_structure:
                detected_type = "event"
            else:
                detected_type = "unknown"

            detected_pages.append(
                DetectedPage(
                    source_url=url,
                    detected_type=detected_type,
                    detected_title=title,
                    detected_author=author,
                    detected_text=text,
                    detected_image=page_image,
                    detected_datetime=event_datetime,
                    detected_european_date=event_eu,
                    detected_location=event_location,
                    detected_guests_list=event_guests,
                    detected_registration=registration,
                )
            )
        except (requests.RequestException, ValueError, TypeError, ValidationError) as exc:
            errors.append(_make_tool_error("discover_text_and_detection_from_url", exc, url=url_str))

    return UrlTextDetectionResult(
        processed_count=len(detected_pages),
        detected_pages=detected_pages,
        errors=errors,
    )


def _summarize_news_urls(payload: UrlSummarizeInput) -> NewsBatchResult:
    selected = [
        page
        for page in payload.detected_pages
        if page.detected_type == "news"
        and _is_any_date_after_min_date([page.detected_european_date, page.detected_datetime], payload.min_date)
    ][: payload.max_items]

    news_items: list[NewsItem] = []
    errors: list[ToolError] = []

    for page in selected:
        try:
            summary = _make_summary(page.detected_text, payload.summary_sentence_count) or page.detected_text
            item = NewsItem(
                news_title=page.detected_title or "N/A",
                news_author=page.detected_author or "N/A",
                news_content_summary=summary or "N/A",
                news_url=page.source_url,
                news_image=page.detected_image,
            )
            news_items.append(item)
        except (ValueError, TypeError, ValidationError) as exc:
            errors.append(_make_tool_error("summarize_news_urls", exc, url=str(page.source_url)))

    return NewsBatchResult(
        processed_count=len(news_items),
        news_items=news_items,
        errors=errors,
    )


def _summarize_event_urls(payload: UrlSummarizeInput) -> EventBatchResult:
    selected = [
        page
        for page in payload.detected_pages
        if page.detected_type == "event"
        and _is_any_date_after_min_date([page.detected_european_date, page.detected_datetime], payload.min_date)
    ][: payload.max_items]

    events_items: list[EventItem] = []
    errors: list[ToolError] = []

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
        except (ValueError, TypeError, ValidationError) as exc:
            errors.append(_make_tool_error("summarize_event_urls", exc, url=str(page.source_url)))

    return EventBatchResult(
        processed_count=len(events_items),
        events_items=events_items,
        errors=errors,
    )


# Ensure MCP instance exists before using @mcp.tool() decorators
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
