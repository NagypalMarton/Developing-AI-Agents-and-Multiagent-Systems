from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import re

from fastmcp import FastMCP
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field


mcp = FastMCP("news-event-extractor")


EVENT_KEYWORDS = (
	"esemény",
	"event",
	"workshop",
	"rendezvény",
	"konferencia",
	"előadás",
	"szeminárium",
	"meeting",
	"találkozó",
)

MONTH_NAME_TO_NUMBER = {
	"január": 1,
	"januárja": 1,
	"február": 2,
	"februárja": 2,
	"március": 3,
	"márciusa": 3,
	"április": 4,
	"áprilisa": 4,
	"május": 5,
	"májusa": 5,
	"június": 6,
	"júniusa": 6,
	"július": 7,
	"júliusa": 7,
	"augusztus": 8,
	"augusztusa": 8,
	"szeptember": 9,
	"szeptembere": 9,
	"október": 10,
	"októbere": 10,
	"november": 11,
	"novembere": 11,
	"december": 12,
	"decembere": 12,
}

DATE_PATTERNS = (
	re.compile(
		r"(?P<year>\d{4})\s*[.\-/]??\s*(?P<month>\d{1,2})\s*[.\-/]??\s*(?P<day>\d{1,2})",
		re.IGNORECASE,
	),
	re.compile(
		r"(?P<year>\d{4})\s*[.\-/]??\s*(?P<month>[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+)\s+(?P<day>\d{1,2})",
		re.IGNORECASE,
	),
	re.compile(
		r"(?P<day>\d{1,2})\s*(?:-|\.\s*)?(?:án|en|ának|ének|jára|ére)?\s+(?P<month>[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+)\s+(?P<year>\d{4})",
		re.IGNORECASE,
	),
)


class NewsItem(BaseModel):
	model_config = ConfigDict(str_strip_whitespace=True)

	source_url: AnyHttpUrl
	title: str = Field(min_length=1)
	item_url: AnyHttpUrl
	category: str = Field(pattern=r"^(news|event|unknown)$")
	published_at: Optional[str] = None
	summary: Optional[str] = None
	location: Optional[str] = None
	image_url: Optional[AnyHttpUrl] = None


class NewsExtractionRequest(BaseModel):
	model_config = ConfigDict(str_strip_whitespace=True)

	urls: list[AnyHttpUrl] = Field(min_length=1)


@dataclass(frozen=True)
class RawItem:
	title: str
	item_url: str
	summary: Optional[str] = None
	published_at: Optional[str] = None
	location: Optional[str] = None
	image_url: Optional[str] = None
	category: str = "unknown"


try:
	from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
	BeautifulSoup = None


class _SimpleHTMLTextExtractor(HTMLParser):
	def __init__(self) -> None:
		super().__init__()
		self.parts: list[str] = []

	def handle_data(self, data: str) -> None:
		text = data.strip()
		if text:
			self.parts.append(text)


def _fetch_html(url: str) -> str:
	request = Request(url, headers={"User-Agent": "Mozilla/5.0 (FastMCP News Extractor)"})
	with urlopen(request, timeout=20) as response:
		body = response.read()
		charset = response.headers.get_content_charset() or "utf-8"
		return body.decode(charset, errors="replace")


def _clean_text(value: Optional[str]) -> Optional[str]:
	if value is None:
		return None
	text = re.sub(r"\s+", " ", unescape(value)).strip()
	return text or None


def _extract_text_from_html(fragment: str) -> str:
	extractor = _SimpleHTMLTextExtractor()
	extractor.feed(fragment)
	return _clean_text(" ".join(extractor.parts)) or ""


def _normalize_date(text: Optional[str]) -> Optional[str]:
	if not text:
		return None

	cleaned = _clean_text(text) or ""
	if not cleaned:
		return None

	for pattern in DATE_PATTERNS:
		match = pattern.search(cleaned)
		if not match:
			continue

		year = int(match.group("year"))
		month_raw = match.group("month").lower()
		if month_raw.isdigit():
			month = int(month_raw)
		else:
			month = MONTH_NAME_TO_NUMBER.get(month_raw)
			if month is None:
				continue

		day = int(match.group("day"))
		try:
			return datetime(year, month, day).date().isoformat()
		except ValueError:
			continue

	return cleaned


def _detect_category(title: str, summary: Optional[str], url: str) -> str:
	text = f"{title} {summary or ''} {url}".lower()
	if any(token in text for token in ("esemeny", "esemenyek", "esemény", "események")):
		return "event"
	if any(keyword in text for keyword in EVENT_KEYWORDS):
		return "event"
	if any(token in text for token in ("hir", "hír", "news", "hirek", "hírek")):
		return "news"
	return "unknown"


def _parse_with_bs4(html: str, base_url: str) -> list[RawItem]:
	if BeautifulSoup is None:
		return []

	soup = BeautifulSoup(html, "html.parser")
	if soup is None:
		return []

	candidates: list[RawItem] = []
	seen: set[tuple[str, str]] = set()

	selectors = [
		"div.news-item",
		"article.node-hir",
		"div.bme_news_card",
		"a.h-p100.d-block",
		"div.event",
		"div.views-row article",
		"div.views-row",
	]

	for selector in selectors:
		for node in soup.select(selector):
			title_node = node.select_one(
				"h2 a, h4 a, .node__title a, .event-title, .bme_event_card-title, h2, h4, .bme_news_card-title"
			)
			if title_node is None:
				continue

			title = _clean_text(title_node.get_text(" ", strip=True))
			href = None
			if title_node.name == "a":
				href = title_node.get("href")
			if not href and node.name == "a":
				href = node.get("href")
			if not href:
				local_link = node.select_one("a[href]")
				if local_link is not None:
					href = local_link.get("href")
			if not href:
				parent_link = node.find_parent("a", href=True)
				if parent_link is not None:
					href = parent_link.get("href")
			if not title or not href:
				continue

			item_url = urljoin(base_url, href)
			dedupe_key = (title.lower(), item_url)
			if dedupe_key in seen:
				continue
			seen.add(dedupe_key)

			summary_node = node.select_one(
				".news-excerpt, .news-content p, .bme_news_card-body p, .bme_event_card-body p, .field-name-body p, .field--name-body p, p"
			)
			summary = _clean_text(summary_node.get_text(" ", strip=True)) if summary_node else None

			date_node = node.select_one(
				".news-date, .event-date, .bme_event_card-date, .field--name-created, .created, time, datetime"
			)
			published_at = _normalize_date(date_node.get_text(" ", strip=True) if date_node else None)

			location_node = node.select_one(".bme_event_card-location")
			location = _clean_text(location_node.get_text(" ", strip=True)) if location_node else None

			image_node = node.select_one("img")
			image_url = None
			if image_node is not None:
				image_src = image_node.get("src") or image_node.get("data-src")
				if image_src:
					image_url = urljoin(base_url, image_src)

			candidates.append(
				RawItem(
					title=title,
					item_url=item_url,
					summary=summary,
					published_at=published_at,
					location=location,
					image_url=image_url,
					category=_detect_category(title, summary, item_url),
				)
			)

	return candidates


def _parse_without_bs4(html: str, base_url: str) -> list[RawItem]:
	candidates: list[RawItem] = []
	seen: set[tuple[str, str]] = set()

	patterns = [
		re.compile(r'<div class="news-item">(.*?)</div>\s*</div>', re.DOTALL | re.IGNORECASE),
		re.compile(r'<article[^>]*node-hir[^>]*>(.*?)</article>', re.DOTALL | re.IGNORECASE),
		re.compile(r'<div class="bme_news_card.*?">(.*?)</div>\s*</div>', re.DOTALL | re.IGNORECASE),
		re.compile(r'<a[^>]*class="[^"]*h-p100[^"]*d-block[^"]*"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE),
		re.compile(r'<div[^>]*class="(?:event|[^"]*\sevent\b[^"]*)"[^>]*>(.*?)</div>', re.DOTALL | re.IGNORECASE),
	]

	for pattern in patterns:
		for match in pattern.finditer(html):
			block = match.group(0)
			title_match = re.search(
				r'<(?:h2|h4)[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
				block,
				re.IGNORECASE | re.DOTALL,
			)
			if not title_match:
				title_match = re.search(
					r'<a[^>]*class="[^"]*event-title[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
					block,
					re.IGNORECASE | re.DOTALL,
				)
			if not title_match:
				title_match = re.search(
					r'<a[^>]*href="([^"]+)"[^>]*>\s*.*?<h4[^>]*bme_event_card-title[^>]*>(.*?)</h4>',
					block,
					re.IGNORECASE | re.DOTALL,
				)
			if not title_match:
				continue

			href = title_match.group(1)
			title = _extract_text_from_html(title_match.group(2))
			if not title:
				continue

			item_url = urljoin(base_url, href)
			dedupe_key = (title.lower(), item_url)
			if dedupe_key in seen:
				continue
			seen.add(dedupe_key)

			summary_match = re.search(
				r'(?:news-excerpt|bme_news_card-body|bme_event_card-body|field-name-body|field--name-body).*?<p>(.*?)</p>',
				block,
				re.IGNORECASE | re.DOTALL,
			)
			summary = _extract_text_from_html(summary_match.group(1)) if summary_match else None

			date_match = re.search(
				r'(?:news-date|event-date|bme_event_card-date|field--name-created|created)[^>]*>(.*?)<',
				block,
				re.IGNORECASE | re.DOTALL,
			)
			published_at = _normalize_date(_extract_text_from_html(date_match.group(1)) if date_match else None)

			location_match = re.search(
				r'(?:bme_event_card-location)[^>]*>(.*?)<',
				block,
				re.IGNORECASE | re.DOTALL,
			)
			location = _extract_text_from_html(location_match.group(1)) if location_match else None

			image_match = re.search(r'<img[^>]+src="([^"]+)"', block, re.IGNORECASE)
			image_url = urljoin(base_url, image_match.group(1)) if image_match else None

			candidates.append(
				RawItem(
					title=title,
					item_url=item_url,
					summary=summary,
					published_at=published_at,
					location=location,
					image_url=image_url,
					category=_detect_category(title, summary, item_url),
				)
			)

	return candidates


def _extract_items_from_html(html: str, base_url: str) -> list[RawItem]:
	if BeautifulSoup is not None:
		items = _parse_with_bs4(html, base_url)
		if items:
			return items
	return _parse_without_bs4(html, base_url)


def _validate_result_item(raw_item: RawItem, source_url: str) -> NewsItem:
	return NewsItem.model_validate(
		{
			"source_url": source_url,
			"title": raw_item.title,
			"item_url": raw_item.item_url,
			"category": raw_item.category,
			"published_at": raw_item.published_at,
			"summary": raw_item.summary,
			"location": raw_item.location,
			"image_url": raw_item.image_url,
		}
	)


@mcp.tool()
def extract_news_and_events(urls: list[AnyHttpUrl]) -> list[dict[str, object]]:
	"""Fetch each URL, parse the page HTML, and return normalized news/event items.

	The tool is meant for BME/VIK/TMIT-style pages that list multiple items in the same
	HTML document. For every input URL it:
	1. downloads the page,
	2. finds news or event blocks,
	3. extracts the title, item URL, publication date, summary, and image URL when present,
	4. normalizes the date to ISO-8601 when possible,
	5. deduplicates repeated items across URLs.

	The result is a JSON-serializable list of validated records with keys:
	source_url, title, item_url, category, published_at, summary, location, and image_url.
	"""

	request = NewsExtractionRequest(urls=urls)
	items: list[NewsItem] = []
	seen: set[tuple[str, str]] = set()

	for url in request.urls:
		source_url = str(url)
		try:
			html = _fetch_html(source_url)
		except (HTTPError, URLError, TimeoutError, ValueError) as exc:
			raise ValueError(f"Failed to fetch {source_url}: {exc}") from exc

		for raw_item in _extract_items_from_html(html, source_url):
			dedupe_key = (raw_item.title.lower(), raw_item.item_url)
			if dedupe_key in seen:
				continue
			seen.add(dedupe_key)
			items.append(_validate_result_item(raw_item, source_url))

	return [item.model_dump(mode="json") for item in items]


@mcp.tool()
def normalize_source_urls(urls: list[AnyHttpUrl]) -> list[str]:
	"""Validate and normalize URLs without fetching them."""

	request = NewsExtractionRequest(urls=urls)
	return [str(url) for url in request.urls]


if __name__ == "__main__":
	mcp.run(transport="stdio")
