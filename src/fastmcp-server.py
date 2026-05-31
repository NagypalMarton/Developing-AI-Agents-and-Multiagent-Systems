from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import unescape
from typing import Optional
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import re

from fastmcp import FastMCP
from bs4 import BeautifulSoup
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


@dataclass(frozen=True)
class RawItem:
	title: str
	item_url: str
	summary: Optional[str] = None
	published_at: Optional[str] = None
	location: Optional[str] = None
	image_url: Optional[str] = None
	category: str = "unknown"


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
	return _clean_text(BeautifulSoup(fragment, "html.parser").get_text(" ", strip=True)) or ""


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
	soup = BeautifulSoup(html, "html.parser")

	candidates: list[RawItem] = []
	seen: set[tuple[str, str]] = set()

	selectors = [
		"div.news-item",
		"article.node-hir",
		"div.bme_news_card",
		"a.h-p100.d-block",
		"div.event",
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
				".news-excerpt, .news-content p, .bme_news_card-body p, .bme_event_card-body p, .field-name-body p, .field--name-body p"
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

	items: list[NewsItem] = []
	seen: set[tuple[str, str]] = set()

	for url in urls:
		source_url = str(url)
		try:
			html = _fetch_html(source_url)
		except (HTTPError, URLError, TimeoutError, ValueError) as exc:
			raise ValueError(f"Failed to fetch {source_url}: {exc}") from exc

		for raw_item in _parse_with_bs4(html, source_url):
			dedupe_key = (raw_item.title.lower(), raw_item.item_url)
			if dedupe_key in seen:
				continue
			seen.add(dedupe_key)
			items.append(
				NewsItem.model_validate(
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
			)

	return [item.model_dump(mode="json") for item in items]
if __name__ == "__main__":
	transport = os.getenv("FASTMCP_TRANSPORT", "stdio")
	if transport in ("http", "streamable-http", "sse"):
		host = os.getenv("FASTMCP_HOST", "0.0.0.0")
		port = int(os.getenv("FASTMCP_PORT", "8000"))
		path = os.getenv("FASTMCP_PATH", "/mcp")
		mcp.run(transport=transport, host=host, port=port, path=path)
	else:
		mcp.run(transport="stdio")
