from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import unescape
import logging
from typing import Literal, Optional
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import re

from fastmcp import FastMCP
from bs4 import BeautifulSoup
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

mcp = FastMCP("news-event-extractor")
logger = logging.getLogger(__name__)

NEWS_ITEM_SELECTORS = (
	"div.news-item",
	"article.node-hir",
	"div.bme_news_card",
	"a.h-p100.d-block",
	"div.field--name-field-paragraphs",
	"div.field--name-body",
	"div.field-name-body",
	"div.page-content",
)

EVENT_ITEM_SELECTORS = (
	"div.event",
	"article.node-event",
	"div.bme_event_card",
	"a.h-p100.d-block.event",
	"div.field--name-body",
	"div.field-name-body",
	"div.page-content",
)

NEWS_ITEM_TITLE_SELECTORS = (
	"h2 a",
	"h4 a",
	".node__title a",
	".event-title",
	".bme_event_card-title",
	"h2",
	"h4",
	".bme_news_card-title",
)

EVENT_ITEM_TITLE_SELECTORS = (
	"h2 a",
	"h4 a",
	".node__title a",
	".event-title",
	".bme_event_card-title",
	"h2",
	"h4",
	".bme_news_card-title",
)

NEWS_ITEM_SUMMARY_SELECTORS = (
	".news-excerpt",
	".news-content p",
	".bme_news_card-body p",
	".field-name-body p",
	".field--name-body p",
)

NEWS_ITEM_DATE_SELECTORS = (
	".news-date",
	".field--name-created",
	".created",
	"time",
	"datetime",
)

EVENT_ITEM_DATE_SELECTORS = (
	".event-date",
	".bme_event_card-date",
	".field--name-created",
	".created",
	"time",
	"datetime",
)

NEWS_ITEM_LOCATION_SELECTORS = (
	".news-location",
	".location",
	"address",
	".field--name-field-location",
)

EVENT_ITEM_LOCATION_SELECTORS = (
	".bme_event_card-location",
	".location",
	"address",
	".field--name-field-location",
)

ENTRY_TEXT_TITLE_SELECTORS = (
	"h1.page-title",
	"article header [property='dc:title']",
	"article h1",
	"h1",
	".page-title",
	"meta[property='og:title']",
)

ENTRY_TEXT_CONTAINER_SELECTORS = (
	"div.field--name-field-paragraphs",
	"div.field--name-field-paragraphs .field--name-field-formatted-text",
	"div.field--name-field-paragraphs .field__item",
	"div.field--name-field-formatted-text",
	"div.field-item[property='content:encoded']",
	"article.node-event .field--name-body",
	"article.node-event .field-name-body",
	"article.node-event .field-items",
	"article.node-hir .field--name-body",
	"article.node-hir .field-name-body",
	"article.node-hir .field-items",
	"div.field--name-field-paragraphs",
	"div.main-page-container .page-content",
	"div.page-content",
	"div.field-name-body",
	"article.node-event",
	"article.node-hir",
	".field-name-body",
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

class NewsEventLinkResponse(BaseModel):
	model_config = ConfigDict(str_strip_whitespace=True)

	item_url: AnyHttpUrl
	category: Literal["news", "event"]


class BaseItemResponse(BaseModel):
	model_config = ConfigDict(str_strip_whitespace=True)

	source_url: AnyHttpUrl
	title: str = Field(min_length=1)
	item_url: AnyHttpUrl
	published_at: Optional[str] = None
	image_url: Optional[AnyHttpUrl] = None


class NewsItemResponse(BaseItemResponse):
	category: Literal["news"] = "news"
	summary: Optional[str] = None


class EventItemResponse(BaseItemResponse):
	category: Literal["event"] = "event"
	text: str = Field(min_length=1)
	location: Optional[str] = None
	guests_list: str = Field(min_length=1)
	registration_url: str = Field(min_length=1)

@dataclass(frozen=True)
class RawItem:
	title: str
	item_url: str
	category: Literal["news", "event"] = "news"
	source_url: Optional[str] = None
	published_at: Optional[str] = None
	summary: Optional[str] = None
	text: Optional[str] = None
	location: Optional[str] = None
	guests_list: Optional[str] = None
	registration_url: Optional[str] = None
	image_url: Optional[str] = None


def _fetch_html(url: str) -> str:
	request = Request(url, headers={"User-Agent": "Mozilla/5.0 (FastMCP News Extractor)"})
	with urlopen(request, timeout=20) as response:
		body = response.read()
		charset = response.headers.get_content_charset() or "utf-8"
		return body.decode(charset, errors="replace")

def _fetch_html_or_error(url: str) -> str:
	try:
		return _fetch_html(url)
	except (HTTPError, URLError, TimeoutError, ValueError) as exc:
		raise ValueError(f"Failed to fetch {url}: {exc}") from exc

def _clean_text(value: Optional[str]) -> Optional[str]:
	if value is None:
		return None
	text = re.sub(r"\s+", " ", unescape(value)).strip()
	return text or None


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


def _first_text_by_selectors(node, selectors: tuple[str, ...]) -> Optional[str]:
	for selector in selectors:
		match = node.select_one(selector)
		if match is None:
			continue
		content = match.get("content")
		if content:
			text = _clean_text(content)
			if text:
				return text
		text = _clean_text(match.get_text(" ", strip=True))
		if text:
			return text
	return None


def _extract_entry_text_block(node) -> Optional[str]:
	container = None
	for selector in ENTRY_TEXT_CONTAINER_SELECTORS:
		container = node.select_one(selector)
		if container is not None:
			break
	if container is None:
		container = node

	lines: list[str] = []
	for element in container.select("p, li"):
		text = _clean_text(element.get_text(" ", strip=True))
		if not text:
			continue
		if element.name == "li":
			text = f"- {text}"
		lines.append(text)

	if not lines:
		raw_text = _clean_text(container.get_text("\n", strip=True))
		if raw_text:
			lines = [line for line in (segment.strip() for segment in raw_text.splitlines()) if line]

	joined = "\n".join(lines).strip()
	return joined or None


def _extract_item_title(node) -> Optional[str]:
	title = _first_text_by_selectors(node, NEWS_ITEM_TITLE_SELECTORS)
	if title:
		return title
	title = _first_text_by_selectors(node, EVENT_ITEM_TITLE_SELECTORS)
	if title:
		return title
	return _first_text_by_selectors(node, ENTRY_TEXT_TITLE_SELECTORS)


def _extract_registration_url(node, base_url: str) -> Optional[str]:
	for link in node.select("a[href]"):
		href = link.get("href")
		if not href:
			continue
		link_text = _clean_text(link.get_text(" ", strip=True)) or ""
		haystack = f"{href} {link_text}".lower()
		if any(keyword in haystack for keyword in ("registration", "regisztráció", "regisztracio", "jelentkezés", "jelentkezes", "signup", "sign-up", "register")):
			return urljoin(base_url, href)

	text = _clean_text(node.get_text(" ", strip=True)) or ""
	if not text:
		return None
	for match in re.finditer(r"https?://\S+", text):
		candidate = match.group(0).rstrip(").,;:")
		if any(keyword in candidate.lower() for keyword in ("registration", "regisztráció", "regisztracio", "jelentkezés", "jelentkezes", "signup", "sign-up", "register")):
			return candidate

	return None


def _extract_event_guests_list(node) -> Optional[str]:
	for selector in ("p", "li", "div", "span", "strong", "b"):
		for element in node.select(selector):
			text = _clean_text(element.get_text(" ", strip=True))
			if not text:
				continue
			lower_text = text.lower()
			if not any(keyword in lower_text for keyword in ("résztvevők", "résztvevő", "előadók", "előadó", "meghívott", "vendég", "guest", "guests", "speaker", "speakers")):
				continue
			cleaned = re.sub(
				r"^(?:résztvevők|résztvevő|előadók|előadó|meghívott|vendég|guest(?:s)?|speaker(?:s)?)\s*[:\-–]\s*",
				"",
				text,
				flags=re.IGNORECASE,
			)
			return cleaned.strip() or None

	raw_text = _clean_text(node.get_text("\n", strip=True))
	if not raw_text:
		return None

	for line in (segment.strip() for segment in raw_text.splitlines()):
		if not line:
			continue
		lower_line = line.lower()
		if not any(keyword in lower_line for keyword in ("résztvevők", "résztvevő", "előadók", "előadó", "meghívott", "vendég", "guest", "guests", "speaker", "speakers")):
			continue
		cleaned = re.sub(
			r"^(?:résztvevők|résztvevő|előadók|előadó|meghívott|vendég|guest(?:s)?|speaker(?:s)?)\s*[:\-–]\s*",
			"",
			line,
			flags=re.IGNORECASE,
		)
		if cleaned.strip():
			return cleaned.strip()

	return None


def _extract_image_url(node, base_url: str) -> Optional[str]:
	image_node = node.select_one("img")
	if image_node is None:
		return None
	image_src = image_node.get("src") or image_node.get("data-src")
	if not image_src:
		return None
	return urljoin(base_url, image_src)


def _detect_category_from_html(node) -> Literal["news", "event"]:
	event_selectors = (
		"div.event",
		".event",
		".event-title",
		".event-date",
		".bme_event_card",
		".bme_event_card-title",
		".bme_event_card-location",
	)
	if any(node.select_one(selector) is not None for selector in event_selectors):
		return "event"
	classes = node.get("class") or []
	if any("event" in cls for cls in classes):
		return "event"
	return "news"


def _extract_detail_page_item(
	soup: BeautifulSoup,
	base_url: str,
	category: Literal["news", "event"],
) -> Optional[RawItem]:
	title = _extract_item_title(soup)
	if not title:
		return None

	published_at = None
	location = None
	image_url = _extract_image_url(soup, base_url)

	if category == "news":
		summary = _extract_entry_text_block(soup)
		date_node = soup.select_one(
			", ".join(NEWS_ITEM_DATE_SELECTORS)
		)
		published_at = _normalize_date(date_node.get_text(" ", strip=True) if date_node else None)
		return RawItem(
			title=title,
			item_url=base_url,
			category="news",
			source_url=base_url,
			published_at=published_at,
			summary=summary,
			image_url=image_url,
		)

	date_node = soup.select_one(
		", ".join(EVENT_ITEM_DATE_SELECTORS)
	)
	published_at = _normalize_date(date_node.get_text(" ", strip=True) if date_node else None)
	location_node = soup.select_one(
		", ".join(EVENT_ITEM_LOCATION_SELECTORS)
	)
	location = _clean_text(location_node.get_text(" ", strip=True)) if location_node else None
	text = _extract_entry_text_block(soup)
	guests_list = _extract_event_guests_list(soup) or "Nem Ismert"
	registration_url = _extract_registration_url(soup, base_url) or "Nem található"

	return RawItem(
		title=title,
		item_url=base_url,
		category="event",
		source_url=base_url,
		published_at=published_at,
		text=text,
		location=location,
		guests_list=guests_list,
		registration_url=registration_url,
		image_url=image_url,
	)


def _parse_with_bs4(
	html: str,
	base_url: str,
	category: Optional[Literal["news", "event"]] = None,
) -> list[RawItem]:
	soup = BeautifulSoup(html, "html.parser")

	candidates: list[RawItem] = []
	seen: set[tuple[str, str]] = set()
	selector_groups = NEWS_ITEM_SELECTORS + EVENT_ITEM_SELECTORS

	for selector in selector_groups:
		for node in soup.select(selector):
			node_category = _detect_category_from_html(node)
			if category is not None and node_category != category:
				continue

			title_selectors = NEWS_ITEM_TITLE_SELECTORS if node_category == "news" else EVENT_ITEM_TITLE_SELECTORS
			title_node = node.select_one(
				", ".join(title_selectors)
			)
			title = _clean_text(title_node.get_text(" ", strip=True)) if title_node else None
			if not title:
				title = _extract_item_title(node)
			if not title:
				continue

			href = None
			if title_node is not None and title_node.name == "a":
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
			if not href:
				continue

			item_url = urljoin(base_url, href)
			dedupe_key = (title.lower(), item_url)
			if dedupe_key in seen:
				continue
			seen.add(dedupe_key)

			source_url = base_url
			published_at = None
			summary = None
			text = None
			location = None
			guests_list = None
			registration_url = None
			image_url = _extract_image_url(node, base_url)

			if node_category == "news":
				summary_node = node.select_one(
					", ".join(NEWS_ITEM_SUMMARY_SELECTORS)
				)
				summary = _clean_text(summary_node.get_text(" ", strip=True)) if summary_node else None
				if not summary:
					summary = _extract_entry_text_block(node)

				date_node = node.select_one(
					", ".join(NEWS_ITEM_DATE_SELECTORS)
				)
				published_at = _normalize_date(date_node.get_text(" ", strip=True) if date_node else None)

				if category in (None, "news"):
					candidates.append(
						RawItem(
							title=title,
							item_url=item_url,
							category="news",
							source_url=source_url,
							published_at=published_at,
							summary=summary,
							image_url=image_url,
						)
					)
			elif node_category == "event":
				date_node = node.select_one(
					", ".join(EVENT_ITEM_DATE_SELECTORS)
				)
				published_at = _normalize_date(date_node.get_text(" ", strip=True) if date_node else None)
				location_node = node.select_one(
					", ".join(EVENT_ITEM_LOCATION_SELECTORS)
				)
				location = _clean_text(location_node.get_text(" ", strip=True)) if location_node else None
				text = _extract_entry_text_block(node)
				guests_list = _extract_event_guests_list(node)
				registration_url = _extract_registration_url(node, base_url)

				if not text or not guests_list or not registration_url:
					try:
						detail_html = _fetch_html(item_url)
						detail_soup = BeautifulSoup(detail_html, "html.parser")
						if not text:
							text = _extract_entry_text_block(detail_soup)
						if not guests_list:
							guests_list = _extract_event_guests_list(detail_soup)
						if not registration_url:
							registration_url = _extract_registration_url(detail_soup, item_url)
					except Exception:
						logger.debug("Failed to fetch detail page for %s", item_url)

				guests_list = guests_list or "Nem Ismert"
				registration_url = registration_url or "Nem található"

				if category in (None, "event"):
					candidates.append(
						RawItem(
							title=title,
							item_url=item_url,
							category="event",
							source_url=source_url,
							published_at=published_at,
							text=text,
							location=location,
							guests_list=guests_list,
							registration_url=registration_url,
							image_url=image_url,
						)
					)

	if not candidates:
		if category is not None:
			detail_item = _extract_detail_page_item(soup, base_url, category)
			if detail_item is not None:
				candidates.append(detail_item)

	if not candidates:
		logger.warning("No news or event items found for %s", base_url)

	return candidates

@mcp.tool()
def extract_news_and_events(urls: list[AnyHttpUrl]) -> list[dict[str, object]]:
	"""Extract item links from list pages that contain multiple news or event cards.

	Use this tool for index/list pages only. For each input URL it:
	1. downloads the HTML,
	2. finds candidate news or event cards using CSS selectors and HTML structure,
	3. extracts the canonical item URL and detected category,
	4. removes duplicates across all input URLs.

	Return value:
	- JSON-serializable list of link records
	- schema: {"item_url": string, "category": "news" | "event"}
	- no title, body text, date, location or image fields
	"""

	items: list[NewsEventLinkResponse] = []
	seen: set[tuple[str, str]] = set()

	for url in urls:
		source_url = str(url)
		html = _fetch_html_or_error(source_url)

		for raw_item in _parse_with_bs4(html, source_url):
			dedupe_key = (raw_item.title.lower(), raw_item.item_url)
			if dedupe_key in seen:
				continue
			seen.add(dedupe_key)
			items.append(
				NewsEventLinkResponse.model_validate(
					{
						"item_url": raw_item.item_url,
						"category": raw_item.category,
					}
				)
			)

	return [item.model_dump(mode="json") for item in items]


@mcp.tool()
def extract_normalized_news_and_events(
	category: Literal["news", "event"],
	urls: list[AnyHttpUrl],
) -> list[dict[str, object]]:
	"""Extract normalized news or event records from page HTML.

	Use this tool when you need a typed record set instead of only links.
	The `category` argument is a hard filter: only records of that type are returned.
	For each input URL the tool:
	1. downloads the page,
	2. finds matching cards using CSS selectors and HTML structure,
	3. extracts and normalizes title, item URL, published date, text fields and image URL,
	4. falls back to the detail page when an event card does not contain enough text data,
	5. removes duplicates across all input URLs.

	Return value:
	- JSON-serializable list of validated records
	- news schema: source_url, title, item_url, category, published_at, summary, image_url
	- event schema: source_url, title, item_url, category, published_at, text,
	  location, guests_list, registration_url, image_url
	"""

	items: list[BaseItemResponse] = []
	seen: set[tuple[str, str]] = set()

	for url in urls:
		source_url = str(url)
		html = _fetch_html_or_error(source_url)

		for raw_item in _parse_with_bs4(html, source_url, category):
			dedupe_key = (raw_item.title.lower(), raw_item.item_url)
			if dedupe_key in seen:
				continue
			seen.add(dedupe_key)

			if category == "news":
				items.append(
					NewsItemResponse.model_validate(
						{
							"source_url": raw_item.source_url or source_url,
							"title": raw_item.title,
							"item_url": raw_item.item_url,
							"category": "news",
							"published_at": raw_item.published_at,
							"summary": raw_item.summary,
							"image_url": raw_item.image_url,
						}
					)
				)
			else:
				text = raw_item.text or raw_item.summary or raw_item.title
				items.append(
					EventItemResponse.model_validate(
						{
							"source_url": raw_item.source_url or source_url,
							"title": raw_item.title,
							"item_url": raw_item.item_url,
							"category": "event",
							"published_at": raw_item.published_at,
							"text": text,
							"location": raw_item.location,
							"guests_list": raw_item.guests_list or "Nem Ismert",
							"registration_url": raw_item.registration_url or "Nem található",
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
