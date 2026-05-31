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
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError


mcp = FastMCP("news-event-extractor")
logger = logging.getLogger(__name__)


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

EVENT_GUEST_KEYWORDS = (
	"résztvevők",
	"résztvevő",
	"előadók",
	"előadó",
	"meghívott",
	"vendég",
	"guest",
	"guests",
	"speaker",
	"speakers",
)

REGISTRATION_URL_KEYWORDS = (
	"registration",
	"regisztráció",
	"regisztracio",
	"jelentkezés",
	"jelentkezes",
	"signup",
	"sign-up",
	"register",
)

UNKNOWN_GUESTS_LIST = "Nem Ismert"
UNKNOWN_REGISTRATION_URL = "Nem található"
NETWORK_ERROR_MESSAGE = "Letöltés sikertelen hálózati hiba miatt!"
SCHEMA_ERROR_MESSAGE = "Az adott kinyert szöveg nem felel meg a Pydantic sémának!"
UNHANDLED_ERROR_MESSAGE = "Kezeletlen hiba keletkezett!"

NEWS_ITEM_SELECTORS = (
	"div.news-item",
	"article.node-hir",
	"div.bme_news_card",
	"a.h-p100.d-block",
	"div.event",
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

NEWS_ITEM_SUMMARY_SELECTORS = (
	".news-excerpt",
	".news-content p",
	".bme_news_card-body p",
	".bme_event_card-body p",
	".field-name-body p",
	".field--name-body p",
)

NEWS_ITEM_DATE_SELECTORS = (
	".news-date",
	".event-date",
	".bme_event_card-date",
	".field--name-created",
	".created",
	"time",
	"datetime",
)

NEWS_ITEM_LOCATION_SELECTORS = (
	".bme_event_card-location",
	".bme_news_card-location",
	".news-location",
	".location",
	"address",
	".field--name-field-location",
)

ENTRY_TEXT_TITLE_SELECTORS = (
	"h1.page-title",
	"article header [property='dc:title']",
	"article h1",
	".page-title",
	"meta[property='og:title']",
)

ENTRY_TEXT_CONTAINER_SELECTORS = (
	"article.node-hir .field--name-body",
	"article.node-hir .field-name-body",
	"article.node-hir .field-items",
	"div.field--name-field-paragraphs",
	"div.main-page-container .page-content",
	"div.page-content",
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


class BaseItemResponse(BaseModel):
	model_config = ConfigDict(str_strip_whitespace=True)

	source_url: AnyHttpUrl
	title: str = Field(min_length=1)
	item_url: AnyHttpUrl
	category: Literal["news", "event"]
	published_at: Optional[str] = None
	summary: Optional[str] = None
	image_url: Optional[AnyHttpUrl] = None


class NewsItemResponse(BaseItemResponse):
	category: Literal["news"] = "news"


class EventItemResponse(BaseItemResponse):
	category: Literal["event"] = "event"
	location: Optional[str] = None
	guests_list: str = Field(default=UNKNOWN_GUESTS_LIST, min_length=1)
	registration_url: str = Field(default=UNKNOWN_REGISTRATION_URL, min_length=1)


class EntryTextResponse(BaseModel):
	model_config = ConfigDict(str_strip_whitespace=True)

	source_url: AnyHttpUrl
	title: Optional[str] = None
	text: str = Field(min_length=1)


@dataclass(frozen=True)
class RawItem:
	title: str
	item_url: str
	summary: Optional[str] = None
	published_at: Optional[str] = None
	location: Optional[str] = None
	guests_list: Optional[str] = None
	registration_url: Optional[str] = None
	image_url: Optional[str] = None
	category: Literal["news", "event"] = "news"


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


def _detect_category(title: str, summary: Optional[str], url: str) -> Literal["news", "event"]:
	text = f"{title} {summary or ''} {url}".lower()
	if any(token in text for token in ("esemeny", "esemenyek", "esemény", "események")):
		return "event"
	if any(keyword in text for keyword in EVENT_KEYWORDS):
		return "event"
	return "news"


def _detect_category_from_html(node, item_url: str) -> Literal["news", "event"]:
	haystack = f"{item_url} {' '.join(node.get('class', [])) if node.get('class') else ''}".lower()
	text = _clean_text(node.get_text(" ", strip=True)) or ""
	text = text.lower()

	event_markers = (
		"div.event",
		".event",
		".event-title",
		".event-date",
		".bme_event_card",
		".bme_event_card-title",
		".bme_event_card-location",
	)
	news_markers = (
		"div.news-item",
		"article.node-hir",
		".news-date",
		".news-excerpt",
		".bme_news_card",
		".bme_news_card-title",
	)
	event_score = 0
	news_score = 0

	if any(token in haystack for token in ("/esemeny/", "/esemény/", "/event/")):
		event_score += 2
	if any(token in haystack for token in ("/hir/", "/hír/", "/news/")):
		news_score += 2
	if node.get("class") and any(cls in {"event", "bme_event_card"} for cls in node.get("class", [])):
		event_score += 3
	if node.get("class") and any(cls in {"news-item", "bme_news_card", "node-hir"} for cls in node.get("class", [])):
		news_score += 3
	if any(node.select_one(selector) is not None for selector in event_markers):
		event_score += 1
	if any(node.select_one(selector) is not None for selector in news_markers):
		news_score += 1
	if any(keyword in text for keyword in EVENT_GUEST_KEYWORDS) or any(keyword in text for keyword in REGISTRATION_URL_KEYWORDS):
		event_score += 1
	if any(token in text for token in ("hír", "hírek", "news", "hirek")):
		news_score += 1
	if event_score >= news_score:
		return "event"
	return "news"


def _normalize_guests_list(text: Optional[str]) -> Optional[str]:
	cleaned = _clean_text(text)
	if not cleaned:
		return None
	if not re.search(r"[:\-–]", cleaned) and not re.search(r"\b(?:és|and)\b|[,;/]", cleaned, flags=re.IGNORECASE):
		return None
	cleaned = re.sub(
		r"^(?:résztvevők|résztvevő|előadók|előadó|meghívott|vendég|guest(?:s)?|speaker(?:s)?)\s*[:\-–]\s*",
		"",
		cleaned,
		flags=re.IGNORECASE,
	)
	parts = [part.strip(" .;:") for part in re.split(r"[,;/]|\s+és\s+|\s+and\s+", cleaned, flags=re.IGNORECASE)]
	names = [part for part in parts if part]
	return ", ".join(names) if names else cleaned


def _extract_event_guests_list(node) -> Optional[str]:
	for selector in ("p", "li", "div", "span", "strong", "b"):
		for element in node.select(selector):
			text = _clean_text(element.get_text(" ", strip=True))
			if not text:
				continue
			lower_text = text.lower()
			if not any(keyword in lower_text for keyword in EVENT_GUEST_KEYWORDS):
				continue
			guests = _normalize_guests_list(text)
			if guests:
				return guests

	raw_text = _clean_text(node.get_text("\n", strip=True))
	if not raw_text:
		return None

	for line in (segment.strip() for segment in raw_text.splitlines()):
		if not line:
			continue
		lower_line = line.lower()
		if not any(keyword in lower_line for keyword in EVENT_GUEST_KEYWORDS):
			continue
		guests = _normalize_guests_list(line)
		if guests:
			return guests

	return None


def _extract_registration_url(node, base_url: str) -> Optional[str]:
	for link in node.select("a[href]"):
		href = link.get("href")
		if not href:
			continue
		link_text = _clean_text(link.get_text(" ", strip=True)) or ""
		haystack = f"{href} {link_text}".lower()
		if any(keyword in haystack for keyword in REGISTRATION_URL_KEYWORDS):
			return urljoin(base_url, href)

	text = _clean_text(node.get_text(" ", strip=True)) or ""
	if not text:
		return None
	for match in re.finditer(r"https?://\S+", text):
		candidate = match.group(0).rstrip(").,;:")
		if any(keyword in candidate.lower() for keyword in REGISTRATION_URL_KEYWORDS):
			return candidate

	return None


def _parse_with_bs4(html: str, base_url: str) -> list[RawItem]:
	soup = BeautifulSoup(html, "html.parser")

	candidates: list[RawItem] = []
	seen: set[tuple[str, str]] = set()

	for selector in NEWS_ITEM_SELECTORS:
		for node in soup.select(selector):
			title_node = node.select_one(
				", ".join(NEWS_ITEM_TITLE_SELECTORS)
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
				", ".join(NEWS_ITEM_SUMMARY_SELECTORS)
			)
			summary = _clean_text(summary_node.get_text(" ", strip=True)) if summary_node else None

			date_node = node.select_one(
				", ".join(NEWS_ITEM_DATE_SELECTORS)
			)
			published_at = _normalize_date(date_node.get_text(" ", strip=True) if date_node else None)

			location_node = node.select_one(", ".join(NEWS_ITEM_LOCATION_SELECTORS))
			location = _clean_text(location_node.get_text(" ", strip=True)) if location_node else None
			category = _detect_category_from_html(node, item_url)
			guests_list = None
			registration_url = None
			if category == "event":
				guests_list = _extract_event_guests_list(node)
				registration_url = _extract_registration_url(node, base_url)
				# Fallback: if not found in the list card, try the item's detail page
				if not guests_list or not registration_url:
					try:
						detail_html = _fetch_html(item_url)
						detail_soup = BeautifulSoup(detail_html, "html.parser")
						if not guests_list:
							guests_list = _extract_event_guests_list(detail_soup)
						if not registration_url:
							registration_url = _extract_registration_url(detail_soup, item_url)
					except Exception:
						logger.debug("Failed to fetch detail page for %s", item_url)
				guests_list = guests_list or UNKNOWN_GUESTS_LIST
				registration_url = registration_url or UNKNOWN_REGISTRATION_URL

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
					guests_list=guests_list,
					registration_url=registration_url,
					image_url=image_url,
					category=category,
				)
			)

	if not candidates:
		logger.warning("No news or event items found for %s", base_url)

	return candidates


def _first_text_by_selectors(soup: BeautifulSoup, selectors: tuple[str, ...]) -> Optional[str]:
	for selector in selectors:
		node = soup.select_one(selector)
		if node is None:
			continue
		content = node.get("content")
		if content:
			text = _clean_text(content)
			if text:
				return text
		text = _clean_text(node.get_text(" ", strip=True))
		if text:
			return text
	return None


def _extract_entry_text_block(soup: BeautifulSoup) -> tuple[Optional[str], str]:
	title = _first_text_by_selectors(
		soup,
		ENTRY_TEXT_TITLE_SELECTORS,
	)
	container = None
	for selector in ENTRY_TEXT_CONTAINER_SELECTORS:
		container = soup.select_one(selector)
		if container is not None:
			break
	if container is None:
		container = soup.body or soup

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

	return title, "\n".join(lines).strip()


@mcp.tool()
def extract_entry_text(urls: list[AnyHttpUrl]) -> list[dict[str, object]]:
	"""Fetch multiple valid URLs and return each page's main entry text.

	For each input URL (BME-S, BME-TMIT, BME-VIK style articles) the tool:
	- downloads the page,
	- extracts the title and the main body text (paragraphs and lists),
	- validates the result with Pydantic, and
	- returns a JSON-serializable list of objects with `source_url`, `title`, and `text`.
	"""

	results: list[dict[str, object]] = []

	for url in urls:
		source = str(url)
		try:
			html = _fetch_html(source)
		except (HTTPError, URLError, TimeoutError):
			results.append({"source_url": source, "error": NETWORK_ERROR_MESSAGE})
			continue
		except Exception as exc:
			results.append({"source_url": source, "error": f"{UNHANDLED_ERROR_MESSAGE} {exc}"})
			continue

		try:
			soup = BeautifulSoup(html, "html.parser")
			title, text = _extract_entry_text_block(soup)
			validated = EntryTextResponse.model_validate(
				{
					"source_url": source,
					"title": title,
					"text": text,
				}
			)
		except ValidationError:
			results.append({"source_url": source, "error": SCHEMA_ERROR_MESSAGE})
			continue
		except Exception as exc:
			results.append({"source_url": source, "error": f"{UNHANDLED_ERROR_MESSAGE} {exc}"})
			continue

		results.append(validated.model_dump(mode="json"))

	return results

@mcp.tool()
def extract_news_and_events(urls: list[AnyHttpUrl]) -> list[dict[str, object]]:
	"""Fetch each URL, parse the page HTML, and return normalized news/event items.

	Use this tool for index/list pages that contain multiple BME/VIK/TMIT news or
	event cards in the same HTML document. For every input URL it:
	1. downloads the page,
	2. finds news or event blocks,
	3. extracts the title, item URL, publication date, summary, location, and image URL when present,
	4. normalizes the date to ISO-8601 when possible,
	5. deduplicates repeated items across URLs.

	The result is a JSON-serializable list of validated records using two schemas:
	- news: source_url, title, item_url, category, published_at, summary, and image_url
	- event: source_url, title, item_url, category, published_at, summary, location, guests_list,
	  registration_url, and image_url
	"""

	items: list[BaseItemResponse] = []
	seen: set[tuple[str, str]] = set()

	for url in urls:
		source_url = str(url)
		html = _fetch_html_or_error(source_url)

		for raw_item in _parse_with_bs4(html, source_url):
			dedupe_key = (raw_item.title.lower(), raw_item.item_url)
			if dedupe_key in seen:
				continue
			seen.add(dedupe_key)
			base_payload = {
				"source_url": source_url,
				"title": raw_item.title,
				"item_url": raw_item.item_url,
				"published_at": raw_item.published_at,
				"summary": raw_item.summary,
				"image_url": raw_item.image_url,
			}
			if raw_item.category == "event":
				items.append(
					EventItemResponse.model_validate(
						{
							**base_payload,
							"category": "event",
							"location": raw_item.location,
							"guests_list": raw_item.guests_list or UNKNOWN_GUESTS_LIST,
							"registration_url": raw_item.registration_url or UNKNOWN_REGISTRATION_URL,
						}
					)
				)
			elif raw_item.category == "news":
				items.append(
					NewsItemResponse.model_validate(
						{
							**base_payload,
							"category": "news",
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
