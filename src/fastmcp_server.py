from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MONTH_ALIASES = {
	"1": 1,
	"01": 1,
	"jan": 1,
	"2": 2,
	"02": 2,
	"feb": 2,
	"febr": 2,
	"3": 3,
	"03": 3,
	"mar": 3,
	"4": 4,
	"04": 4,
	"apr": 4,
	"5": 5,
	"05": 5,
	"may": 5,
	"6": 6,
	"06": 6,
	"jun": 6,
	"7": 7,
	"07": 7,
	"jul": 7,
	"8": 8,
	"08": 8,
	"aug": 8,
	"9": 9,
	"09": 9,
	"sep": 9,
	"sept": 9,
	"10": 10,
	"oct": 10,
	"okt": 10,
	"11": 11,
	"nov": 11,
	"12": 12,
	"dec": 12,
	"dez": 12,
}

CANONICAL_MONTHS = [
	"Jan",
	"Feb",
	"Mar",
	"Apr",
	"May",
	"Jun",
	"Jul",
	"Aug",
	"Sep",
	"Oct",
	"Nov",
	"Dec",
]


class RawHTML(BaseModel):
	pages: dict[str, str] = Field(default_factory=dict)


class News(BaseModel):
	title: str
	date: str
	content: str
	url: str


class Event(BaseModel):
	event_title: str
	event_date: str
	event_location: str | None = None
	event_description: str = Field(min_length=1)
	event_guests_list: list[str] = Field(default_factory=list)
	event_registration_url: str = Field(min_length=1)
	event_url: str


mcp = FastMCP("BME News")


def _strip_accents(text: str) -> str:
	normalized = unicodedata.normalize("NFKD", text)
	return "".join(character for character in normalized if not unicodedata.combining(character))


def _clean_text(text: str | None) -> str:
	if not text:
		return ""
	return re.sub(r"\s+", " ", text).strip()


def _normalize_url(raw_url: str) -> str:
	"""Validate and slightly canonicalize a URL.

	Returns a normalized URL string using lowercased scheme/netloc and
	a trimmed path (no trailing slash except root). Raises ValueError
	for unsupported schemes or missing netloc.
	"""
	parsed = urlparse(raw_url)
	if parsed.scheme not in {"http", "https"} or not parsed.netloc:
		raise ValueError(f"Unsupported URL: {raw_url}")

	scheme = parsed.scheme.lower()
	netloc = parsed.netloc.lower()
	# trim trailing slash from path for normalization (but keep single /)
	path = parsed.path
	if path and path != "/":
		path = path.rstrip("/")

	normalized = parsed._replace(scheme=scheme, netloc=netloc, path=path).geturl()
	return normalized


def _canonical_month_name(month_number: int) -> str:
	if not 1 <= month_number <= 12:
		raise ValueError(f"Invalid month number: {month_number}")
	return CANONICAL_MONTHS[month_number - 1]


def _month_number(token: str) -> int:
	normalized = _strip_accents(token).lower().strip().strip(".")
	if normalized.isdigit():
		month_number = int(normalized)
		if 1 <= month_number <= 12:
			return month_number
	if normalized in MONTH_ALIASES:
		return MONTH_ALIASES[normalized]
	for alias, month_number in MONTH_ALIASES.items():
		if alias.isalpha() and normalized.startswith(alias):
			return month_number
	raise ValueError(f"Unsupported month value: {token}")


def _canonical_date(year: int, month: int, day: int) -> str:
	return f"{year:04d}.{_canonical_month_name(month)}.{day:02d}"


def _normalize_date_string(date_text: str) -> str:
	text = _clean_text(date_text)
	if not text:
		raise ValueError("Date string is empty")

	patterns = (
		re.compile(
			r"(?P<year>20\d{2})[./\-\s]+(?P<month>[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+|\d{1,2})[./\-\s]+(?P<day>\d{1,2})"
		),
		re.compile(
			r"(?P<day>\d{1,2})[./\-\s]+(?P<month>[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+|\d{1,2})[./\-\s]+(?P<year>20\d{2})"
		),
		re.compile(
			r"(?P<month>[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+|\d{1,2})\s+(?P<day>\d{1,2}),?\s+(?P<year>20\d{2})"
		),
	)

	for pattern in patterns:
		match = pattern.search(text)
		if not match:
			continue
		year = int(match.group("year"))
		month = _month_number(match.group("month"))
		day = int(match.group("day"))
		return _canonical_date(year, month, day)

	raise ValueError(f"Unsupported date format: {date_text}")


def _extract_meta_content(soup: BeautifulSoup, selectors: Iterable[str]) -> str:
	for selector in selectors:
		element = soup.select_one(selector)
		if element is None:
			continue
		if element.name == "meta":
			content = element.get("content")
			if content:
				return _clean_text(content)
		else:
			text = _clean_text(element.get_text(" ", strip=True))
			if text:
				return text
	return ""


def _extract_visible_text(node) -> str:
	for tag in node.find_all(["script", "style", "noscript"]):
		tag.decompose()
	return _clean_text(node.get_text(" ", strip=True))


def _extract_page_title(soup: BeautifulSoup) -> str:
	return _extract_meta_content(
		soup,
		[
			"meta[property='og:title']",
			"meta[name='title']",
			"meta[name='dc.title']",
			"main h1",
			"article h1",
			"h1",
			"h2",
		],
	)


def _extract_page_date(soup: BeautifulSoup) -> str:
	selectors = [
		"meta[property='article:published_time']",
		"meta[name='article:published_time']",
		"meta[name='pubdate']",
		"meta[name='date']",
		"time[datetime]",
		".submitted",
		".date",
		".field--name-field-date",
		".field-name-field-date",
		"main time",
	]
	for selector in selectors:
		element = soup.select_one(selector)
		if element is None:
			continue
		raw_value = element.get("content") if element.name == "meta" else element.get("datetime") or element.get_text(" ", strip=True)
		raw_value = _clean_text(raw_value)
		if not raw_value:
			continue
		try:
			if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?", raw_value):
				return _canonical_date(int(raw_value[:4]), int(raw_value[5:7]), int(raw_value[8:10]))
			return _normalize_date_string(raw_value)
		except ValueError:
			continue

	for text in (
		_extract_meta_content(soup, ["meta[name='description']"]),
		_extract_meta_content(soup, ["main"]),
		_extract_meta_content(soup, ["article"]),
	):
		try:
			return _normalize_date_string(text)
		except ValueError:
			continue

	return ""





def _looks_like_article_link(href: str, base_url: str) -> bool:
	parsed = urlparse(urljoin(base_url, href))
	path = parsed.path.rstrip("/")
	if not path:
		return False

	if "bme.hu" in parsed.netloc:
		return bool(
			re.search(r"/(hirek|hir)/\d+", path)
			or re.search(r"/node/\d+", path)
			or re.fullmatch(r"/(TopN-\d{4}|TIPP_\d{4})", path)
		)

	return bool(re.search(r"/node/\d+", path))


def _find_summary_text(container, title: str, date_text: str) -> str:
	full_text = _extract_visible_text(container)
	for text in (title, date_text):
		if text:
			full_text = full_text.replace(text, " ")
	return _clean_text(full_text)


def _parse_article_page(html: str, url: str) -> list[News]:
	soup = BeautifulSoup(html, "html.parser")
	title = _extract_page_title(soup)
	if not title:
		return []

	content_text = ""
	for selector in ["article", "main", ".node__content", ".field--name-body", ".content", "#content"]:
		element = soup.select_one(selector)
		if element is None:
			continue
		candidate = _extract_visible_text(element)
		if len(candidate) > len(content_text):
			content_text = candidate

	if not content_text:
		content_text = _extract_visible_text(soup)

	date_text = _extract_page_date(soup)
	if not date_text:
		for source_text in (content_text, title):
			try:
				date_text = _normalize_date_string(source_text)
				break
			except ValueError:
				continue

	return [
		News(
			title=title,
			date=date_text,
			content=content_text,
			url=url,
		)
	]


def _parse_listing_page(html: str, url: str) -> list[News]:
	soup = BeautifulSoup(html, "html.parser")
	articles: list[News] = []
	seen_urls: set[str] = set()

	for anchor in soup.find_all("a", href=True):
		href = anchor.get("href", "")
		absolute_url = urljoin(url, href)
		if absolute_url in seen_urls:
			continue
		if not _looks_like_article_link(href, url):
			continue
		if anchor.find_parent(["nav", "header", "footer", "aside", "form", "script"]):
			continue

		container = anchor.find_parent(["article", "li", "section", "div", "main"]) or anchor.parent
		if container is None:
			continue

		title = _clean_text(anchor.get_text(" ", strip=True))
		if not title or len(title) < 4:
			continue

		container_text = _extract_visible_text(container)
		if len(container_text) < len(title) + 20:
			continue

		date_text = ""
		for selector in ["time", ".date", ".submitted", ".field--name-field-date", ".field-name-field-date"]:
			element = container.select_one(selector)
			if element is None:
				continue
			candidate = element.get("datetime") if element.name == "time" else element.get_text(" ", strip=True)
			candidate = _clean_text(candidate)
			if not candidate:
				continue
			try:
				if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?", candidate):
					date_text = _canonical_date(int(candidate[:4]), int(candidate[5:7]), int(candidate[8:10]))
				else:
					date_text = _normalize_date_string(candidate)
				break
			except ValueError:
				continue

		if not date_text:
			for pattern in (
				re.search(r"\d{4}[./\- ]+[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+[./\- ]+\d{1,2}", container_text),
				re.search(r"\d{4}[./\- ]+\d{1,2}[./\- ]+\d{1,2}", container_text),
				re.search(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+\s+\d{1,2},?\s+20\d{2}", container_text),
			):
				if pattern:
					try:
						date_text = _normalize_date_string(pattern.group(0))
						break
					except ValueError:
						continue

		content_text = _find_summary_text(container, title, date_text) or title

		articles.append(
			News(
				title=title,
				date=date_text,
				content=content_text,
				url=absolute_url,
			)
		)
		seen_urls.add(absolute_url)

	return articles


def _parse_news_html(html: str, url: str) -> list[News]:
	if not html.strip():
		return []

	path = urlparse(url).path
	if re.search(r"/(node|hir)/\d+", path) or re.fullmatch(r"/(TopN-\d{4}|TIPP_\d{4})", path):
		parsed = _parse_article_page(html, url)
		if parsed:
			return parsed

	listing_items = _parse_listing_page(html, url)
	if listing_items:
		return listing_items

	return _parse_article_page(html, url)


@mcp.tool()
def fetch_html(urls: list[str]) -> RawHTML:
	pages: dict[str, str] = {}
	for raw_url in urls:
		try:
			url = _normalize_url(raw_url)
		except ValueError as exc:
			logger.warning("Skipping invalid URL %s: %s", raw_url, exc)
			pages[raw_url] = ""
			continue

		request = Request(
			url,
			headers={
				"User-Agent": (
					"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
					"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
				)
			},
		)
		try:
			with urlopen(request, timeout=20) as response:
				charset = response.headers.get_content_charset() or "utf-8"
				pages[url] = response.read().decode(charset, errors="replace")
		except (HTTPError, URLError, TimeoutError, OSError) as exc:
			logger.warning("Failed to fetch %s: %s", url, exc)
			pages[url] = ""
	return RawHTML(pages=pages)


@mcp.tool()
def parse_news(html: str, url: str) -> list[News]:
	_normalize_url(url)
	return _parse_news_html(html, url)


@mcp.tool()
def filter_items_by_date(items: list[News] | list[Event], date: str) -> list[News | Event]:
	target_date = _normalize_date_string(date)
	filtered: list[News | Event] = []
	for item in items:
		item_date = item.date if isinstance(item, News) else item.event_date
		if item_date == target_date:
			filtered.append(item)
	return filtered


@mcp.tool()
def filter_news_by_date(news: list[News], date: str) -> list[News]:
	"""Deprecated: use filter_items_by_date for unified date filtering.

	Kept for backward compatibility for news-only callers.
	"""
	filtered = filter_items_by_date(news, date)
	return [item for item in filtered if isinstance(item, News)]


@mcp.tool()
def get_today_news(urls: list[str], date: str) -> list[News]:
	fetched = fetch_html(urls)
	parsed_news: list[News] = []
	for source_url, html in fetched.pages.items():
		parsed_news.extend(_parse_news_html(html, source_url))

	filtered = filter_news_by_date(parsed_news, date)
	deduped: list[News] = []
	seen: set[tuple[str, str, str]] = set()
	for item in filtered:
		identity = (item.title, item.date, item.url)
		if identity in seen:
			continue
		seen.add(identity)
		deduped.append(item)
	return deduped


def _parse_events_html(html: str, url: str) -> list[Event]:
	_normalize_url(url)
	soup = BeautifulSoup(html, "html.parser")
	events: list[Event] = []
	seen: set[tuple[str, str]] = set()

	# Candidate containers: elements that often contain event info
	candidates = soup.find_all(["article", "li", "section", "div"])
	for container in candidates:
		classes = " ".join(container.get("class") or [])
		text = _extract_visible_text(container)
		if (
			"event" not in classes.lower()
			and "vevent" not in classes.lower()
			and "program" not in classes.lower()
			and "agenda" not in classes.lower()
			and not container.select_one("time")
			and "event" not in container.get_text(" ", strip=True).lower()
		):
			continue

		# Title
		title = ""
		for sel in ("h1", "h2", "h3", "a", "strong"):
			el = container.select_one(sel)
			if el and _clean_text(el.get_text(" ", strip=True)):
				title = _clean_text(el.get_text(" ", strip=True))
				break
		if not title:
			continue

		# URL
		anchor = container.find("a", href=True)
		absolute_url = urljoin(url, anchor.get("href")) if anchor else url

		# Date extraction
		start_date = ""
		date_text = ""
		time_el = container.select_one("time")
		if time_el:
			raw = time_el.get("datetime") or time_el.get_text(" ", strip=True)
			raw = _clean_text(raw)
			try:
				if raw:
					if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?", raw):
						start_date = _canonical_date(int(raw[:4]), int(raw[5:7]), int(raw[8:10]))
					else:
						start_date = _normalize_date_string(raw)
					date_text = raw
			except ValueError:
				start_date = ""

		if not start_date:
			# try to find date-like text in container
			for pattern in (
				re.search(r"\d{4}[./\- ]+[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+[./\- ]+\d{1,2}", text),
				re.search(r"\d{4}[./\- ]+\d{1,2}[./\- ]+\d{1,2}", text),
				re.search(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+\s+\d{1,2},?\s+20\d{2}", text),
			):
				if pattern:
					try:
						start_date = _normalize_date_string(pattern.group(0))
						date_text = pattern.group(0)
						break
					except ValueError:
						continue

		# Location and description
		location = ""
		for sel in (".location", ".venue", ".place", ".helyszin", "[class*='helysz']"):
			el = container.select_one(sel)
			if el:
				location = _clean_text(el.get_text(" ", strip=True))
				break

		description = _find_summary_text(container, title, date_text)
		if not description:
			continue

		# try to find a registration link inside the container
		reg_url = ""
		for a in container.find_all("a", href=True):
			href = a.get("href") or ""
			text = _clean_text(a.get_text(" ", strip=True)).lower()
			if any(tok in href.lower() for tok in ("register", "regisztr", "jelent", "apply", "signup")) or any(tok in text for tok in ("regisztr", "jelent", "register", "sign up", "apply")):
				reg_url = urljoin(url, href)
				break
		if not reg_url:
			continue

		event = Event(
			event_title=title,
			event_date=start_date or "",
			event_location=location or None,
			event_description=description,
			event_guests_list=[],
			event_registration_url=reg_url,
			event_url=absolute_url,
		)

		identity = (event.event_title, event.event_date)
		if identity in seen:
			continue
		seen.add(identity)
		events.append(event)

	return events


@mcp.tool()
def parse_events(html: str, url: str) -> list[Event]:
	return _parse_events_html(html, url)


@mcp.tool()
def filter_events_by_date(events: list[Event], date: str) -> list[Event]:
	"""Deprecated: use filter_items_by_date for unified date filtering.

	Kept for backward compatibility for event-only callers.
	"""
	filtered = filter_items_by_date(events, date)
	return [e for e in filtered if isinstance(e, Event)]


@mcp.tool()
def get_today_events(urls: list[str], date: str) -> list[Event]:
	fetched = fetch_html(urls)
	parsed: list[Event] = []
	for source_url, html in fetched.pages.items():
		parsed.extend(parse_events(html, source_url))

	filtered_items = filter_items_by_date(parsed, date)
	filtered = [item for item in filtered_items if isinstance(item, Event)]
	deduped: list[Event] = []
	seen: set[tuple[str, str]] = set()
	for item in filtered:
		identity = (item.event_title, item.event_date)
		if identity in seen:
			continue
		seen.add(identity)
		deduped.append(item)
	return deduped


def main() -> None:
	host = os.getenv("FASTMCP_HOST", "0.0.0.0")
	port = int(os.getenv("FASTMCP_PORT", "8000"))
	path = os.getenv("FASTMCP_PATH", "/mcp")
	mcp.settings.host = host
	mcp.settings.port = port
	mcp.settings.streamable_http_path = path
	if mcp.settings.transport_security is None:
		mcp.settings.transport_security = TransportSecuritySettings()
	mcp.settings.transport_security.allowed_hosts = [
		"127.0.0.1:*",
		"localhost:*",
		"[::1]:*",
		"fastmcp-server:*",
	]
	mcp.settings.transport_security.allowed_origins = [
		"http://127.0.0.1:*",
		"http://localhost:*",
		"http://[::1]:*",
		"http://fastmcp-server:*",
	]
	mcp.run(transport="streamable-http")


if __name__ == "__main__":
	main()
