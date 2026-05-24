from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
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
	topics: list[str] = Field(default_factory=list)
	url: str


mcp = FastMCP("BME News")


def _strip_accents(text: str) -> str:
	normalized = unicodedata.normalize("NFKD", text)
	return "".join(character for character in normalized if not unicodedata.combining(character))


def _clean_text(text: str | None) -> str:
	if not text:
		return ""
	return re.sub(r"\s+", " ", text).strip()


def _normalize_url(raw_url: str) -> str:
	parsed = urlparse(raw_url)
	if parsed.scheme not in {"http", "https"} or not parsed.netloc:
		raise ValueError(f"Unsupported URL: {raw_url}")
	return raw_url


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


def _extract_topics(soup: BeautifulSoup, title: str, content: str, url: str) -> list[str]:
	topics: list[str] = []

	meta_keywords = _extract_meta_content(
		soup,
		[
			"meta[name='keywords']",
			"meta[name='news_keywords']",
			"meta[property='article:tag']",
		],
	)
	if meta_keywords:
		for token in re.split(r"[;,/|]", meta_keywords):
			cleaned = _clean_text(token)
			if cleaned and cleaned not in topics:
				topics.append(cleaned)

	breadcrumb_text = " ".join(
		_clean_text(node.get_text(" ", strip=True))
		for node in soup.select("nav.breadcrumb, .breadcrumb, .breadcrumbs, .path")
		if _clean_text(node.get_text(" ", strip=True))
	)
	if breadcrumb_text:
		for token in re.split(r"\s{2,}|[>»/|]", breadcrumb_text):
			cleaned = _clean_text(token)
			if cleaned and cleaned not in topics:
				topics.append(cleaned)

	haystack = f"{title} {content} {url}".lower()
	keyword_map = [
		("AI", ["mesterséges intelligencia", " ai", "llm", "chatgpt", "robot", "nóra", "nora"]),
		("Kiberbiztonság", ["kiber", "cyber", "security", "adatvédelem", "ssl"]),
		("Esemény", ["workshop", "konferencia", "esemény", "event", "rendezv", "előadás", "eloadas"]),
		("Oktatás", ["hallgat", "felvételi", "záróvizsga", "diploma", "ösztöndíj", "kurzus", "education"]),
		("Kutatás", ["kutatás", "projekt", "fejleszt", "innováció", "labor", "research"]),
		("Űrkutatás", ["űr", "urkutatas", "space", "satellite"]),
		("Mobil", ["mobil", "telefon", "iphone", "android", "app"]),
		("Energetika", ["energia", "villamos", "emc", "power", "távközl", "telekom"]),
	]
	for label, keywords in keyword_map:
		if any(keyword in haystack for keyword in keywords) and label not in topics:
			topics.append(label)

	if not topics:
		if "vik.bme.hu" in url:
			topics.append("VIK")
		elif "tmit.bme.hu" in url:
			topics.append("TMIT")
		elif "bme.hu" in url:
			topics.append("BME")

	return topics[:5]


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
			topics=_extract_topics(soup, title, content_text, url),
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
				topics=_extract_topics(soup, title, content_text, absolute_url),
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
		url = _normalize_url(raw_url)
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
def filter_news_by_date(news: list[News], date: str) -> list[News]:
	target_date = _normalize_date_string(date)
	return [item for item in news if item.date == target_date]


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


def main() -> None:
	host = os.getenv("FASTMCP_HOST", "0.0.0.0")
	port = int(os.getenv("FASTMCP_PORT", "8000"))
	path = os.getenv("FASTMCP_PATH", "/mcp")
	mcp.run(transport="sse", host=host, port=port, path=path)


if __name__ == "__main__":
	main()
