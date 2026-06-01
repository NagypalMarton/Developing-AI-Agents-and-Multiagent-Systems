from __future__ import annotations

from dataclasses import dataclass
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

class NewsEventLinkResponse(BaseModel):
	model_config = ConfigDict(str_strip_whitespace=True)

	item_url: AnyHttpUrl
	category: Literal["news", "event"]

@dataclass(frozen=True)
class RawItem:
	title: str
	item_url: str
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

			# Determine category solely from CSS selectors/classes
			event_selectors = (
				"div.event",
				".event",
				".event-title",
				".event-date",
				".bme_event_card",
				".bme_event_card-title",
				".bme_event_card-location",
			)
			is_event = any(node.select_one(sel) is not None for sel in event_selectors)
			if not is_event:
				classes = node.get("class") or []
				if any("event" in cls for cls in classes):
					is_event = True

			category = "event" if is_event else "news"

			candidates.append(RawItem(title=title, item_url=item_url, category=category))

	if not candidates:
		logger.warning("No news or event items found for %s", base_url)

	return candidates

@mcp.tool()
def extract_news_and_events(urls: list[AnyHttpUrl]) -> list[dict[str, object]]:
	"""Fetch each URL, parse the page HTML, and return news/event link records.

	Use this tool for index/list pages that contain multiple BME/VIK/TMIT news or
	event cards in the same HTML document. For every input URL it:
	1. downloads the page,
	2. finds news or event blocks,
	3. extracts the item URL and category,
	4. deduplicates repeated items across URLs.

	The result is a JSON-serializable list of validated records using one schema:
	- news/event links: item_url, category
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
if __name__ == "__main__":
	transport = os.getenv("FASTMCP_TRANSPORT", "stdio")
	if transport in ("http", "streamable-http", "sse"):
		host = os.getenv("FASTMCP_HOST", "0.0.0.0")
		port = int(os.getenv("FASTMCP_PORT", "8000"))
		path = os.getenv("FASTMCP_PATH", "/mcp")
		mcp.run(transport=transport, host=host, port=port, path=path)
	else:
		mcp.run(transport="stdio")
