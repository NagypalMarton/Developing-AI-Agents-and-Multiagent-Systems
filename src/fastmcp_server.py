from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from html import unescape
from typing import Literal
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from fastmcp import FastMCP
from pydantic import BaseModel, Field, HttpUrl


mcp = FastMCP("news-events-agent")


NEWS_KEYWORDS = {
	"news",
	"hir",
	"article",
	"aktualis",
	"blog",
	"post",
	"announcement",
	"announcements",
}

EVENT_KEYWORDS = {
	"event",
	"events",
	"esemeny",
	"esemenyek",
	"calendar",
	"program",
	"conference",
	"workshop",
	"meetup",
	"webinar",
}


class UrlExtractionResult(BaseModel):
	source_url: HttpUrl | None = None
	news_urls: list[HttpUrl] = Field(default_factory=list)
	event_urls: list[HttpUrl] = Field(default_factory=list)
	ignored_urls: list[str] = Field(default_factory=list)


class PageSummary(BaseModel):
	url: HttpUrl
	title: str
	text: str
	image_url: HttpUrl | None = None
	summary: str


class SummaryBatchResult(BaseModel):
	kind: Literal["news", "events"]
	items: list[PageSummary] = Field(default_factory=list)
	failed_urls: list[str] = Field(default_factory=list)


class EndToEndResult(BaseModel):
	extracted: UrlExtractionResult
	news: SummaryBatchResult
	events: SummaryBatchResult


@dataclass
class _LinkCandidate:
	url: str
	news_score: int
	event_score: int


def _normalize_text(value: str) -> str:
	value = unescape(value or "")
	value = value.lower().strip()
	value = re.sub(r"\s+", " ", value)
	return value


def _keyword_score(text: str, keywords: set[str]) -> int:
	if not text:
		return 0
	normalized = _normalize_text(text)
	return sum(1 for kw in keywords if kw in normalized)


def _extract_best_title(soup: BeautifulSoup) -> str:
	og_title = soup.select_one('meta[property="og:title"]')
	if og_title and og_title.get("content"):
		return og_title["content"].strip()

	h1 = soup.find("h1")
	if h1 and h1.get_text(strip=True):
		return h1.get_text(strip=True)

	if soup.title and soup.title.get_text(strip=True):
		return soup.title.get_text(strip=True)

	return "Untitled"


def _extract_best_image_url(soup: BeautifulSoup, page_url: str) -> str | None:
	og_image = soup.select_one('meta[property="og:image"]')
	if og_image and og_image.get("content"):
		return urljoin(page_url, og_image["content"].strip())

	# Fallback: first large-looking image in article/main/body.
	scopes = [soup.find("article"), soup.find("main"), soup.body, soup]
	for scope in scopes:
		if not scope:
			continue
		for img in scope.find_all("img"):
			src = (img.get("src") or "").strip()
			if src:
				return urljoin(page_url, src)

	return None


def _extract_best_text(soup: BeautifulSoup) -> str:
	for trash_selector in ["script", "style", "noscript", "svg"]:
		for node in soup.select(trash_selector):
			node.decompose()

	scopes = [soup.find("article"), soup.find("main"), soup.body, soup]
	chunks: list[str] = []
	for scope in scopes:
		if not scope:
			continue

		for tag in scope.find_all(["p", "li", "blockquote"]):
			text = tag.get_text(" ", strip=True)
			if len(text) >= 40:
				chunks.append(text)

		if chunks:
			break

	if not chunks:
		fallback = soup.get_text(" ", strip=True)
		return re.sub(r"\s+", " ", fallback).strip()

	merged = "\n".join(dict.fromkeys(chunks))
	return re.sub(r"\s+", " ", merged).strip()


def _summarize_text(text: str, max_sentences: int = 3, max_chars: int = 500) -> str:
	text = re.sub(r"\s+", " ", text or "").strip()
	if not text:
		return ""

	sentences = re.split(r"(?<=[.!?])\s+", text)
	summary = " ".join(sentences[:max_sentences]).strip()
	if len(summary) > max_chars:
		return summary[: max_chars - 3].rstrip() + "..."
	return summary


def _fetch_html(url: str, timeout_seconds: int = 20) -> str:
	request = Request(
		url,
		headers={
			"User-Agent": "Mozilla/5.0 (compatible; FastMCPNewsAgent/1.0)",
			"Accept": "text/html,application/xhtml+xml",
		},
	)

	with urlopen(request, timeout=timeout_seconds) as response:
		raw = response.read()
		encoding = response.headers.get_content_charset() or "utf-8"

	try:
		return raw.decode(encoding, errors="replace")
	except LookupError:
		return raw.decode("utf-8", errors="replace")


def _collect_link_candidates(html: str, base_url: str | None = None) -> tuple[list[_LinkCandidate], list[str]]:
	soup = BeautifulSoup(html, "html.parser")
	candidates: list[_LinkCandidate] = []
	ignored: list[str] = []
	seen: set[str] = set()

	for link in soup.find_all("a"):
		href = (link.get("href") or "").strip()
		if not href or href.startswith("#"):
			continue

		resolved_url = urljoin(base_url, href) if base_url else href
		if not resolved_url.startswith(("http://", "https://")):
			ignored.append(href)
			continue

		if resolved_url in seen:
			continue
		seen.add(resolved_url)

		anchor_text = link.get_text(" ", strip=True)
		parent_text = ""
		if link.parent:
			parent_text = link.parent.get_text(" ", strip=True)

		class_blob = " ".join(link.get("class", []))
		id_blob = str(link.get("id") or "")
		score_source = " ".join([resolved_url, anchor_text, parent_text, class_blob, id_blob])

		news_score = _keyword_score(score_source, NEWS_KEYWORDS)
		event_score = _keyword_score(score_source, EVENT_KEYWORDS)
		if news_score == 0 and event_score == 0:
			continue

		candidates.append(
			_LinkCandidate(
				url=resolved_url,
				news_score=news_score,
				event_score=event_score,
			)
		)

	return candidates, ignored


def _split_news_and_event_urls(
	candidates: list[_LinkCandidate], max_per_type: int
) -> tuple[list[str], list[str]]:
	news: list[str] = []
	events: list[str] = []

	for candidate in sorted(candidates, key=lambda item: (item.news_score + item.event_score), reverse=True):
		if candidate.news_score >= candidate.event_score and len(news) < max_per_type:
			news.append(candidate.url)
		if candidate.event_score > candidate.news_score and len(events) < max_per_type:
			events.append(candidate.url)

		# If both are strong matches, include in both lists when there is room.
		if candidate.news_score > 0 and candidate.event_score > 0:
			if candidate.url not in news and len(news) < max_per_type:
				news.append(candidate.url)
			if candidate.url not in events and len(events) < max_per_type:
				events.append(candidate.url)

	return news, events


def _summarize_pages(urls: list[str], kind: Literal["news", "events"], timeout_seconds: int) -> SummaryBatchResult:
	items: list[PageSummary] = []
	failed: list[str] = []

	for url in urls:
		try:
			html = _fetch_html(url=url, timeout_seconds=timeout_seconds)
			soup = BeautifulSoup(html, "html.parser")

			title = _extract_best_title(soup)
			text = _extract_best_text(soup)
			image_url = _extract_best_image_url(soup, page_url=url)

			items.append(
				PageSummary(
					url=url,
					title=title,
					text=text,
					image_url=image_url,
					summary=_summarize_text(text),
				)
			)
		except Exception:
			failed.append(url)

	return SummaryBatchResult(kind=kind, items=items, failed_urls=failed)


@mcp.tool()
def extract_news_and_event_urls(
	html: str,
	base_url: str | None = None,
	max_per_type: int = 30,
) -> dict:
	"""
	Extract likely news and event URLs from a provided HTML page.

	Args:
		html: Raw HTML string from user input.
		base_url: Optional base URL used to resolve relative links.
		max_per_type: Maximum number of URLs to return for each category.
	"""
	candidates, ignored = _collect_link_candidates(html=html, base_url=base_url)
	news_urls, event_urls = _split_news_and_event_urls(candidates=candidates, max_per_type=max_per_type)

	result = UrlExtractionResult(
		source_url=base_url,
		news_urls=news_urls,
		event_urls=event_urls,
		ignored_urls=ignored,
	)
	return result.model_dump(mode="json")


@mcp.tool()
def summarize_news_from_urls(urls: list[str], timeout_seconds: int = 20) -> dict:
	"""
	Download and summarize news pages.

	For each URL the tool extracts title, main text, first image URL, and a short summary.
	"""
	result = _summarize_pages(urls=urls, kind="news", timeout_seconds=timeout_seconds)
	return result.model_dump(mode="json")


@mcp.tool()
def summarize_events_from_urls(urls: list[str], timeout_seconds: int = 20) -> dict:
	"""
	Download and summarize event pages.

	For each URL the tool extracts title, main text, first image URL, and a short summary.
	"""
	result = _summarize_pages(urls=urls, kind="events", timeout_seconds=timeout_seconds)
	return result.model_dump(mode="json")


@mcp.tool()
def process_news_and_events_from_html(
	html: str,
	base_url: str | None = None,
	max_per_type: int = 15,
	timeout_seconds: int = 20,
) -> dict:
	"""
	End-to-end tool: extract links from HTML, then fetch and summarize both news and event pages.
	"""
	extracted = extract_news_and_event_urls(html=html, base_url=base_url, max_per_type=max_per_type)

	news_urls = extracted.get("news_urls", [])
	event_urls = extracted.get("event_urls", [])

	news_result = _summarize_pages(urls=news_urls, kind="news", timeout_seconds=timeout_seconds)
	event_result = _summarize_pages(urls=event_urls, kind="events", timeout_seconds=timeout_seconds)

	final = EndToEndResult(
		extracted=UrlExtractionResult.model_validate(extracted),
		news=news_result,
		events=event_result,
	)
	return final.model_dump(mode="json")


@mcp.tool()
def health() -> dict:
	"""Simple health-check tool."""
	return {"status": "ok"}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="FastMCP server for news and events processing")
	parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
	parser.add_argument("--port", type=int, default=8000, help="Port to bind")
	parser.add_argument("--path", default="/mcp", help="MCP HTTP path")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	# Newer fastmcp versions commonly use streamable-http transport.
	try:
		mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)
	except TypeError:
		# Fallback for fastmcp variants with a different run signature.
		mcp.run(host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
	main()
