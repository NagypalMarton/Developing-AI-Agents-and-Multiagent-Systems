"""Basic FastMCP server with Pydantic-validated tools.

Run as MCP server (HTTP):
	python src/fastmcp_server.py

Custom host/port/path:
	python src/fastmcp_server.py --host 127.0.0.1 --port 8000 --path /mcp

Required packages:
	pip install fastmcp pydantic requests beautifulsoup4
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import date
from typing import Iterable
from urllib.parse import urljoin

from fastmcp import FastMCP
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup


mcp = FastMCP("basic-tools")


class FetchHtmlInput(BaseModel):
	url: str = Field(description="Target URL to fetch")
	timeout_seconds: int = Field(
		default=15, ge=3, le=60, description="HTTP timeout in seconds"
	)


class ExtractNewsBlocksInput(BaseModel):
	html: str = Field(description="Raw HTML content of a news homepage")
	base_url: str = Field(
		description="Base URL of the page to resolve relative article links"
	)
	limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")


class NewsBlock(BaseModel):
	news_title: str
	news_date: str
	news_content: str
	news_topics: list[str]
	news_url: str


class SummarizeTextInput(BaseModel):
	text: str = Field(description="Long text to summarize")
	max_sentences: int = Field(
		default=3,
		ge=1,
		le=8,
		description="Maximum number of sentences in the output summary",
	)


def _clean_text(value: str) -> str:
	return re.sub(r"\s+", " ", value).strip()


def _split_sentences(text: str) -> list[str]:
	normalized = _clean_text(text)
	if not normalized:
		return []
	parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
	return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> list[str]:
	return re.findall(r"[^\W\d_]{3,}", text.lower(), flags=re.UNICODE)


def _iter_news_candidates(soup: BeautifulSoup) -> Iterable:
	selectors = [
		"article",
		"div[class*='news']",
		"div[class*='story']",
		"div[class*='post']",
		"li[class*='news']",
		"li[class*='story']",
		"li[class*='post']",
	]
	seen = set()
	for selector in selectors:
		for node in soup.select(selector):
			node_id = id(node)
			if node_id in seen:
				continue
			seen.add(node_id)
			yield node


def _unique_topics(raw_topics: list[str]) -> list[str]:
	seen: set[str] = set()
	result: list[str] = []
	for topic in raw_topics:
		clean = _clean_text(topic)
		if not clean:
			continue
		key = clean.lower()
		if key in seen:
			continue
		seen.add(key)
		result.append(clean)
	return result


def _extract_node_hir_articles(soup: BeautifulSoup, base_url: str) -> list[NewsBlock]:
	items: list[NewsBlock] = []
	for article in soup.select("article.node-hir"):
		title_link = article.select_one("h1 a, h2 a, h3 a, h4 a")
		if not title_link:
			continue

		news_title = _clean_text(title_link.get_text(" ", strip=True))
		if len(news_title) < 8:
			continue

		href = title_link.get("href")
		if not href:
			continue
		news_url = urljoin(base_url, href)

		body_node = article.select_one(
			".field-name-body p, .field--name-body p, .field-type-text-with-summary p"
		)
		news_content = _clean_text(body_node.get_text(" ", strip=True)) if body_node else news_title

		date_node = article.find("time") or article.find(
			attrs={"class": re.compile(r"date|time|datum|created", re.I)}
		)
		news_date = _clean_text(date_node.get_text(" ", strip=True)) if date_node else ""

		topic_nodes = article.select(
			"a[rel='tag'], a[class*='tag'], a[class*='topic'], a[class*='category'], .field-name-field-tags li"
		)
		topics = _unique_topics([node.get_text(" ", strip=True) for node in topic_nodes])

		items.append(
			NewsBlock(
				news_title=news_title,
				news_date=news_date,
				news_content=news_content,
				news_topics=topics,
				news_url=news_url,
			)
		)
	return items


def _extract_bme_news_cards(soup: BeautifulSoup, base_url: str) -> list[NewsBlock]:
	items: list[NewsBlock] = []
	for link in soup.find_all("a", href=True):
		if not link.find(class_=re.compile(r"bme_news_card", re.I)):
			continue

		title_node = link.select_one("h4.bme_news_card-title, h3.bme_news_card-title, h4, h3")
		if not title_node:
			continue

		news_title = _clean_text(title_node.get_text(" ", strip=True))
		if len(news_title) < 8:
			continue

		news_url = urljoin(base_url, link["href"])

		content_node = link.select_one(".bme_news_card-body p, .bme_news_card-body")
		news_content = _clean_text(content_node.get_text(" ", strip=True)) if content_node else news_title

		date_node = link.select_one("datetime .field--name-created, .field--name-created")
		news_date = _clean_text(date_node.get_text(" ", strip=True)) if date_node else ""

		topic_nodes = link.select(".bme_news_card-tags li, .field--name-field-tags li")
		topics = _unique_topics([node.get_text(" ", strip=True) for node in topic_nodes])

		items.append(
			NewsBlock(
				news_title=news_title,
				news_date=news_date,
				news_content=news_content,
				news_topics=topics,
				news_url=news_url,
			)
		)
	return items


STOPWORDS = {
	"a",
	"az",
	"egy",
	"es",
	"hogy",
	"de",
	"ha",
	"is",
	"mint",
	"vagy",
	"van",
	"volt",
	"to",
	"the",
	"and",
	"for",
	"with",
	"this",
	"that",
	"from",
	"are",
	"was",
	"were",
	"you",
	"your",
	"their",
	"about",
	"into",
	"over",
	"under",
	"can",
	"will",
	"not",
	"have",
	"has",
	"had",
}


@mcp.tool
def get_today() -> str:
	"""Return the current date in ISO format."""
	return date.today().isoformat()


@mcp.tool
def fetch_html(payload: FetchHtmlInput) -> str:
	"""Download raw HTML from a URL using HTTP GET."""
	headers = {
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
			"AppleWebKit/537.36 (KHTML, like Gecko) "
			"Chrome/124.0.0.0 Safari/537.36"
		)
	}
	response = requests.get(payload.url, timeout=payload.timeout_seconds, headers=headers)
	response.raise_for_status()
	return response.text


@mcp.tool
def extract_news_blocks(payload: ExtractNewsBlocksInput) -> list[NewsBlock]:
	"""Extract news fields from homepage HTML in a structured JSON-friendly format."""
	soup = BeautifulSoup(payload.html, "html.parser")
	results: list[NewsBlock] = []
	seen_keys = set()

	for block in _extract_node_hir_articles(soup, payload.base_url):
		key = (block.news_title.lower(), block.news_url)
		if key in seen_keys:
			continue
		seen_keys.add(key)
		results.append(block)
		if len(results) >= payload.limit:
			return results

	for block in _extract_bme_news_cards(soup, payload.base_url):
		key = (block.news_title.lower(), block.news_url)
		if key in seen_keys:
			continue
		seen_keys.add(key)
		results.append(block)
		if len(results) >= payload.limit:
			return results

	for candidate in _iter_news_candidates(soup):
		title_node = candidate.find(["h1", "h2", "h3", "h4"]) or candidate.find("a")
		if not title_node:
			continue

		title = _clean_text(title_node.get_text(" ", strip=True))
		if len(title) < 12:
			continue

		link_node = title_node.find("a") if title_node.name != "a" else title_node
		if not link_node:
			link_node = candidate.find("a", href=True)
		if not link_node or not link_node.get("href"):
			continue

		news_url = urljoin(payload.base_url, link_node["href"])  # resolve relative links

		date_node = candidate.find("time")
		if date_node and date_node.get("datetime"):
			news_date = _clean_text(date_node["datetime"])
		elif date_node:
			news_date = _clean_text(date_node.get_text(" ", strip=True))
		else:
			alt_date = candidate.find(attrs={"class": re.compile(r"date|time|datum", re.I)})
			news_date = _clean_text(alt_date.get_text(" ", strip=True)) if alt_date else ""

		desc_node = candidate.find("p")
		news_content = _clean_text(desc_node.get_text(" ", strip=True)) if desc_node else ""
		if not news_content:
			news_content = title
		if len(news_content) > 420:
			news_content = news_content[:417].rstrip() + "..."

		topic_nodes = candidate.select("a[rel='tag'], a[class*='tag'], a[class*='topic'], a[class*='category']")
		news_topics = _unique_topics([node.get_text(" ", strip=True) for node in topic_nodes])

		if not news_topics:
			meta_keywords = candidate.find("meta", attrs={"name": re.compile(r"keywords", re.I)})
			if meta_keywords and meta_keywords.get("content"):
				news_topics = _unique_topics([
					_clean_text(part)
					for part in meta_keywords["content"].split(",")
					if _clean_text(part)
				])[:5]

		key = (title.lower(), news_url)
		if key in seen_keys:
			continue
		seen_keys.add(key)
		results.append(
			NewsBlock(
				news_title=title,
				news_date=news_date,
				news_content=news_content,
				news_topics=news_topics,
				news_url=news_url,
			)
		)

		if len(results) >= payload.limit:
			break

	return results


@mcp.tool
def summarize_text(payload: SummarizeTextInput) -> str:
	"""Create a concise extractive summary from long text."""
	sentences = _split_sentences(payload.text)
	if not sentences:
		return ""
	if len(sentences) <= payload.max_sentences:
		return " ".join(sentences)

	tokens = [token for token in _tokenize(payload.text) if token not in STOPWORDS]
	if not tokens:
		return " ".join(sentences[: payload.max_sentences])

	frequencies = Counter(tokens)
	max_freq = max(frequencies.values())
	normalized = {word: score / max_freq for word, score in frequencies.items()}

	scored: list[tuple[int, float]] = []
	for idx, sentence in enumerate(sentences):
		sentence_tokens = [t for t in _tokenize(sentence) if t not in STOPWORDS]
		if not sentence_tokens:
			continue
		score = sum(normalized.get(token, 0.0) for token in sentence_tokens)
		scored.append((idx, score))

	if not scored:
		return " ".join(sentences[: payload.max_sentences])

	top_indices = sorted(
		[idx for idx, _ in sorted(scored, key=lambda item: item[1], reverse=True)[: payload.max_sentences]]
	)
	return " ".join(sentences[idx] for idx in top_indices)


def run_server(host: str, port: int, path: str) -> None:
	"""Start FastMCP server in HTTP mode for localhost/remote clients."""
	mcp.run(transport="http", host=host, port=port, path=path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Basic FastMCP HTTP server with Pydantic tools"
	)
	parser.add_argument(
		"--host",
		default="127.0.0.1",
		help="Host to bind the HTTP MCP server to (default: 127.0.0.1)",
	)
	parser.add_argument(
		"--port",
		type=int,
		default=8000,
		help="Port for the HTTP MCP server (default: 8000)",
	)
	parser.add_argument(
		"--path",
		default="/mcp",
		help="HTTP path for MCP endpoint (default: /mcp)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_server(host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
	main()
