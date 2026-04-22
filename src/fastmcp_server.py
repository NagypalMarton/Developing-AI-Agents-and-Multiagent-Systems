"""Basic FastMCP server with Pydantic-validated tools.

Run as MCP server (HTTP):
	python src/fastmcp_server.py

Custom host/port/path:
	python src/fastmcp_server.py --host 127.0.0.1 --port 8000 --path /mcp

Required packages:
	pip install fastmcp pydantic beautifulsoup4 arize-phoenix arize-phoenix-otel pydantic-evals
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from difflib import SequenceMatcher
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import urljoin

from fastmcp import FastMCP
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import requests


mcp = FastMCP("basic-tools")


class FetchUrlInput(BaseModel):
	url: str = Field(description="Target URL")


class FetchUrlOutput(BaseModel):
	status: int
	content_type: str
	body: str


class ExtractHtmlInput(BaseModel):
	html: str = Field(description="Raw HTML document")
	mode: Literal["teaser", "article"] = Field(
		description="Extraction mode: teaser or article"
	)


class TeaserItem(BaseModel):
	title: str
	summary: str
	url: str
	image: str


class TeaserExtractionOutput(BaseModel):
	items: list[TeaserItem]


class ArticleExtractionOutput(BaseModel):
	title: str
	lead: str
	body: list[str]
	images: list[str]
	links: list[str]


class ExtractEntitiesInput(BaseModel):
	text: str = Field(description="Input text for entity extraction")


class EntitiesOutput(BaseModel):
	dates: list[str]
	locations: list[str]
	events: list[str]
	persons: list[str]
	organizations: list[str]
	urls: list[str]


class DetectEventInput(BaseModel):
	entities: EntitiesOutput


class DetectEventOutput(BaseModel):
	event_detected: bool
	event_name: str
	event_date: str
	event_location: str
	registration_url: str


class SummarizeInput(BaseModel):
	text: str
	length: Literal["short", "medium", "long"]


class SummarizeOutput(BaseModel):
	summary: str


class RewriteForPlatformInput(BaseModel):
	platform: str
	content: str


class RewriteForPlatformOutput(BaseModel):
	post: str


class GenerateCtaInput(BaseModel):
	event_name: str
	registration_url: str


class GenerateCtaOutput(BaseModel):
	cta: str


class GetTodayInput(BaseModel):
	# Empty object input to keep OpenAI-style function schema uniform across tools.
	pass


class EvalCaseInput(BaseModel):
	name: str = Field(description="Human-readable test case name")
	input_text: str = Field(description="Input prompt to evaluate")
	expected_output: str = Field(description="Expected answer for the input")


class RunPhoenixPydanticEvalsInput(BaseModel):
	project_name: str = Field(
		default="default",
		description="Phoenix project name to query spans and upload annotations",
	)
	llm_judge_model: str = Field(
		default="openai:gpt-4o-mini",
		description="Model identifier used by Pydantic Evals LLMJudge",
	)
	fuzzy_threshold: float = Field(
		default=0.8,
		ge=0.0,
		le=1.0,
		description="Similarity threshold for fuzzy matching evaluator",
	)
	cases: list[EvalCaseInput] = Field(
		default_factory=lambda: [
			EvalCaseInput(
				name="capital of France",
				input_text="What is the capital of France?",
				expected_output="Paris",
			),
			EvalCaseInput(
				name="author of Romeo and Juliet",
				input_text="Who wrote Romeo and Juliet?",
				expected_output="William Shakespeare",
			),
			EvalCaseInput(
				name="largest planet",
				input_text="What is the largest planet in our solar system?",
				expected_output="Jupiter",
			),
		],
		description="Evaluation dataset cases",
	)


class EvalCaseResult(BaseModel):
	name: str
	input_text: str
	expected_output: str
	output: str
	exact_match: bool
	fuzzy_match: bool
	llm_match: bool | None = None


class RunPhoenixPydanticEvalsOutput(BaseModel):
	project_name: str
	total_cases: int
	exact_match_rate: float
	fuzzy_match_rate: float
	llm_match_rate: float | None = None
	uploaded_annotations: list[str]
	notes: list[str]
	cases: list[EvalCaseResult]
	report: dict[str, object]


class NewsBlock(BaseModel):
	news_title: str
	news_date: str
	news_content: str
	news_topics: list[str]
	news_url: str


def _clean_text(value: str) -> str:
	return re.sub(r"\s+", " ", value).strip()


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


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
	seen: set[str] = set()
	result: list[str] = []
	for value in values:
		clean = _clean_text(value)
		if not clean:
			continue
		key = clean.lower()
		if key in seen:
			continue
		seen.add(key)
		result.append(clean)
	return result


def _extract_teaser_items(soup: BeautifulSoup) -> list[TeaserItem]:
	items: list[TeaserItem] = []
	seen: set[tuple[str, str]] = set()

	for candidate in _iter_news_candidates(soup):
		title_node = candidate.find(["h1", "h2", "h3", "h4"]) or candidate.find("a")
		if not title_node:
			continue

		title = _clean_text(title_node.get_text(" ", strip=True))
		if len(title) < 8:
			continue

		link_node = title_node.find("a") if title_node.name != "a" else title_node
		if not link_node:
			link_node = candidate.find("a", href=True)
		url = _clean_text(link_node.get("href", "")) if link_node else ""

		summary_node = candidate.find("p")
		summary = _clean_text(summary_node.get_text(" ", strip=True)) if summary_node else ""
		if not summary:
			summary = title

		image_node = candidate.find("img")
		image = ""
		if image_node:
			image = _clean_text(
				image_node.get("src")
				or image_node.get("data-src")
				or image_node.get("data-original")
				or ""
			)

		key = (title.lower(), url)
		if key in seen:
			continue
		seen.add(key)

		items.append(TeaserItem(title=title, summary=summary, url=url, image=image))
		if len(items) >= 30:
			break

	return items


def _extract_article(soup: BeautifulSoup) -> ArticleExtractionOutput:
	title_node = soup.select_one("article h1, main h1, h1")
	if not title_node:
		title_node = soup.find("title")
	title = _clean_text(title_node.get_text(" ", strip=True)) if title_node else ""

	lead_node = soup.select_one(
		"article .lead, article p.lead, main .lead, .article-lead, .entry-summary"
	)
	if not lead_node:
		lead_node = soup.select_one("article p, main p")
	lead = _clean_text(lead_node.get_text(" ", strip=True)) if lead_node else ""

	body_nodes = soup.select("article p, main p")
	if not body_nodes:
		body_nodes = soup.select("p")
	body = _dedupe_keep_order(
		[
			_clean_text(node.get_text(" ", strip=True))
			for node in body_nodes
			if _clean_text(node.get_text(" ", strip=True))
		]
	)

	image_urls = _dedupe_keep_order(
		[
			_clean_text(img.get("src") or img.get("data-src") or "")
			for img in soup.select("article img, main img, img")
		]
	)

	links = _dedupe_keep_order(
		[_clean_text(a.get("href", "")) for a in soup.select("article a[href], main a[href], a[href]")]
	)

	return ArticleExtractionOutput(
		title=title,
		lead=lead,
		body=body,
		images=image_urls,
		links=links,
	)


def _split_sentences(text: str) -> list[str]:
	parts = re.split(r"(?<=[.!?])\s+", _clean_text(text))
	return [part.strip() for part in parts if part.strip()]


def _extract_entities_from_text(text: str) -> EntitiesOutput:
	clean = _clean_text(text)

	date_patterns = [
		r"\b\d{4}-\d{2}-\d{2}\b",
		r"\b\d{4}\.\d{1,2}\.\d{1,2}\.?\b",
		r"\b\d{1,2}\.\d{1,2}\.\d{4}\.?\b",
	]
	dates = _dedupe_keep_order(
		match for pattern in date_patterns for match in re.findall(pattern, clean)
	)

	url_pattern = r"https?://[^\s)\]>'\"]+"
	urls = _dedupe_keep_order(re.findall(url_pattern, clean))

	person_pattern = (
		r"\b[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+\b"
	)
	persons = _dedupe_keep_order(re.findall(person_pattern, clean))

	org_keyword_pattern = (
		r"\b[A-ZÁÉÍÓÖŐÚÜŰ][\wÁÉÍÓÖŐÚÜŰáéíóöőúüű.-]*(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][\wÁÉÍÓÖŐÚÜŰáéíóöőúüű.-]*)*\s+"
		r"(?:University|Egyetem|Kar|Tanszék|Intézet|Kft|Zrt|Ltd|Inc|GmbH)\b"
	)
	organizations = _dedupe_keep_order(re.findall(org_keyword_pattern, clean))

	acronyms = _dedupe_keep_order(re.findall(r"\b[A-Z]{2,}\b", clean))
	for acronym in acronyms:
		if acronym not in organizations:
			organizations.append(acronym)

	event_pattern = (
		r"\b[\wÁÉÍÓÖŐÚÜŰáéíóöőúüű\- ]{3,}"
		r"(?:konferencia|workshop|szeminárium|seminar|meetup|hackathon|nyílt nap|verseny|előadás|fórum)\b"
	)
	events = _dedupe_keep_order(re.findall(event_pattern, clean, flags=re.IGNORECASE))

	locations: list[str] = []
	for match in re.findall(
		r"\b(?:in|at|on|Budapesten|Budapesten|Budapest|Debrecenben|Szegeden|Győrben)\s+"
		r"([A-ZÁÉÍÓÖŐÚÜŰ][\wÁÉÍÓÖŐÚÜŰáéíóöőúüű-]*(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][\wÁÉÍÓÖŐÚÜŰáéíóöőúüű-]*)*)",
		clean,
	):
		locations.append(match)
	locations = _dedupe_keep_order(locations)

	return EntitiesOutput(
		dates=dates,
		locations=locations,
		events=events,
		persons=persons,
		organizations=organizations,
		urls=urls,
	)


def _summarize_text(text: str, length: str) -> str:
	char_limits = {"short": 240, "medium": 600, "long": 1200}
	limit = char_limits[length]
	if len(_clean_text(text)) <= limit:
		return _clean_text(text)

	sentences = _split_sentences(text)
	if not sentences:
		return _clean_text(text)[: max(0, limit - 3)] + "..."

	selected: list[str] = []
	current_len = 0
	for sentence in sentences:
		needed = len(sentence) + (1 if selected else 0)
		if current_len + needed > limit:
			break
		selected.append(sentence)
		current_len += needed

	if not selected:
		return _clean_text(text)[: max(0, limit - 3)] + "..."

	result = " ".join(selected)
	if len(result) < len(_clean_text(text)):
		result = result.rstrip(" .") + "..."
	return result


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


def _matches_iso_date(news_date_text: str, iso_date: str | None) -> bool:
	if not iso_date:
		return True
	if not news_date_text:
		return False

	clean = _clean_text(news_date_text)
	if iso_date in clean:
		return True

	iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", clean)
	if iso_match and iso_match.group(0) == iso_date:
		return True

	dotted_match = re.search(r"\b(\d{4})\.(\d{1,2})\.(\d{1,2})\b", clean)
	if dotted_match:
		year, month, day = dotted_match.groups()
		normalized = f"{year}-{int(month):02d}-{int(day):02d}"
		if normalized == iso_date:
			return True

	return False


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
	for card in soup.select("div.bme_news_card, article.bme_news_card, section.bme_news_card"):
		link = card.find_parent("a", href=True)
		if not link:
			continue

		title_node = card.select_one("h4.bme_news_card-title, h3.bme_news_card-title, h4, h3")
		if not title_node:
			continue

		news_title = _clean_text(title_node.get_text(" ", strip=True))
		if len(news_title) < 8:
			continue

		news_url = urljoin(base_url, link["href"])

		content_node = card.select_one(".bme_news_card-body p, .bme_news_card-body")
		news_content = _clean_text(content_node.get_text(" ", strip=True)) if content_node else news_title

		date_node = card.select_one("datetime .field--name-created, .field--name-created")
		news_date = _clean_text(date_node.get_text(" ", strip=True)) if date_node else ""

		topic_nodes = card.select(".bme_news_card-tags li, .field--name-field-tags li")
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


def _download_html(url: str, timeout_seconds: int) -> tuple[str, str]:
	headers = {
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
			"AppleWebKit/537.36 (KHTML, like Gecko) "
			"Chrome/124.0.0.0 Safari/537.36"
		)
	}
	response = requests.get(url, timeout=timeout_seconds, headers=headers)
	response.raise_for_status()
	return response.text, response.url


def _extract_news_blocks_from_html(
	html: str,
	base_url: str,
	limit: int,
	published_on: str | None = None,
) -> list[NewsBlock]:
	soup = BeautifulSoup(html, "html.parser")
	results: list[NewsBlock] = []
	seen_keys = set()

	for block in _extract_node_hir_articles(soup, base_url):
		if not _matches_iso_date(block.news_date, published_on):
			continue
		key = (block.news_title.lower(), block.news_url)
		if key in seen_keys:
			continue
		seen_keys.add(key)
		results.append(block)
		if len(results) >= limit:
			return results

	for block in _extract_bme_news_cards(soup, base_url):
		if not _matches_iso_date(block.news_date, published_on):
			continue
		key = (block.news_title.lower(), block.news_url)
		if key in seen_keys:
			continue
		seen_keys.add(key)
		results.append(block)
		if len(results) >= limit:
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

		news_url = urljoin(base_url, link_node["href"])

		date_node = candidate.find("time")
		if date_node and date_node.get("datetime"):
			news_date = _clean_text(date_node["datetime"])
		elif date_node:
			news_date = _clean_text(date_node.get_text(" ", strip=True))
		else:
			alt_date = candidate.find(attrs={"class": re.compile(r"date|time|datum", re.I)})
			news_date = _clean_text(alt_date.get_text(" ", strip=True)) if alt_date else ""

		if not _matches_iso_date(news_date, published_on):
			continue

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

		if len(results) >= limit:
			return results

	return results


def _message_content(value: object, preferred_role: str | None = None) -> str:
	if not isinstance(value, list):
		return ""
	for item in value:
		if not isinstance(item, dict):
			continue
		message = item.get("message", item)
		if not isinstance(message, dict):
			continue
		role = message.get("role")
		if preferred_role and role != preferred_role:
			continue
		content = message.get("content")
		if isinstance(content, str):
			return _clean_text(content)
		if isinstance(content, list):
			parts: list[str] = []
			for part in content:
				if isinstance(part, dict):
					text = part.get("text")
					if isinstance(text, str) and text.strip():
						parts.append(text)
			if parts:
				return _clean_text(" ".join(parts))
	return ""


def _similarity_ratio(a: str, b: str) -> float:
	return SequenceMatcher(None, a, b).ratio()


def _script_span_message_content(value: object, index: int) -> str:
	"""Extract message content with the same index-based logic as the reference script."""
	if not isinstance(value, list) or len(value) <= index:
		return ""
	item = value[index]
	if not isinstance(item, dict):
		return ""
	message = item.get("message")
	if not isinstance(message, dict):
		return ""
	content = message.get("content")
	return _clean_text(content) if isinstance(content, str) else ""


def _memory_root(root_value: str) -> Path:
	root = Path(root_value)
	if not root.is_absolute():
		root = Path.cwd() / root
	root = root.resolve()
	root.mkdir(parents=True, exist_ok=True)
	return root


def _memory_safe_path(root: Path, raw_path: str) -> Path:
	relative = raw_path.strip().replace("\\", "/").lstrip("/")
	if not relative:
		raise ValueError("Memory path must not be empty.")
	target = (root / relative).resolve()
	if root != target and root not in target.parents:
		raise ValueError("Memory path escapes configured memory root.")
	return target


def _memory_view(root: Path, path: str) -> str | list[str]:
	target = _memory_safe_path(root, path)
	if target.is_dir():
		entries: list[str] = []
		for child in sorted(target.iterdir(), key=lambda item: item.name.lower()):
			suffix = "/" if child.is_dir() else ""
			entries.append(f"{child.name}{suffix}")
		return entries
	if not target.exists():
		return ""
	return target.read_text(encoding="utf-8")


def _memory_create(root: Path, path: str, file_text: str) -> str:
	target = _memory_safe_path(root, path)
	if target.exists():
		raise ValueError("Memory create failed: path already exists.")
	target.parent.mkdir(parents=True, exist_ok=True)
	target.write_text(file_text, encoding="utf-8")
	return f"Created {path}."


def _memory_str_replace(root: Path, path: str, old_str: str, new_str: str) -> str:
	target = _memory_safe_path(root, path)
	text = target.read_text(encoding="utf-8")
	count = text.count(old_str)
	if count != 1:
		raise ValueError("str_replace requires old_str to appear exactly once.")
	target.write_text(text.replace(old_str, new_str), encoding="utf-8")
	return f"Updated {path}."


def _memory_insert(root: Path, path: str, insert_line: int, insert_text: str) -> str:
	target = _memory_safe_path(root, path)
	lines = target.read_text(encoding="utf-8").splitlines()
	if insert_line < 0 or insert_line > len(lines):
		raise ValueError("insert_line is out of range.")
	new_lines = lines[:insert_line] + insert_text.splitlines() + lines[insert_line:]
	target.write_text("\n".join(new_lines), encoding="utf-8")
	return f"Inserted text into {path}."


def _memory_delete(root: Path, path: str) -> str:
	target = _memory_safe_path(root, path)
	if not target.exists():
		return f"Path not found: {path}."
	if target.is_dir():
		shutil.rmtree(target)
	else:
		target.unlink()
	return f"Deleted {path}."


def _memory_rename(root: Path, old_path: str, new_path: str) -> str:
	source = _memory_safe_path(root, old_path)
	target = _memory_safe_path(root, new_path)
	target.parent.mkdir(parents=True, exist_ok=True)
	source.rename(target)
	return f"Renamed {old_path} to {new_path}."


def _run_memory_command(command: dict[str, Any], root: Path) -> str | list[str]:
	command_name = str(command.get("command", "")).strip().lower()
	if command_name == "view":
		path = str(command.get("path", ""))
		return _memory_view(root, path)
	if command_name == "create":
		return _memory_create(
			root,
			str(command.get("path", "")),
			str(command.get("file_text", "")),
		)
	if command_name == "str_replace":
		return _memory_str_replace(
			root,
			str(command.get("path", "")),
			str(command.get("old_str", "")),
			str(command.get("new_str", "")),
		)
	if command_name == "insert":
		insert_line = int(command.get("insert_line", 0))
		insert_text = str(command.get("insert_text", ""))
		return _memory_insert(
			root,
			str(command.get("path", "")),
			insert_line,
			insert_text,
		)
	if command_name == "delete":
		return _memory_delete(root, str(command.get("path", "")))
	if command_name == "rename":
		return _memory_rename(
			root,
			str(command.get("old_path", "")),
			str(command.get("new_path", "")),
		)
	if command_name == "clear_all_memory":
		for child in root.iterdir():
			if child.is_dir():
				shutil.rmtree(child)
			else:
				child.unlink()
		return "Cleared all memory."
	raise ValueError(f"Unsupported memory command: {command_name or '<empty>'}")


def setup_telemetry() -> None:
	"""Configure OpenTelemetry export to Arize Phoenix using OTLP/HTTP."""
	tracer_provider = TracerProvider()
	trace.set_tracer_provider(tracer_provider)

	collector_base = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006").rstrip("/")
	endpoint = f"{collector_base}/v1/traces"

	headers: dict[str, str] | None = None
	phoenix_api_key = os.getenv("PHOENIX_API_KEY")
	if phoenix_api_key:
		headers = {"Authorization": f"Bearer {phoenix_api_key}"}

	exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
	tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
	tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))


@mcp.tool
def get_today(payload: GetTodayInput) -> str:
	"""Return the current date in ISO format."""
	_ = payload
	return date.today().isoformat()


@mcp.tool
def fetch_url(payload: FetchUrlInput) -> FetchUrlOutput:
	"""HTTP GET wrapper that returns status code, content-type and body."""
	headers = {
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
			"AppleWebKit/537.36 (KHTML, like Gecko) "
			"Chrome/124.0.0.0 Safari/537.36"
		)
	}
	response = requests.get(payload.url, timeout=20, headers=headers)
	content_type = _clean_text(response.headers.get("Content-Type", ""))
	return FetchUrlOutput(
		status=response.status_code,
		content_type=content_type,
		body=response.text,
	)


@mcp.tool
def extract_html(payload: ExtractHtmlInput) -> TeaserExtractionOutput | ArticleExtractionOutput:
	"""Extract teaser blocks or article content from raw HTML."""
	soup = BeautifulSoup(payload.html, "html.parser")
	if payload.mode == "teaser":
		return TeaserExtractionOutput(items=_extract_teaser_items(soup))
	return _extract_article(soup)


@mcp.tool
def extract_entities(payload: ExtractEntitiesInput) -> EntitiesOutput:
	"""Extract basic entities from input text using deterministic heuristics."""
	return _extract_entities_from_text(payload.text)


@mcp.tool
def detect_event(payload: DetectEventInput) -> DetectEventOutput:
	"""Detect whether extracted entities indicate an event and return structured fields."""
	entities = payload.entities
	event_detected = bool(
		entities.dates and (entities.events or entities.locations or entities.urls)
	)

	event_name = entities.events[0] if entities.events else ""
	if not event_name and entities.organizations:
		event_name = f"{entities.organizations[0]} event"

	registration_url = ""
	for url in entities.urls:
		if re.search(r"register|registration|signup|jelentke", url, re.IGNORECASE):
			registration_url = url
			break
	if not registration_url and entities.urls:
		registration_url = entities.urls[0]

	return DetectEventOutput(
		event_detected=event_detected,
		event_name=event_name,
		event_date=entities.dates[0] if entities.dates else "",
		event_location=entities.locations[0] if entities.locations else "",
		registration_url=registration_url,
	)


@mcp.tool
def summarize(payload: SummarizeInput) -> SummarizeOutput:
	"""Generate a short, medium or long summary suitable for social channels."""
	return SummarizeOutput(summary=_summarize_text(payload.text, payload.length))


@mcp.tool
def rewrite_for_platform(payload: RewriteForPlatformInput) -> RewriteForPlatformOutput:
	"""Rewrite summary text for a target social platform."""
	platform = payload.platform.strip().lower()
	content = _clean_text(payload.content)

	if platform in {"x", "twitter"}:
		post = content
		if len(post) > 280:
			post = post[:277].rstrip() + "..."
		post = f"{post} #university #news"
	elif platform == "facebook":
		post = f"Friss hír: {content}\n\nOszd meg azokkal is, akiket érdekelhet."
	elif platform == "linkedin":
		post = (
			f"Egyetemi/tanszéki hír összefoglaló:\n{content}\n\n"
			"Ha releváns számodra, csatlakozz a szakmai beszélgetéshez."
		)
	else:
		post = content

	return RewriteForPlatformOutput(post=post)


@mcp.tool
def generate_cta(payload: GenerateCtaInput) -> GenerateCtaOutput:
	"""Generate a short call-to-action for an event."""
	event_name = _clean_text(payload.event_name)
	registration_url = _clean_text(payload.registration_url)

	if event_name and registration_url:
		cta = f"Ne maradj le: jelentkezz a(z) {event_name} eseményre itt: {registration_url}"
	elif registration_url:
		cta = f"Regisztrálj most: {registration_url}"
	elif event_name:
		cta = f"Csatlakozz a(z) {event_name} eseményhez!"
	else:
		cta = "Csatlakozz az eseményhez!"

	return GenerateCtaOutput(cta=cta)


@mcp.tool
def run_phoenix_pydantic_evals(payload: RunPhoenixPydanticEvalsInput) -> RunPhoenixPydanticEvalsOutput:
	"""Run Pydantic Evals over Phoenix traces and optionally upload evaluation labels."""
	from phoenix.otel import register
	from phoenix.client import Client
	from phoenix.client.types.spans import SpanQuery
	from pydantic_evals import Case, Dataset
	from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge

	notes: list[str] = []
	uploaded_annotations: list[str] = []

	register(project_name=payload.project_name, auto_instrument=True)

	query = SpanQuery().select("llm.input_messages", "llm.output_messages")
	phoenix_client = Client()
	spans = phoenix_client.spans.get_spans_dataframe(query=query, project_name=payload.project_name)

	if spans.empty:
		return RunPhoenixPydanticEvalsOutput(
			project_name=payload.project_name,
			total_cases=0,
			exact_match_rate=0.0,
			fuzzy_match_rate=0.0,
			llm_match_rate=None,
			uploaded_annotations=[],
			notes=["No spans found in Phoenix for the selected project."],
			cases=[],
			report={},
		)

	spans = spans.rename(columns={"llm.input_messages": "input", "llm.output_messages": "output"})
	spans["input"] = spans["input"].apply(lambda x: _script_span_message_content(x, 1) or _message_content(x, preferred_role="user"))
	spans["output"] = spans["output"].apply(lambda x: _script_span_message_content(x, 0) or _message_content(x, preferred_role="assistant"))

	lookup: dict[str, str] = {}
	for _, row in spans.iterrows():
		input_text = _clean_text(str(row.get("input", "")))
		output_text = _clean_text(str(row.get("output", "")))
		if input_text and output_text and input_text not in lookup:
			lookup[input_text] = output_text

	available_cases = [
		case for case in payload.cases if _clean_text(case.input_text) in lookup
	]
	missing_case_inputs = [
		case.input_text for case in payload.cases if _clean_text(case.input_text) not in lookup
	]

	if missing_case_inputs:
		notes.append(
			f"Skipped {len(missing_case_inputs)} case(s) with no matching traced input."
		)

	if not available_cases:
		return RunPhoenixPydanticEvalsOutput(
			project_name=payload.project_name,
			total_cases=0,
			exact_match_rate=0.0,
			fuzzy_match_rate=0.0,
			llm_match_rate=None,
			uploaded_annotations=[],
			notes=notes + ["No evaluable cases remained after matching against traced inputs."],
			cases=[],
			report={},
		)

	class MatchesExpectedOutput(Evaluator[str, str]):
		def evaluate(self, ctx: EvaluatorContext[str, str]) -> bool:
			return _clean_text(ctx.expected_output) == _clean_text(ctx.output)

	class FuzzyMatchesOutput(Evaluator[str, str]):
		def __init__(self, threshold: float) -> None:
			self.threshold = threshold

		def evaluate(self, ctx: EvaluatorContext[str, str]) -> bool:
			score = _similarity_ratio(_clean_text(ctx.expected_output).lower(), _clean_text(ctx.output).lower())
			return score >= self.threshold

	def task(input_text: str) -> str:
		result = lookup.get(_clean_text(input_text))
		if result is None:
			# This should not happen after prefiltering, but keep a safe fallback.
			return ""
		return result

	cases = [
		Case(name=case.name, inputs=case.input_text, expected_output=case.expected_output)
		for case in available_cases
	]
	dataset = Dataset(cases=cases, evaluators=[MatchesExpectedOutput(), FuzzyMatchesOutput(payload.fuzzy_threshold)])

	dataset.add_evaluator(
		LLMJudge(
			rubric=(
				"Output and Expected Output should represent the same answer, "
				"even if the text does not match exactly"
			),
			include_input=True,
			model=payload.llm_judge_model,
		)
	)

	report = dataset.evaluate_sync(task)
	report_data = report.model_dump()

	case_results: list[EvalCaseResult] = []
	exact_hits = 0
	fuzzy_hits = 0
	llm_hits = 0
	llm_count = 0

	for case in report_data.get("cases", []):
		assertions = case.get("assertions", {})
		exact_match = bool(assertions.get("MatchesExpectedOutput", {}).get("value", False))
		fuzzy_match = bool(assertions.get("FuzzyMatchesOutput", {}).get("value", False))
		llm_value_raw = assertions.get("LLMJudge", {}).get("value")
		llm_match = bool(llm_value_raw) if llm_value_raw is not None else None

		if exact_match:
			exact_hits += 1
		if fuzzy_match:
			fuzzy_hits += 1
		if llm_match is not None:
			llm_count += 1
			if llm_match:
				llm_hits += 1

		input_text = str(case.get("inputs", ""))
		case_results.append(
			EvalCaseResult(
				name=str(case.get("name", "")),
				input_text=input_text,
				expected_output=str(case.get("expected_output", "")),
				output=lookup.get(_clean_text(input_text), ""),
				exact_match=exact_match,
				fuzzy_match=fuzzy_match,
				llm_match=llm_match,
			)
		)

	evaluated_inputs = {
		_clean_text(str(case.get("inputs", ""))) for case in report_data.get("cases", [])
	}
	eval_spans = spans[spans["input"].apply(_clean_text).isin(evaluated_inputs)].copy()

	eval_frames = {
		"Direct Match Eval": eval_spans.copy(),
		"Fuzzy Match Eval": eval_spans.copy(),
		"LLM Match Eval": eval_spans.copy(),
	}

	for eval_case in report_data.get("cases", []):
		assertions = eval_case.get("assertions", {})
		input_text = str(eval_case.get("inputs", ""))

		labels: dict[str, object] = {
			"Direct Match Eval": assertions.get("MatchesExpectedOutput", {}).get("value"),
			"Fuzzy Match Eval": assertions.get("FuzzyMatchesOutput", {}).get("value"),
			"LLM Match Eval": assertions.get("LLMJudge", {}).get("value"),
		}

		for annotation_name, label_value in labels.items():
			df = eval_frames[annotation_name]
			df.loc[df["input"] == input_text, "label"] = str(label_value)

	for annotation_name, df in eval_frames.items():
		if df.empty:
			notes.append(f"Skipped annotation upload for {annotation_name}: no evaluated spans.")
			continue
		df["score"] = df["label"].apply(lambda x: 1 if str(x) == "True" else 0)
		annotator_kind = "LLM" if annotation_name == "LLM Match Eval" else "CODE"
		phoenix_client.spans.log_span_annotations_dataframe(
			dataframe=df,
			annotation_name=annotation_name,
			annotator_kind=annotator_kind,
		)
		uploaded_annotations.append(annotation_name)

	total_cases = len(case_results)
	exact_match_rate = (exact_hits / total_cases) if total_cases else 0.0
	fuzzy_match_rate = (fuzzy_hits / total_cases) if total_cases else 0.0
	llm_match_rate = (llm_hits / llm_count) if llm_count else None

	if not lookup:
		notes.append("No usable input/output rows were extracted from spans.")

	return RunPhoenixPydanticEvalsOutput(
		project_name=payload.project_name,
		total_cases=total_cases,
		exact_match_rate=exact_match_rate,
		fuzzy_match_rate=fuzzy_match_rate,
		llm_match_rate=llm_match_rate,
		uploaded_annotations=uploaded_annotations,
		notes=notes,
		cases=case_results,
		report=report_data,
	)


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
	setup_telemetry()
	args = parse_args()
	run_server(host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
	main()
