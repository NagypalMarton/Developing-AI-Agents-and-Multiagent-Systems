import hashlib
import json
import logging
import os
import re
import threading
from collections import deque
from datetime import UTC, date, datetime
from typing import Any, Literal

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl, ValidationError, field_validator

from fastmcp import FastMCP


LOGGER = logging.getLogger("fastmcp-server")
logging.basicConfig(level=logging.INFO)

MISTRAL_API_BASE = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

DEDUP_CACHE: set[str] = set()
PUBLISH_LOG: deque[dict[str, Any]] = deque(maxlen=1000)
STATE_LOCK = threading.Lock()

STATE_PUBLISHED = "published"
STATE_DUPLICATE = "duplicate"
STATE_FAILED = "failed"

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class EventInfo(BaseModel):
    is_event: bool
    confidence: float = Field(ge=0.0, le=1.0)


class NewsItem(BaseModel):
    source: str = Field(min_length=1)
    source_url: HttpUrl
    title: str = Field(min_length=3)
    summary: str = Field(min_length=3)
    full_text: str | None = None
    published_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    event: EventInfo


class ParsedRawItem(BaseModel):
    source: str = Field(min_length=1)
    source_url: HttpUrl
    title: str = Field(min_length=3)
    summary: str = Field(min_length=3)
    full_text: str | None = None
    published_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class UnifiedNewsPayload(BaseModel):
    news_title: str = Field(min_length=3)
    news_date: date | None = None
    news_content: str = Field(min_length=3)
    news_topics: list[str] = Field(default_factory=list)
    news_source: HttpUrl
    news_tags: list[str] = Field(default_factory=list)

    @field_validator("news_topics", mode="before")
    @classmethod
    def default_topic_if_empty(cls, value: Any) -> list[str]:
        if value is None:
            return ["egyeb"]
        if isinstance(value, list) and len(value) == 0:
            return ["egyeb"]
        return value


class GeneratedPost(BaseModel):
    text: str = Field(min_length=10, max_length=280)
    hashtags: list[str] = Field(min_length=3, max_length=5)
    link: HttpUrl
    cta_used: bool

    @field_validator("text")
    @classmethod
    def validate_no_emoji(cls, value: str) -> str:
        if EMOJI_RE.search(value):
            raise ValueError("Post text must not contain emoji")
        return value


class PipelineResult(BaseModel):
    source_url: HttpUrl
    processed_at: datetime
    total_raw: int
    published_count: int
    duplicate_count: int
    failed_count: int
    items: list[dict[str, Any]]


def _require_api_key() -> None:
    if not MISTRAL_API_KEY:
        raise RuntimeError("Missing MISTRAL_API_KEY environment variable")


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _dedup_hash(title: str, link: str) -> str:
    payload = f"{title}|{link}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_and_mark_duplicate(title: str, source: str) -> bool:
    digest = _dedup_hash(title, source)
    with STATE_LOCK:
        if digest in DEDUP_CACHE:
            return True
        DEDUP_CACHE.add(digest)
    return False


def _fetch_url(url: str, timeout_seconds: int = 30) -> str:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def _parse_rss(source_name: str, source_url: str, xml_text: str) -> list[ParsedRawItem]:
    soup = BeautifulSoup(xml_text, "xml")
    parsed: list[ParsedRawItem] = []

    for item in soup.find_all("item"):
        title = _clean_text(item.title.text if item.title else "")
        link = _clean_text(item.link.text if item.link else source_url)
        summary = _clean_text(item.description.text if item.description else "")
        pub_date_raw = _clean_text(item.pubDate.text if item.pubDate else "")

        published_at: datetime | None = None
        if pub_date_raw:
            try:
                published_at = datetime.strptime(pub_date_raw, "%a, %d %b %Y %H:%M:%S %z")
            except ValueError:
                published_at = None

        if not title or not summary:
            continue

        try:
            parsed.append(
                ParsedRawItem(
                    source=source_name,
                    source_url=link,
                    title=title,
                    summary=summary,
                    published_at=published_at,
                    tags=[],
                )
            )
        except ValidationError as exc:
            LOGGER.warning("RSS item validation failed: %s", exc)

    return parsed


def _parse_html_drupal_node_hir(source_name: str, source_url: str, html_text: str) -> list[ParsedRawItem]:
    soup = BeautifulSoup(html_text, "html.parser")
    cards = soup.select(".node")
    parsed: list[ParsedRawItem] = []

    for card in cards:
        title_el = card.select_one(".node-title a")
        summary_el = card.select_one(".field-name-body p")

        if not title_el or not summary_el:
            continue

        title = _clean_text(title_el.get_text(" ", strip=True))
        summary = _clean_text(summary_el.get_text(" ", strip=True))
        href = _clean_text(title_el.get("href", ""))
        link = href if href.startswith("http") else source_url.rstrip("/") + "/" + href.lstrip("/")

        try:
            parsed.append(
                ParsedRawItem(
                    source=source_name,
                    source_url=link,
                    title=title,
                    summary=summary,
                    tags=[],
                )
            )
        except ValidationError as exc:
            LOGGER.warning("Drupal item validation failed: %s", exc)

    return parsed


def _parse_html_bme_news_card(source_name: str, source_url: str, html_text: str) -> list[ParsedRawItem]:
    soup = BeautifulSoup(html_text, "html.parser")
    cards = soup.select(".bme_news_card")
    parsed: list[ParsedRawItem] = []

    for card in cards:
        title_el = card.select_one(".bme_news_card-title")
        summary_el = card.select_one(".bme_news_card-body p")
        date_el = card.select_one(".field--name-created")
        tag_els = card.select(".field--name-field-tags li")
        link_el = card.select_one("a[href]")

        if not title_el or not summary_el or not link_el:
            continue

        title = _clean_text(title_el.get_text(" ", strip=True))
        summary = _clean_text(summary_el.get_text(" ", strip=True))
        href = _clean_text(link_el.get("href", ""))
        link = href if href.startswith("http") else source_url.rstrip("/") + "/" + href.lstrip("/")
        tags = [_clean_text(tag.get_text(" ", strip=True)) for tag in tag_els if _clean_text(tag.get_text())]

        published_at: datetime | None = None
        if date_el:
            date_raw = _clean_text(date_el.get_text(" ", strip=True))
            for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%d.%m.%Y"):
                try:
                    published_at = datetime.strptime(date_raw, fmt).replace(tzinfo=UTC)
                    break
                except ValueError:
                    continue

        try:
            parsed.append(
                ParsedRawItem(
                    source=source_name,
                    source_url=link,
                    title=title,
                    summary=summary,
                    published_at=published_at,
                    tags=tags,
                )
            )
        except ValidationError as exc:
            LOGGER.warning("BME card validation failed: %s", exc)

    return parsed


def _parse_html_auto(source_name: str, source_url: str, html_text: str) -> list[ParsedRawItem]:
    bme_items = _parse_html_bme_news_card(source_name, source_url, html_text)
    drupal_items = _parse_html_drupal_node_hir(source_name, source_url, html_text)

    merged: list[ParsedRawItem] = []
    seen_keys: set[str] = set()
    for item in [*bme_items, *drupal_items]:
        key = _dedup_hash(item.title, str(item.source_url))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(item)
    return merged


def _fetch_and_parse(source_name: str, source_url: str, source_type: Literal["rss", "html"]) -> list[ParsedRawItem]:
    body = _fetch_url(source_url)
    if source_type == "rss":
        return _parse_rss(source_name, source_url, body)
    return _parse_html_auto(source_name, source_url, body)


def _payload_to_raw_item(payload: UnifiedNewsPayload) -> ParsedRawItem:
    published_at = None
    if payload.news_date:
        published_at = datetime.combine(payload.news_date, datetime.min.time(), tzinfo=UTC)

    merged_tags = list(dict.fromkeys([*payload.news_topics, *payload.news_tags]))
    return ParsedRawItem(
        source="api",
        source_url=payload.news_source,
        title=payload.news_title,
        summary=payload.news_content,
        full_text=payload.news_content,
        published_at=published_at,
        tags=merged_tags,
    )


def _mistral_chat_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    _require_api_key()
    response = requests.post(
        f"{MISTRAL_API_BASE}/chat/completions",
        timeout=30,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MISTRAL_MODEL,
            "temperature": 0.4,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    return json.loads(content)


def _detect_event_with_llm(item: ParsedRawItem) -> EventInfo:
    system_prompt = (
        "Task: classify university news as event or not. Return JSON only: "
        '{"is_event": bool, "confidence": float}.'
    )
    user_prompt = (
        f"title: {item.title}\n"
        f"summary: {item.summary}\n"
        f"tags: {', '.join(item.tags) if item.tags else 'none'}"
    )
    result = _mistral_chat_json(system_prompt, user_prompt)
    event = EventInfo(**result)
    if event.confidence < 0.6:
        return EventInfo(is_event=False, confidence=event.confidence)
    return event


def _generate_post_with_llm(item: NewsItem) -> GeneratedPost:
    system_prompt = (
        "Generate a Hungarian social post for university news. Return JSON only: "
        '{"text": str, "hashtags": [str], "link": str, "cta_used": bool}. '
        "Rules: <=280 chars, no emoji, 3-5 hashtags, mandatory link, "
        "event=true requires CTA, event=false must not contain CTA."
    )
    user_prompt = (
        f"source: {item.source}\n"
        f"title: {item.title}\n"
        f"summary: {item.summary}\n"
        f"link: {item.source_url}\n"
        f"event: {item.event.is_event}"
    )
    result = _mistral_chat_json(system_prompt, user_prompt)
    post = GeneratedPost(**result)

    if item.event.is_event and not post.cta_used:
        raise ValueError("Event item requires CTA")
    if not item.event.is_event and post.cta_used:
        raise ValueError("Non-event item cannot use CTA")
    if str(post.link) != str(item.source_url):
        raise ValueError("Post link must match source_url")

    return post


def _publish_mock(post: str | GeneratedPost, platform: str = "mock-social") -> dict[str, Any]:
    post_payload: Any = post
    if isinstance(post, GeneratedPost):
        post_payload = post.model_dump(mode="json")

    publish_record = {
        "platform": platform,
        "published_at": datetime.now(tz=UTC).isoformat(),
        "post": post_payload,
        "status": "published",
        "id": "",
    }
    with STATE_LOCK:
        publish_record["id"] = f"mock_{len(PUBLISH_LOG) + 1}"
        PUBLISH_LOG.append(publish_record)
    return publish_record


def _to_simple_news_dict(item: ParsedRawItem) -> dict[str, Any]:
    return {
        "news_title": item.title,
        "news_content": item.summary,
        "news_source": str(item.source_url),
        "news_date": item.published_at.isoformat() if item.published_at else None,
        "news_topics": item.tags,
        "news_tags": item.tags,
    }


def _run_advanced_pipeline_item(
    raw: ParsedRawItem, platform: str
) -> tuple[dict[str, Any], Literal["published", "duplicate", "failed"]]:
    item_log: dict[str, Any] = {
        "title": raw.title,
        "source_url": str(raw.source_url),
        "status": "started",
        "steps": {},
    }

    try:
        if _check_and_mark_duplicate(raw.title, str(raw.source_url)):
            item_log["status"] = "skipped_duplicate"
            item_log["steps"]["dedup"] = "duplicate"
            return item_log, STATE_DUPLICATE
        item_log["steps"]["dedup"] = "ok"
    except Exception as exc:
        item_log["status"] = "failed"
        item_log["steps"]["dedup"] = f"error: {exc}"
        return item_log, STATE_FAILED

    try:
        event = _detect_event_with_llm(raw)
        item_log["steps"]["event_detection"] = event.model_dump()
    except Exception as exc:
        item_log["status"] = "failed"
        item_log["steps"]["event_detection"] = f"error: {exc}"
        return item_log, STATE_FAILED

    try:
        validated_news = NewsItem(
            source=raw.source,
            source_url=raw.source_url,
            title=raw.title,
            summary=raw.summary,
            full_text=raw.full_text,
            published_at=raw.published_at,
            tags=raw.tags,
            event=event,
        )
        item_log["steps"]["pydantic_validation"] = "ok"
    except Exception as exc:
        item_log["status"] = "failed"
        item_log["steps"]["pydantic_validation"] = f"error: {exc}"
        return item_log, STATE_FAILED

    try:
        post = _generate_post_with_llm(validated_news)
        item_log["steps"]["post_generation"] = post.model_dump(mode="json")
    except Exception as exc:
        item_log["status"] = "failed"
        item_log["steps"]["post_generation"] = f"error: {exc}"
        return item_log, STATE_FAILED

    try:
        publish_result = _publish_mock(post, platform=platform)
        item_log["status"] = "published"
        item_log["steps"]["publish"] = publish_result
        return item_log, STATE_PUBLISHED
    except Exception as exc:
        item_log["status"] = "failed"
        item_log["steps"]["publish"] = f"error: {exc}"
        return item_log, STATE_FAILED


mcp = FastMCP("university-news-pipeline")


# -------- Simple tools from src/server.py --------


@mcp.tool()
def fetch_and_parse(url: str) -> dict[str, Any]:
    try:
        html = _fetch_url(url, timeout_seconds=20)
        parsed = _parse_html_auto("web", url, html)
        if not parsed:
            return {"error": "Nem talalhato tamogatott HTML template a lapon."}

        items = [_to_simple_news_dict(item) for item in parsed]
        first = items[0]
        return {
            **first,
            "items": items,
            "count": len(items),
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def validate_news(data: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = UnifiedNewsPayload(**data)
        return payload.model_dump(mode="json")
    except ValidationError as exc:
        return {"error": exc.errors()}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def deduplicate(data: dict[str, Any]) -> dict[str, Any]:
    try:
        title = str(data.get("news_title", ""))
        source = str(data.get("news_source", ""))

        if _check_and_mark_duplicate(title, source):
            return {"duplicate": True}
        return {"duplicate": False, "data": data}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def detect_event(data: dict[str, Any]) -> dict[str, Any]:
    try:
        text = f"{data.get('news_title', '')} {data.get('news_content', '')}".lower()
        keywords = ["esemeny", "konferencia", "workshop", "eloadas", "előadás", "esemény"]
        hit = any(keyword in text for keyword in keywords)
        confidence = 0.8 if hit else 0.3
        is_event = hit and confidence >= 0.6
        return {"event": {"is_event": is_event, "confidence": confidence}}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def generate_post(data: dict[str, Any]) -> dict[str, Any]:
    try:
        title = str(data.get("news_title", "")).strip()
        content = str(data.get("news_content", "")).strip()
        source = str(data.get("news_source", "")).strip()
        topics = data.get("news_topics", []) or []

        hashtags = []
        for topic in topics[:3]:
            normalized = str(topic).strip().replace(" ", "")
            if not normalized:
                continue
            if not normalized.startswith("#"):
                normalized = f"#{normalized}"
            hashtags.append(normalized)

        summary = content[:120].rstrip()
        post = f"{title} - {summary}... {' '.join(hashtags)} {source}".strip()
        if len(post) > 280:
            post = post[:277].rstrip() + "..."

        return {"post": post}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def publish_post(post: str) -> dict[str, Any]:
    try:
        _publish_mock(post)
        return {"status": "published"}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def process_news(url: str) -> dict[str, Any]:
    try:
        step_1 = fetch_and_parse(url)
        if "error" in step_1:
            return step_1

        items = step_1.get("items", [])
        if not items:
            return {"error": "Nincs feldolgozhato hirelem."}

        results: list[dict[str, Any]] = []
        published_count = 0
        duplicate_count = 0
        failed_count = 0

        for item in items:
            step_2 = validate_news(item)
            if "error" in step_2:
                failed_count += 1
                results.append({"status": "failed", "step": "validate_news", "error": step_2["error"], "item": item})
                continue

            step_3 = deduplicate(step_2)
            if "error" in step_3:
                failed_count += 1
                results.append({"status": "failed", "step": "deduplicate", "error": step_3["error"], "item": step_2})
                continue
            if step_3.get("duplicate") is True:
                duplicate_count += 1
                results.append({"status": "skipped_duplicate", "title": step_2.get("news_title")})
                continue

            step_4 = detect_event(step_3["data"])
            if "error" in step_4:
                failed_count += 1
                results.append({"status": "failed", "step": "detect_event", "error": step_4["error"], "item": step_3["data"]})
                continue

            payload = dict(step_3["data"])
            payload["event"] = step_4["event"]

            step_5 = generate_post(payload)
            if "error" in step_5:
                failed_count += 1
                results.append({"status": "failed", "step": "generate_post", "error": step_5["error"], "item": payload})
                continue

            step_6 = publish_post(step_5["post"])
            if "error" in step_6:
                failed_count += 1
                results.append({"status": "failed", "step": "publish_post", "error": step_6["error"], "post": step_5["post"]})
                continue

            published_count += 1
            results.append(
                {
                    "status": "ok",
                    "event": step_4["event"],
                    "post": step_5["post"],
                    "publish": step_6,
                }
            )

        return {
            "status": "ok",
            "total_items": len(items),
            "published_count": published_count,
            "duplicate_count": duplicate_count,
            "failed_count": failed_count,
            "items": results,
        }
    except Exception as exc:
        return {"error": str(exc)}


# -------- Advanced tools from src/fastmcp.py --------


@mcp.tool()
def process_news_pipeline(
    source_name: str,
    source_url: str,
    source_type: Literal["rss", "html"],
    limit: int = 10,
    platform: str = "mock-social",
) -> dict[str, Any]:
    items_result: list[dict[str, Any]] = []
    published_count = 0
    duplicate_count = 0
    failed_count = 0

    try:
        raw_items = _fetch_and_parse(source_name, source_url, source_type)
    except Exception as exc:
        return {
            "status": "failed",
            "step": "scraping",
            "error": str(exc),
            "source_url": source_url,
        }

    for raw in raw_items[: max(0, limit)]:
        item_log, state = _run_advanced_pipeline_item(raw, platform=platform)
        items_result.append(item_log)

        if state == STATE_PUBLISHED:
            published_count += 1
        elif state == STATE_DUPLICATE:
            duplicate_count += 1
        else:
            failed_count += 1

    try:
        result = PipelineResult(
            source_url=source_url,
            processed_at=datetime.now(tz=UTC),
            total_raw=len(raw_items),
            published_count=published_count,
            duplicate_count=duplicate_count,
            failed_count=failed_count,
            items=items_result,
        )
        return result.model_dump(mode="json")
    except Exception as exc:
        return {
            "status": "failed",
            "step": "pipeline_result",
            "error": str(exc),
            "items": items_result,
        }


@mcp.tool()
def process_news_payload(payload: dict[str, Any], platform: str = "mock-social") -> dict[str, Any]:
    try:
        api_payload = UnifiedNewsPayload(**payload)
    except Exception as exc:
        return {"status": "failed", "step": "api_validation", "error": str(exc)}

    try:
        raw = _payload_to_raw_item(api_payload)
    except Exception as exc:
        return {"status": "failed", "step": "mapping", "error": str(exc)}

    item_log, _state = _run_advanced_pipeline_item(raw, platform=platform)
    # Keep api-specific steps visible for this endpoint.
    item_log.setdefault("steps", {})
    item_log["steps"]["api_validation"] = "ok"
    item_log["steps"]["mapping"] = "ok"
    return item_log


@mcp.tool()
def get_api_contract() -> dict[str, Any]:
    return {
        "input_schema": {
            "news_title": "str",
            "news_date": "YYYY-MM-DD | null",
            "news_content": "str",
            "news_topics": ["str"],
            "news_source": "https://...",
            "news_tags": ["str"],
        },
        "pipeline_tools": ["process_news", "process_news_payload", "process_news_pipeline"],
    }


@mcp.tool()
def get_runtime_state() -> dict[str, Any]:
    return {
        "dedup_cache_size": len(DEDUP_CACHE),
        "published_size": len(PUBLISH_LOG),
        "last_published": PUBLISH_LOG[-1] if PUBLISH_LOG else None,
        "mistral_model": MISTRAL_MODEL,
        "mistral_api_base": MISTRAL_API_BASE,
    }


if __name__ == "__main__":
    mcp.run()
