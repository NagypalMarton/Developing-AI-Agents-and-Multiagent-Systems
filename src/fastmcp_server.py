 
import re
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator, HttpUrl
from bs4 import BeautifulSoup, Tag
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

FACEBOOK_MAX_LENGTH = 63206
LINKEDIN_HEADLINE_MAX = 200
X_MAX_LENGTH = 280
INSTAGRAM_MAX_LENGTH = 2200
DISCORD_DESCRIPTION_MAX = 2048
DISCORD_TITLE_MAX = 256
DISCORD_DEFAULT_COLOR = 3447003

# HTML Parser config
HTML_PARSER_CONFIG = {
    "features": "html.parser",
    "from_encoding": "utf-8"
}

# HTML Selectors configuration - centralized for easy maintenance
HTML_SELECTORS = {
    "tmit": {
        "article": {"tag": "article", "class": "node-hir"},
        "title": {"tag": "h2", "class": "node-title"},
        "content": {"tag": "div", "attrs": {"property": "content:encoded"}},
        "image": {"tag": "img"}
    },
    "vik": {
        "title": {"tag": "h2", "class": "news-title-important"},
        "date": {"tag": "span", "class": "news-date"},
        "excerpt": {"tag": "div", "class": "news-excerpt"},
        "image": {"tag": "img", "class": "news-image"},
        "parent": {"tag": "div", "class": "news-item"}
    },
    "bme_news": {
        "card": {"tag": "div", "class": "bme_news_card"},
        "title": {"tag": "h4", "class": "bme_news_card-title"},
        "date": {"tag": "span", "class": "field--name-created"},
        "body": {"tag": "div", "class": "bme_news_card-body"},
        "image": {"tag": "img"}
    },
    "bme_event": {
        "title": {"tag": "h4", "class": "bme_event_card-title"},
        "date_container": {"tag": "div", "class": "bme_event_card-date"},
        "date": {"tag": "span", "class": "nowrap"},
        "location": {"tag": "p", "class": "bme_event_card-location"},
        "body": {"tag": "div", "class": "bme_event_card-body"},
        "parent": {"tag": "div", "class": "px-5"}
    },
    "simple_event": {
        "date": {"tag": "span", "class": "event-date"},
        "title": {"tag": "a", "class": "event-title"},
        "container": {"tag": "div", "class": "event"},
        "participants": {"tag": "p", "class": "event-participants"}
    }
}

# Social templates (use format placeholders: {title}, {content}, {url})
SOCIAL_TEMPLATES = {
    "facebook": "Kedves közösség! 🎓\n\n{title}\n\n{content}\n\nTudj meg többet: {url}\n",
    "linkedin": "{title}\n\n{content}\n\n📌 Tudj meg többet:\n{url}\n\n#BME #Hírek #Oktatás\n",
    "x": "{title}\n\n{content_slice}...\n\n{url}",
    "instagram": "{title}\n.\n{content}\n\n#BME #Egyetem #Hírek",
    "discord": {
        "title": "{title}",
        "description": "{content}",
        "url": "{url}"
    }
}

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class EventType(str, Enum):
    LECTURE = "lecture"
    WORKSHOP = "workshop"
    CONFERENCE = "conference"
    DEADLINE = "deadline"
    CONCERT = "concert"
    DOCTORAL_DEFENSE = "doctoral_defense"
    OTHER = "other"

class PlatformType(str, Enum):
    """Supported social media platforms"""
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    X = "x"
    INSTAGRAM = "instagram"
    DISCORD = "discord"

class SocialMediaPosts(BaseModel):
    """Social media posts for all platforms"""
    facebook: Optional[str] = None
    linkedin: Optional[str] = None
    x: Optional[str] = None
    instagram: Optional[str] = None
    discord: Optional[str] = None

class HTMLParseRequest(BaseModel):
    """Validated HTML parsing request"""
    html_content: str = Field(..., min_length=1, description="HTML tartalom")
    source_url: HttpUrl = Field(..., description="Érvényes URL cím")
    
    @field_validator("html_content")
    @classmethod
    def validate_html_not_empty(cls, v: str) -> str:
        """Ensure HTML content is not just whitespace"""
        if not v or not v.strip():
            raise ValueError("HTML tartalom nem lehet üres")
        return v

class ParseHTMLResponse(BaseModel):
    """Response model for parse_html_and_extract_news"""
    news_count: int
    event_count: int
    news_items: List[Dict[str, Any]]
    events_items: List[Dict[str, Any]]
    status: str
    error: Optional[str] = None

class EventDetected(BaseModel):
    """Detektált esemény - minden mező kötelező"""
    title: str = Field(..., description="Esemény címe")
    date: str = Field(..., description="Esemény dátuma (YYYY.MMM.DD. formátumban)")
    location: str = Field(..., description="Esemény helyszíne")
    event_type: EventType = Field(..., description="Esemény típusa")
    content: str = Field(..., description="Esemény tartalma/leírása")
    registration_url: str = Field(..., description="Regisztrációs URL")
    source_url: str = Field(..., description="Forrás URL")
    social_posts: SocialMediaPosts = Field(..., description="Social media posztok")
    participants: List[str] = Field(..., description="Résztvevők listája")

class NewsItem(BaseModel):
    """Kinyert hírelemek"""
    title: str
    content: str
    image_url: Optional[str] = None
    registration_url: Optional[str] = None
    source_url: str
    publish_date: Optional[str] = None
    events: List[EventDetected] = Field(default_factory=list)
    social_posts: Optional[SocialMediaPosts] = Field(None, description="Social media posztok")

# NOTE: Social post models (SocialPostFacebook, etc.) are generated dynamically as dicts
# in generate_social_posts() function. Legacy Pydantic models removed (code smell #3).

# ============================================================================
# HELPER FUNCTIONS - Common parsing utilities
# ============================================================================

def create_error_response(status: str, error: str, news_items: Optional[List] = None, events: Optional[List] = None) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "status": status,
        "error": error,
        "news_items": news_items or [],
        "events_items": events or [],
        "news_count": 0,
        "event_count": 0
    }

def create_success_response(news_items: List, events: List) -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        "status": "success",
        "news_items": news_items,
        "events_items": events,
        "news_count": len(news_items),
        "event_count": len(events)
    }

# ============================================================================
# EXTRACTION HELPERS - Code smell #4 refactoring
# ============================================================================

def safe_extract_news(soup, source_url: str, extract_func: Callable, selector_key: str, log_name: str) -> List[Dict[str, Any]]:
    """Safe extraction wrapper with error handling and logging"""
    news_items = []
    try:
        sel = HTML_SELECTORS[selector_key]
        container_selector = sel.get("article") or sel.get("parent") or sel.get("card") or sel.get("container")
        if not container_selector:
            raise KeyError(f"No container selector configured for {selector_key}")

        class_name = container_selector.get("class") or ""
        items = soup.select(f"{container_selector['tag']}.{class_name}") if class_name else soup.select(container_selector['tag'])
        for item in items:
            html_str = str(item)
            news = extract_func(html_str, source_url)
            if news:
                news_items.append(news.model_dump(mode="json", exclude_none=False))
        logger.info(f"Found {len(items)} {log_name}")
    except (AttributeError, TypeError, ValueError, KeyError) as e:
        logger.error(f"Error parsing {log_name}: {e}")
    return news_items

def _extract_tmit_news(soup, source_url: str) -> List[Dict[str, Any]]:
    """Extract TMIT news articles from soup"""
    return safe_extract_news(soup, source_url, parse_tmit_news, "tmit", "TMIT articles")

def _extract_vik_news(soup, source_url: str) -> List[Dict[str, Any]]:
    """Extract VIK news from soup"""
    return safe_extract_news(soup, source_url, parse_vik_news, "vik", "VIK news items")

def _extract_bme_news(soup, source_url: str) -> List[Dict[str, Any]]:
    """Extract BME news cards from soup"""
    return safe_extract_news(soup, source_url, parse_bme_news, "bme_news", "BME news cards")

def _extract_bme_events(soup, source_url: str) -> List[Dict[str, Any]]:
    """Extract BME events from soup"""
    events = []
    try:
        sel = HTML_SELECTORS["bme_event"]
        dc = sel["date_container"]
        dc_class = dc.get("class") or ""
        bme_events = soup.select(f"{dc['tag']}.{dc_class}") if dc_class else soup.select(dc['tag'])
        for event_card in bme_events:
            parent = event_card.find_parent(sel["parent"]["tag"]) if sel.get("parent") else None
            if parent and sel.get("parent"):
                parent_classes = parent.get("class", [])
                if sel["parent"].get("class") not in parent_classes:
                    parent = None
            if parent:
                html_str = str(parent)
                event = parse_bme_event(html_str, source_url)
                if event:
                    events.append(event.model_dump(mode="json", exclude_none=False))
        logger.info(f"Found {len(bme_events)} BME events")
    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Error parsing BME events: {e}")
    return events

def _extract_simple_events(soup, source_url: str) -> List[Dict[str, Any]]:
    """Extract simple events from soup"""
    events = []
    try:
        sel = HTML_SELECTORS["simple_event"]
        c = sel["container"]
        c_class = c.get("class") or ""
        simple_events = soup.select(f"{c['tag']}.{c_class}") if c_class else soup.select(c['tag'])
        for event_elem in simple_events:
            html_str = str(event_elem)
            event = parse_simple_event(html_str, source_url)
            if event:
                events.append(event.model_dump(mode="json", exclude_none=False))
        logger.info(f"Found {len(simple_events)} simple events")
    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Error parsing simple events: {e}")
    return events

# ============================================================================
# PARSING HELPER FUNCTIONS
# ============================================================================

def safe_find_text(element, tag: str, class_name: Optional[str] = None, attrs: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Safely extract and strip text from BeautifulSoup element using CSS selectors."""
    if element is None:
        return None
    try:
        selector = tag
        if class_name:
            selector = f"{tag}.{class_name}"
        elif attrs:
            # build attribute selector for the first attr
            parts = [f'[{k}="{v}"]' for k, v in attrs.items()]
            selector = tag + ''.join(parts)

        found = element.select_one(selector)
        if found:
            text = found.get_text(strip=True)
            return text if text else None
        return None
    except (AttributeError, TypeError) as e:
        logger.warning(f"Error extracting text: {e}")
        return None

def safe_get_attr(element, attr: str) -> Optional[str]:
    """Safely extract attribute from BeautifulSoup element"""
    if element is None:
        return None
    try:
        return element.get(attr)
    except (AttributeError, TypeError) as e:
        logger.warning(f"Error getting attribute {attr}: {e}")
        return None

def validate_url_safety(url: Optional[str]) -> Optional[str]:
    """Validate and sanitize URL to prevent XSS"""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            logger.warning(f"Invalid URL scheme: {parsed.scheme}")
            return None
        return url
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"URL validation failed: {e}")
        return None


def extract_registration_url_from_text(*text_candidates: Optional[str]) -> Optional[str]:
    """Extract the most likely registration URL from visible text or raw HTML."""
    parts = [text.strip() for text in text_candidates if isinstance(text, str) and text.strip()]
    if not parts:
        return None

    combined_text = "\n".join(parts)

    hint_match = REGEX_REGISTRATION_HINT.search(combined_text)
    if hint_match:
        hinted_url = validate_url_safety(hint_match.group(1).rstrip(').,;:!>\"\''))
        if hinted_url:
            return hinted_url

    for url_match in REGEX_URL.finditer(combined_text):
        candidate_url = validate_url_safety(url_match.group(0).rstrip(').,;:!>\"\''))
        if candidate_url:
            return candidate_url

    return None

# Pre-compiled regex patterns
REGEX_HUNGARIAN_DATE = re.compile(r'(\d{4})\D+(\d{1,2})\D+(\d{1,2})')
REGEX_HUMAN_DATE = re.compile(r'(\d{4})[.\-/\s]+([A-Za-z]{3,9})[.\-/\s]+(\d{1,2})(?:\.)?')
REGEX_TIME = re.compile(r'(\d{1,2}):(\d{2})')
REGEX_URL = re.compile(r'https?://[^\s<>"\'\]]+', re.IGNORECASE)
REGEX_REGISTRATION_HINT = re.compile(
    r'(?:regisztráci(?:ó|os)|jelentkez(?:és|esi)|register|registration|sign\s*up)'
    r'[^\n]{0,120}?(https?://[^\s<>"\'\]]+)',
    re.IGNORECASE,
)

MONTH_ABBREVIATIONS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _format_human_date(date_value: datetime) -> str:
    return f"{date_value.year}.{MONTH_ABBREVIATIONS[date_value.month]}.{date_value.day:02d}."


# Helper to create BeautifulSoup consistently
def make_soup(html_content: str) -> BeautifulSoup:
    return BeautifulSoup(html_content, features=HTML_PARSER_CONFIG.get("features", "html.parser"))


def _month_name_to_number(month_name: str) -> Optional[int]:
    normalized = month_name.strip().lower().rstrip(".")
    return MONTH_NAME_TO_NUMBER.get(normalized)


def _build_human_date_string(year: int, month: int, day: int) -> Optional[str]:
    try:
        return _format_human_date(datetime(year, month, day))
    except ValueError as e:
        logger.warning(f"Invalid date components: {year}-{month}-{day}: {e}")
        return None

# ============================================================================
# HTML PARSER UTILS
# ============================================================================

def parse_tmit_news(html_content: str, source_url: str) -> Optional[NewsItem]:
    """TMIT (node-hir) típusú hír parsése"""
    try:
        sel = HTML_SELECTORS["tmit"]
        if isinstance(html_content, Tag):
            article = html_content
        else:
            soup = make_soup(html_content)
            art_class = sel["article"].get("class") or ""
            article = soup.select_one(f"{sel['article']['tag']}.{art_class}") if art_class else soup.select_one(sel["article"]["tag"])
        if not article:
            logger.debug("TMIT article not found")
            return None

        t_class = sel["title"].get("class") or ""
        title_elem = article.select_one(f"{sel['title']['tag']}.{t_class}") if t_class else article.select_one(sel["title"]["tag"])
        title = title_elem.get_text(strip=True) if title_elem else "N/A"
        image_elem = article.find(sel["image"]["tag"]) if sel.get("image") else None
        image_url = validate_url_safety(safe_get_attr(image_elem, 'src'))
        # content may use attribute selectors
        content_attrs = sel["content"].get("attrs") if sel.get("content") else None
        if content_attrs:
            # take first attribute
            k, v = next(iter(content_attrs.items()))
            content_elem = article.select_one(f"{sel['content']['tag']}[{k}='{v}']")
        else:
            content_elem = article.select_one(sel["content"]["tag"]) if sel.get("content") else None
        content = content_elem.get_text(strip=True) if content_elem else ""
        raw_text = article.get_text(separator=" ", strip=True)
        registration_url = extract_registration_url_from_text(content, raw_text, str(article))
        
        return NewsItem(
            title=title,
            content=content,
            image_url=image_url,
            registration_url=registration_url,
            source_url=str(source_url),
            social_posts=None
        )
    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Error parsing TMIT news: {e}")
        return None

def parse_vik_news(html_content: str, source_url: str) -> Optional[NewsItem]:
    """VIK (news-title-important) típusú hír parsése"""
    try:
        sel = HTML_SELECTORS["vik"]
        if isinstance(html_content, Tag):
            container = html_content
        else:
            container = make_soup(html_content)

        t_class = sel["title"].get("class") or ""
        title_elem = container.select_one(f"{sel['title']['tag']}.{t_class}") if t_class else container.select_one(sel["title"]["tag"])
        if not title_elem:
            logger.debug("VIK title not found")
            return None

        title = title_elem.get_text(strip=True) or "N/A"
        publish_date = safe_find_text(container, sel["date"]["tag"], class_name=sel["date"].get("class"))
        content = safe_find_text(container, sel["excerpt"]["tag"], class_name=sel["excerpt"].get("class")) or ""
        i_class = sel["image"].get("class") if sel.get("image") else None
        image_elem = container.select_one(f"{sel['image']['tag']}.{i_class}") if i_class else container.select_one(sel["image"]["tag"]) if sel.get("image") else None
        image_url = validate_url_safety(safe_get_attr(image_elem, 'src'))
        raw_text = container.get_text(separator=" ", strip=True)
        registration_url = extract_registration_url_from_text(content, raw_text, str(container))
        
        return NewsItem(
            title=title,
            content=content,
            image_url=image_url,
            registration_url=registration_url,
            source_url=str(source_url),
            publish_date=publish_date,
            social_posts=None
        )
    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Error parsing VIK news: {e}")
        return None

def parse_bme_news(html_content: str, source_url: str) -> Optional[NewsItem]:
    """BME (bme_news_card) típusú hír parsése"""
    try:
        sel = HTML_SELECTORS["bme_news"]
        if isinstance(html_content, Tag):
            news_card = html_content
        else:
            container = make_soup(html_content)
            c_class = sel["card"].get("class") or ""
            news_card = container.select_one(f"{sel['card']['tag']}.{c_class}") if c_class else container.select_one(sel["card"]["tag"])
        if not news_card:
            logger.debug("BME news card not found")
            return None
        
        title = safe_find_text(news_card, sel["title"]["tag"], class_name=sel["title"].get("class")) or "N/A"
        publish_date = safe_find_text(news_card, sel["date"]["tag"], class_name=sel["date"].get("class"))
        content = safe_find_text(news_card, sel["body"]["tag"], class_name=sel["body"].get("class")) or ""
        image_elem = news_card.find(sel["image"]["tag"])
        image_url = validate_url_safety(safe_get_attr(image_elem, 'src'))
        raw_text = news_card.get_text(separator=" ", strip=True)
        registration_url = extract_registration_url_from_text(content, raw_text, str(news_card))
        
        return NewsItem(
            title=title,
            content=content,
            image_url=image_url,
            registration_url=registration_url,
            source_url=str(source_url),
            publish_date=publish_date,
            social_posts=None
        )
    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Error parsing BME news: {e}")
        return None

def parse_bme_event(html_content: str, source_url: str) -> Optional[EventDetected]:
    """BME event card parsése"""
    try:
        sel = HTML_SELECTORS["bme_event"]
        if isinstance(html_content, Tag):
            container = html_content
        else:
            container = make_soup(html_content)

        title = safe_find_text(container, sel["title"]["tag"], class_name=sel["title"].get("class")) or "N/A"
        dc = sel.get("date_container")
        if dc:
            dc_class = dc.get("class") or ""
            date_elem = container.select_one(f"{dc['tag']}.{dc_class}") if dc_class else container.select_one(dc['tag'])
        else:
            date_elem = None
        date_str = safe_find_text(date_elem, sel["date"]["tag"], class_name=sel["date"].get("class")) or ""
        location = safe_find_text(container, sel["location"]["tag"], class_name=sel["location"].get("class")) or "MISSING"
        content = safe_find_text(container, sel["body"]["tag"], class_name=sel["body"].get("class")) or "MISSING"
        raw_text = container.get_text(separator=" ", strip=True)

        # Extract registration link from the visible text / raw HTML source
        reg_url = extract_registration_url_from_text(content, raw_text, str(container)) or "MISSING"
        participants = ["MISSING"]

        part_sel = sel.get("participants")
        if part_sel:
            p_class = part_sel.get("class") or ""
            part_elem = container.select_one(f"{part_sel['tag']}.{p_class}") if p_class else container.select_one(part_sel['tag'])
            if part_elem:
                text = part_elem.get_text(separator=",", strip=True)
                participants = [p.strip() for p in text.split(",") if p.strip()] or ["MISSING"]
        
        event_type = detect_event_type(title)
        
        return EventDetected(
            title=title,
            date=parse_date_string(date_str),
            location=location,
            event_type=event_type,
            content=content,
            registration_url=reg_url,
            source_url=str(source_url),
            social_posts=SocialMediaPosts(),
            participants=participants
        )
    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Error parsing BME event: {e}")
        return None

def parse_simple_event(html_content: str, source_url: str) -> Optional[EventDetected]:
    """Egyszerű event formátum parsése (VIK)"""
    try:
        sel = HTML_SELECTORS["simple_event"]
        if isinstance(html_content, Tag):
            container = html_content
        else:
            container = make_soup(html_content)

        date_str = safe_find_text(container, sel["date"]["tag"], class_name=sel["date"].get("class")) or ""
        title = safe_find_text(container, sel["title"]["tag"], class_name=sel["title"].get("class")) or "N/A"

        # registration and participants
        reg_url = "MISSING"
        participants = ["MISSING"]
        reg_sel = sel.get("registration")
        if reg_sel:
            r_class = reg_sel.get("class") or ""
            reg_elem = container.select_one(f"{reg_sel['tag']}.{r_class}") if r_class else container.select_one(reg_sel['tag'])
            reg_href = safe_get_attr(reg_elem, "href") if reg_elem is not None else None
            reg_url = validate_url_safety(reg_href) or "MISSING"

        part_sel = sel.get("participants")
        if part_sel:
            p_class = part_sel.get("class") or ""
            part_elem = container.select_one(f"{part_sel['tag']}.{p_class}") if p_class else container.select_one(part_sel['tag'])
            if part_elem:
                text = part_elem.get_text(separator=",", strip=True)
                participants = [p.strip() for p in text.split(",") if p.strip()] or ["MISSING"]

        return EventDetected(
            title=title,
            date=parse_date_string(date_str),
            location="MISSING",
            event_type=EventType.OTHER,
            content="MISSING",
            registration_url=reg_url,
            source_url=str(source_url),
            social_posts=SocialMediaPosts(),
            participants=participants
        )
    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Error parsing simple event: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_date_string(date_str: str) -> str:
    """Dátum stringet YYYY.MMM.DD. formátumra alakít - ROBUST ERROR HANDLING"""
    if not date_str or not isinstance(date_str, str):
        logger.warning(f"Invalid date_str: {date_str}")
        return _format_human_date(datetime.now())
    
    try:
        human_match = REGEX_HUMAN_DATE.search(date_str)
        if human_match:
            year_str, month_name, day_str = human_match.groups()
            month_number = _month_name_to_number(month_name)
            if month_number:
                result = _build_human_date_string(int(year_str), month_number, int(day_str))
                if result:
                    return result

        date_match = REGEX_HUNGARIAN_DATE.search(date_str)
        if date_match:
            year_str, month_str, day_str = date_match.groups()
            result = _build_human_date_string(int(year_str), int(month_str), int(day_str))
            if result:
                return result

        logger.debug(f"No date pattern found in: {date_str}")
        return _format_human_date(datetime.now())
        
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Date parsing validation failed for '{date_str}': {e}")
        return _format_human_date(datetime.now())

def detect_event_type(title: str) -> EventType:
    """Eseménytípus detektálása a címből - TYPE SAFE"""
    if not title or not isinstance(title, str):
        return EventType.OTHER

    title_lower = title.lower()

    keywords_map = {
        EventType.DOCTORAL_DEFENSE: ['doktori', 'védés', 'phd'],
        EventType.CONCERT: ['koncert', 'zene', 'szimfónia'],
        EventType.CONFERENCE: ['konferencia', 'symposium'],
        EventType.WORKSHOP: ['workshop', 'tanfolyam', 'képzés'],
        EventType.LECTURE: ['előadás', 'lecture', 'diasor'],
        EventType.DEADLINE: ['határidő', 'deadline', 'pályázat'],
    }

    for event_type, keywords in keywords_map.items():
        if any(keyword in title_lower for keyword in keywords):
            return event_type

    return EventType.OTHER

# ============================================================================
# MCP SERVER SETUP
# ============================================================================

mcp = FastMCP("news-to-social-agent")
mcp.settings.streamable_http_path = "/sse"
mcp.settings.transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "127.0.0.1:*",
        "localhost:*",
        "[::1]:*",
        "fastmcp-server:*",
    ],
    allowed_origins=[
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
        "http://fastmcp-server:*",
    ],
)

mcp_app = mcp.streamable_http_app()

@mcp.tool()
async def parse_html_and_extract_news(html_content: str, source_url: str) -> Dict[str, Any]:
    """
    HTML feldolgozás és hírek + események extraktálása.
    
    Támogatott formátumok:
    - TMIT: <article class="node-hir">
    - VIK: <h2 class="news-title-important">
    - BME: <div class="bme_news_card">
    - Events: BME event cards, simple events
    
    Raises:
    - ValueError: Ha az input nem megfelelő
    - Exception: Az egy-egy parser-specifikus hibákat loggol, nem dobja
    """
    try:
        # Input validation
        if not html_content or not html_content.strip():
            logger.warning("Empty HTML content provided")
            return {
                "news_count": 0,
                "event_count": 0,
                "news_items": [],
                "events_items": [],
                "status": "error",
                "error": "HTML tartalom nem lehet üres"
            }
        
        if not source_url:
            logger.warning("Missing source URL")
            return {
                "news_count": 0,
                "event_count": 0,
                "news_items": [],
                "events_items": [],
                "status": "error",
                "error": "Source URL szükséges"
            }
        
        soup = make_soup(html_content)
        
        # Extract news and events using helper functions
        tmit_news = _extract_tmit_news(soup, source_url)
        vik_news = _extract_vik_news(soup, source_url)
        bme_news = _extract_bme_news(soup, source_url)
        bme_events = _extract_bme_events(soup, source_url)
        simple_events = _extract_simple_events(soup, source_url)
        
        # Combine all results
        news_items = tmit_news + vik_news + bme_news
        events = bme_events + simple_events
        
        logger.info(f"Total extracted: {len(news_items)} news, {len(events)} events")
        return {
            "news_count": len(news_items),
            "event_count": len(events),
            "news_items": news_items,
            "events_items": events,
            "status": "success"
        }
    
    except ValueError as e:
        logger.error(f"Validation error in parse_html_and_extract_news: {e}")
        return {
            "status": "error",
            "error": f"Validation error: {str(e)}",
            "news_items": [],
            "events_items": [],
            "news_count": 0,
            "event_count": 0
        }
    except Exception as e:
        logger.exception(f"Unexpected error in parse_html_and_extract_news: {e}")
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "news_items": [],
            "events_items": [],
            "news_count": 0,
            "event_count": 0
        }

@mcp.tool()
async def generate_social_posts(
    news_title: str,
    news_content: str,
    source_url: str,
    events: Optional[List[Dict[str, Any]]] = None,
    platforms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Platform-specifikus szociálmédiai posztok generálása.
    
    Az n8n Mistral LLM-ét hívja meg a konkrét szöveg generálásához,
    ez a tool csak a Pydantic szerkezeteket validálja.
    
    platforms: ["facebook", "linkedin", "x", "instagram", "discord"]
    
    Raises:
    - ValueError: Ha input nem megfelelő
    """
    try:
        # Input validation
        if not news_title or not isinstance(news_title, str):
            logger.warning("Invalid news_title for post generation")
            raise ValueError("news_title must be non-empty string")
        
        if not news_content or not isinstance(news_content, str):
            logger.warning("Invalid news_content for post generation")
            raise ValueError("news_content must be non-empty string")
        
        if not source_url or not isinstance(source_url, str):
            logger.warning("Invalid source_url for post generation")
            raise ValueError("source_url must be non-empty string")
        
        # Validate URL
        validated_source_url = validate_url_safety(source_url)
        if not validated_source_url:
            logger.warning("Source URL failed validation")
            raise ValueError("Invalid source URL")
        
        if not platforms:
            platforms = ["facebook", "linkedin", "x", "instagram", "discord"]
        
        if not events:
            events = []

        news_registration_url = extract_registration_url_from_text(news_content, news_title)
        event_registration_url = events[0].get("registration_url") if events else None
        resolved_registration_url = validate_url_safety(event_registration_url or news_registration_url)
        
        # Validate platform names
        valid_platforms = {p.value for p in PlatformType}
        for platform in platforms:
            if platform not in valid_platforms:
                logger.warning(f"Invalid platform: {platform}")
                raise ValueError(f"Invalid platform: {platform}")
        
        # Fill templates from SOCIAL_TEMPLATES
        templates = {}
        for key, val in SOCIAL_TEMPLATES.items():
            if isinstance(val, dict):
                templates[key] = {k: v.format(title=news_title, content=news_content, url=validated_source_url) for k, v in val.items()}
            else:
                if key == "x":
                    templates[key] = val.format(title=news_title, content=news_content, content_slice=news_content[:200], url=validated_source_url)
                else:
                    templates[key] = val.format(title=news_title, content=news_content, url=validated_source_url)
        
        result = {}
        
        try:
            if "facebook" in platforms:
                result["facebook"] = {
                    "content": templates["facebook"][:FACEBOOK_MAX_LENGTH],
                    "hashtags": ["#BME", "#Hírek"],
                    "cta_button": "Tudj meg többet" if resolved_registration_url else None,
                    "cta_url": resolved_registration_url
                }
            
            if "linkedin" in platforms:
                result["linkedin"] = {
                    "headline": news_title[:LINKEDIN_HEADLINE_MAX],
                    "body": templates["linkedin"],
                    "hashtags": ["BME", "Hírek", "Oktatás"],
                    "cta_text": "Regisztrálj",
                    "cta_url": resolved_registration_url or validated_source_url
                }
            
            if "x" in platforms:
                result["x"] = {
                    "content": templates["x"][:X_MAX_LENGTH],
                    "hashtags": ["BME", "Hírek"]
                }
            
            if "instagram" in platforms:
                result["instagram"] = {
                    "caption": templates["instagram"][:INSTAGRAM_MAX_LENGTH],
                    "hashtags": ["BME", "Egyetem", "Hírek", "Oktatás"],
                    "emoji_usage": "🎓📚🏫"
                }
            
            if "discord" in platforms:
                result["discord"] = {
                    "embed_title": news_title[:DISCORD_TITLE_MAX],
                    "embed_description": news_content[:DISCORD_DESCRIPTION_MAX],
                    "embed_fields": {
                        "Forrás": validated_source_url,
                        "Típus": "Hír"
                    },
                    "embed_color": DISCORD_DEFAULT_COLOR
                }
            
            logger.info(f"Generated {len(result)} social posts for platforms: {list(result.keys())}")
            
            return {
                "status": "success",
                "posts": result,
                "platforms_generated": list(result.keys()),
                "event_count": len(events)
            }
        
        except KeyError as e:
            logger.error(f"Missing field in template generation: {e}")
            raise ValueError(f"Template generation error: {str(e)}")
    
    except ValueError as e:
        logger.error(f"Validation error in generate_social_posts: {e}")
        return {
            "status": "error",
            "error": f"Validation error: {str(e)}",
            "posts": {},
            "platforms_generated": [],
            "event_count": 0
        }
    except Exception as e:
        logger.exception(f"Unexpected error in generate_social_posts: {e}")
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "posts": {},
            "platforms_generated": [],
            "event_count": 0
        }

# ============================================================================
# SERVER RUN - Lifespan Event Handler (FastAPI 0.93+)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for startup and shutdown"""
    logger.info("=" * 60)
    logger.info("News to Social Media MCP Server starting up")
    logger.info(f"HTML Parser: {HTML_PARSER_CONFIG}")
    logger.info("=" * 60)

    async with mcp.session_manager.run():
        yield

    logger.info("=" * 60)
    logger.info("News to Social Media MCP Server shutting down")
    logger.info("=" * 60)

# FastAPI app MCP SSE transporttal
app = FastAPI(
    title="News to Social Media MCP Server",
    description="MCP szerver FastAPI-val és hivatalos MCP Streamable HTTP transporttal",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/mcp", mcp_app)

@app.get("/health")
async def health():
    """Health check - no auth required"""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "service": "News to Social Media MCP Server",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint - API information"""
    logger.info("Root endpoint accessed")
    return {
        "status": "running",
        "service": "News to Social Media MCP Server",
        "version": "1.0.0",
        "endpoints": {
            "mcp_streamable_http": "http://fastmcp-server:8000/mcp/sse",
            "health": "http://fastmcp-server:8000/health"
        },
        "auth_required": False,
        "api_docs": "http://fastmcp-server:8000/docs"
    }

def run_server():
    """HTTP szerver indítása (szinkron)"""
    # Uvicorn saját event loop-ot kezel; futtatjuk szinkron módon
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    # Indítsuk el a HTTP szervert
    run_server()